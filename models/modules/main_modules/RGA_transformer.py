# ------------------------------------------------------------------------------------
# Modified from GeoTransformer (https://github.com/qinzheng93/GeoTransformer)
# Originally authored by Zheng Qin (Copyright (c) 2022 Zheng Qin)
# Licensed under the MIT License.
# Modifications copyright (c) 2026 Qingyang Zhou
# ------------------------------------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn

from models.modules.ops import pairwise_distance
from models.modules.transformer import (
    SinusoidalPositionalEmbedding, 
    RPEConditionalTransformer, 
    FineRPEConditionalTransformer,
    FourierEmbedding, 
    AttentionLayer,
    RPEAttentionLayer,
)

class RGSEmbedding(nn.Module):
    def __init__(self, hidden_dim):
        super(RGSEmbedding, self).__init__()

        num_freq = 8
        self.fourier = FourierEmbedding(num_freq)
        self.proj = nn.Linear(num_freq * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, points):
        dist = torch.sqrt(pairwise_distance(points, points) + 1e-6)  # (B, N, N)
        dist = dist/2.0  # scale
        dist = dist.unsqueeze(-1)  # (B,R,R,1)
        emb = self.fourier(dist)   # (B,R,R,2*num_freq)
        emb = self.proj(emb)
        emb = self.norm(emb)
        emb = emb*0.1
        return emb  # (B,R,R,D)     
    
class GeometricStructureEmbedding(nn.Module):
    def __init__(self, hidden_dim, sigma_d, sigma_a, angle_k, reduction_a='max'):
        super(GeometricStructureEmbedding, self).__init__()
        self.sigma_d = sigma_d
        self.sigma_a = sigma_a
        self.factor_a = 180.0 / (self.sigma_a * np.pi)
        self.angle_k = angle_k

        self.embedding = SinusoidalPositionalEmbedding(hidden_dim)
        self.proj_d = nn.Linear(hidden_dim, hidden_dim)
        self.proj_a = nn.Linear(hidden_dim, hidden_dim)

        self.reduction_a = reduction_a
        if self.reduction_a not in ['max', 'mean']:
            raise ValueError(f'Unsupported reduction mode: {self.reduction_a}.')

    @torch.no_grad()
    def get_embedding_indices(self, points,if_ankle=True):
        r"""Compute the indices of pair-wise distance embedding and triplet-wise angular embedding.

        Args:
            points: torch.Tensor (B, N, 3), input point cloud

        Returns:
            d_indices: torch.FloatTensor (B, N, N), distance embedding indices
            a_indices: torch.FloatTensor (B, N, N, k), angular embedding indices
        """
        batch_size, num_point, _ = points.shape

        dist_map = torch.sqrt(pairwise_distance(points, points))  # (B, N, N)
        d_indices = dist_map / self.sigma_d

        if not if_ankle:
            return d_indices, None
        else:
            k = self.angle_k
            knn_indices = dist_map.topk(k=k + 1, dim=2, largest=False)[1][:, :, 1:]  # (B, N, k) 
            knn_indices = knn_indices.unsqueeze(3).expand(batch_size, num_point, k, 3)  # (B, N, k, 3)
            expanded_points = points.unsqueeze(1).expand(batch_size, num_point, num_point, 3)  # (B, N, N, 3)
            knn_points = torch.gather(expanded_points, dim=2, index=knn_indices)  # (B, N, k, 3)
            ref_vectors = knn_points - points.unsqueeze(2)  # (B, N, k, 3) 
            anc_vectors = points.unsqueeze(1) - points.unsqueeze(2)  # (B, N, N, 3) 
            ref_vectors = ref_vectors.unsqueeze(2).expand(batch_size, num_point, num_point, k, 3)  # (B, N, N, k, 3)
            anc_vectors = anc_vectors.unsqueeze(3).expand(batch_size, num_point, num_point, k, 3)  # (B, N, N, k, 3)
            sin_values = torch.linalg.norm(torch.cross(ref_vectors, anc_vectors, dim=-1), dim=-1)  # (B, N, N, k) 
            cos_values = torch.sum(ref_vectors * anc_vectors, dim=-1)  # (B, N, N, k)
            angles = torch.atan2(sin_values, cos_values)  # (B, N, N, k) 
            a_indices = angles * self.factor_a

            return d_indices, a_indices

    def forward(self, points,if_region = False, if_ankle=True):
        if if_region:
            self.sigma_d=self.sigma_d*2
            if_ankle=False

        d_indices, a_indices = self.get_embedding_indices(points,if_ankle=if_ankle)  # (B, N, N), (B, N, N, k)

        d_embeddings = self.embedding(d_indices)  # (B, N, N, d_model).detach()
        d_embeddings = self.proj_d(d_embeddings)  

        if if_ankle:
            a_embeddings = self.embedding(a_indices)
            a_embeddings = self.proj_a(a_embeddings)  
            if self.reduction_a == 'max':
                a_embeddings = a_embeddings.max(dim=3)[0]
            else:
                a_embeddings = a_embeddings.mean(dim=3)

            embeddings = d_embeddings + a_embeddings
        else:
            embeddings = d_embeddings

        return embeddings  #(B,N,N,d_model)


class RGA_transformer(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        num_heads,
        nblocks,
        sigma_d,
        sigma_a,
        angle_k,
        dropout=None,
        focusing_factor=3,
        activation_fn='ReLU',
        reduction_a='max',
    ):
        super(RGA_transformer, self).__init__()

        self.d_model = hidden_dim
        self.embedding = GeometricStructureEmbedding(hidden_dim, sigma_d, sigma_a, angle_k, reduction_a=reduction_a)
        self.RGSseed_embedding = RGSEmbedding(hidden_dim)

        self.in_proj = nn.Linear(input_dim, hidden_dim)

        self.region_transformers = []
        for _ in range(2):
            self.region_transformers.append(RGSTransformer(
                d_model=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
        )
        self.region_transformers = nn.ModuleList(self.region_transformers)
        
        self.transformers = []
        self.nblocks = nblocks
        for _ in range(self.nblocks):
            self.transformers.append(RPEConditionalTransformer(
                blocks=['self', 'cross'],
                d_model=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                activation_fn=activation_fn,
                parallel=False,
                return_attention_scores=False,
            )
        )
        self.transformers = nn.ModuleList(self.transformers)

        self.out_proj1 = nn.Linear(hidden_dim, output_dim)
        self.out_proj2 = nn.Linear(hidden_dim, output_dim)

    def attention_mask(self, region_points, points,region_idx):
        B = points.size(0)
        R = region_points.size(1)  # BG:BG+R
        N = points.size(1)  # BG+R:
        region_mask = torch.zeros(B,R,N+1).bool().to(points.device)  
        region_mask[torch.arange(B, device=points.device)[:, None, None].expand(B, R, region_idx.size(2)).to(points.device), 
                    torch.arange(R, device=points.device)[None, :, None].expand(B, R, region_idx.size(2)).to(points.device),
                    region_idx
                    ] = True
        region_mask = region_mask[:, :, :-1]   # (B,R,N)
        return region_mask

    def RGSseed(self,raw_point_feats,region_mask):
        point_feats = raw_point_feats[:,None,:,:].expand(-1,region_mask.size(1),-1,-1)
        region_feats = point_feats*region_mask[:,:,:,None].float()

        RGSseed = region_feats.sum(dim=2) / (region_mask.sum(dim=2,keepdim=True)+1e-6)
        return RGSseed
    def forward(
        self,
        ref_points,
        src_points,
        ref_feats,
        src_feats,
        ref_c_region_points,  # (B, N, 3)
        src_c_region_points,  # (B, M, 3)
        ref_c_region_idx,
        src_c_region_idx,
        ref_masks=None,
        src_masks=None,
    ):
        B = ref_points.size(0)
        R_ref = ref_c_region_points.size(1)
        R_src = src_c_region_points.size(1)

        ref_c_region_mask = self.attention_mask(ref_c_region_points, ref_points, ref_c_region_idx)  # (B,R_ref,N-ref)
        src_c_region_mask = self.attention_mask(src_c_region_points, src_points, src_c_region_idx)
        ref_embeddings = self.embedding(ref_points)  # (B,N,N,d_model)
        src_embeddings = self.embedding(src_points)  # (B,N,N,d_model)

        ref_feats = self.in_proj(ref_feats)
        src_feats = self.in_proj(src_feats)
        ref_RGSseed = self.RGSseed(ref_feats.detach(), ref_c_region_mask)
        src_RGSseed = self.RGSseed(src_feats.detach(), src_c_region_mask)
        ref_RGSseed_embeddings = self.RGSseed_embedding(ref_c_region_points)  # (B,N,N,d_model)
        src_RGSseed_embeddings = self.RGSseed_embedding(src_c_region_points)  # (B,N,N,d_model)

        for idx in range(self.nblocks):
            ref_feats, src_feats = self.transformers[idx](ref_feats, src_feats, ref_embeddings, src_embeddings)
            
            if idx==0 or idx==1:
                ref_RGSseed, src_RGSseed, ref_feats, src_feats = self.region_transformers[idx](ref_RGSseed, src_RGSseed, ref_feats, src_feats, ref_RGSseed_embeddings,
                    src_RGSseed_embeddings,ref_c_region_mask, src_c_region_mask)

        ref_feats = self.out_proj1(ref_feats)
        src_feats = self.out_proj1(src_feats)
        ref_RGSseed = self.out_proj2(ref_RGSseed)
        src_RGSseed = self.out_proj2(src_RGSseed)

        return ref_feats, src_feats, ref_RGSseed, src_RGSseed, ref_c_region_mask, src_c_region_mask 
    
class RGSTransformer(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        dropout=None,
    ):
        super(RGSTransformer, self).__init__()
        layers = []
        layers.append(AttentionLayer(d_model=d_model, num_heads=num_heads, dropout=dropout))
        layers.append(RPEAttentionLayer(d_model=d_model, num_heads=num_heads, dropout=dropout))
        layers.append(AttentionLayer(d_model=d_model, num_heads=num_heads, dropout=dropout))
        layers.append(AttentionLayer(d_model=d_model, num_heads=num_heads, dropout=dropout, learnable_residual_scale=True))

        self.layers = nn.ModuleList(layers)

    def forward(self, ref_RGSseed, src_RGSseed, ref_feats, src_feats, ref_RGSseed_embeddings,
                src_RGSseed_embeddings,ref_c_region_mask, src_c_region_mask):
        ref_RGSseed,_ = self.layers[0](ref_RGSseed, ref_feats.detach(),
                                            attention_factors=None, 
                                            attention_masks=ref_c_region_mask) 
        src_RGSseed,_ = self.layers[0](src_RGSseed, src_feats.detach(), 
                                            attention_factors=None, 
                                            attention_masks=src_c_region_mask)

        ref_RGSseed,_ = self.layers[1](ref_RGSseed, ref_RGSseed, 
                                            ref_RGSseed_embeddings)
        src_RGSseed,_ = self.layers[1](src_RGSseed, src_RGSseed, 
                                            src_RGSseed_embeddings)
        ref_RGSseed,_ = self.layers[2](ref_RGSseed, src_RGSseed, 
                                            attention_factors=None)
        src_RGSseed,_ = self.layers[2](src_RGSseed, ref_RGSseed, 
                                            attention_factors=None)

        ref_feats,_ = self.layers[3](ref_feats, ref_RGSseed, 
                                    attention_factors=None,
                                    attention_masks=ref_c_region_mask.transpose(1,2))
        src_feats,_ = self.layers[3](src_feats, src_RGSseed, 
                                    attention_factors=None,
                                    attention_masks=src_c_region_mask.transpose(1,2))
        return ref_RGSseed, src_RGSseed, ref_feats, src_feats
    
class FineRGA_transformer(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        num_heads,
        blocks,
        sigma_d,
        sigma_a,
        angle_k,
        dropout=None,
        activation_fn='ReLU',
        reduction_a='max',
    ):
        super(FineRGA_transformer, self).__init__()

        self.embedding = GeometricStructureEmbedding(hidden_dim, sigma_d*0.25, sigma_a*2/3, angle_k, reduction_a=reduction_a)
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.transformer = FineRPEConditionalTransformer(
            blocks, hidden_dim, output_dim, num_heads, dropout=dropout, activation_fn=activation_fn
        )

    def forward(
        self,
        ref_points,
        src_points,
        ref_feats,
        src_feats,
        ref_masks=None,
        src_masks=None,
    ):
        ref_embeddings = self.embedding(ref_points)  # (B,N,N,d_model)
        src_embeddings = self.embedding(src_points)  # (B,N,N,d_model)

        ref_feats = self.in_proj(ref_feats)
        src_feats = self.in_proj(src_feats)
        
        attention_list, score_list, saliency_list = self.transformer(
            ref_feats,  # (B,N,hidden)
            src_feats,  # (B,N,hidden)
            ref_embeddings,  # (B,N,N,d_model)
            src_embeddings,  # (B,N,N,d_model)
            masks0=ref_masks,  # None
            masks1=src_masks,  # None
        )

        return attention_list, score_list, saliency_list