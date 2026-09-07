# ------------------------------------------------------------------------------------
# Modified from GeoTransformer (https://github.com/qinzheng93/GeoTransformer)
# Originally authored by Zheng Qin (Copyright (c) 2022 Zheng Qin)
# Licensed under the MIT License.
# Modifications copyright (c) 2026 Qingyang Zhou
# ------------------------------------------------------------------------------------
import torch.nn as nn
import torch
from einops import rearrange
import torch.nn.functional as F

from models.modules.transformer.lrpe_transformer import LRPETransformerLayer
from models.modules.transformer.pe_transformer import PETransformerLayer
from models.modules.transformer.rpe_transformer import RPETransformerLayer
from models.modules.transformer.vanilla_transformer import TransformerLayer
from models.modules.transformer.output_layer import AttentionOutput
from models.modules.layers import build_act_layer, build_dropout_layer

def _check_block_type(block):
    if block not in ['self', 'cross']:
        raise ValueError('Unsupported block type "{}".'.format(block))


class VanillaConditionalTransformer(nn.Module):
    def __init__(self, blocks, d_model, num_heads, dropout=None, activation_fn='ReLU', return_attention_scores=False):
        super(VanillaConditionalTransformer, self).__init__()
        self.blocks = blocks
        layers = []
        for block in self.blocks:
            _check_block_type(block)
            layers.append(TransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn))
        self.layers = nn.ModuleList(layers)
        self.return_attention_scores = return_attention_scores

    def forward(self, feats0, feats1, masks0=None, masks1=None):
        attention_scores = []
        for i, block in enumerate(self.blocks):
            if block == 'self':
                feats0, scores0 = self.layers[i](feats0, feats0, memory_masks=masks0)
                feats1, scores1 = self.layers[i](feats1, feats1, memory_masks=masks1)
            else:
                feats0, scores0 = self.layers[i](feats0, feats1, memory_masks=masks1)
                feats1, scores1 = self.layers[i](feats1, feats0, memory_masks=masks0)
            if self.return_attention_scores:
                attention_scores.append([scores0, scores1])
        if self.return_attention_scores:
            return feats0, feats1, attention_scores
        else:
            return feats0, feats1


class PEConditionalTransformer(nn.Module):
    def __init__(self, blocks, d_model, num_heads, dropout=None, activation_fn='ReLU', return_attention_scores=False):
        super(PEConditionalTransformer, self).__init__()
        self.blocks = blocks
        layers = []
        for block in self.blocks:
            _check_block_type(block)
            if block == 'self':
                layers.append(PETransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn))
            else:
                layers.append(TransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn))
        self.layers = nn.ModuleList(layers)
        self.return_attention_scores = return_attention_scores

    def forward(self, feats0, feats1, embeddings0, embeddings1, masks0=None, masks1=None):
        attention_scores = []
        for i, block in enumerate(self.blocks):
            if block == 'self':
                feats0, scores0 = self.layers[i](feats0, feats0, embeddings0, embeddings0, memory_masks=masks0)
                feats1, scores1 = self.layers[i](feats1, feats1, embeddings1, embeddings1, memory_masks=masks1)
            else:
                feats0, scores0 = self.layers[i](feats0, feats1, memory_masks=masks1)
                feats1, scores1 = self.layers[i](feats1, feats0, memory_masks=masks0)
            if self.return_attention_scores:
                attention_scores.append([scores0, scores1])
        if self.return_attention_scores:
            return feats0, feats1, attention_scores
        else:
            return feats0, feats1


class RPEConditionalTransformer(nn.Module):
    def __init__(
        self,
        blocks,
        d_model,
        num_heads,
        dropout=None,
        activation_fn='ReLU',
        return_attention_scores=False,
        parallel=False,
    ):
        super(RPEConditionalTransformer, self).__init__()
        self.blocks = blocks
        layers = []
        for block in self.blocks:
            _check_block_type(block)
            if block == 'self':
                layers.append(RPETransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn))
            else:
                layers.append(TransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn))
        self.layers = nn.ModuleList(layers)
        self.return_attention_scores = return_attention_scores
        self.parallel = parallel

    def cross_mask(self, mask0, mask1):
        if mask0 is None or mask1 is None:
            return None
        B = mask0.size(0)

        cross_mask = torch.ones(B,1+mask0.size(1)+mask0.size(2),1+mask1.size(1)+mask1.size(2)).bool().to(mask0.device)
        cross_mask[:,1:1+mask0.size(1),1+mask1.size(1):] = False
        cross_mask[:,1+mask1.size(1):,1:1+mask0.size(1)] = False
        return cross_mask
    
    def forward(self, feats0, feats1, embeddings0, embeddings1, masks0=None, masks1=None):
        attention_scores = []
        self_mask0 = masks0  
        self_mask1 = masks1
        cross_mask0 = self.cross_mask(masks0,masks1)
        cross_mask1 = self.cross_mask(masks1,masks0)
        for i, block in enumerate(self.blocks):
            if block == 'self':
                feats0, scores0 = self.layers[i](feats0, feats0, embeddings0, memory_masks=self_mask0)  # ref
                feats1, scores1 = self.layers[i](feats1, feats1, embeddings1, memory_masks=self_mask1)  # src
            else:
                if self.parallel:
                    new_feats0, scores0 = self.layers[i](feats0, feats1, attention_masks=cross_mask0)
                    new_feats1, scores1 = self.layers[i](feats1, feats0, attention_masks=cross_mask1)
                    feats0 = new_feats0
                    feats1 = new_feats1
                else:
                    feats0, scores0 = self.layers[i](feats0, feats1, attention_masks=cross_mask0)
                    feats1, scores1 = self.layers[i](feats1, feats0, attention_masks=cross_mask1)
            if self.return_attention_scores:
                attention_scores.append([scores0, scores1])
        if self.return_attention_scores:
            return feats0, feats1, attention_scores
        else:
            return feats0, feats1

class FineRPEConditionalTransformer(nn.Module):
    def __init__(
        self,
        blocks,
        d_model,
        output_dim,
        num_heads,
        dropout=None,
        activation_fn='ReLU',
        return_attention_scores=False,
        parallel=False,
    ):
        super(FineRPEConditionalTransformer, self).__init__()
        self.blocks = blocks
        layers = []
        for block in self.blocks:
            _check_block_type(block)
            if block == 'self':
                layers.append(RPETransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn))
            else:
                layers.append(TransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn))
        self.layers = nn.ModuleList(layers)
        self.return_attention_scores = return_attention_scores
        self.parallel = parallel

        self.out_proj = nn.Linear(d_model, output_dim)

        self.score_heads = []
        self.nblock = len(blocks)//2
        for _ in range(self.nblock):
            self.score_heads.append(nn.Linear(d_model, 1))
        self.score_heads = nn.ModuleList(self.score_heads)
        self.sigmoid = nn.Sigmoid()

    def forward(self, feats0, feats1, embeddings0, embeddings1, masks0=None, masks1=None):
        attention_list = []
        score_list = []
        saliency_list = []
        for i, block in enumerate(self.blocks):
            if block == 'self':
                feats0, scores0 = self.layers[i](feats0, feats0, embeddings0, memory_masks=masks0)  # ref
                feats1, scores1 = self.layers[i](feats1, feats1, embeddings1, memory_masks=masks1)  # src
            else:
                if self.parallel:
                    new_feats0, scores0 = self.layers[i](feats0, feats1, memory_masks=masks1)
                    new_feats1, scores1 = self.layers[i](feats1, feats0, memory_masks=masks0)
                    feats0 = new_feats0
                    feats1 = new_feats1
                else:
                    feats0, scores0 = self.layers[i](feats0, feats1, memory_masks=masks1)
                    feats1, scores1 = self.layers[i](feats1, feats0, memory_masks=masks0)
            if i%2 == 1:  # Assuming score heads are only used for 'cross' blocks
                idx = i // 2
                scores = self.score_heads[idx](torch.cat((feats0, feats1), dim=1))
                if self.return_attention_scores or self.training or i == len(self.blocks) - 1:
                    # calculate attention scores
                    feats0_proj = self.out_proj(feats0)
                    feats1_proj = self.out_proj(feats1)
                    feats0_proj = F.normalize(feats0_proj, p=2, dim=2)
                    feats1_proj = F.normalize(feats1_proj, p=2, dim=2)
                    atten_mat = feats0_proj @ feats1_proj.transpose(1, 2)
                    atten_mat = atten_mat / 0.1
                    attention_list.append(atten_mat)
                    # calculate score
                    s1, s2 = scores[:, :feats0.shape[1]], scores[:,feats0.shape[1]:]  # bs, n1, 1; bs, n2, 1
                    score = torch.cat((s1, s2), dim=1).squeeze(-1)  # bs, n1+n2
                    score = torch.clamp(self.sigmoid(score), min=0, max=1)
                    score_list.append(score)
                    # calculate saliency
                    m1 = torch.matmul(F.softmax(attention_list[-1][:, :, :], dim=2), s2)  # bs, n1, 1
                    m2 = torch.matmul(F.softmax(attention_list[-1][:, :, :].transpose(1, 2), dim=2), s1)  # bs, n2, 1
                    saliency = torch.cat((m1, m2), dim=1).squeeze(-1)
                    saliency = torch.clamp(self.sigmoid(saliency), min=0, max=1)
                    saliency_list.append(saliency)

        return attention_list, score_list, saliency_list

class LRPEConditionalTransformer(nn.Module):
    def __init__(
        self,
        blocks,
        d_model,
        num_heads,
        num_embeddings,
        dropout=None,
        activation_fn='ReLU',
        return_attention_scores=False,
    ):
        super(LRPEConditionalTransformer, self).__init__()
        self.blocks = blocks
        layers = []
        for block in self.blocks:
            _check_block_type(block)
            if block == 'self':
                layers.append(
                    LRPETransformerLayer(
                        d_model, num_heads, num_embeddings, dropout=dropout, activation_fn=activation_fn
                    )
                )
            else:
                layers.append(TransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn))
        self.layers = nn.ModuleList(layers)
        self.return_attention_scores = return_attention_scores

    def forward(self, feats0, feats1, emb_indices0, emb_indices1, masks0=None, masks1=None):
        attention_scores = []
        for i, block in enumerate(self.blocks):
            if block == 'self':
                feats0, scores0 = self.layers[i](feats0, feats0, emb_indices0, memory_masks=masks0)
                feats1, scores1 = self.layers[i](feats1, feats1, emb_indices1, memory_masks=masks1)
            else:
                feats0, scores0 = self.layers[i](feats0, feats1, memory_masks=masks1)
                feats1, scores1 = self.layers[i](feats1, feats0, memory_masks=masks0)
            if self.return_attention_scores:
                attention_scores.append([scores0, scores1])
        if self.return_attention_scores:
            return feats0, feats1, attention_scores
        else:
            return feats0, feats1

class LinearTransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None, activation_fn='ReLU', focusing_factor=3):
        super(LinearTransformerLayer, self).__init__()
        self.attention = LinearAttentionLayer(d_model, num_heads, dropout=dropout, focusing_factor=focusing_factor)
        self.output = AttentionOutput(d_model, dropout=dropout, activation_fn=activation_fn)

    def forward(
        self,
        input_states,
        memory_states
    ):
        hidden_states = self.attention(
            input_states,
            memory_states
        )
        output_states = self.output(hidden_states)
        return output_states
    
class LinearAttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=False, focusing_factor=3):
        super(LinearAttentionLayer, self).__init__()
        self.attention = LinearAttention(d_model, num_heads, focusing_factor=focusing_factor)
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = build_dropout_layer(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        input_states,
        memory_states,
    ):
        hidden_states= self.attention(
            input_states,
            memory_states,
            memory_states,
        )
        hidden_states = self.linear(hidden_states)
        hidden_states = self.dropout(hidden_states)
        output_states = self.norm(hidden_states + input_states)
        return output_states
    
class LinearAttention(nn.Module):
    def __init__(self, d_model, num_heads, focusing_factor=3):
        super(LinearAttention, self).__init__()
        if d_model % num_heads != 0:
            raise ValueError('`d_model` ({}) must be a multiple of `num_heads` ({}).'.format(d_model, num_heads))
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_model_per_head = d_model // num_heads
        self.focusing_factor = focusing_factor
        self.kernel_function = nn.ReLU()

        self.proj_q = nn.Linear(self.d_model, self.d_model)
        self.proj_k = nn.Linear(self.d_model, self.d_model)
        self.proj_v = nn.Linear(self.d_model, self.d_model)
        self.scale = nn.Parameter(torch.zeros(size=(1, 1, self.d_model)))

    def forward(self, input_q, input_k, input_v):

        q = self.proj_q(input_q)
        k = self.proj_k(input_k)
        v = self.proj_v(input_v)
        scale = nn.Softplus()(self.scale)

        q = self.kernel_function(q) + 1e-6
        k = self.kernel_function(k) + 1e-6
        q = q / scale
        k = k / scale
        q_norm = q.norm(dim=-1, keepdim=True)
        k_norm = k.norm(dim=-1, keepdim=True)
        q = q ** self.focusing_factor
        k = k ** self.focusing_factor
        q = (q / q.norm(dim=-1, keepdim=True)) * q_norm
        k = (k / k.norm(dim=-1, keepdim=True)) * k_norm

        q, k, v = (rearrange(x, "b n (h c) -> (b h) n c", h=self.num_heads) for x in [q, k, v])
        i, j, c, d = q.shape[-2], k.shape[-2], k.shape[-1], v.shape[-1]

        z = 1 / (torch.einsum("b i c, b c -> b i", q, k.sum(dim=1)) + 1e-6)
        if i * j * (c + d) > c * d * (i + j):
            kv = torch.einsum("b j c, b j d -> b c d", k, v)
            x = torch.einsum("b i c, b c d, b i -> b i d", q, kv, z)
        else:
            qk = torch.einsum("b i c, b j c -> b i j", q, k)
            x = torch.einsum("b i j, b j d, b i -> b i d", qk, v, z)
        x = rearrange(x, "(b h) n c -> b n (h c)", h=self.num_heads)

        return x