# ------------------------------------------------------------------------------------
# Modified from GeoTransformer (https://github.com/qinzheng93/GeoTransformer)
# Originally authored by Zheng Qin (Copyright (c) 2022 Zheng Qin)
# Licensed under the MIT License.
# Modifications copyright (c) 2026 Qingyang Zhou
# ------------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.utils.torch import to_cuda
from models.utils.data import precompute_data_stack_mode
from models.modules.ops import point_to_node_partition, index_select
from models.modules.registration import get_node_correspondences,get_region_correspondences,WeightedProcrustes
from models.modules.ops import apply_transform, inverse_transform
from models.utils.pointcloud import random_sample_transform
from models.modules.main_modules import (
    RGA_transformer,
    FineRGA_transformer,
    SuperPointTargetGenerator,
)
from models.modules.FPS import FPS
from models.outlierrejection import (
    OutlierRejection,
    RegionOutlierRejection,
)

from backbone import KPConvFPN
from pointnet2_ops import pointnet2_utils

class RGA(nn.Module):
    def __init__(self, cfg):
        super(RGA, self).__init__()
        self.num_points_in_patch = cfg.model.num_points_in_patch 
        self.matching_radius = cfg.model.ground_truth_matching_radius
        self.neighbor_limits = cfg.backbone.neighbor_limits

        self.backbone = KPConvFPN(
            cfg.backbone.input_dim,
            cfg.backbone.output_dim,
            cfg.backbone.init_dim,
            cfg.backbone.kernel_size,
            cfg.backbone.init_radius,
            cfg.backbone.init_sigma,
            cfg.backbone.group_norm,
        )

        self.transformer = RGA_transformer(
            cfg.transformer.input_dim,
            cfg.transformer.output_dim,
            cfg.transformer.hidden_dim,
            cfg.transformer.num_heads,
            cfg.transformer.nblocks,
            cfg.transformer.sigma_d,
            cfg.transformer.sigma_a,
            cfg.transformer.angle_k,
            focusing_factor=cfg.transformer.focusing_factor,
            reduction_a=cfg.transformer.reduction_a,
        )

        self.fine_transformer = FineRGA_transformer(
            cfg.transformer.output_dim,
            cfg.transformer.output_dim,
            cfg.transformer.hidden_dim,
            cfg.transformer.num_heads,
            cfg.transformer.blocks,
            cfg.transformer.sigma_d,
            cfg.transformer.sigma_a,
            cfg.transformer.angle_k,
            reduction_a=cfg.transformer.reduction_a,
        )

        self.fps_radius = cfg.model.fps_radius
        self.fps_ngroup = cfg.model.fps_ngroup
        self.fps_max_numsamples = cfg.model.num_points_in_region
        self.region_seg = FPS(fps_ngroup = self.fps_ngroup, fps_radius = self.fps_radius, fps_max_numsamples = self.fps_max_numsamples)
        self.outlier_rejection = OutlierRejection(
            num_correspondences=cfg.coarse_matching.num_correspondences,
            local_topk = cfg.outlierrejection.local_topk,
            first_num_correspondences = cfg.coarse_matching.first_num_correspondences,
            dual_normalization = cfg.coarse_matching.dual_normalization,
            pcl_resolution = cfg.backbone.init_voxel_size,
        )
        self.region_outlier_rejection = RegionOutlierRejection(
            pcl_resolution = cfg.backbone.init_voxel_size,
        )

        self.coarse_target = SuperPointTargetGenerator(
            cfg.coarse_matching.num_targets, cfg.coarse_matching.overlap_threshold
        ) 

        self.procrustes = WeightedProcrustes(return_transform=True)

        

    def forward_coarse(self, data_dict, output_dict):
        # Downsample point clouds
        feats = data_dict['features'].detach()  
        transform = data_dict['transform'].detach() 

        ref_length_c = data_dict['lengths'][-1][0].item() 
        src_length_c = data_dict['lengths'][-1][-1].item() 
        ref_length_f = data_dict['lengths'][0][0].item() 
        ref_length = data_dict['lengths'][0][0].item() 
        points_c = data_dict['points'][-1].detach() 
        points_f = data_dict['points'][0].detach() 
        points = data_dict['points'][0].detach()  

        ref_points_c = points_c[:ref_length_c] 
        src_points_c = points_c[ref_length_c:]  
        ref_points_f = points_f[:ref_length_f]  
        src_points_f = points_f[ref_length_f:]  
        ref_points = points[:ref_length] 
        src_points = points[ref_length:]  

        output_dict['ref_points_c'] = ref_points_c 
        output_dict['src_points_c'] = src_points_c 
        output_dict['ref_points_f'] = ref_points_f
        output_dict['src_points_f'] = src_points_f
        output_dict['ref_points'] = ref_points
        output_dict['src_points'] = src_points 

        # 1. Generate ground truth node correspondences
        _, ref_node_masks, ref_node_knn_indices, ref_node_knn_masks = point_to_node_partition(
            ref_points_f, ref_points_c, self.num_points_in_patch
        ) 
        _, src_node_masks, src_node_knn_indices, src_node_knn_masks = point_to_node_partition(
            src_points_f, src_points_c, self.num_points_in_patch
        )

        ref_padded_points_f = torch.cat([ref_points_f, torch.zeros_like(ref_points_f[:1])], dim=0) 
        src_padded_points_f = torch.cat([src_points_f, torch.zeros_like(src_points_f[:1])], dim=0)
        ref_node_knn_points = index_select(ref_padded_points_f, ref_node_knn_indices, dim=0)  # (_c, K, 3)
        src_node_knn_points = index_select(src_padded_points_f, src_node_knn_indices, dim=0)  # (_c, K, 3)

        gt_node_corr_indices, gt_node_corr_overlaps = get_node_correspondences(
            ref_points_c,  # tensor(_c,3)
            src_points_c,  # tensor(_c,3)
            ref_node_knn_points,  # (_c, K, 3) 
            src_node_knn_points,  # (_c, K, 3) 
            transform,
            self.matching_radius,
            ref_masks=ref_node_masks,  # (_c,) 
            src_masks=src_node_masks,  # (_c,) 
            ref_knn_masks=ref_node_knn_masks,  # (_c, K)
            src_knn_masks=src_node_knn_masks,  # (_c, K) 
        )
        output_dict['gt_node_corr_indices'] = gt_node_corr_indices  
        output_dict['gt_node_corr_overlaps'] = gt_node_corr_overlaps 

        # 2. KPFCNN Encoder
        feats_list = self.backbone(feats, data_dict)
        feats_c = feats_list[-1]  # latent_s3

        # 3. Conditional Transformer
        ref_feats_c = feats_c[:ref_length_c]
        src_feats_c = feats_c[ref_length_c:]
        with torch.no_grad():
            # FPS
            _, ref_c_center_pos, ref_c_region_idx, _ = self.region_seg(ref_points_c.unsqueeze(0),ref_points_f.unsqueeze(0))
            _, src_c_center_pos, src_c_region_idx, _ = self.region_seg(src_points_c.unsqueeze(0),src_points_f.unsqueeze(0))

        ref_feats_c, src_feats_c, ref_region_token, src_region_token, ref_c_region_mask, src_c_region_mask = self.transformer(
            ref_points_c.unsqueeze(0), 
            src_points_c.unsqueeze(0),  
            ref_feats_c.unsqueeze(0), 
            src_feats_c.unsqueeze(0),  
            ref_c_center_pos.unsqueeze(0),
            src_c_center_pos.unsqueeze(0),
            ref_c_region_idx.unsqueeze(0),
            src_c_region_idx.unsqueeze(0),
        )  # (B,N,hidden)(B,M,hidden)
        ref_feats_c_norm = F.normalize(ref_feats_c.squeeze(0), p=2, dim=1)  # L2
        src_feats_c_norm = F.normalize(src_feats_c.squeeze(0), p=2, dim=1)  # L2
        ref_region_token_norm = F.normalize(ref_region_token.squeeze(0), p=2, dim=1)  # L2
        src_region_token_norm = F.normalize(src_region_token.squeeze(0), p=2, dim=1)  # L2
        output_dict['ref_region_token'] = ref_region_token_norm  # loss
        output_dict['src_region_token'] = src_region_token_norm  # loss
        output_dict['ref_feats_c'] = ref_feats_c_norm  # loss
        output_dict['src_feats_c'] = src_feats_c_norm  # loss
        
        # GT region matching
        gt_region_indices, gt_region_overlaps = get_region_correspondences(ref_c_region_idx,src_c_region_idx,gt_node_corr_indices, ref_c_region_mask, src_c_region_mask)
        output_dict['gt_region_indices'] = gt_region_indices # loss
        output_dict['gt_region_overlaps'] = gt_region_overlaps # loss

        with torch.no_grad():
            if not self.training:
                # 6. Select topk nearest node correspondences   and use RCCR to select
                all_region_ref_corr_indices, all_region_src_corr_indices, region_to_corr_indices, region_to_corr_mask, all_region_to_corr_scores, corr_scores, corr_ref_pts, corr_src_pts = self.outlier_rejection(ref_feats_c_norm, src_feats_c_norm, ref_points_c, 
                                                                                                        src_points_c, ref_masks=ref_node_masks, src_masks=src_node_masks, 
                                                                                                        ref_region_idx=ref_c_region_idx, ref_c_region_mask=ref_c_region_mask)
                if all_region_ref_corr_indices is None and all_region_src_corr_indices is None and region_to_corr_indices is None and region_to_corr_mask is None and all_region_to_corr_scores is None and corr_scores is None: 
                    raise ValueError("cannot find correspondences of the REGION")
                
                # 7. choose which regions can be together
                ref_node_corr_indices, src_node_corr_indices, node_corr_scores, ref_centroid_pts, src_centroid_pts = self.region_outlier_rejection(corr_ref_pts,
                                                                corr_src_pts,
                                                                region_to_corr_indices, 
                                                                region_to_corr_mask,  # (R_ref'',C)mask
                                                                all_region_to_corr_scores,
                                                                all_region_ref_corr_indices,  # (C,)ref-idx
                                                                all_region_src_corr_indices,  # (C,)src-idx
                                                                corr_scores,  # (C)feats-scores
                                                                )
                output_dict['ref_node_corr_indices'] = ref_node_corr_indices  
                output_dict['src_node_corr_indices'] = src_node_corr_indices 
                output_dict['ref_centroid_pts'] = ref_centroid_pts
                output_dict['src_centroid_pts'] = src_centroid_pts

                if ref_node_corr_indices.shape[0]!=0:
                    for i in range(ref_node_corr_indices.size(0)):
                        group_ref_node_corr_indices = ref_node_corr_indices[i,:]  
                        group_ref_node_corr_indices = group_ref_node_corr_indices[group_ref_node_corr_indices!=-1]
                        group_src_node_corr_indices = src_node_corr_indices[i,:]  
                        group_src_node_corr_indices = group_src_node_corr_indices[group_src_node_corr_indices!=-1]
                        group_node_corr_scores = node_corr_scores[i,:]  
                        group_node_corr_scores = group_node_corr_scores[group_node_corr_scores!=0]

                        # using _c-ref&_c-src to weighted SVD
                        group_src_corr_points = index_select(src_points_c, group_src_node_corr_indices, dim=0)
                        group_ref_corr_points = index_select(ref_points_c, group_ref_node_corr_indices, dim=0)
                        group_estimated_transform = self.procrustes(group_src_corr_points, group_ref_corr_points, group_node_corr_scores)

                        if i==0:
                            estimated_transform = group_estimated_transform.unsqueeze(0)
                            ref_corr_points = group_ref_corr_points
                            src_corr_points = group_src_corr_points
                            corr_scores = group_node_corr_scores
                        else:
                            estimated_transform = torch.cat([estimated_transform, group_estimated_transform.unsqueeze(0)], dim=0)
                            ref_corr_points = torch.cat([ref_corr_points, group_ref_corr_points], dim=0)
                            src_corr_points = torch.cat([src_corr_points, group_src_corr_points], dim=0)
                            corr_scores = torch.cat([corr_scores, group_node_corr_scores], dim=0)
                            
                    output_dict['coarse_est_transform'] = estimated_transform
                else:
                    output_dict['coarse_est_transform'] = torch.eye(4).unsqueeze(0).cuda()
                    output_dict['ref_centroid_pts'] = ref_c_center_pos.unsqueeze(0) 
                    output_dict['src_centroid_pts'] = src_c_center_pos.unsqueeze(0)  
            else:
                # 8 Random select ground truth node correspondences during training
                ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_target(
                    gt_node_corr_indices, gt_node_corr_overlaps
                )  # gt_ref.topk.index: tensor(num_corre) gt_src.topk.index: tensor(num_corre) gt_scores: tensor(num_corre)
                output_dict['ref_node_corr_indices'] = ref_node_corr_indices  # val
                output_dict['src_node_corr_indices'] = src_node_corr_indices  # val

                if gt_region_indices.size(0)==0:
                    num_to_select_neg = 1
                    gt_region_indices_set = set(map(tuple, gt_region_indices.tolist()))
                    selected_neg_region_pair = []
                    while len(selected_neg_region_pair)<num_to_select_neg:
                        pair = (torch.randint(0, gt_region_overlaps.shape[0], (1,)).item(), 
                                torch.randint(0, gt_region_overlaps.shape[1], (1,)).item())
                        if pair not in gt_region_indices_set and pair not in selected_neg_region_pair:
                            selected_region_nsample_ref = ref_c_region_idx[pair[0],:] 
                            selected_region_nsample_src = src_c_region_idx[pair[1],:]
                            if selected_region_nsample_ref[selected_region_nsample_ref!=-1].unique().numel()>3 and selected_region_nsample_src[selected_region_nsample_src!=-1].unique().numel()>3:
                                selected_neg_region_pair.append(pair)
                    selected_neg_region_pair = torch.tensor(selected_neg_region_pair).cuda() 
                    selected_pair_fine_input = selected_neg_region_pair     
                else:
                    num_to_select_neg = 1
                    gt_region_indices_set = set(map(tuple, gt_region_indices.tolist()))
                    selected_neg_region_pair = []
                    while len(selected_neg_region_pair)<num_to_select_neg:
                        pair = (torch.randint(0, gt_region_overlaps.shape[0], (1,)).item(), 
                                torch.randint(0, gt_region_overlaps.shape[1], (1,)).item())
                        if pair not in gt_region_indices_set and pair not in selected_neg_region_pair:
                            selected_region_nsample_ref = ref_c_region_idx[pair[0],:] 
                            selected_region_nsample_src = src_c_region_idx[pair[1],:]
                            if selected_region_nsample_ref[selected_region_nsample_ref!=-1].unique().numel()>3 and selected_region_nsample_src[selected_region_nsample_src!=-1].unique().numel()>3:
                                selected_neg_region_pair.append(pair)
                    selected_neg_region_pair = torch.tensor(selected_neg_region_pair).cuda()  

                    num_to_select_pos = min(gt_region_indices.size(0),2)
                    bool_flag = True
                    count = 0
                    while (bool_flag and count<10):
                        count+=1
                        indices_pos = torch.randint(0, gt_region_indices.size(0), (num_to_select_pos,)).cuda()
                        # selected_ref_c_region_idx = ref_c_region_idx[indices_pos]
                        bool_flag = False
                        for row in range(indices_pos.size(0)):
                            selected_region_nsample_ref = ref_c_region_idx[gt_region_indices[indices_pos[row],0],:]
                            selected_region_nsample_src = src_c_region_idx[gt_region_indices[indices_pos[row],1],:]
                            if selected_region_nsample_ref[selected_region_nsample_ref!=-1].unique().numel()<=3:
                                bool_flag = True
                                break
                            if selected_region_nsample_src[selected_region_nsample_src!=-1].unique().numel()<=3:
                                bool_flag = True
                                break
                        if bool_flag==False:
                            selected_gt_region_pair = gt_region_indices[indices_pos]  
                    if not bool_flag:
                        selected_pair_fine_input = torch.cat([selected_gt_region_pair, selected_neg_region_pair], dim=0)  
                    else:
                        selected_pair_fine_input = selected_neg_region_pair  
                ref_centroid_pts = ref_c_center_pos[selected_pair_fine_input[:,0]] 
                src_centroid_pts = src_c_center_pos[selected_pair_fine_input[:,1]] 
                output_dict['ref_centroid_pts'] = ref_centroid_pts.unsqueeze(1)
                output_dict['src_centroid_pts'] = src_centroid_pts.unsqueeze(1)

                self.rotation_magnitude=35
                self.translation_magnitude=0.2
                estimated_transform = torch.eye(4).unsqueeze(0).repeat(ref_centroid_pts.size(0),1,1).cuda() 
                for i in range(ref_centroid_pts.size(0)):
                    aug_transform = random_sample_transform(self.rotation_magnitude, self.translation_magnitude)   
                    aug_transform = torch.tensor(aug_transform).cuda().float()
                    inv_aug_transform = inverse_transform(aug_transform) 
                    inv_aug_transform = inv_aug_transform.cuda().float()
                    estimated_transform[i,:,:] = inv_aug_transform @ transform 
                output_dict['coarse_est_transform'] = estimated_transform
        
        return output_dict
    
    def forward_fine(self, data_dict, output_dict, group_id):
        ref_centroid_pts = output_dict['ref_centroid_pts'].detach()
        src_centroid_pts = output_dict['src_centroid_pts'].detach()
        init_transform = output_dict['coarse_est_transform'].detach()
        transform = data_dict['transform'].detach()  

        feats = data_dict['features'].detach() 

        points_f = data_dict['points'][0].detach()  
        ref_length_f = data_dict['lengths'][0][0].item() 
        ref_points_f = points_f[:ref_length_f]  
        src_points_f = points_f[ref_length_f:]  

        group_init_T = init_transform[group_id] 
        group_src_points_f = apply_transform(src_points_f, group_init_T)
        group_ref_points_f = ref_points_f

        points_list_precompute = [group_ref_points_f,group_src_points_f]
        lengths_precompute = torch.LongTensor([points.shape[0] for points in points_list_precompute])  
        points_precompute = torch.cat(points_list_precompute, dim=0)  
        input_dict = {}
        input_dict = precompute_data_stack_mode(points_precompute, lengths_precompute, self.neighbor_limits)
        input_dict = to_cuda(input_dict)

        feats_list = self.backbone(feats, input_dict)
        feats_f = feats_list[0]  # latent_s1
        ref_feats_f = feats_f[:ref_length_f]
        src_feats_f = feats_f[ref_length_f:]

        group_ref_centroid_pts = ref_centroid_pts[group_id]
        group_ref_centroid_pts = group_ref_centroid_pts[group_ref_centroid_pts!=-100].reshape(1,-1,3)
        ref_f_idx = pointnet2_utils.ball_query(self.fps_radius*2, 400//group_ref_centroid_pts.shape[1], group_ref_points_f.unsqueeze(0), group_ref_centroid_pts)  
        ref_f_idx = ref_f_idx.view(-1).unique()  
        if ref_f_idx.shape[0]<=3:
            return output_dict, False
        group_ref_f_pts = index_select(group_ref_points_f, ref_f_idx, dim=0)
        group_ref_f_feats = index_select(ref_feats_f, ref_f_idx, dim=0)

        group_src_centroid_pts = src_centroid_pts[group_id]
        group_src_centroid_pts = group_src_centroid_pts[group_src_centroid_pts!=-100].reshape(1,-1,3)
        src_f_idx = pointnet2_utils.ball_query(self.fps_radius*2, 400//group_src_centroid_pts.shape[1], src_points_f.unsqueeze(0), group_src_centroid_pts)  
        src_f_idx = src_f_idx.view(-1).unique() 
        if src_f_idx.shape[0]<=3:
            return output_dict, False
        group_src_f_pts = index_select(group_src_points_f, src_f_idx, dim=0) 
        group_src_f_feats = index_select(src_feats_f, src_f_idx, dim=0)

        group_attention_list, group_score_list, group_saliency_list = self.fine_transformer(group_ref_f_pts.unsqueeze(0),
                                group_src_f_pts.unsqueeze(0),
                                group_ref_f_feats.unsqueeze(0),
                                group_src_f_feats.unsqueeze(0),
                                )
        for key in ['fine_score_list', 'fine_saliency_list', 'fine_attention_list']:
            if key not in output_dict:
                output_dict[key] = []
        output_dict['fine_score_list'].append(group_score_list)
        output_dict['fine_saliency_list'].append(group_saliency_list)
        output_dict['fine_attention_list'].append(group_attention_list)
            
        with torch.no_grad():
            group_attention = group_attention_list[-1].float()  # (1,N,M)
            group_assginment_mat = torch.softmax(group_attention, dim=2) * torch.softmax(group_attention, dim=1)
            group_ref_score = group_score_list[-1][:,:group_ref_f_pts.shape[0]].float()  # (1,N)
            group_src_score = group_score_list[-1][:,group_ref_f_pts.shape[0]:].float()  # (1,M)
            group_ref_score = group_ref_score[:, :, None].repeat(1, 1, group_src_f_pts.shape[0])  # (1,N,M)
            group_src_score = group_src_score[:, None, :].repeat(1, group_ref_f_pts.shape[0], 1)  # (1,N,M)
            group_assginment_mat = group_assginment_mat * group_ref_score * group_src_score
            # compute pose
            label1 = torch.max(group_assginment_mat[:, :, :], dim=2)[1]
            label2 = torch.max(group_assginment_mat[:, :, :], dim=1)[1]
            group_assginment_mat = group_assginment_mat[:, :, :] * (label1 > 0).float().unsqueeze(2) * (label2 > 0).float().unsqueeze(1)
            group_normalized_assginment_mat = group_assginment_mat / (group_assginment_mat.sum(2, keepdim=True) + 1e-6)
            group_pred_src_f_pts = group_normalized_assginment_mat @ group_src_f_pts.unsqueeze(0)
            group_assginment_score = group_assginment_mat.sum(2)
            group_estimated_transform = self.procrustes(group_pred_src_f_pts.squeeze(0), group_ref_f_pts, group_assginment_score.squeeze(0))  

            group_estimated_transform = group_estimated_transform @ group_init_T  # src->ref
            # for val
            inverse_group_init_T = inverse_transform(group_init_T)  # src1->src
            group_src_corr_points = apply_transform(group_src_f_pts, inverse_group_init_T)  # src1->src
        
        for key in ['ref_corr_points', 'src_corr_points', 'estimated_transform']:
            if key not in output_dict:
                output_dict[key] = []
        output_dict['ref_corr_points'].append(group_ref_f_pts)
        output_dict['src_corr_points'].append(group_src_corr_points)
        output_dict['estimated_transform'].append(group_estimated_transform)


        return output_dict, True
    

    def forward(self, data_dict, output_dict,group_id=0):
        if output_dict=={}:
            return self.forward_coarse(data_dict,output_dict)
        else:
            return self.forward_fine(data_dict,output_dict, group_id)

def create_model(cfg):
    model = RGA(cfg)
    return model

def main():
    from config import make_cfg

    cfg = make_cfg()
    model = create_model(cfg)
    print(model.state_dict().keys())
    print(model)

if __name__ == '__main__':
    main()
