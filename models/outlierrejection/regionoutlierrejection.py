import torch
import torch.nn as nn
import RCCR_py_inter
import numpy as np
from models.modules.ops import index_select
from models.modules.registration import WeightedProcrustes

class RegionOutlierRejection():
    def __init__(self,pcl_resolution = 0.025,):
        super(RegionOutlierRejection, self).__init__()
        self.resolution = pcl_resolution*4*2  # distance tolerance in RCCR, 2x point cloud resolution

    def region_corr_weighted_center(self,
                                    corr_ref_pts,
                                    corr_src_pts,
                                    region_to_corr_indices,  # (R_ref'', C) corr-idx contains -1
                                    all_region_to_corr_scores,  # (R_ref'', C) feats-scores are 0 if not selected by region
                                    #  all_region_ref_corr_indices, 
                                    #  all_region_src_corr_indices,
                                    #  all_region_node_corr_scores,
                                    #  ref_pts, 
                                    #  src_pts,
                                    eps=1e-10):
        # region_center = torch.zeros((all_region_ref_corr_indices.shape[0],3)).cuda()
        
        region_to_corr_indices[region_to_corr_indices ==-1] = corr_ref_pts.shape[0]  # tensor(R_ref'', C)
        corr_ref_pts = torch.cat([corr_ref_pts,torch.zeros((1,3), device=corr_ref_pts.device, dtype=corr_ref_pts.dtype)], dim=0)  # (C+1, 3)
        ref_corr_pts = index_select(corr_ref_pts, region_to_corr_indices, dim=0)  # (R_ref'', C, 3) contains (0,0,0)

        corr_src_pts = torch.cat([corr_src_pts,torch.zeros((1,3), device=corr_src_pts.device, dtype=corr_src_pts.dtype)], dim=0)  # (C+1, 3)
        src_corr_pts = index_select(corr_src_pts, region_to_corr_indices, dim=0)  # (R_ref'', C, 3) contains (0,0,0)
        
        weights = all_region_to_corr_scores / (torch.sum(all_region_to_corr_scores, dim=1, keepdim=True) + eps)  # normalized within patch
        weights = weights.unsqueeze(2)  # (R_ref'', C, 1)

        ref_centroid = torch.sum(ref_corr_pts * weights, dim=1, keepdim=True)  # (R_ref'', 1, 3)
        src_centroid = torch.sum(src_corr_pts * weights, dim=1, keepdim=True)  # (R_ref'', 1, 3)

        ref_centroid = ref_centroid.squeeze(1)
        src_centroid = src_centroid.squeeze(1)
        return ref_centroid, src_centroid
    
    def __call__(self,
                #  all_region_ref_node_corr_indices, 
                #  all_region_src_node_corr_indices,
                #  all_region_node_corr_scores,
                 corr_ref_pts,
                 corr_src_pts,
                 region_to_corr_indices,  # (R_ref'', C) corr-idx contains -1
                 region_to_corr_mask,  # (R_ref'', C) mask
                 all_region_to_corr_scores,  # (R_ref'', C) feats-scores are 0 if not selected by region
                 ref_indices,  # (C,) ref-idx
                 src_indices,  # (C,) src-idx
                 corr_scores,  # (C) feats-scores
                ):
        '''
        input:
            corr_ref_pts:                   tensor(C, 3)       
            corr_src_pts:                   tensor(C, 3)        
            region_to_corr_indices:         tensor(R_ref'', C)       corr-idx contains -1
            region_to_corr_mask:            tensor(R_ref'', C)       corr-mask
            all_region_to_corr_scores:      tensor(R_ref'', C)       feats-scores are 0 if not selected by region
            ref_indices:                    tensor(C,)               ref-idx
            src_indices:                    tensor(C,)               src-idx
            corr_scores:                    tensor(C,)               feats-scores 
        output:
        '''
        corr_ref_pts_bac = corr_ref_pts.clone()
        corr_src_pts_bac = corr_src_pts.clone()
        region_to_corr_indices_bac = region_to_corr_indices.clone()  # tensor(R_ref'', C)  
        region_to_corr_mask_bac = region_to_corr_mask.clone()  # tensor(R_ref'', C)  

        ref_centroid, src_centroid = self.region_corr_weighted_center(corr_ref_pts_bac,corr_src_pts_bac,region_to_corr_indices_bac, all_region_to_corr_scores)  # (R_ref'', 3)  (R_ref'', 3)  
        
        cmp_thresh = np.exp(-0.005)
        resolution = self.resolution
        top_corr_idx, top_clique_weight = RCCR_py_inter.interRCCR(ref_centroid, src_centroid, corr_ref_pts_bac, corr_src_pts_bac, region_to_corr_indices_bac, resolution, cmp_thresh)
        if top_corr_idx.numel()==0 and top_clique_weight.numel()==0:
            top_corr_idx[:] = -1
        # return 0
        sorted_top_corr_idx, sorted_indices = torch.sort(top_corr_idx, descending=False)
        single_num = torch.count_nonzero(sorted_top_corr_idx == -1).item()  # number of -1s
        if single_num == 0:
            ref_node_corr_indices = torch.full((sorted_top_corr_idx.unique().numel(),corr_ref_pts_bac.shape[0]), -1, dtype=torch.long, device=corr_ref_pts_bac.device)  # (N, C) ref-idx
            src_node_corr_indices = torch.full((sorted_top_corr_idx.unique().numel(),corr_ref_pts_bac.shape[0]), -1, dtype=torch.long, device=corr_ref_pts_bac.device)  # (N, C) src-idx
            node_corr_scores = torch.zeros((sorted_top_corr_idx.unique().numel(),corr_ref_pts_bac.shape[0]), dtype=torch.float, device=corr_ref_pts_bac.device)  # (N, C) scores
            ref_centroid_pts = torch.full((sorted_top_corr_idx.unique().numel(),ref_centroid.shape[0],3), -100, dtype=torch.float, device=corr_ref_pts_bac.device)  # (N, R-ref'') ref-center-idx
            src_centroid_pts = torch.full((sorted_top_corr_idx.unique().numel(),src_centroid.shape[0],3), -100, dtype=torch.float, device=corr_ref_pts_bac.device)  # (N, R-src'') src-center-idx
        
        else:
            ref_node_corr_indices = torch.full((sorted_top_corr_idx.unique().numel()+single_num-1,corr_ref_pts_bac.shape[0]), -1, dtype=torch.long, device=corr_ref_pts_bac.device)  # (N, C) ref-idx
            src_node_corr_indices = torch.full((sorted_top_corr_idx.unique().numel()+single_num-1,corr_ref_pts_bac.shape[0]), -1, dtype=torch.long, device=corr_ref_pts_bac.device)  # (N, C) src-idx
            node_corr_scores = torch.zeros((sorted_top_corr_idx.unique().numel()+single_num-1,corr_ref_pts_bac.shape[0]), dtype=torch.float, device=corr_ref_pts_bac.device)  # (N, C) scores
            ref_centroid_pts = torch.full((sorted_top_corr_idx.unique().numel()+single_num-1,ref_centroid.shape[0],3), -100, dtype=torch.float, device=corr_ref_pts_bac.device)  # (N, R-ref'') ref-center-idx
            src_centroid_pts = torch.full((sorted_top_corr_idx.unique().numel()+single_num-1,src_centroid.shape[0],3), -100, dtype=torch.float, device=corr_ref_pts_bac.device)  # (N, R-src'') src-center-idx
        
        group_idx = 0
        for group_id in sorted_top_corr_idx.unique():
            if group_id == -1:  # single point region
                idx = (sorted_top_corr_idx == group_id).nonzero(as_tuple=True)[0]
                sorted_region_indices = sorted_indices[idx]  # (M,) region-idx
                group_to_corr_mask = region_to_corr_mask_bac[sorted_region_indices,:]  # (M, C) mask
                ref_indices_expand = ref_indices.unsqueeze(0).expand(single_num, -1)
                src_indices_expand = src_indices.unsqueeze(0).expand(single_num, -1)
                corr_scores_expand = corr_scores.unsqueeze(0).expand(single_num, -1)

                ref_node_corr_indices[group_idx:group_idx+single_num, :][group_to_corr_mask] = ref_indices_expand[group_to_corr_mask]   # (M, C) ref-idx contains -1
                src_node_corr_indices[group_idx:group_idx+single_num, :][group_to_corr_mask] = src_indices_expand[group_to_corr_mask]  # (M, C) src-idx contains -1
                node_corr_scores[group_idx:group_idx+single_num, :][group_to_corr_mask] = corr_scores_expand[group_to_corr_mask].float()  # (M, C) scores
                ref_centroid_pts[group_idx:group_idx+single_num, 0:1,:] = ref_centroid[sorted_region_indices].unsqueeze(1)  # (M, 1, 3) ref-center-pos contains -100
                src_centroid_pts[group_idx:group_idx+single_num, 0:1,:] = src_centroid[sorted_region_indices].unsqueeze(1)  # (M, 1, 3) src-center-pos contains -100
                group_idx += single_num
                continue
            idx = (sorted_top_corr_idx == group_id).nonzero(as_tuple=True)[0]
            sorted_region_indices = sorted_indices[idx]  # region-idx
            group_to_corr_mask = region_to_corr_mask_bac[sorted_region_indices,:].any(dim=0)  # (C) mask

            ref_node_corr_indices[group_idx:group_idx+1, group_to_corr_mask] = ref_indices[group_to_corr_mask]  # (C) ref-idx contains -1
            src_node_corr_indices[group_idx:group_idx+1, group_to_corr_mask] = src_indices[group_to_corr_mask]
            node_corr_scores[group_idx:group_idx+1, group_to_corr_mask] = corr_scores[group_to_corr_mask].float()  # (C) scores
            ref_centroid_pts[group_idx:group_idx+1, 0:sorted_region_indices.shape[0],:] = ref_centroid[sorted_region_indices].unsqueeze(0)  # (1, 1, 3) ref-center-pos contains -100
            src_centroid_pts[group_idx:group_idx+1, 0:sorted_region_indices.shape[0],:] = src_centroid[sorted_region_indices].unsqueeze(0)   # (1, 1, 3) src-center-pos contains -100
            group_idx += 1

        return ref_node_corr_indices, src_node_corr_indices, node_corr_scores,ref_centroid_pts,src_centroid_pts
        # (N, C) ref-idx contains -1   (N, C) src-idx contains -1    (N, C) scores contains 0