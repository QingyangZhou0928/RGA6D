import torch
import torch.nn as nn
import RCCR_py_intra

import numpy as np
from models.modules.ops import pairwise_distance

class OutlierRejection():
    def __init__(self,               
                 num_correspondences=128,  
                 local_topk=3,
                 first_num_correspondences=1.5,
                 dual_normalization=True,
                 region_to_corr_threshold=0.1,
                 pcl_resolution = 0.025,
                 ):
        super(OutlierRejection, self).__init__()

        self.num_correspondences = num_correspondences
        self.local_topk = local_topk
        self.dual_normalization = dual_normalization
        self.region_to_corr_threshold = region_to_corr_threshold
        self.first_num_correspondences = first_num_correspondences  # first select 4x num_correspondences as candidates
        self.resolution = pcl_resolution*4*1.5  # distance tolerance in RCCR, 1.5x point cloud resolution

    # @torch.no_grad()
    def select_correspondences(self, ref_feats, src_feats, ref_masks=None, src_masks=None):
        if ref_masks is None:
            ref_masks = torch.ones(size=(ref_feats.shape[0],), dtype=torch.bool).cuda()
        if src_masks is None:
            src_masks = torch.ones(size=(src_feats.shape[0],), dtype=torch.bool).cuda()
        ref_indices = torch.nonzero(ref_masks, as_tuple=True)[0]  # indices of nodes matched by _f (nearest distance)
        src_indices = torch.nonzero(src_masks, as_tuple=True)[0]
        sel_ref_feats = ref_feats[ref_indices]  # (N, hidden) features of nodes matched by _f (nearest distance)
        sel_src_feats = src_feats[src_indices]  # (M, hidden)

        # select global topk
        sel_matching_scores = torch.exp(-pairwise_distance(sel_ref_feats, sel_src_feats, normalized=True))  # feature distance matrix between _c-ref and _c-src (matched by _f with nearest distance) (exp(-d))
        if self.dual_normalization:
            sel_ref_matching_scores = sel_matching_scores / sel_matching_scores.sum(dim=1, keepdim=True)  # row normalization
            sel_src_matching_scores = sel_matching_scores / sel_matching_scores.sum(dim=0, keepdim=True)  # column normalization
            sel_matching_scores = sel_ref_matching_scores * sel_src_matching_scores  # element-wise product after row and column normalization
        _, sel_global_corr_indices = sel_matching_scores.view(-1).topk(k=min(int(self.first_num_correspondences*self.num_correspondences), 
                                                                             sel_matching_scores.numel()), largest=True)
        sel_global_corr_ref_indices = sel_global_corr_indices // sel_matching_scores.shape[1]
        sel_global_corr_src_indices = sel_global_corr_indices % sel_matching_scores.shape[1]
        sel_corr_mask = torch.zeros((sel_ref_feats.shape[0], sel_src_feats.shape[0]), dtype=torch.bool).cuda()  
        sel_corr_mask[sel_global_corr_ref_indices, sel_global_corr_src_indices] = True

        # select local top3 (specified)
        _, sel_local_corr_indices = sel_matching_scores.topk(k=self.local_topk, dim=1, largest=True)  # locally select supplementary correspondences
        sel_corr_mask[torch.arange(sel_local_corr_indices.shape[0]).view(-1,1).expand(-1,self.local_topk).cuda(), 
                      sel_local_corr_indices] = True  
        
        # both globally and locally select
        sel_ref_corr_indices = torch.nonzero(sel_corr_mask, as_tuple=True)[0]  # indices of nodes matched by _f (nearest distance) (under global scope)
        sel_src_corr_indices = torch.nonzero(sel_corr_mask, as_tuple=True)[1]
        _c_corr_scores = sel_matching_scores[sel_ref_corr_indices, sel_src_corr_indices]
        ref_corr_indices = ref_indices[sel_ref_corr_indices]  # (C,) reference point indices of selected correspondences in global scope
        src_corr_indices = src_indices[sel_src_corr_indices]  # (C,) source point indices of selected correspondences in global scope

        return ref_corr_indices, src_corr_indices, _c_corr_scores

    # @torch.no_grad()
    def get_region_to_corr(
        self,
        ref_region_idx: torch.tensor,  # tensor(R-ref, N-idx)
        ref_c_region_mask: torch.tensor, 
        ref_corr_indices: torch.tensor,  # (C,)
        _c_corr_scores: torch.tensor,  # (C,)
    ):
        """
        ref_region_idx : (R_ref, N_ref)  point indices of each region per row
        src_region_idx : (R_src, N_src)  point indices of each region per row
        gt_corr_indices : (C, 2)         point-level correspondences (ref_idx, src_idx)
        
        Returns:
            valid_pairs : [(r_ref, r_src), ...]  indices of valid region pairs
            overlap_mat : (R_ref, R_src)         matching ratio (average number of points in region)
        """
        R_ref = ref_region_idx.size(0)
        if ref_c_region_mask.dim()==3:
            ref_c_region_mask = ref_c_region_mask.squeeze(0)  # (R, N)
        N_ref = ref_c_region_mask.size(1)

        ref_point2region = ref_c_region_mask.transpose(0,1)  # (N, R)
        # get the corresponding region of matches
        ref_regions = ref_point2region[ref_corr_indices, :]  # (C, R_ref) mask
    
        region_to_corr_mask = ref_regions.transpose(0,1)  # (R_ref, C) mask
        region_to_corr_scores = region_to_corr_mask.float() * _c_corr_scores[None]

        # remove 0-corr nodes
        local_corr_counts = region_to_corr_mask.sum(dim=-1)  # (R_ref,)
        region_mask = torch.gt(local_corr_counts, 0)  # (R_ref,)
        sel_region_to_corr_mask = region_to_corr_mask[region_mask]  # (R_ref', C)

        sel_region_to_corr_indices = torch.arange(region_to_corr_mask.shape[1])[None,:].expand(sel_region_to_corr_mask.shape[0],-1).cuda()  # (R_ref', C)
        sel_region_to_corr_indices = sel_region_to_corr_indices.masked_fill(~sel_region_to_corr_mask, -1)  # (R_ref', C)

        return sel_region_to_corr_mask, sel_region_to_corr_indices, region_to_corr_scores[region_mask], region_mask
    
    def __call__(self, ref_feats, src_feats, ref_pts, src_pts, 
                ref_masks=None, src_masks=None, 
                ref_region_idx=None, ref_c_region_mask=None):
        '''
        input:
            ref_feats: tensor(M, C) already L2 normalized
            src_feats: tensor(N, C) already L2 normalized
            ref_pts: tensor(M, 3) 
            src_pts: tensor(N, 3)
            ref_masks (BoolTensor=None): masks of the superpoints in reference point cloud (False if empty).
            src_masks (BoolTensor=None): masks of the superpoints in source point cloud (False if empty).
        output:
        '''
        # select point correspondences
        ref_corr_indices, src_corr_indices, _c_corr_scores = self.select_correspondences(
            ref_feats, src_feats, ref_masks=ref_masks, src_masks=src_masks
        )  # (C,) ref-idx, (C,) src-idx
        ref_keypts = ref_pts[ref_corr_indices]  # (C, 3)
        src_keypts = src_pts[src_corr_indices]  # (C, 3)

        sel_region_to_corr_mask, sel_region_to_corr_indices, sel_region_to_corr_scores, region_mask = self.get_region_to_corr(
            ref_region_idx,
            ref_c_region_mask, 
            ref_corr_indices,
            _c_corr_scores,
        )  # (R_ref', C) mask, (R_ref', C) corr-idx with -1, (R_ref', C) feats-scores, (R_ref,): under global and region selection, retain only corr > 0.1 and drop regions without any valid corr

        # with torch.no_grad():
        # outlier rejection
        cmp_thresh = np.exp(-0.005)
        resolution = self.resolution
        for region_idx in range(sel_region_to_corr_mask.size(0)):
            raw_region_corr_idx = sel_region_to_corr_indices[region_idx,:]  
            region_corr_idx = raw_region_corr_idx[raw_region_corr_idx!=-1]  # tensor(R-C,) corr-idx
            region_ref_corr_pts = ref_keypts[region_corr_idx,:]  # tensor(R-C, 3)
            region_src_corr_pts = src_keypts[region_corr_idx,:]  # tensor(R-C, 3)
            top_corr_idx, top_clique_weight = RCCR_py_intra.intraRCCR(region_ref_corr_pts, region_src_corr_pts, resolution, cmp_thresh)
            if top_corr_idx.numel()==0 and top_clique_weight.item()==0:
                region_mask[region_idx] = False
            else:
                top_corr_idx = region_corr_idx[top_corr_idx]  # tensor(R-C',) corr-idx (global)
                sel_region_to_corr_mask[region_idx,:] = False
                sel_region_to_corr_mask[region_idx,top_corr_idx] = True  # (R_ref', C) mask in-place modification
                sel_region_to_corr_indices[region_idx,:] = raw_region_corr_idx.masked_fill(~sel_region_to_corr_mask[region_idx,:], -1)  # (R_ref', C) corr-idx with -1, in-place modification
        if (region_mask == False).all():  # No valid region remaining
            return None, None, None, None, None, None
        sel_region_to_corr_mask = sel_region_to_corr_mask[region_mask]  # (R_ref'', C) mask
        sel_region_to_corr_indices = sel_region_to_corr_indices[region_mask]  # (R_ref'', C) corr-idx with -1
        sel_region_to_corr_scores = sel_region_to_corr_scores[region_mask]  # (R_ref'', C) feats-scores
        region_mask = region_mask  # (R_ref,) under global and region selection, retain only corr > 0.1, drop regions without any corr, and exclude mutually isolated inliers in RCCR

        # adapt output for secondary selection
        sel_region_to_corr_scores[sel_region_to_corr_indices ==-1] = 0
    
        return ref_corr_indices, src_corr_indices, sel_region_to_corr_indices, sel_region_to_corr_mask, sel_region_to_corr_scores, _c_corr_scores, ref_keypts, src_keypts
        # (C,) ref-idx, (C,) src-idx, (R_ref'', C) corr-idx with -1, (R_ref'', C) corr-mask, (R_ref'', C) feats-scores set to 0 if unselected, (C,) feats-scores