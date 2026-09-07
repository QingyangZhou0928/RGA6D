from pointnet2_ops import pointnet2_utils
import torch
import torch
import torch.nn as nn
from models.modules.ops import pairwise_distance

class FPS():    
    def __init__(self, 
                 fps_ngroup,  
                 fps_radius,
                 fps_max_numsamples,
                ):
        super(FPS, self).__init__()
        self.fps_ngroup = fps_ngroup  
        self.fps_radius = fps_radius
        self.fps_max_numsamples = fps_max_numsamples
        
    # @torch.no_grad()
    def __call__(self, points_c: torch.Tensor,points_f:torch.Tensor) -> torch.Tensor:
        """
        input: points_c - Tensor (N, 3) or (1, N, 3) or (B, N, 3)
              points_f - Tensor (M, 3) or (1, M, 3) or (B, M, 3)
        """
        # original_shape = points_c.shape
        points_c = points_c.reshape(1, -1, 3)  # (1, N, 3)
        points_f = points_f.reshape(1, -1, 3)  # (1, M, 3)
        
        points_cf = torch.cat([points_c,points_f],dim=1)  
        fps_ngroup = self.fps_ngroup
        unique_count_history=0

        dist = torch.cdist(points_cf, points_cf, p=2)  # [B, N, M]
        neighbor_count = (dist <= 0.05).sum(-1)  # [B, N]
        # idx = pointnet2_utils.ball_query(0.025, self.fps_max_numsamples, points_cf, points_cf)
        # neighbor_count = (idx != -1).sum(-1)  # (1,N)
        min_neighbors = 2
        valid_mask = neighbor_count > min_neighbors
        valid_points = points_cf[:,valid_mask[0]]
        valid_indices = valid_mask[0].nonzero(as_tuple=False).squeeze(1).to(torch.int32)

        while True:
            fps_idx = pointnet2_utils.furthest_point_sample(valid_points, fps_ngroup)  
            fps_idx = valid_indices[fps_idx.view(-1)].unique().reshape(1,-1)  
            sub_pc = pointnet2_utils.gather_operation(points_cf.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous() 
            idx = pointnet2_utils.ball_query(self.fps_radius, self.fps_max_numsamples, valid_points, sub_pc) 
            idx = valid_indices[idx.view(-1)].reshape(1, fps_idx.size(1), -1) 
            unique_count = idx.view(-1).unique().numel()
            if unique_count >= 0.8*valid_points.size(1) or fps_ngroup>=valid_points.size(1):  
                idx[idx>=points_c.size(1)] = -1  
                break
            elif unique_count==unique_count_history:
                idx[idx>=points_c.size(1)] = -1  
                break
            else:
                unique_count_history = unique_count
                fps_ngroup+=10

        fps_idx = fps_idx.squeeze(0)
        sub_pc = sub_pc.squeeze(0)
        idx = idx.squeeze(0)
        region_num = idx.size(0)

        region_mask= torch.ones_like(fps_idx,dtype=torch.bool,device=fps_idx.device)
        for region_idx,center_idx in enumerate(fps_idx):
            region = idx[region_idx,:].unique()
            if region.shape[0]==1:
                region_mask[region_idx]=False
                region_num-=1
        fps_idx=fps_idx[region_mask]
        sub_pc=sub_pc[region_mask]
        idx=idx[region_mask]

        for region_idx,center_idx in enumerate(fps_idx):
            if center_idx>=points_c.size(1): 
                center_f_pos = points_f[0,center_idx-points_c.size(1),:]  # (3,)
                region = idx[region_idx,:].unique()
                region = region[region!=-1]
                center_c_pos = (points_c.squeeze(0))[region].unsqueeze(0)
 
                dist = pairwise_distance(center_f_pos.reshape(1,1,-1), center_c_pos, normalized=False)
                fps_idx[region_idx] = region[dist.min(dim=2)[1].item()]
                sub_pc[region_idx,:] = center_c_pos[0,dist.min(dim=2)[1].item(),:]
                
        return fps_idx, sub_pc, idx, region_num
