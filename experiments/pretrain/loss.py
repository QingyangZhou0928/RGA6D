# ------------------------------------------------------------------------------------
# Modified from GeoTransformer (https://github.com/qinzheng93/GeoTransformer)
# Originally authored by Zheng Qin (Copyright (c) 2022 Zheng Qin)
# Licensed under the MIT License.
# Modifications copyright (c) 2026 Qingyang Zhou
# ------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.ops import apply_transform, pairwise_distance
from models.modules.loss import WeightedCircleLoss
from models.modules.registration.metrics import isotropic_transform_error

class CoarseMatchingLoss(nn.Module):
    def __init__(self, cfg):
        super(CoarseMatchingLoss, self).__init__()
        self.weighted_circle_loss = WeightedCircleLoss(
            cfg.coarse_loss.positive_margin,
            cfg.coarse_loss.negative_margin,
            cfg.coarse_loss.positive_optimal,
            cfg.coarse_loss.negative_optimal,
            cfg.coarse_loss.log_scale,
        )
        self.positive_overlap = cfg.coarse_loss.positive_overlap

    def forward(self, output_dict):
        ref_feats = output_dict['ref_feats_c']  # (_c-ref,_c-ref-feats)
        src_feats = output_dict['src_feats_c']  # (_c-src,_c-src-feats)
        gt_node_corr_indices = output_dict['gt_node_corr_indices']
        gt_node_corr_overlaps = output_dict['gt_node_corr_overlaps']  # (_c-ref&_c-src，)
        gt_ref_node_corr_indices = gt_node_corr_indices[:, 0]  # (_c-ref&_c-src，)
        gt_src_node_corr_indices = gt_node_corr_indices[:, 1]  # (_c-ref&_c-src，)

        feat_dists = torch.sqrt(pairwise_distance(ref_feats, src_feats, normalized=True))  

        overlaps = torch.zeros_like(feat_dists)
        overlaps[gt_ref_node_corr_indices, gt_src_node_corr_indices] = gt_node_corr_overlaps
        pos_masks = torch.gt(overlaps, self.positive_overlap)  # (_c-ref,_c-src)
        neg_masks = torch.eq(overlaps, 0)  # (_c-ref,_c-src)
        pos_scales = torch.sqrt(overlaps * pos_masks.float())  # (_c-ref,_c-src) 

        loss = self.weighted_circle_loss(pos_masks, neg_masks, feat_dists, pos_scales)

        return loss

class RegionCoarseMatchingLoss(nn.Module):
    def __init__(self, cfg):
        super(RegionCoarseMatchingLoss, self).__init__()
        self.weighted_circle_loss = WeightedCircleLoss(
            cfg.coarse_loss.region_positive_margin,
            cfg.coarse_loss.region_negative_margin,
            cfg.coarse_loss.region_positive_optimal,
            cfg.coarse_loss.region_negative_optimal,
            cfg.coarse_loss.region_log_scale,
        )
        self.positive_overlap = cfg.coarse_loss.region_positive_overlap

    def forward(self, output_dict):
        ref_region_feats = output_dict['ref_region_token']  # (R,D)
        src_region_feats = output_dict['src_region_token']  # (R,D)
        gt_region_indices = output_dict['gt_region_indices']
        gt_region_overlaps = output_dict['gt_region_overlaps']  


        feat_dists = torch.sqrt(pairwise_distance(ref_region_feats, src_region_feats, normalized=True)) 

        pos_masks = torch.gt(gt_region_overlaps, self.positive_overlap)  # (_c-ref,_c-src) 
        neg_masks = torch.eq(gt_region_overlaps, 0)  # (_c-ref,_c-src) 
        pos_scales = torch.sqrt(gt_region_overlaps * pos_masks.float())  # (_c-ref,_c-src) 

        loss = self.weighted_circle_loss(pos_masks, neg_masks, feat_dists, pos_scales)

        return loss

class CoarselLoss(nn.Module):
    def __init__(self, cfg):
        super(CoarselLoss, self).__init__()
        self.coarse_loss = CoarseMatchingLoss(cfg)
        self.region_loss = RegionCoarseMatchingLoss(cfg)
        self.weight_coarse_loss = cfg.loss.weight_coarse_loss
        self.weight_fine_loss = cfg.loss.weight_fine_loss

    def forward(self, output_dict, data_dict):
        coarse_loss = self.coarse_loss(output_dict)  # Overlap-aware circle loss
        region_loss = self.region_loss(output_dict)
        
        loss = self.weight_coarse_loss * coarse_loss + self.weight_coarse_loss * 0.1 * region_loss
        
        if isinstance(loss, float):
            loss = torch.tensor(loss, device=coarse_loss.device if isinstance(coarse_loss, torch.Tensor) else 'cuda', requires_grad=True)

        return {
            'coarse_loss': loss,
            'c_loss': coarse_loss,
            'r_loss': region_loss,
        }

class CoarseLoss_Without_Region(nn.Module):
    def __init__(self, cfg):
        super(CoarseLoss_Without_Region, self).__init__()
        self.coarse_loss = CoarseMatchingLoss(cfg)
        self.weight_coarse_loss = cfg.loss.weight_coarse_loss

    def forward(self, output_dict, data_dict):
        coarse_loss = self.coarse_loss(output_dict)  # Overlap-aware circle loss

        loss = self.weight_coarse_loss * coarse_loss 
        
        return {
            'coarse_loss': loss,
            'c_loss': coarse_loss,
        }
   
class FineLoss(nn.Module):
    '''
    Modified from UNOPose (https://github.com/shanice-l/UNOPose)
    Originally authored by Xingyu Liu, Gu Wang, and other UNOPose authors (Copyright (c) 2025 Xingyu Liu, Gu Wang, and other UNOPose authors)
    Licensed under the MIT License.
    Modifications copyright (c) 2026 Qingyang Zhou
    '''
    def __init__(self, cfg):
        super(FineLoss, self).__init__()
        self.positive_radius = cfg.fine_loss.positive_radius
    def get_weighted_bce_loss(self,prediction, gt):
        loss = nn.BCELoss(reduction="none")
        class_loss = loss(prediction, gt)

        weights = torch.ones_like(gt)
        w_negative = gt.sum(1) / gt.size(1)
        w_positive = 1 - w_negative

        w_positive = w_positive[:, None].repeat(1, gt.shape[1])
        w_negative = w_negative[:, None].repeat(1, gt.shape[1])

        weights[gt >= 0.5] = w_positive[gt >= 0.5]
        weights[gt < 0.5] = w_negative[gt < 0.5]
        w_class_loss = (weights * class_loss).mean(1)

        return w_class_loss

    def forward(self, output_dict, data_dict): 
        CE = nn.CrossEntropyLoss(reduction="none")
        fine_score_loss = [0.0 for _ in range(len(output_dict['fine_score_list'][-1]))]  
        fine_saliency_loss = [0.0 for _ in range(len(output_dict['fine_score_list'][-1]))]
        fine_attention_loss = [0.0 for _ in range(len(output_dict['fine_score_list'][-1]))]
        
        ref_node_corr_knn_points = output_dict['ref_corr_points'][-1].unsqueeze(0)  # (B,M,3) 
        src_node_corr_knn_points = output_dict['src_corr_points'][-1].unsqueeze(0)  # (B,M,3) 
        transform = data_dict['transform']  

        src_node_corr_knn_points = apply_transform(src_node_corr_knn_points, transform)
        dists = pairwise_distance(ref_node_corr_knn_points, src_node_corr_knn_points)  
        corr = torch.stack(torch.where(dists[0] <= self.positive_radius ** 2), dim=-1)
        idx1, idx2 = torch.unique(corr[:, 0]), torch.unique(corr[:, 1])
        idx2 += ref_node_corr_knn_points.shape[1]
        idx = torch.cat((idx1, idx2), dim=0)
        gt_overlap = torch.zeros([1,ref_node_corr_knn_points.shape[1]+src_node_corr_knn_points.shape[1]]).cuda()  # bs, n1+n2
        gt_overlap[:,idx] = 1

        # calculate score loss
        score_list = output_dict['fine_score_list'][-1]  
        for idx, score in enumerate(score_list):
            score = score.float()
            fine_score_loss[idx]=self.get_weighted_bce_loss(score, gt_overlap)
        # calculate saliency loss
        saliency_list = output_dict['fine_saliency_list'][-1]  
        for idx, saliency in enumerate(saliency_list):
            saliency = saliency.float()
            fine_saliency_loss[idx]=self.get_weighted_bce_loss(saliency, gt_overlap)
        # calculate matching loss
        attention_list = output_dict['fine_attention_list'][-1] 
        pos_mask = dists <= self.positive_radius ** 2   # (B, M, M)
       
        for idx, atten in enumerate(attention_list):
            log_prob = F.log_softmax(atten.float(), dim=-1)  # row-wise
            loss_row = -(log_prob * pos_mask.float()).sum(-1) / (pos_mask.sum(-1) + 1e-6)
            loss_row = loss_row.mean()
            log_prob_t = F.log_softmax(atten.transpose(1, 2), dim=-1)
            loss_col = -(log_prob_t * pos_mask.transpose(1, 2).float()).sum(-1) / (pos_mask.sum(1) + 1e-6)
            loss_col = loss_col.mean()
            fine_attention_loss[idx] = 0.5 * (loss_row + loss_col)
        
        if len(output_dict['fine_score_list'][0])==3:
            fine_score_loss = 0.1*fine_score_loss[0] + 0.2*fine_score_loss[1] + 0.7*fine_score_loss[2]
            fine_saliency_loss = 0.1*fine_saliency_loss[0] + 0.2*fine_saliency_loss[1] + 0.7*fine_saliency_loss[2]
            fine_attention_loss = 0.1*fine_attention_loss[0] + 0.2*fine_attention_loss[1] + 0.7*fine_attention_loss[2]
        else:
            fine_score_loss = fine_score_loss[-1] 
            fine_saliency_loss = fine_saliency_loss[-1] 
            fine_attention_loss = fine_attention_loss[-1] 
        fine_loss = fine_score_loss + fine_saliency_loss + fine_attention_loss
        return {
            'f_loss': fine_loss,
            'fine_score_loss': fine_score_loss,
            'fine_saliency_loss': fine_saliency_loss,
            'fine_attention_loss': fine_attention_loss,
        }

class Evaluator(nn.Module):
    def __init__(self, cfg):
        super(Evaluator, self).__init__()
        self.acceptance_overlap = cfg.eval.acceptance_overlap
        self.acceptance_radius = cfg.eval.acceptance_radius
        self.acceptance_rre = cfg.eval.rre_threshold
        self.acceptance_rte = cfg.eval.rte_threshold

    @torch.no_grad()
    def evaluate_coarse(self, output_dict):
        ref_length_c = output_dict['ref_points_c'].shape[0]
        src_length_c = output_dict['src_points_c'].shape[0]
        gt_node_corr_overlaps = output_dict['gt_node_corr_overlaps']
        gt_node_corr_indices = output_dict['gt_node_corr_indices']
        masks = torch.gt(gt_node_corr_overlaps, self.acceptance_overlap) 
        gt_node_corr_indices = gt_node_corr_indices[masks] 
        gt_ref_node_corr_indices = gt_node_corr_indices[:, 0]
        gt_src_node_corr_indices = gt_node_corr_indices[:, 1]
        gt_node_corr_map = torch.zeros(ref_length_c, src_length_c).cuda()
        gt_node_corr_map[gt_ref_node_corr_indices, gt_src_node_corr_indices] = 1.0
        ref_node_corr_indices = output_dict['ref_node_corr_indices']
        src_node_corr_indices = output_dict['src_node_corr_indices']

        if ref_node_corr_indices.dim() != 1 or src_node_corr_indices.dim() != 1:
            precision = gt_node_corr_map[ref_node_corr_indices[ref_node_corr_indices!=-1].view(-1), src_node_corr_indices[src_node_corr_indices!=-1].view(-1)].mean()
        else:
            precision = gt_node_corr_map[ref_node_corr_indices[ref_node_corr_indices!=-1], src_node_corr_indices[src_node_corr_indices!=-1]].mean()

        return precision

    @torch.no_grad()
    def evaluate_fine(self, output_dict, data_dict, top_idx,positive_radius=0.005):
        group_idx = top_idx
        
        ref_node_corr_knn_points = output_dict['ref_corr_points'][group_idx].unsqueeze(0)  
        src_node_corr_knn_points = output_dict['src_corr_points'][group_idx].unsqueeze(0) 
        transform = data_dict['transform']  
        src_node_corr_knn_points = apply_transform(src_node_corr_knn_points, transform)
        dists = pairwise_distance(ref_node_corr_knn_points, src_node_corr_knn_points)  
        
        attention_list = output_dict['fine_attention_list'][group_idx] 
        dis1, label1 = dists.min(2)
        fg_label1 = (dis1 <= positive_radius ** 2).float()  
        label1 = (fg_label1 * (label1.float() + 1.0)).long() 

        # acc
        pred_label = torch.max(attention_list[-1][:, :, :], dim=2)[1]
        group_precision=(pred_label == label1).float().mean(1)
        
        return group_precision

    @torch.no_grad()
    def evaluate_registration(self, output_dict, data_dict, top_idx):
        transform = data_dict['transform']
        src_points = output_dict['src_points']
        est_transform = output_dict['estimated_transform']
        ref_region_idx = top_idx
        
        group_est_transform = est_transform[ref_region_idx]
        rre, rte = isotropic_transform_error(transform, group_est_transform)
        recall = torch.logical_and(torch.lt(rre, self.acceptance_rre), torch.lt(rte, self.acceptance_rte)).float()

        gt_src_points = apply_transform(src_points, transform)
        est_src_points = apply_transform(src_points, group_est_transform)
        rmse = torch.linalg.norm(est_src_points - gt_src_points, dim=1).mean()

        return rre, rte, rmse, recall 
    
    def transform_ranking(self,data_dict,output_dict):
        ref_length = data_dict['lengths'][0][0].item()  
        points = data_dict['points'][0].detach()   

        ref_points = points[:ref_length] 
        src_points = points[ref_length:]  
        estimated_transform = output_dict['estimated_transform']

        estimated_transform = torch.stack(estimated_transform, dim=0)
        N_T = estimated_transform.shape[0]
        batch_src_points = apply_transform(src_points[None,:,:].expand(N_T,-1,-1),estimated_transform)
        dist = torch.cdist(ref_points[None,:,:].expand(N_T,-1,-1),batch_src_points)  
        min_dist = dist.min(dim=2)[0]  # nearest neighbor (N_T, N_c)
        D_h = min_dist.mean(dim=1)  # average (N_T)
        xi = 1.0 / (D_h + 1e-10)  # ranking score
        xi, rank = torch.sort(xi, descending=True)
        return rank, xi

    def forward(self, output_dict, data_dict):
        rank, xi = self.transform_ranking(data_dict, output_dict)
        top_idx = rank[0]
        c_precision = self.evaluate_coarse(output_dict)
        f_precision = self.evaluate_fine(output_dict, data_dict, top_idx)
        rre, rte, rmse, recall = self.evaluate_registration(output_dict, data_dict, top_idx)
        return {
            'PIR': c_precision,
            'IR': f_precision,
            'RRE': rre,
            'RTE': rte,
            'RMSE': rmse,
            'RR': recall,
        }
