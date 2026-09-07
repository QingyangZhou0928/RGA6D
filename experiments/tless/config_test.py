# ------------------------------------------------------------------------------------
# Modified from GeoTransformer (https://github.com/qinzheng93/GeoTransformer)
# Originally authored by Zheng Qin (Copyright (c) 2022 Zheng Qin)
# Licensed under the MIT License.
# Modifications copyright (c) 2026 Qingyang Zhou
# ------------------------------------------------------------------------------------

import os
import os.path as osp
import argparse

from easydict import EasyDict as edict

from models.utils.common import ensure_dir

_C = edict()

# common
_C.seed = 7351

# dirs
_C.working_dir = osp.dirname(osp.realpath(__file__))
_C.root_dir = osp.dirname(osp.dirname(_C.working_dir))
_C.exp_name = osp.basename(_C.working_dir)
_C.output_dir = osp.join(_C.root_dir, "output", "o&g_partial_frozen_backbone_tless-1_16train-17_20val-21_30test")
_C.snapshot_dir = osp.join(_C.output_dir, "snapshots")
_C.log_dir = osp.join(_C.output_dir, "logs")
_C.event_dir = osp.join(_C.output_dir, "events")

ensure_dir(_C.output_dir)
ensure_dir(_C.snapshot_dir)
ensure_dir(_C.log_dir)
ensure_dir(_C.event_dir)

# data
_C.data = edict()
_C.data.dataset_root = osp.join(f'./datasets/tless', "tless-1_16train-17_20val-21_30test")  
_C.data.ref_num_points = 1024 
_C.data.src_num_points = 2048
_C.data.twice_sample = True

# train data
_C.train = edict()
_C.train.batch_size = 1
_C.train.num_workers = 8

# test data
_C.test = edict()
_C.test.batch_size = 1
_C.test.num_workers = 8

# evaluation
_C.eval = edict()
_C.eval.acceptance_overlap = 0.0
_C.eval.acceptance_radius = 0.1
# _C.eval.inlier_ratio_threshold = 0.05  
_C.eval.rre_threshold = 1.0
_C.eval.rte_threshold = 0.1

# optim
_C.optim = edict()
_C.optim.lr = 1e-4
_C.optim.weight_decay = 1e-6
_C.optim.warmup_steps = 10000
_C.optim.eta_init = 0.1
_C.optim.eta_min = 0.1
_C.optim.max_iteration = 400000000000000 
_C.optim.snapshot_steps = 10000
_C.optim.grad_acc_steps = 1

# model - backbone
_C.backbone = edict()
_C.backbone.num_stages = 3  
_C.backbone.init_voxel_size = 0.025 
_C.backbone.kernel_size = 15
_C.backbone.base_radius = 2.5
_C.backbone.base_sigma = 2.0
_C.backbone.init_radius = _C.backbone.base_radius * _C.backbone.init_voxel_size  
_C.backbone.init_sigma = _C.backbone.base_sigma * _C.backbone.init_voxel_size
_C.backbone.group_norm = 32
_C.backbone.input_dim = 1
_C.backbone.init_dim = 64 
_C.backbone.output_dim = 256  

# model - Global
_C.model = edict()
_C.model.ground_truth_matching_radius = 0.05 
_C.model.num_points_in_patch = 128  
_C.model.fps_ngroup = 20 
_C.model.fps_radius = 0.25 
_C.model.num_points_in_region = 512 

# model - Coarse Matching
_C.coarse_matching = edict()
_C.coarse_matching.num_targets = 256 
_C.coarse_matching.overlap_threshold = 0.1  
_C.coarse_matching.first_num_correspondences = 1.5 
_C.coarse_matching.num_correspondences = 256 
_C.coarse_matching.dual_normalization = True

# model 
_C.transformer = edict()
_C.transformer.input_dim = 512
_C.transformer.hidden_dim = 256  
_C.transformer.output_dim = 256  
_C.transformer.num_heads = 4
_C.transformer.blocks = ["self", "cross", "self", "cross", "self", "cross"]
_C.transformer.nblocks = 3
_C.transformer.focusing_factor = 3
_C.transformer.sigma_d = 0.2 
_C.transformer.sigma_a = 15
_C.transformer.angle_k = 3  
_C.transformer.reduction_a = "max"  

# outlierrejection
_C.outlierrejection = edict()
_C.outlierrejection.local_topk = 3

# loss - Coarse level
_C.coarse_loss = edict()
_C.coarse_loss.positive_margin = 0.1  
_C.coarse_loss.negative_margin = 1.4 
_C.coarse_loss.positive_optimal = 0.1  
_C.coarse_loss.negative_optimal = 1.4 
_C.coarse_loss.log_scale = 24
_C.coarse_loss.positive_overlap = 0.1  
_C.coarse_loss.region_positive_margin = 0.1
_C.coarse_loss.region_negative_margin = 1.0
_C.coarse_loss.region_positive_optimal = 0.1
_C.coarse_loss.region_negative_optimal = 1.0
_C.coarse_loss.region_log_scale = 12
_C.coarse_loss.region_positive_overlap = 0.8

# loss - Fine level
_C.fine_loss = edict()
_C.fine_loss.positive_radius = 0.05 

# loss - Overall
_C.loss = edict()
_C.loss.weight_coarse_loss = 1.0
_C.loss.weight_fine_loss = 1.0


def make_cfg():
    return _C


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--link_output", dest="link_output", action="store_true", help="link output dir")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = make_cfg()
    if args.link_output:
        os.symlink(cfg.output_dir, "output")


if __name__ == "__main__":
    main()
