# ------------------------------------------------------------------------------------
# Modified from GeoTransformer (https://github.com/qinzheng93/GeoTransformer)
# Originally authored by Zheng Qin (Copyright (c) 2022 Zheng Qin)
# Licensed under the MIT License.
# Modifications copyright (c) 2026 Qingyang Zhou
# ------------------------------------------------------------------------------------

from models.datasets.objaverse_gso.dataset import Objaverse_GSO_Dataset
from models.utils.data import (
    registration_collate_fn_stack_mode,
    calibrate_neighbors_stack_mode,
    build_dataloader_stack_mode,
)

def train_valid_data_loader(cfg, distributed):
    train_dataset = Objaverse_GSO_Dataset(
        cfg.data.dataset_root,
        "train",
        ref_num_points=cfg.data.ref_num_points,
        src_num_points=cfg.data.src_num_points,
        noise_magnitude=cfg.train.noise_magnitude,
        deterministic=False,
        twice_sample=cfg.data.twice_sample,
        return_occupancy=True,
        check_overlap=cfg.data.check_overlap,
        min_overlap=cfg.data.min_overlap,
    )
    neighbor_limits = calibrate_neighbors_stack_mode(
        train_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
    )
    cfg.backbone.neighbor_limits = neighbor_limits
    train_loader = build_dataloader_stack_mode(
        train_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        neighbor_limits,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        distributed=distributed,  
    )
    val_dataset = Objaverse_GSO_Dataset(
        cfg.data.dataset_root,
        "val",
        ref_num_points=cfg.data.ref_num_points,
        src_num_points=cfg.data.src_num_points,
        noise_magnitude=cfg.test.noise_magnitude,
        deterministic=True,  
        twice_sample=cfg.data.twice_sample,
        return_occupancy=True,
        check_overlap=cfg.data.check_overlap,
        min_overlap=cfg.data.min_overlap,
    )
    val_loader = build_dataloader_stack_mode(
        val_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        neighbor_limits,
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False,
        distributed=distributed,
    )

    return train_loader, val_loader, neighbor_limits


def test_data_loader(cfg):
    train_dataset = Objaverse_GSO_Dataset(
        cfg.data.dataset_root,
        "train",
        ref_num_points=cfg.data.ref_num_points,
        src_num_points=cfg.data.src_num_points,
        noise_magnitude=cfg.train.noise_magnitude,
        deterministic=False,
        twice_sample=cfg.data.twice_sample,
        return_occupancy=True,
        check_overlap=cfg.data.check_overlap,
        min_overlap=cfg.data.min_overlap,
    )
    neighbor_limits = calibrate_neighbors_stack_mode(
        train_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
    )
    cfg.backbone.neighbor_limits = neighbor_limits
    test_dataset = Objaverse_GSO_Dataset(
        cfg.data.dataset_root,
        "test",
        ref_num_points=cfg.data.ref_num_points,
        src_num_points=cfg.data.src_num_points,
        deterministic=True,
        twice_sample=cfg.data.twice_sample,
        return_occupancy=True,
        check_overlap=cfg.data.check_overlap,
        min_overlap=cfg.data.min_overlap,
    )
    test_loader = build_dataloader_stack_mode(
        test_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        neighbor_limits,
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False,
    )
    return test_loader, neighbor_limits
