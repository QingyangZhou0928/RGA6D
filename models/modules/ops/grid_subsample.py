# ------------------------------------------------------------------------------------
# Adapted from GeoTransformer (https://github.com/qinzheng93/GeoTransformer)
# Copyright (c) 2022 Zheng Qin
# Licensed under the MIT License.
# ------------------------------------------------------------------------------------
import importlib
import torch
import numpy as np

ext_module = importlib.import_module('geotransformer.ext')


def grid_subsample(points, lengths, voxel_size):
    """Grid subsampling in stack mode.

    This function is implemented on CPU.

    Args:
        points (Tensor): stacked points. (N, 3)
        lengths (Tensor): number of points in the stacked batch. (B,)
        voxel_size (float): voxel size.

    Returns:
        s_points (Tensor): stacked subsampled points (M, 3)
        s_lengths (Tensor): numbers of subsampled points in the batch. (B,)
    """
    if torch.is_tensor(points) and points.is_cuda:
        points = points.cpu()
    else:
        if isinstance(points, np.ndarray):
            points = torch.from_numpy(points).float()
    if torch.is_tensor(lengths) and lengths.is_cuda:
        lengths = lengths.cpu()
    else:
        if isinstance(lengths, np.ndarray):
            lengths = torch.from_numpy(lengths)
    s_points, s_lengths = ext_module.grid_subsampling(points, lengths, voxel_size)  
    return s_points, s_lengths
