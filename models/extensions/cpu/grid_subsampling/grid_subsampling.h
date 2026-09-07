// ------------------------------------------------------------------------------------
// Adapted from GeoTransformer (https://github.com/qinzheng93/GeoTransformer)
// Copyright (c) 2022 Zheng Qin
// Licensed under the MIT License.
// ------------------------------------------------------------------------------------
#pragma once

#include <vector>
#include "../../common/torch_helper.h"

std::vector<at::Tensor> grid_subsampling(
  at::Tensor points,
  at::Tensor lengths,
  float voxel_size
);
