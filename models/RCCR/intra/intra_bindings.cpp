// ------------------------------------------------------------------------------------
// Modified from MAC (https://github.com/zhangxy0517/3D-Registration-with-Maximal-Cliques)
// Originally authored by Xiyu Zhang (Copyright (c) 2023 zhangxy0517)
// Licensed under the MIT License.
// Modifications copyright (c) 2026 Qingyang Zhou
// ------------------------------------------------------------------------------------
#include <torch/extension.h>
#include "Eva.h"
#include <vector>

std::pair<std::vector<int>, float>
intraRCCR(std::vector<Corre_3DMatch>& correspondence,
         float resolution,
         float cmp_thresh);

std::tuple<torch::Tensor, torch::Tensor>
intraRCCR_wrapper(torch::Tensor src,
            torch::Tensor ref,
            float resolution,
            float cmp_thresh)
{
    auto device = src.device();
    src = src.contiguous();
    ref = ref.contiguous();
    if (src.is_cuda())
        src = src.cpu();
    if (ref.is_cuda())
        ref = ref.cpu();

    int C = src.size(0);

    std::vector<Corre_3DMatch> corr(C);

    auto src_ptr = src.data_ptr<float>();
    auto ref_ptr = ref.data_ptr<float>();

    for (int i = 0; i < C; ++i)
    {
        corr[i].src.x = src_ptr[i*3+0];
        corr[i].src.y = src_ptr[i*3+1];
        corr[i].src.z = src_ptr[i*3+2];

        corr[i].des.x = ref_ptr[i*3+0];
        corr[i].des.y = ref_ptr[i*3+1];
        corr[i].des.z = ref_ptr[i*3+2];
    }

    auto result = intraRCCR(corr, resolution, cmp_thresh);
    std::vector<int> top_idx = result.first;
    float max_weight = result.second;

    torch::Tensor idx_tensor =
        torch::from_blob(top_idx.data(),
                         {(long)top_idx.size()},
                         torch::kInt32).clone();

    torch::Tensor weight_tensor =
        torch::tensor({max_weight}, torch::kFloat32);

    if (device.is_cuda())
    {
        idx_tensor = idx_tensor.to(device);
        weight_tensor = weight_tensor.to(device);
    }

    return {idx_tensor, weight_tensor};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("intraRCCR",
          &intraRCCR_wrapper,
          "6D Registration");
}