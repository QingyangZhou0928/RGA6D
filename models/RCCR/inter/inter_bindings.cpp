// ------------------------------------------------------------------------------------
// Modified from MAC (https://github.com/zhangxy0517/3D-Registration-with-Maximal-Cliques)
// Originally authored by Xiyu Zhang (Copyright (c) 2023 zhangxy0517)
// Licensed under the MIT License.
// Modifications copyright (c) 2026 Qingyang Zhou
// ------------------------------------------------------------------------------------
#include <torch/extension.h>
#include "Eva.h"
#include <vector>

std::tuple<std::vector<int>, std::vector<float>>
interRCCR(std::vector<Corre_3DMatch>& correspondence,
         std::vector<Corre_3DMatch>& all_correspondences,
         torch::Tensor& region_to_corr_indices, 
         float resolution,
         float cmp_thresh);

std::tuple<torch::Tensor, torch::Tensor>
interRCCR_wrapper(torch::Tensor src,
            torch::Tensor ref,
            torch::Tensor src_all,         
            torch::Tensor ref_all,        
            torch::Tensor corr_indices,   
            float resolution,
            float cmp_thresh)
{
    auto device = src.device();
    src = src.contiguous();
    ref = ref.contiguous();
    src_all = src_all.contiguous();
    ref_all = ref_all.contiguous();
    corr_indices = corr_indices.to(torch::kInt32);
    corr_indices = corr_indices.contiguous();
    if (src.is_cuda())
        src = src.cpu();
    if (ref.is_cuda())
        ref = ref.cpu();
    if (src_all.is_cuda())
        src_all = src_all.cpu();
    if (ref_all.is_cuda())
        ref_all = ref_all.cpu();
    if (corr_indices.is_cuda())
        corr_indices = corr_indices.cpu();

    int region_num = src.size(0);
    std::vector<Corre_3DMatch> corr_regions(region_num);
    auto src_ptr = src.data_ptr<float>();
    auto ref_ptr = ref.data_ptr<float>();
    for (int i = 0; i < region_num; ++i)
    {
        corr_regions[i].src.x = src_ptr[i*3+0];
        corr_regions[i].src.y = src_ptr[i*3+1];
        corr_regions[i].src.z = src_ptr[i*3+2];

        corr_regions[i].des.x = ref_ptr[i*3+0];
        corr_regions[i].des.y = ref_ptr[i*3+1];
        corr_regions[i].des.z = ref_ptr[i*3+2];
    }
    int total_num = src_all.size(0);
    std::vector<Corre_3DMatch> corr_all(total_num);
    auto src_all_ptr = src_all.data_ptr<float>();
    auto ref_all_ptr = ref_all.data_ptr<float>();
    for (int i = 0; i < total_num; ++i) {
        corr_all[i].src.x = src_all_ptr[i*3+0];
        corr_all[i].src.y = src_all_ptr[i*3+1];
        corr_all[i].src.z = src_all_ptr[i*3+2];

        corr_all[i].des.x = ref_all_ptr[i*3+0];
        corr_all[i].des.y = ref_all_ptr[i*3+1];
        corr_all[i].des.z = ref_all_ptr[i*3+2];
    }


    auto result = interRCCR(corr_regions, corr_all, corr_indices, resolution, cmp_thresh);
    std::vector<int> node_clique = std::get<0>(result);
    std::vector<float> node_weight = std::get<1>(result);

    torch::Tensor clique_tensor =
        torch::from_blob(node_clique.data(),
                         {(long)node_clique.size()},
                         torch::kInt32).clone();

    torch::Tensor weight_tensor =
    torch::from_blob(node_weight.data(),
                     {(long)node_weight.size()},
                     torch::kFloat32).clone();

    if (device.is_cuda())
    {
        clique_tensor = clique_tensor.to(device);
        weight_tensor = weight_tensor.to(device);
    }

    return {clique_tensor, weight_tensor};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("interRCCR",
          &interRCCR_wrapper,
          "6D Registration");
}