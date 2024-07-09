/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <torch/extension.h> // torch的扩展库
#include "rasterize_points.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // 在这里调用CUDA的方法
  m.def("rasterize_gaussians", &RasterizeGaussiansCUDA); // 前向传播 // 或许我们直接看 cuda的forward
  m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA); // 反向传播 // 直接看cuda的backward的方法
  m.def("mark_visible", &markVisible);
}