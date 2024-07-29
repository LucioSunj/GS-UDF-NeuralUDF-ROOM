#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from gsBranch.scene.gaussian_model import GaussianModel
from gsBranch.utils.sh_utils import eval_sh
from diff_gaussian_rasterization_rade import GaussianRasterizationSettings_rade, GaussianRasterizer_rade

# 这一段就是计算2D平面上的样子
def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    # 进行配置
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )
    # CUDA 渲染的核心对象
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    # Gaussian 坐标
    means3D = pc.get_xyz
    means2D = screenspace_points
    # Gaussian 透明度
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            # 颜色
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to images, obtain their radii (on screen).
    # 进行Cuda渲染（光栅化）
    # 就是直接进行forward了
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}


# def render_rade(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0):
#     """
#     Render the scene.
#
#     Background tensor (bg_color) must be on GPU!
#     """
#
#     # Set up rasterization configuration
#     tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
#     tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
#     screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
#     try:
#         screenspace_points.retain_grad()
#     except:
#         pass
#
#     raster_settings = GaussianRasterizationSettings(
#         image_height=int(viewpoint_camera.image_height),
#         image_width=int(viewpoint_camera.image_width),
#         tanfovx=tanfovx,
#         tanfovy=tanfovy,
#         bg=bg_color,
#         scale_modifier=scaling_modifier,
#         viewmatrix=viewpoint_camera.world_view_transform,
#         projmatrix=viewpoint_camera.full_proj_transform,
#         sh_degree=pc.active_sh_degree,
#         campos=viewpoint_camera.camera_center,
#         prefiltered=False,
#         debug=pipe.debug
#     )
#
#     rasterizer = GaussianRasterizer(raster_settings=raster_settings)
#
#     means3D = pc.get_xyz
#     means2D = screenspace_points
#
#     # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
#     # scaling / rotation by the rasterizer.
#     scales = None
#     rotations = None
#     cov3D_precomp = None
#     scales, opacity = pc.get_scaling_n_opacity_with_3D_filter
#     rotations = pc.get_rotation
#
#     # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
#     # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
#
#     # 这两个的配置部分有些不同
#     shs = pc.get_features
#     colors_precomp = None
#
#     # 这里就是调用了GaussianRasterizer的forward函数，因为pytorch的nn.Module就是可以这样调用的
#     rendered_image, radii, rendered_depth, rendered_middepth, rendered_alpha, rendered_normal, depth_distortion = rasterizer(
#         means3D=means3D,
#         means2D=means2D,
#         shs=shs,
#         colors_precomp=colors_precomp,
#         opacities=opacity,
#         scales=scales,
#         rotations=rotations,
#         cov3D_precomp=cov3D_precomp)
#
#     # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
#     # They will be excluded from value updates used in the splitting criteria.
#     return {"render": rendered_image,
#             "mask": rendered_alpha,
#             "depth": rendered_depth,
#             "middepth": rendered_middepth,
#             "viewspace_points": means2D,
#             "visibility_filter": radii > 0,
#             "radii": radii,
#             "normal": rendered_normal,
#             "depth_distortion": depth_distortion,
#             }

def render_rade(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, kernel_size, scaling_modifier=1.0,
                geo_reg: bool = True, require_depth: bool = True):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    raster_settings = GaussianRasterizationSettings_rade(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        kernel_size=kernel_size,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        geo_reg=geo_reg,
        require_depth=require_depth,
        debug=pipe.debug
    )


    rasterizer = GaussianRasterizer_rade(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    scales, opacity = pc.get_scaling_n_opacity_with_3D_filter
    rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = pc.get_features
    colors_precomp = None

    rendered_image, radii, rendered_expected_coord, rendered_median_coord, rendered_expected_depth, rendered_median_depth, rendered_alpha, rendered_normal, depth_distortion = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)



    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "mask": rendered_alpha,
            "expected_coord": rendered_expected_coord,
            "median_coord": rendered_median_coord,
            "expected_depth": rendered_expected_depth,  # 还是得到了深度的
            "median_depth": rendered_median_depth,
            "viewspace_points": means2D,
            # "visibility_filter": radii > 0,
            "radii": radii,
            "normal": rendered_normal,
            }

