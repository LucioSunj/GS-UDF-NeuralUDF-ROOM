import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
# import pymcubes as mcubes # 这里源码是import mcubes
from icecream import ic
import skimage.measure
import pdb
# TODO 解决marching cubes的兼容问题，或者我们不要marching cube了？
import mcubes
from udfBranch.models.patch_projector import PatchProjector

from udfBranch.models.fields import color_blend


def extract_fields(bound_min, bound_max, resolution, query_func, device):
    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1).to(device)
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u


def extract_gradient_fields(bound_min, bound_max, resolution, query_func, device):
    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution, 3], dtype=np.float32)

    for xi, xs in enumerate(X):
        for yi, ys in enumerate(Y):
            for zi, zs in enumerate(Z):
                xx, yy, zz = torch.meshgrid(xs, ys, zs)
                pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1).to(device)
                val = query_func(pts).reshape(len(xs), len(ys), len(zs), 3).detach().cpu().numpy()
                u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func, device):
    print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func, device)

    vertices, triangles = mcubes.marching_cubes(u, threshold)
    # vertices, triangles, normals, values = skimage.measure.marching_cubes(u, threshold)

    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles


# TODO 或许我们不用改这个函数？直接利用edge和gs改变传入的weights就行？
# 根据weights计算出对应的pdf概率密度函数，然后得到cdf，然后通过cdf进行插值上采样得到插值后的点作为上采样的结果返回
def sample_pdf(bins, weights, n_samples, det=False):
    """
    Sample new points based on a given probability density function (PDF) using inverse transform sampling.
    This method is often used in volume rendering for NeRF (Neural Radiance Fields).

    Arguments:
    - bins: A tensor of bin edges used for sampling (shape: [batch_size, num_bins]).
    - weights: A tensor of weights for each bin, representing the PDF (shape: [batch_size, num_bins - 1]).
    - n_samples: The number of samples to generate.
    - det: If True, perform deterministic sampling (uniform intervals); if False, perform random sampling.

    Returns:
    - samples: A tensor of sampled points (shape: [batch_size, n_samples]).
    """

    device = weights.device

    # Prevent division by zero and NaN values by adding a small constant to weights.
    # 而在有alph_occ的情况下，就用occ作为weights，也就是每个采样点的权重
    weights = weights + 1e-5

    # Compute the normalized PDF.
    # 根据权重，计算pdf概率密度函数
    pdf = weights / torch.sum(weights, -1, keepdim=True)

    # Compute the cumulative distribution function (CDF) from the PDF.
    # 根据PDF概率密度函数计算得到累加概率密度函数
    cdf = torch.cumsum(pdf, -1)

    # Add a zero at the beginning of the CDF to account for the first bin.
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

    # Generate sample positions 'u' in the CDF space.
    if det:
        # Deterministic sampling with uniform intervals.
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples).to(device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        # Random sampling.
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples]).to(device)

    # Ensure 'u' is contiguous in memory for efficient processing.
    u = u.contiguous()

    # Find the indices of the CDF values that are just above 'u'.
    inds = torch.searchsorted(cdf, u, right=True)

    # Ensure indices are within valid range.
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)

    # Combine below and above indices for gathering.
    inds_g = torch.stack([below, above], -1)  # Shape: [batch_size, n_samples, 2]

    # Expand CDF and bins tensors to match the shape required for gathering values.
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]

    # Gather CDF values for the computed indices.
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)

    # Gather bin values for the computed indices.
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    # Calculate the denominator for interpolation, preventing division by zero.
    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)

    # Compute the interpolation factor 't' within the bins.
    t = (u - cdf_g[..., 0]) / denom

    # Compute the final samples using linear interpolation.
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    # Check for NaN values in the samples and pause execution for debugging if any are found.
    flag = torch.any(torch.isnan(samples)).cpu().numpy().item()
    if flag:
        print("z_vals", samples[torch.isnan(samples)])
        print('z_samples have nan values')
        pdb.set_trace()  # Pause for debugging
        # raise Exception("z_samples have nan values")

    # Return the sampled points.
    return samples


class UDFRendererBlending:
    def __init__(self,
                 nerf,
                 udf_network,
                 deviation_network,
                 color_network,
                 beta_network,
                 n_samples,
                 n_importance,
                 n_outside,
                 up_sample_steps,
                 perturb,
                 sdf2alpha_type='numerical',  # * numerical is better
                 upsampling_type='classical',  # classical is better for DTU
                 sparse_scale_factor=25000,
                 h_patch_size=3,
                 use_norm_grad_for_cosine=False
                 ):
        # the networks
        self.nerf = nerf
        self.udf_network = udf_network
        self.deviation_network = deviation_network
        self.color_network = color_network
        self.beta_network = beta_network  # use to detect zero-level set

        # sampling setting
        self.n_samples = n_samples
        self.n_importance = n_importance
        self.n_outside = n_outside
        self.perturb = perturb
        self.up_sample_steps = up_sample_steps

        self.sdf2alpha_type = sdf2alpha_type
        self.upsampling_type = upsampling_type
        self.sparse_scale_factor = sparse_scale_factor

        # the setting of patch blending
        self.h_patch_size = h_patch_size
        self.patch_projector = PatchProjector(self.h_patch_size)

        self.use_norm_grad_for_cosine = use_norm_grad_for_cosine

        self.sigmoid = nn.Sigmoid()

    def udf2logistic(self, udf, inv_s, gamma=20, abs_cos_val=1.0, cos_anneal_ratio=None):
        """
        将用户定义的函数(UDF)转换为逻辑斯谛函数的形式。

        这种转换常用于将连续值转换为概率值，逻辑斯谛函数是一个常用的激活函数，
        在神经网络中用于预测输出的概率。

        参数:
        udf (Tensor): 用户定义的函数的输出，这是一个标量张量。
        inv_s (Tensor): 用于调整曲线斜率的参数，也是一个标量张量。
        gamma (float, 可选): 用于缩放输出的参数，默认值为20。
        abs_cos_val (float, 可选): 初始的绝对余弦值，用于控制曲线的形状，默认值为1.0。
        cos_anneal_ratio (float, 可选): 余弦退火比例，用于动态调整abs_cos_val的值，默认为None。

        返回:
        Tensor: 逻辑斯谛函数的输出，与输入udf具有相同的形状。
        """

        # 如果cos_anneal_ratio给出，则根据公式动态调整abs_cos_val的值
        if cos_anneal_ratio is not None:
            abs_cos_val = (abs_cos_val * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) + \
                          abs_cos_val * cos_anneal_ratio  # always non-positive

        # 计算逻辑斯谛函数的原始输出
        raw = abs_cos_val * inv_s * torch.exp(-inv_s * udf) / (1 + torch.exp(-inv_s * udf)) ** 2

        # 将原始输出乘以gamma进行缩放
        raw = raw * gamma

        return raw

    def render_core_outside(self, rays_o, rays_d, z_vals, sample_dist, nerf, background_rgb=None):
        batch_size, n_samples = z_vals.shape

        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape)], -1)

        mid_z_vals = z_vals + dists * 0.5

        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # batch_size, n_samples, 3

        if self.n_outside > 0:
            dis_to_center = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).clip(1.0, 1e10)
            pts = torch.cat([pts / dis_to_center, 1.0 / dis_to_center], dim=-1)  # batch_size, n_samples, 4

        dirs = rays_d[:, None, :].expand(batch_size, n_samples, 3)

        pts = pts.reshape(-1, 3 + int(self.n_outside > 0))
        dirs = dirs.reshape(-1, 3)

        raw, sampled_color = nerf(pts, dirs)
        alpha = 1.0 - torch.exp(-F.relu(raw.reshape(batch_size, n_samples)) * dists)
        alpha = alpha.reshape(batch_size, n_samples)
        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:,
                          :-1]  # n_rays, n_samples
        sampled_color = sampled_color.reshape(batch_size, n_samples, 3)
        color = (weights[:, :, None] * sampled_color).sum(dim=1)
        if background_rgb is not None:
            color = color + background_rgb * (1.0 - weights.sum(dim=-1, keepdim=True))

        return {
            'color': color,
            'sampled_color': sampled_color,
            'alpha': alpha,
            'weights': weights,
        }

    def cat_z_vals(self, rays_o, rays_d, z_vals, new_z_vals, udf, net_gradients=None, last=False):
        """
        合并原始和新采样点的z值，并更新相应的udf值。

        参数:
        rays_o: 光源方向向量，形状为(batch_size, 3)。
        rays_d: 射线方向向量，形状为(batch_size, 3)。
        z_vals: 原始采样位置，形状为(batch_size, n_samples)。
        new_z_vals: 新采样位置，形状为(batch_size, n_importance)。
        udf: 原始采样点的单层透明度估计，形状为(batch_size, n_samples)。
        net_gradients: 神经网络的梯度（未使用）。
        last: 是否是最后一次采样标志。

        返回:
        z_vals: 合并并排序后的z值，形状为(batch_size, n_samples + n_importance)。
        udf: 合并并排序后的udf值，形状为(batch_size, n_samples + n_importance)。
        """
        # 获取批量大小和采样点数量
        batch_size, n_samples = z_vals.shape
        # 获取新重要采样点的数量
        _, n_importance = new_z_vals.shape

        # 计算新采样点的位置
        pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]

        # 将原始采样位置和新采样位置合并
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        # 对合并后的z值进行排序，并获取排序后的索引
        z_vals, index = torch.sort(z_vals, dim=-1)

        # 检查是否是最后一次采样
        if not last:
            # 通过UDF网络计算新采样点的udf值
            # 将计算的新采样点位置 pts 输入 UDF 网络 ( udf_network )，获得相应的新UDF值，并调整输出的形状，使其与其他数据对齐
            new_udf_output = self.udf_network(pts.reshape(-1, 3))
            new_udf_output = new_udf_output.reshape(batch_size, n_importance, -1)
            new_udf = new_udf_output[:, :, 0]

            # 将原始udf值和新udf值合并
            udf = torch.cat([udf, new_udf], dim=-1)

            # 获取展开的索引，便于根据排序后的索引重新排序udf
            xx = torch.arange(batch_size)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
            index = index.reshape(-1)
            udf = udf[(xx, index)].reshape(batch_size, n_samples + n_importance)

        # 返回合并并排序后的z值和udf值
        return z_vals, udf

    def sdf2alpha(self, sdf, true_cos, dists, inv_s, cos_anneal_ratio=None, udf_eps=None):
        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.

        # 若提供了 cos_anneal_ratio，则在训练初期通过调整 cos 值进行退火操作，以提升初期训练收敛效果。
        if cos_anneal_ratio is not None:
            # 计算 iter_cos，取决于 cos_anneal_ratio，确保其总为非正数。
            iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                         F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive
        else:
            # 若未提供 cos_anneal_ratio，则直接使用 true_cos。
            iter_cos = true_cos

        # 获取 iter_cos 的绝对值。
        abs_cos_val = iter_cos.abs()

        # 若提供了 udf_eps，则进行以下处理。
        if udf_eps is not None:
            # ! 在接近 udf=0 时，abs_cos_val 可能不准确。
            mask = sdf.abs() < udf_eps
            # 对于满足条件的点，将 abs_cos_val 设为默认值 1。
            abs_cos_val[mask] = 1.  # {udf < udf_eps} use default abs_cos_val value

        if self.sdf2alpha_type == 'numerical':
            # todo: 不能很好地处理靠近表面的情况。
            # 估计下一个 SDF 值。
            estimated_next_sdf = (sdf + iter_cos * dists * 0.5)
            # 估计上一个 SDF 值。
            estimated_prev_sdf = (sdf - iter_cos * dists * 0.5)

            # 计算上一个和下一个 CDF。
            prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
            next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

            # 计算 p 和 c 的差值，这决定了光线的方向。
            p = prev_cdf - next_cdf  # ! this decides the ray shoot direction
            c = prev_cdf

            # 计算 alpha 值，并进行截断处理。
            alpha = ((p + 1e-5) / (c + 1e-5))
            alpha = alpha.clip(0.0, 1.0)
        elif self.sdf2alpha_type == 'theorical':
            # 理论方式计算 alpha 值。
            raw = abs_cos_val * inv_s * (1 - self.sigmoid(sdf * inv_s))
            # 计算 alpha 值，确保其非负。
            alpha = 1.0 - torch.exp(-F.relu(raw) * dists)

        # 返回计算的 alpha 值。
        return alpha

    def render_core(self,
                    rays_o,
                    rays_d,
                    z_vals,
                    sample_dist,
                    udf_network,
                    deviation_network,
                    color_network,
                    beta_network=None,
                    cos_anneal_ratio=None,
                    background_rgb=None,
                    background_alpha=None,
                    background_sampled_color=None,
                    flip_saturation=0.0,
                    # * blending params
                    color_maps=None,
                    w2cs=None,
                    intrinsics=None,
                    query_c2w=None,
                    img_index=None,
                    rays_uv=None,
                    ):
        device = z_vals.device
        _, n_samples = z_vals.shape
        batch_size, _ = rays_o.shape
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([sample_dist]).to(device).expand(dists[..., :1].shape)], -1)

        mid_z_vals = z_vals + dists * 0.5

        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # n_rays, n_samples, 3

        dirs = rays_d[:, None, :].expand(pts.shape)

        pts = pts.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)

        udf_nn_output = udf_network(pts)
        udf = udf_nn_output[:, :1]
        feature_vector = udf_nn_output[:, 1:]

        gradients = udf_network.gradient(pts).reshape(batch_size * n_samples, 3)

        gradients_mag = torch.linalg.norm(gradients, ord=2, dim=-1, keepdim=True)
        gradients_norm = gradients / (gradients_mag + 1e-5)  # normalize to unit vector

        inv_s = deviation_network(torch.zeros([1, 3]).to(device))[:, :1].clip(1e-6, 1e6)  # Single parameter
        inv_s = inv_s.expand(batch_size * n_samples, 1)

        beta = beta_network.get_beta().clip(1e-6, 1e6)
        gamma = beta_network.get_gamma().clip(1e-6, 1e6)

        # ? use gradient w/o normalization
        if self.use_norm_grad_for_cosine:
            true_cos = (dirs * gradients_norm).sum(-1, keepdim=True)  # [N, 1]
        else:
            true_cos = (dirs * gradients).sum(-1, keepdim=True)  # [N, 1]

        with torch.no_grad():
            cos = (dirs * gradients_norm).sum(-1, keepdim=True)  # [N, 1]
            flip_sign = torch.sign(cos) * -1  # used for visualize the surface normal
            flip_sign[flip_sign == 0] = 1

        vis_prob = None
        alpha_occ = None

        # * the probability of occlusion
        raw_occ = self.udf2logistic(udf, beta, 1.0, 1.0).reshape(batch_size, n_samples)

        # near 0levelset alpha_acc -> 1 others -> 0
        alpha_occ = 1.0 - torch.exp(-F.relu(raw_occ) * gamma * dists)

        # ! consider the direction of gradients to alleviate the early zero of vis_prob
        vis_mask = torch.ones_like(true_cos).to(true_cos.device).to(true_cos.dtype) * (
                true_cos < 0.01).float()
        vis_mask = vis_mask.reshape(batch_size, n_samples)

        # shift one pt
        vis_mask = torch.cat([vis_mask[:, 1:], torch.ones([batch_size, 1]).to(device)], dim=-1)

        vis_prob = torch.cumprod(
            torch.cat([torch.ones([batch_size, 1]).to(device),
                       ((1. - alpha_occ + flip_saturation * vis_mask).clip(0, 1)) + 1e-7],
                      -1), -1)[:, :-1]  # before udf=0 -> 1; after udf=0 -> 0

        vis_prob = vis_prob.clip(0, 1)

        alpha_plus = self.sdf2alpha(udf, -1 * torch.abs(true_cos), dists.view(-1, 1), inv_s,
                                    cos_anneal_ratio).reshape(batch_size, n_samples)
        alpha_minus = self.sdf2alpha(-udf, -1 * torch.abs(true_cos), dists.view(-1, 1), inv_s,
                                     cos_anneal_ratio).reshape(batch_size, n_samples)

        alpha = alpha_plus * vis_prob + alpha_minus * (1 - vis_prob)

        udf = udf.reshape(batch_size, n_samples)

        alpha = alpha.reshape(batch_size, n_samples)

        sampled_color_base, sampled_color, blending_weights = color_network(pts, gradients_norm, dirs,
                                                                            feature_vector)
        sampled_color_base = sampled_color_base.reshape(batch_size, n_samples, 3)
        sampled_color = sampled_color.reshape(batch_size, n_samples, 3)
        blending_weights = blending_weights.reshape(batch_size, n_samples, -1)

        #########################  Pixel Patch blending    #############################
        # - extract pixel
        if_pixel_blending = False if color_maps is None else True
        pts_pixel_color, pts_pixel_mask = None, None
        if if_pixel_blending:
            pts_pixel_color, pts_pixel_mask = self.patch_projector.pixel_warp(
                pts.reshape(batch_size, n_samples, 3), color_maps, intrinsics,
                w2cs, img_wh=None)  # [N_rays, n_samples, N_views,  3] , [N_rays, n_samples, N_views]

        # - extract patch
        if_patch_blending = False if rays_uv is None else True
        pts_patch_color, pts_patch_mask = None, None
        if if_patch_blending:
            pts_patch_color, pts_patch_mask = self.patch_projector.patch_warp(
                pts.reshape([batch_size, n_samples, 3]),
                rays_uv,
                flip_sign.reshape([batch_size, n_samples, 1]) * gradients_norm.reshape([batch_size, n_samples, 3]),
                # * flip the direction of gradients
                color_maps,
                intrinsics[0], intrinsics,
                query_c2w, torch.inverse(w2cs), img_wh=None,
                detach_normal=True   # detach the normals to avoid unstable optimization
            )  # (N_rays, n_samples, N_src, Npx, 3), (N_rays, n_samples, N_src, Npx)
            N_src, Npx = pts_patch_mask.shape[2:]
            pts_patch_color = pts_patch_color.view(batch_size, n_samples, N_src, Npx, 3)
            pts_patch_mask = pts_patch_mask.view(batch_size, n_samples, N_src, Npx)

        if if_pixel_blending or if_patch_blending:
            sampled_color_pixel, sampled_color_pixel_mask, \
            sampled_color_patch, sampled_color_patch_mask = color_blend(
                blending_weights,
                img_index=img_index,
                pts_pixel_color=pts_pixel_color,
                pts_pixel_mask=pts_pixel_mask,
                pts_patch_color=pts_patch_color,
                pts_patch_mask=pts_patch_mask
            )  # [n, 3], [n, 1]

        if if_pixel_blending:
            sampled_color_pixel = sampled_color_pixel.view(batch_size, n_samples, 3)
            # sampled_color_pixel_mask = sampled_color_pixel_mask.view(batch_size, n_samples)
        else:
            sampled_color_pixel = None

        # patch blending
        if if_patch_blending:
            sampled_color_patch = sampled_color_patch.view(batch_size, n_samples, Npx, 3)
            sampled_color_patch_mask = sampled_color_patch_mask.view(batch_size, n_samples)
        else:
            sampled_color_patch, sampled_color_patch_mask = None, None

        ############################# finish pixel patch extraction  ##########################################

        pts_norm = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).reshape(batch_size, n_samples)
        inside_sphere = (pts_norm < 1.0).float().detach()
        relax_inside_sphere = (pts_norm < 1.2).float().detach()
        near_surface = (udf < 0.05).float().detach()

        # Render with background
        if background_alpha is not None:
            # ! introduce biased depth; since first two or three points are outside of sphere; not use inside_sphere
            # alpha = alpha * inside_sphere + background_alpha[:, :n_samples] * (1.0 - inside_sphere)
            alpha = torch.cat([alpha, background_alpha[:, n_samples:]], dim=-1)

            # sampled_color_base = sampled_color_base * inside_sphere[:, :, None] + \
            #                      background_sampled_color[:, :n_samples] * (1.0 - inside_sphere)[:, :, None]
            sampled_color_base = torch.cat([sampled_color_base, background_sampled_color[:, n_samples:]], dim=1)

            # sampled_color = sampled_color * inside_sphere[:, :, None] + \
            #                 background_sampled_color[:, :n_samples] * (1.0 - inside_sphere)[:, :, None]
            sampled_color = torch.cat([sampled_color, background_sampled_color[:, n_samples:]], dim=1)

            if sampled_color_pixel is not None:
                sampled_color_pixel = sampled_color_pixel * inside_sphere[:, :, None] + \
                                      background_sampled_color[:, :n_samples] * (1.0 - inside_sphere)[:, :, None]
                sampled_color_pixel = torch.cat([sampled_color_pixel, background_sampled_color[:, n_samples:]], dim=1)

        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]).to(device), 1. - alpha + 1e-7], -1), -1)[
                          :, :-1]

        weights_sum = weights.sum(dim=-1, keepdim=True)

        color_base = (sampled_color_base * weights[:, :, None]).sum(dim=1)
        color = (sampled_color * weights[:, :, None]).sum(dim=1)

        color_pixel = None
        if sampled_color_pixel is not None:
            color_pixel = (sampled_color_pixel * weights[:, :, None]).sum(dim=1)

        fused_patch_colors, fused_patch_mask = None, None
        if sampled_color_patch is not None:
            fused_patch_colors = (sampled_color_patch * weights[:, :n_samples, None, None]).sum(
                dim=1)  # [batch_size, Npx, 3]
            fused_patch_mask = (sampled_color_patch_mask.float() * weights[:, :n_samples]).sum(dim=1)  # [batch_size]

        depth = (mid_z_vals * weights[:, :n_samples]).sum(dim=1, keepdim=True)
        if background_rgb is not None:  # Fixed background, usually black
            color = color + background_rgb * (1.0 - weights_sum)

        # Eikonal loss
        gradient_error_ = (torch.linalg.norm(gradients.reshape(batch_size, n_samples, 3), ord=2,
                                             dim=-1) - 1.0) ** 2
        gradient_error = (relax_inside_sphere * gradient_error_).sum() / (relax_inside_sphere.sum() + 1e-5)

        # calculate the eikonal loss near zero level set
        gradient_error_near_surface = (near_surface * gradient_error_).sum() / (near_surface.sum() + 1e-5)

        gradients = gradients.reshape(batch_size, n_samples, 3)

        # gradients = gradients / (
        #         torch.linalg.norm(gradients, ord=2, dim=-1, keepdim=True) + 1e-5)  # normalize to unit vector

        if torch.any(torch.isnan(gradient_error)).cpu().numpy().item():
            pdb.set_trace()

        if vis_prob is not None:
            # gradients_flip = gradients * vis_prob[:, :, None] + gradients * (1 - vis_prob[:, :, None]) * -1
            gradients_flip = flip_sign.reshape([batch_size, n_samples, 1]) * gradients
        else:
            gradients_flip = gradients

        # geo regularization, encourages the UDF have clear surfaces
        sparse_error = torch.exp(-self.sparse_scale_factor * udf).sum(dim=1, keepdim=False).mean()

        return {
            'color_base': color_base,
            'color': color,
            'color_pixel': color_pixel,
            'patch_colors': fused_patch_colors,
            'patch_mask': fused_patch_mask,
            'weights': weights,
            's_val': 1.0 / inv_s,
            'beta': 1.0 / beta,
            'gamma': gamma,
            'depth': depth,
            'gradient_error': gradient_error,
            'gradient_error_near_surface': gradient_error_near_surface,
            'normals': (gradients_flip * weights[:, :n_samples, None]).sum(dim=1, keepdim=False),
            'gradients': gradients,
            'gradients_flip': gradients_flip,
            'inside_sphere': inside_sphere,
            'udf': udf,
            'gradient_mag': gradients_mag.reshape(batch_size, n_samples),
            'true_cos': true_cos.reshape(batch_size, n_samples),
            'vis_prob': vis_prob.reshape(batch_size, n_samples) if vis_prob is not None else None,
            'alpha': alpha[:, :n_samples],
            'alpha_plus': alpha_plus[:, :n_samples],
            'alpha_minus': alpha_minus[:, :n_samples],
            'mid_z_vals': mid_z_vals,
            'dists': dists,
            'sparse_error': sparse_error,
            'alpha_occ': alpha_occ,
            'raw_occ': raw_occ.reshape(batch_size, n_samples)
        }

    def render(self, rays_o, rays_d, near, far,
               cos_anneal_ratio=None,
               perturb_overwrite=-1, background_rgb=None,
               flip_saturation=0,
               # * blending params
               color_maps=None,
               w2cs=None,
               intrinsics=None,
               query_c2w=None,
               img_index=None,
               rays_uv=None,
               ):
        """
        Render the image given the input rays and other rendering parameters.

        Parameters:
        rays_o: Origin of the rays, a tensor of shape (batch_size, 3).
        rays_d: Direction of the rays, a tensor of shape (batch_size, 3).
        near: Near clipping plane distance, a scalar or tensor.
        far: Far clipping plane distance, a scalar or tensor.
        cos_anneal_ratio: Cosine annealing ratio for rendering, used to adjust the sampling strategy.
        perturb_overwrite: Override value for ray perturbation, -1 means using the class default value.
        background_rgb: Background color in RGB, used when a ray intersects nothing.
        flip_saturation: Flag for flipping the saturation of colors.
        color_maps: Color maps used for rendering.
        w2cs: World to camera space transformations.
        intrinsics: Camera intrinsics matrix.
        query_c2w: Camera to world space transformation for querying the scene.
        img_index: Image index for multi-image rendering.
        rays_uv: UV coordinates of the rays.

        Returns:
        A dictionary containing various rendering results, such as colors, depths, normals, etc.
        """
        # Determine the device where the data is located
        device = rays_o.device
        # Determine the batch size
        batch_size = len(rays_o)

        # Ensure near and far are tensors for subsequent operations
        if not isinstance(near, torch.Tensor):
            near = torch.Tensor([near]).view(1, 1).to(device)
            far = torch.Tensor([far]).view(1, 1).to(device)

        # Calculate the average sampling distance
        # 给定范围内采样点之间的平均距离
        sample_dist = ((far - near) / self.n_samples).mean().item()
        # Generate uniformly distributed sample points in the [0, 1] range
        # 平均从0-1的范围中取n个sample点，z_vals中存了这些sample点的z值

        z_vals = torch.linspace(0.0, 1.0, self.n_samples).to(device)

        # Convert sample points from [0, 1] to [near, far] range
        # 将0-1的范围值切换到near-far的范围
        z_vals = near + (far - near) * z_vals[None, :]
        # Prepare for potentially generating additional sample points outside the [near, far] range
        z_vals_outside = None
        if self.n_outside > 0:
            z_vals_outside = torch.linspace(1e-3, 1.0 - 1.0 / (self.n_outside + 1.0), self.n_outside)

        # Initialize parameters for ray sampling
        # 为ray sampling准备初始化参数
        n_samples = self.n_samples # 采样点数
        perturb = self.perturb # 扰动的参数
        # Override the perturbation value if specified
        if perturb_overwrite >= 0:
            perturb = perturb_overwrite
        # Add perturbation to sample points if required
        if perturb > 0:
            # Generate random perturbation offsets
            t_rand = (torch.rand([batch_size, 1]) - 0.5).to(device)
            z_vals = z_vals + t_rand * 2.0 / self.n_samples

            if self.n_outside > 0:
                # Stratified sampling for the additional sample points outside
                mids = .5 * (z_vals_outside[..., 1:] + z_vals_outside[..., :-1])
                upper = torch.cat([mids, z_vals_outside[..., -1:]], -1)
                lower = torch.cat([z_vals_outside[..., :1], mids], -1)
                t_rand = torch.rand(z_vals_outside.shape).to(device)
                z_vals_outside = lower + (upper - lower) * t_rand

        # Adjust the position of sample points outside the clipping plane
        if self.n_outside > 0:
            z_vals_outside = far / torch.flip(z_vals_outside, dims=[-1]) + 1.0 / self.n_samples

        # Initialize background parameters
        background_alpha = None
        background_sampled_color = None
        background_color = torch.zeros([1, 3])

        # Perform importance sampling if required
        # TODO 这里可以用于修改取样点的密集程度，比如根据Gaussians的分布、Edge Map的分布，来进行上采样
        if self.n_importance > 0:
            # Choose the appropriate importance sampling method
            # z_vals中存储的会是最终的采样点的z values
            # 这两个函数之间最大的区别在于上采样的策略不同，第一个的上采样更加精细一些
            if self.upsampling_type == 'classical':
                z_vals = self.importance_sample(rays_o, rays_d, z_vals, sample_dist)
            elif self.upsampling_type == 'mix':
                z_vals = self.importance_sample_mix(rays_o, rays_d, z_vals, sample_dist)

            # Update the number of sample points
            # 采样点的最终数量等于n_samples+n_importance（平均采样点数量+权重采样点数量）
            n_samples = self.n_samples + self.n_importance

        # 采样完毕，开始进入rendering的部分
        # Render the background if additional sample points are used
        if self.n_outside > 0:
            # Combine inside and outside sample points for background rendering
            z_vals_feed = torch.cat([z_vals, z_vals_outside], dim=-1)
            z_vals_feed, _ = torch.sort(z_vals_feed, dim=-1)
            # Call the core rendering function for the background
            ret_outside = self.render_core_outside(rays_o, rays_d, z_vals_feed, sample_dist, self.nerf,
                                                   background_rgb=background_rgb)

            # Extract the rendered background color and alpha
            background_sampled_color = ret_outside['sampled_color']
            background_alpha = ret_outside['alpha']

        # Call the core rendering function for the foreground objects
        ret_fine = self.render_core(rays_o,
                                    rays_d,
                                    z_vals,
                                    sample_dist,
                                    self.udf_network,
                                    self.deviation_network,
                                    self.color_network,
                                    beta_network=self.beta_network,
                                    cos_anneal_ratio=cos_anneal_ratio,
                                    background_rgb=background_rgb,
                                    background_alpha=background_alpha,
                                    background_sampled_color=background_sampled_color,
                                    flip_saturation=flip_saturation,
                                    color_maps=color_maps,
                                    w2cs=w2cs,
                                    intrinsics=intrinsics,
                                    query_c2w=query_c2w,
                                    img_index=img_index,
                                    rays_uv=rays_uv,
                                    )

        # Calculate the error of sparse sampling
        sparse_error = ret_fine['sparse_error']

        # Initialize parameters for random error estimation
        sparse_random_error = 0.0
        udf_random = None
        # Generate random points for estimating the random error
        pts_random = torch.rand([1024, 3]).float().to(device) * 2 - 1  # normalized to (-1, 1)
        udf_random = self.udf_network.udf(pts_random)
        # Calculate the average error based on the random points
        if (udf_random < 0.01).sum() > 10:
            sparse_random_error = torch.exp(-self.sparse_scale_factor * udf_random[udf_random < 0.01]).mean()

        # Return all rendering results
        return {
            'color_base': ret_fine['color_base'],
            'color': ret_fine['color'],
            'color_pixel': ret_fine['color_pixel'],
            'patch_colors': ret_fine['patch_colors'],
            'patch_mask': ret_fine['patch_mask'],
            'weight_sum': ret_fine['weights'][:, :n_samples].sum(dim=-1, keepdim=True),
            'weight_sum_fg_bg': ret_fine['weights'][:, :].sum(dim=-1, keepdim=True),
            'depth': ret_fine['depth'],
            'variance': ret_fine['s_val'],
            'beta': ret_fine['beta'],
            'gamma': ret_fine['gamma'],
            'normals': ret_fine['normals'],
            'gradients': ret_fine['gradients'],
            'gradients_flip': ret_fine['gradients_flip'],
            'weights': ret_fine['weights'],
            'gradient_error': ret_fine['gradient_error'],
            'gradient_error_near_surface': ret_fine['gradient_error_near_surface'],
            'inside_sphere': ret_fine['inside_sphere'],
            'udf': ret_fine['udf'],
            'z_vals': z_vals,
            'gradient_mag': ret_fine['gradient_mag'],
            'true_cos': ret_fine['true_cos'],
            'vis_prob': ret_fine['vis_prob'],
            'alpha': ret_fine['alpha'],
            'alpha_plus': ret_fine['alpha_plus'],
            'alpha_minus': ret_fine['alpha_minus'],
            'mid_z_vals': ret_fine['mid_z_vals'],
            'dists': ret_fine['dists'],
            'sparse_error': sparse_error,
            'alpha_occ': ret_fine['alpha_occ'],
            'raw_occ': ret_fine['raw_occ'],
            'sparse_random_error': sparse_random_error
        }




    '''
    以下两个函数就是用于根据UDF等方式进行上采样的
    '''
    # TODO 因此这两个函数的位置，就可以成为我们edge map，gs_guide_udf的突破口
    @torch.no_grad()
    def importance_sample(self, rays_o, rays_d, z_vals, sample_dist):
        """
        执行重要性采样，通过递归方式增加采样点，以提高光线追踪的精度。

        参数:
        rays_o: 光源位置的批次数组。
        rays_d: 光线方向的批次数组。
        z_vals: 初始的z值采样点批次数组。
        sample_dist: 采样距离的分布。

        返回:
        更新后的z值采样点批次数组。
        """
        # 获取批次大小
        batch_size = rays_o.shape[0]

        # 是否使用无偏向上采样
        up_sample = self.up_sample_unbias

        # 计算采样点的位置
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]

        # 通过UDF网络预测采样点的属性
        udf_nn_output = self.udf_network(pts.reshape(-1, 3))
        udf_nn_output = udf_nn_output.reshape(batch_size, self.n_samples, -1)
        udf = udf_nn_output[:, :, 0]

        # 迭代执行向上采样
        # 与NeuS中一样，多次循环进行上采样
        for i in range(self.up_sample_steps):
            # 执行向上采样步骤，增加采样点
            new_z_vals = up_sample(rays_o,
                                   rays_d,
                                   z_vals,
                                   udf,
                                   sample_dist,
                                   self.n_importance // self.up_sample_steps,
                                   64 * 2 ** i,
                                   # ! important; use larger beta **(i+1); otherwise sampling will be biased
                                   64 * 2 ** (i + 1),
                                   # ! important; use much larger beta **(i+1); otherwise sampling will be biased
                                   gamma=np.clip(20 * 2 ** (self.up_sample_steps - i), 20, 320),
                                   )
            # 合并新的采样点与旧的采样点
            z_vals, udf = self.cat_z_vals(rays_o,
                                          rays_d,
                                          z_vals,
                                          new_z_vals,
                                          udf,
                                          last=(i + 1 == self.up_sample_steps))

        # 返回最终的采样点数组
        return z_vals

    def extract_geometry(self, bound_min, bound_max, resolution, threshold=0.01, device='cpu'):
        ret = extract_geometry(bound_min, bound_max, resolution, threshold,
                               lambda pts: self.udf_network.udf(pts)[:, 0], device)
        return ret

    @torch.no_grad()
    def importance_sample_mix(self, rays_o, rays_d, z_vals, sample_dist):
        """
        Importance sampling for mixture models.

        This method aims to improve the sampling efficiency by adaptively adjusting the sampling points along the rays.
        It first initializes the sampling points and then iteratively refines the sampling distribution through a neural network to estimate the volume density.

        Parameters:
        rays_o: Ray origin tensor.
        rays_d: Ray direction tensor.
        z_vals: Initial sampling points tensor.
        sample_dist: Distribution used for initial sampling.

        Returns:
        The refined sampling points tensor.
        """
        """
        This sampling can make optimization avoid bad initialization of early stage
        make optimization more robust
        Parameters
        ----------
        rays_o :
        rays_d :
        z_vals :
        sample_dist :

        Returns
        -------

        """
        # Get the batch size
        batch_size = rays_o.shape[0]

        # Use the initial sampling points as the base for subsequent refinement
        # 以最初的平均采样点作为后续 refinment 的基础
        base_z_vals = z_vals

        # Calculate the intermediate points along the rays
        # Up sample

        # 这一步就是进行平均采样点的计算，计算在每条rays上相同位置的采样点（距离每条光线的起点距离相同）
        # 生成每条光线在各个采样深度点的空间坐标
        # 这些None就是用于增加一个维度，用于这个式子的 broadcast 运算
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None] # (N_rays, M_samples, 3) N条光线上M个采样点，每个采样点有xyz三个坐标

        # Use the neural network to estimate the volume density at the intermediate points
        # 将 pts 由形状 (N_rays, M_samples, 3) 展平为 (N_rays * M_samples, 3)，以便喂入神经网络
        udf_nn_output = self.udf_network(pts.reshape(-1, 3)) # 输出每个点的UDF估计值
        '''
        这里 batch_size 默认是 N_rays，self.n_samples 是 M_samples，所以 reshape 成 (N_rays, M_samples, -1) 的形状。
        假定神经网络的输出是 (N_rays * M_samples, C) 的形状
        '''
        udf_nn_output = udf_nn_output.reshape(batch_size, self.n_samples, -1)
        # 假设神经网络的输出的第一个通道（即 udf_nn_output[:, :, 0]）是我们需要的体积密度值，将其赋给 udf
        # 也就是得到每个采样点对应的UDF值
        udf = udf_nn_output[:, :, 0]
        base_udf = udf

        # Get the beta and gamma parameters from the beta network, clipping their values to a reasonable range
        # 这两个参数是用于在后续进行计算的时候可以进行梯度下降来训练的
        beta = self.beta_network.get_beta().clip(1e-6, 1e6)
        gamma = self.beta_network.get_gamma().clip(1e-6, 1e6)

        # Perform iterative refinement of the sampling points, focusing on increasing the sampling density
        # * not occlussion-aware sample to avoid missing the true surface
        for i in range(self.up_sample_steps):
            # Perform upsampling without considering occlusion, gradually increasing the sampling density
            new_z_vals = self.up_sample_no_occ_aware(rays_o,
                                                     rays_d,
                                                     z_vals,
                                                     udf,
                                                     sample_dist,
                                                     self.n_importance // (self.up_sample_steps + 1),
                                                     64 * 2 ** i,
                                                     64 * 2 ** (i + 1),
                                                     gamma,
                                                     )
            # Merge the new sampling points with the existing ones
            z_vals, udf = self.cat_z_vals(rays_o,
                                          rays_d,
                                          z_vals,
                                          new_z_vals,
                                          udf,
                                          )

        # In the final iteration, switch to unbiased sampling to further refine the sampling points
        for i in range(self.up_sample_steps - 1, self.up_sample_steps):
            # Adjust the parameters to ensure the sampling is not biased
            new_z_vals = self.up_sample_unbias(rays_o,
                                               rays_d,
                                               z_vals,
                                               udf,
                                               sample_dist,
                                               self.n_importance // (self.up_sample_steps + 1),
                                               64 * 2 ** i,
                                               # ! important; use larger beta **(i+1); otherwise sampling will be biased
                                               64 * 2 ** (i + 1),
                                               # ! important; use much larger beta **(i+1); otherwise sampling will be biased
                                               gamma=20 if i < 4 else 10,
                                               )
            # Merge the new sampling points, marking the last iteration
            z_vals, udf = self.cat_z_vals(rays_o,
                                          rays_d,
                                          z_vals,
                                          new_z_vals,
                                          udf,
                                          last=(i + 1 == self.up_sample_steps))

        # Return the refined sampling points
        return z_vals

    # TODO 或许从这里改Sample的方式也是极好的（或者利用这个代码重新写）
    def up_sample_unbias(self, rays_o, rays_d, z_vals, udf, sample_dist,
                         n_importance, inv_s, beta, gamma, debug=False):
        """
        对射线进行上采样，以减少偏差。

        该方法通过在可能的第一个表面交点处添加更多采样点，来改善NeuS风格的上采样策略。
        它处理了透明物体内部的采样问题，避免了由于透明度估计不准确导致的采样偏差。

        参数:
        rays_o: 光源方向向量，形状为(batch_size, 3)。
        rays_d: 射线方向向量，形状为(batch_size, 3)。
        z_vals: 透明度估计的采样位置，形状为(batch_size, n_samples)。
        udf: 单层透明度估计，形状为(batch_size, n_samples)。
        sample_dist: 最初的采样距离。
        n_importance: 重要采样的数量。
        inv_s: 透明度函数的缩放参数。
        beta: 控制透明度函数形状的参数。
        gamma: 控制重要采样权重的参数。
        debug: 是否开启调试模式。

        返回:
        上采样后的z值，形状为(batch_size, n_importance)。
        """
        # 获取设备信息和批量大小、样本数量
        device = z_vals.device
        batch_size, n_samples = z_vals.shape

        # 计算采样点的位置
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
        # 计算每个采样点距离原点的距离
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)

        # 判断采样点是否在单位球内
        inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)

        # 调整udf的形状以方便后续计算
        udf = udf.reshape(batch_size, n_samples)

        # 计算相邻采样点的距离
        dists_raw = z_vals[..., 1:] - z_vals[..., :-1]
        # 将最后一个采样点距离设置为 sample_dist
        dists_raw = torch.cat([dists_raw, torch.Tensor([sample_dist]).to(device).expand(dists_raw[..., :1].shape)], -1)

        # 扩展射线方向向量以匹配采样点数量
        dirs = rays_d[:, None, :].expand(pts.shape)

        # 计算相邻采样点的udf和z值
        prev_udf, next_udf = udf[:, :-1], udf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_udf = (prev_udf + next_udf) * 0.5
        mid_z_vals = (prev_z_vals + next_z_vals) * 0.5

        # 计算真实的距离间隔
        dists = (next_z_vals - prev_z_vals)

        # 使用udf近似Signed Distance Function (SDF)
        fake_sdf = udf
        prev_sdf, next_sdf = fake_sdf[:, :-1], fake_sdf[:, 1:]

        # 计算法向量的cos值，表示透明度的变化率
        true_cos = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)
        cos_val = -1 * torch.abs(true_cos)

        # 处理边界条件，确保cos值在合理范围内（保证都在单位球内）
        cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere

        # 创建可视掩码，用于处理透明度近似的问题
        vis_mask = torch.ones_like(true_cos).to(true_cos.device).to(true_cos.dtype) * (true_cos < 0.05).float()
        vis_mask = vis_mask.reshape(batch_size, n_samples - 1)
        vis_mask = torch.cat([torch.ones([batch_size, 1]).to(device), vis_mask], dim=-1)

        # 计算遮挡概率
        raw_occ = self.udf2logistic(udf, beta, 1.0, 1.0).reshape(batch_size, n_samples)
        alpha_occ = 1.0 - torch.exp(-F.relu(raw_occ) * gamma * dists_raw)

        # 使用可视掩码调整采样分布，使其更加均匀
        vis_prob = torch.cumprod(
            torch.cat([torch.ones([batch_size, 1]).to(device), (1. - alpha_occ + vis_mask).clip(0, 1) + 1e-7], -1), -1)[
                   :, :-1]

        # 根据透明度和法向量方向计算alpha值
        # 区别就出现在了这里，这里的alpha值计算是根据了udf计算出来的遮挡概率、法向量、可视掩码的
        # 因此这也导致了后续的采样权重也是受这些因素的影响的
        signs_prob = vis_prob[:, :-1]
        sdf_plus = mid_udf
        sdf_minus = mid_udf * -1
        alpha_plus = self.sdf2alpha(sdf_plus, cos_val, dists, inv_s)
        alpha_minus = self.sdf2alpha(sdf_minus, cos_val, dists, inv_s)
        alpha = alpha_plus * signs_prob + alpha_minus * (1 - signs_prob)
        alpha = alpha.reshape(batch_size, n_samples - 1)

        # 计算权重并进行上采样
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1]).to(device), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()

        # 检查是否存在NaN值
        flag = torch.any(torch.isnan(z_samples)).cpu().numpy().item()
        if flag:
            print("z_vals", z_samples[torch.isnan(z_samples)])
            print('z_vals have nan values')
            pdb.set_trace()
            # raise Exception("gradients have nan values")

        return z_samples
    def up_sample_no_occ_aware(self, rays_o, rays_d, z_vals, udf, sample_dist,
                               n_importance, inv_s, beta, gamma):
        """
        This function upsamples points along rays for volume rendering, focusing on areas where the unsigned distance function (UDF) is close to zero.
        Unlike SDF (Signed Distance Function), UDF does not have clear sign changes, so it may miss true surfaces.
        # 就是对于UDF小的地方进行更多地采样


        Arguments:
        - rays_o: Original points of the rays (tensor of shape [batch_size, 3]).
        - rays_d: Directions of the rays (tensor of shape [batch_size, 3]).
        - z_vals: Depth values along each ray (tensor of shape [batch_size, n_samples]).
        - udf: Unsigned Distance Function values at sampled points (tensor of shape [batch_size * n_samples]).
        - sample_dist: Distance between samples.
        - n_importance: Number of important samples to draw.
        - inv_s: Inverse scale parameter (not used in this function).
        - beta, gamma: Parameters for the logistic function.

        Returns:
        - z_samples: New depth samples along the rays.
        """

        device = z_vals.device
        batch_size, n_samples = z_vals.shape

        # Calculate points along rays using origin and direction.
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # Shape: [batch_size, n_samples, 3]

        # Compute the distance from each point to the origin.
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)

        # Determine which points are inside the unit sphere.
        inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)

        # Reshape UDF values to match batch and sample dimensions.
        udf = udf.reshape(batch_size, n_samples)

        # Compute distances between consecutive depth samples.
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([sample_dist]).to(device).expand(dists[..., :1].shape)], -1)

        # Expanding ray directions to match the shape of points.
        dirs = rays_d[:, None, :].expand(pts.shape)

        # * The probability of occlusion calculation.
        # Convert UDF values to a logistic distribution for occlusion probability.
        # 通过每个点的udf值（加一点计算），计算得到每个点的被遮挡概率
        raw_occ = self.udf2logistic(udf, beta, gamma, 1.0)

        # Compute alpha values representing the accumulated opacity. Near zero UDF values mean alpha -> 1,
        # otherwise alpha -> 0.
        # 就是论文中的surface existence probability h(r(t))
        alpha_occ = 1.0 - torch.exp(-F.relu(raw_occ.reshape(batch_size, n_samples)) * dists)

        # Sample new depth values based on the computed alpha values.
        z_samples = sample_pdf(z_vals, alpha_occ[:, :-1], n_importance, det=True).detach()

        # Check for NaN values in the samples and pause execution for debugging if any are found.
        flag = torch.any(torch.isnan(z_samples)).cpu().numpy().item()
        if flag:
            print("z_vals", z_samples[torch.isnan(z_samples)])
            print('z_vals have nan values')
            pdb.set_trace()  # Pause for debugging

        return z_samples
    def render_with_gs(self, rays_o, rays_d, near, far,
               cos_anneal_ratio=None,
               perturb_overwrite=-1, background_rgb=None,
               flip_saturation=0,
               # * blending params
               color_maps=None,
               w2cs=None,
               intrinsics=None,
               query_c2w=None,
               img_index=None,
               rays_uv=None,
               ):
        """
        Render the image given the input rays and other rendering parameters.

        Parameters:
        rays_o: Origin of the rays, a tensor of shape (batch_size, 3).
        rays_d: Direction of the rays, a tensor of shape (batch_size, 3).
        near: Near clipping plane distance, a scalar or tensor.
        far: Far clipping plane distance, a scalar or tensor.
        cos_anneal_ratio: Cosine annealing ratio for rendering, used to adjust the sampling strategy.
        perturb_overwrite: Override value for ray perturbation, -1 means using the class default value.
        background_rgb: Background color in RGB, used when a ray intersects nothing.
        flip_saturation: Flag for flipping the saturation of colors.
        color_maps: Color maps used for rendering.
        w2cs: World to camera space transformations.
        intrinsics: Camera intrinsics matrix.
        query_c2w: Camera to world space transformation for querying the scene.
        img_index: Image index for multi-image rendering.
        rays_uv: UV coordinates of the rays.

        Returns:
        A dictionary containing various rendering results, such as colors, depths, normals, etc.
        """
        # Determine the device where the data is located
        device = rays_o.device
        # Determine the batch size
        batch_size = len(rays_o)

        # Ensure near and far are tensors for subsequent operations
        if not isinstance(near, torch.Tensor):
            near = torch.Tensor([near]).view(1, 1).to(device)
            far = torch.Tensor([far]).view(1, 1).to(device)

        # Calculate the average sampling distance
        # 给定范围内采样点之间的平均距离
        sample_dist = ((far - near) / self.n_samples).mean().item()
        # Generate uniformly distributed sample points in the [0, 1] range
        # 平均从0-1的范围中取n个sample点，z_vals中存了这些sample点的z值

        z_vals = torch.linspace(0.0, 1.0, self.n_samples).to(device)

        # Convert sample points from [0, 1] to [near, far] range
        # 将0-1的范围值切换到near-far的范围
        z_vals = near + (far - near) * z_vals[None, :]
        # Prepare for potentially generating additional sample points outside the [near, far] range
        z_vals_outside = None
        if self.n_outside > 0:
            z_vals_outside = torch.linspace(1e-3, 1.0 - 1.0 / (self.n_outside + 1.0), self.n_outside)

        # Initialize parameters for ray sampling
        # 为ray sampling准备初始化参数
        n_samples = self.n_samples # 采样点数
        perturb = self.perturb # 扰动的参数
        # Override the perturbation value if specified
        if perturb_overwrite >= 0:
            perturb = perturb_overwrite
        # Add perturbation to sample points if required
        if perturb > 0:
            # Generate random perturbation offsets
            t_rand = (torch.rand([batch_size, 1]) - 0.5).to(device)
            z_vals = z_vals + t_rand * 2.0 / self.n_samples

            if self.n_outside > 0:
                # Stratified sampling for the additional sample points outside
                mids = .5 * (z_vals_outside[..., 1:] + z_vals_outside[..., :-1])
                upper = torch.cat([mids, z_vals_outside[..., -1:]], -1)
                lower = torch.cat([z_vals_outside[..., :1], mids], -1)
                t_rand = torch.rand(z_vals_outside.shape).to(device)
                z_vals_outside = lower + (upper - lower) * t_rand

        # Adjust the position of sample points outside the clipping plane
        if self.n_outside > 0:
            z_vals_outside = far / torch.flip(z_vals_outside, dims=[-1]) + 1.0 / self.n_samples

        # Initialize background parameters
        background_alpha = None
        background_sampled_color = None
        background_color = torch.zeros([1, 3])

        # Perform importance sampling if required
        # TODO 这里可以用于修改取样点的密集程度，比如根据Gaussians的分布、Edge Map的分布，来进行上采样
        # if self.n_importance > 0:
        #     # Choose the appropriate importance sampling method
        #     # z_vals中存储的会是最终的采样点的z values
        #     # 这两个函数之间最大的区别在于上采样的策略不同，第一个的上采样更加精细一些
        #     if self.upsampling_type == 'classical':
        #         z_vals = self.importance_sample(rays_o, rays_d, z_vals, sample_dist)
        #     elif self.upsampling_type == 'mix':
        #         z_vals = self.importance_sample_mix(rays_o, rays_d, z_vals, sample_dist)
        #
        #     # Update the number of sample points
        #     # 采样点的最终数量等于n_samples+n_importance（平均采样点数量+权重采样点数量）
        #     n_samples = self.n_samples + self.n_importance

        # 采样完毕，开始进入rendering的部分
        # Render the background if additional sample points are used
        if self.n_outside > 0:
            # Combine inside and outside sample points for background rendering
            z_vals_feed = torch.cat([z_vals, z_vals_outside], dim=-1)
            z_vals_feed, _ = torch.sort(z_vals_feed, dim=-1)
            # Call the core rendering function for the background
            ret_outside = self.render_core_outside(rays_o, rays_d, z_vals_feed, sample_dist, self.nerf,
                                                   background_rgb=background_rgb)

            # Extract the rendered background color and alpha
            background_sampled_color = ret_outside['sampled_color']
            background_alpha = ret_outside['alpha']

        # Call the core rendering function for the foreground objects
        ret_fine = self.render_core(rays_o,
                                    rays_d,
                                    z_vals,
                                    sample_dist,
                                    self.udf_network,
                                    self.deviation_network,
                                    self.color_network,
                                    beta_network=self.beta_network,
                                    cos_anneal_ratio=cos_anneal_ratio,
                                    background_rgb=background_rgb,
                                    background_alpha=background_alpha,
                                    background_sampled_color=background_sampled_color,
                                    flip_saturation=flip_saturation,
                                    color_maps=color_maps,
                                    w2cs=w2cs,
                                    intrinsics=intrinsics,
                                    query_c2w=query_c2w,
                                    img_index=img_index,
                                    rays_uv=rays_uv,
                                    )

        # Calculate the error of sparse sampling
        sparse_error = ret_fine['sparse_error']

        # Initialize parameters for random error estimation
        sparse_random_error = 0.0
        udf_random = None
        # Generate random points for estimating the random error
        pts_random = torch.rand([1024, 3]).float().to(device) * 2 - 1  # normalized to (-1, 1)
        udf_random = self.udf_network.udf(pts_random)
        # Calculate the average error based on the random points
        if (udf_random < 0.01).sum() > 10:
            sparse_random_error = torch.exp(-self.sparse_scale_factor * udf_random[udf_random < 0.01]).mean()

        # Return all rendering results
        return {
            'color_base': ret_fine['color_base'],
            'color': ret_fine['color'],
            'color_pixel': ret_fine['color_pixel'],
            'patch_colors': ret_fine['patch_colors'],
            'patch_mask': ret_fine['patch_mask'],
            'weight_sum': ret_fine['weights'][:, :n_samples].sum(dim=-1, keepdim=True),
            'weight_sum_fg_bg': ret_fine['weights'][:, :].sum(dim=-1, keepdim=True),
            'depth': ret_fine['depth'],
            'variance': ret_fine['s_val'],
            'beta': ret_fine['beta'],
            'gamma': ret_fine['gamma'],
            'normals': ret_fine['normals'],
            'gradients': ret_fine['gradients'],
            'gradients_flip': ret_fine['gradients_flip'],
            'weights': ret_fine['weights'],
            'gradient_error': ret_fine['gradient_error'],
            'gradient_error_near_surface': ret_fine['gradient_error_near_surface'],
            'inside_sphere': ret_fine['inside_sphere'],
            'udf': ret_fine['udf'],
            'z_vals': z_vals,
            'gradient_mag': ret_fine['gradient_mag'],
            'true_cos': ret_fine['true_cos'],
            'vis_prob': ret_fine['vis_prob'],
            'alpha': ret_fine['alpha'],
            'alpha_plus': ret_fine['alpha_plus'],
            'alpha_minus': ret_fine['alpha_minus'],
            'mid_z_vals': ret_fine['mid_z_vals'],
            'dists': ret_fine['dists'],
            'sparse_error': sparse_error,
            'alpha_occ': ret_fine['alpha_occ'],
            'raw_occ': ret_fine['raw_occ'],
            'sparse_random_error': sparse_random_error
        }