- 获取sample点的流程：

```python
def training():
# 随机重排图像顺序，以增加训练的多样性
        image_perm = torch.randperm(self.udf_dataset.n_images)
```



```python
# 根据当前迭代步数获取图像索引，并准备随机射线和对应的Ground truth
        img_idx = image_perm[self.udf_iter_step % len(image_perm)]
        # 随机获取一些随机射线和对应的Ground truth用于训练
        sample = self.udf_dataset.gen_random_rays_patches_at(
            img_idx, self.batch_size,
            crop_patch=color_patch_weight > 0.0, 	 h_patch_size=self.udf_color_loss_func.h_patch_size)
```



```python
    def gen_random_rays_patches_at(self, img_idx, batch_size, importance_sample=False, h_patch_size=3,
                                   crop_patch=False):
        """
        Generate random rays in world space from a specified camera.

        Arguments:
        - img_idx: Index of the image from which to generate rays.
        - batch_size: Number of rays to generate.
        - importance_sample: If True, perform importance sampling in the valid mask regions.
        - h_patch_size: Half the size of the patch to crop if crop_patch is True.
        - crop_patch: If True, crop patches from images.

        Returns:
        A dictionary containing the sampled rays and associated data.
        """

        if not importance_sample:
            # Generate random pixel locations within the image dimensions
            pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])
            pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])
        elif importance_sample and self.masks is not None:
            # Generate some random pixel locations within the entire image
            pixels_x_1 = torch.randint(low=0, high=self.W, size=[batch_size // 4])
            pixels_y_1 = torch.randint(low=0, high=self.H, size=[batch_size // 4])

            # Generate a full grid of pixel coordinates
            ys, xs = torch.meshgrid(torch.linspace(0, self.H - 1, self.H), torch.linspace(0, self.W - 1, self.W))
            p = torch.stack([xs, ys], dim=-1)  # Shape: (H, W, 2)

            # Extract the pixel coordinates where the mask is valid
            p_valid = p[self.masks[img_idx][:, :, 0] > 0]  # Shape: (num_valid_points, 2)

            # Randomly sample pixel coordinates from the valid mask regions
            random_idx = torch.randint(low=0, high=p_valid.shape[0], size=[batch_size // 4 * 3])
            p_select = p_valid[random_idx]  # Shape: (3 * batch_size / 4, 2)

            pixels_x_2 = p_select[:, 0]
            pixels_y_2 = p_select[:, 1]

            # Concatenate the random pixels and the importance sampled pixels
            pixels_x = torch.cat([pixels_x_1, pixels_x_2], dim=0).to(torch.int64)
            pixels_y = torch.cat([pixels_y_1, pixels_y_2], dim=0).to(torch.int64)

        patch_color, patch_mask = None, None
        if crop_patch:
            # Build patch offsets
            offsets = build_patch_offset(h_patch_size)
            # Create grid of patches based on sampled pixels plus offsets
            grid_patch = torch.stack([pixels_x, pixels_y], dim=-1).view(-1, 1, 2) + offsets.float()

            # Create a mask to discard patches that fall outside image boundaries
            patch_mask = (pixels_x > h_patch_size) * (pixels_x < (self.W - h_patch_size)) * (
                    pixels_y > h_patch_size) * (pixels_y < self.H - h_patch_size)

            # Normalize pixel coordinates to [-1, 1] range for grid_sample
            grid_patch_u = 2 * grid_patch[:, :, 0] / (self.W - 1) - 1
            grid_patch_v = 2 * grid_patch[:, :, 1] / (self.H - 1) - 1
            grid_patch_uv = torch.stack([grid_patch_u, grid_patch_v], dim=-1)

            # Sample patch colors using grid_sample
            patch_color = F.grid_sample(self.images[img_idx][None, :, :, :].cuda().permute(0, 3, 1, 2),
                                        grid_patch_uv[None, :, :, :], mode='bilinear', padding_mode='zeros')[0]
            patch_color = patch_color.permute(1, 2, 0).contiguous().cuda()
            patch_mask = patch_mask.view(-1, 1).cuda()

        # Normalize pixel coordinates to [-1, 1] range
        ndc_u = 2 * pixels_x / (self.W - 1) - 1
        ndc_v = 2 * pixels_y / (self.H - 1) - 1
        rays_ndc_uv = torch.stack([ndc_u, ndc_v], dim=-1).view(-1, 2).float()

        # Extract color from image at sampled pixel locations
        color = self.images[img_idx][(pixels_y, pixels_x)]  # Shape: (batch_size, 3)

        # Extract mask value at sampled pixel locations
        mask = (self.masks[img_idx][(pixels_y, pixels_x)] > 0).to(torch.float32)  # Shape: (batch_size, 3)

        # Create homogeneous coordinates for the sampled pixels
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # Shape: (batch_size, 3)

        # Convert image space coordinates to camera space coordinates
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3],
                         p[:, :, None]).squeeze()  # Shape: (batch_size, 3)

        # Compute ray directions from camera coordinates
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # Shape: (batch_size, 3)
        rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3],
                              rays_v[:, :, None]).squeeze()  # Shape: (batch_size, 3)

        # Get camera origins
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape)  # Shape: (batch_size, 3)

        # Concatenate all ray components into a single tensor
        rays = torch.cat([rays_o.cuda(), rays_v.cuda(), color.cuda(), mask[:, :1].cuda()],
                         dim=-1)  # Shape: (batch_size, 10)

        # Create the output sample dictionary
        sample = {
            'rays': rays,
            'rays_ndc_uv': rays_ndc_uv.cuda(),
            'rays_norm_XYZ_cam': p.cuda(),  # The normalized XYZ coordinates in camera space
            'rays_patch_color': patch_color,
            'rays_patch_mask': patch_mask
        }

        return sample
```

- 我们可以通过在sample字典中添加：
  ```python
  'pixels_x': pixels_x.cuda(),  # Add the pixel x coordinates to the sample
  'pixels_y': pixels_y.cuda(),  # Add the pixel y coordinates to the sample
  ```

  从而得到用了哪些pixels，然后我们又已知image_idx，这样我们就可以确定使用了哪些pixels了，然后再对这些pixels计算per pixel  gaussian depth即可