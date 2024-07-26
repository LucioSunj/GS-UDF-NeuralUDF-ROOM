import math
import numpy as np
from plyfile import PlyData
import torch
from scipy.spatial import cKDTree
from scipy.stats import multivariate_normal
import numpy as np
import torch
from plyfile import PlyData
from sklearn.neighbors import NearestNeighbors
from termcolor import colored
from icecream import ic
import torch.nn.functional as F

# class PointCloudProcessor:
#
#     def voxelize_sample(self, data=None, voxel_size=1):
#         """
#         对给定的数据进行体素化处理。
#
#         体素化是将三维空间中的数据点转换成一定大小的体素(三维像素)的过程。这个函数首先对数据点进行随机打乱，
#         然后根据指定的体素大小进行体素化处理，确保结果中的每个数据点代表一个独特的体素中心。
#
#         参数:
#         data: numpy数组，包含需要进行体素化处理的数据点坐标。默认为None。
#         voxel_size: float，表示体素的大小。默认为0.01。
#
#         返回:
#         numpy数组，经过体素化处理后的数据点坐标。
#         """
#         # 对数据点进行随机打乱，以便后续处理
#         np.random.shuffle(data)
#
#         # 将数据点按体素大小进行量化，并去除重复的体素中心
#
#         data = np.unique(np.round(data / voxel_size), axis=0) * voxel_size
#
#         # 返回体素化处理后的数据
#         return data
#
#     def load_ply(self, path):
#         print("Loading ply")
#         plydata = PlyData.read(path)
#
#         xyz = np.stack((np.asarray(plydata.elements[0]['x']),
#                         np.asarray(plydata.elements[0]['y']),
#                         np.asarray(plydata.elements[0]['z'])), axis=1)
#         opacities = np.asarray(plydata.elements[0]['opacity'])[..., np.newaxis]
#
#         features_dc = np.zeros((xyz.shape[0], 3, 1))
#         features_dc[:, 0, 0] = np.asarray(plydata.elements[0]['f_dc_0'])
#         features_dc[:, 1, 0] = np.asarray(plydata.elements[0]['f_dc_1'])
#         features_dc[:, 2, 0] = np.asarray(plydata.elements[0]['f_dc_2'])
#
#         extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith('f_rest_')]
#         extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
#         features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
#         for idx, attr_name in enumerate(extra_f_names):
#             features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
#         features_extra = features_extra.reshape(
#             (features_extra.shape[0], 3, (4 ** 2 - 1)))  # Assuming max_sh_degree is 3
#
#         scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith('scale_')]
#         scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
#         scales = np.zeros((xyz.shape[0], len(scale_names)))
#         for idx, attr_name in enumerate(scale_names):
#             scales[:, idx] = np.asarray(plydata.elements[0][attr_name])
#
#         rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith('rot')]
#         rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
#         rots = np.zeros((xyz.shape[0], len(rot_names)))
#         for idx, attr_name in enumerate(rot_names):
#             rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
#         print("Finishing loading ply")
#         pcd = {
#             "xyz": torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True),
#             "features_dc": torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1,
#                                                                                                  2).contiguous().requires_grad_(
#                 True),
#             "features_rest": torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1,
#                                                                                                       2).contiguous().requires_grad_(
#                 True),
#             "opacity": torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True),
#             "scaling": torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True),
#             "rotation": torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True),
#             "active_sh_degree": 3
#         }
#         return pcd
#
#     def voxelize_and_structure(self, pcd, voxel_size=1, batch_size=10000):
#         """
#         对点云进行体素化并分批处理，以防内存耗尽。
#         """
#         xyz = pcd["xyz"].cpu().detach().numpy()
#         num_batches = math.ceil(xyz.shape[0] / batch_size)
#         # 用于存放所有批次体素化后的体素中心点的集合（all_voxel_centers）。
#         # 用于记录每个体素中心对应的点的索引的字典（point_map）。
#         all_voxel_centers = []
#         point_map = {}
#
#         for i in range(num_batches):
#             start_idx = i * batch_size
#             end_idx = min((i + 1) * batch_size, xyz.shape[0])
#
#             batch_xyz = xyz[start_idx:end_idx]
#             voxelized_batch = self.voxelize_sample(batch_xyz, voxel_size)
#             all_voxel_centers.append(voxelized_batch)
#
#             for voxel in voxelized_batch:
#                 idxs = np.all(np.round(batch_xyz / voxel_size) * voxel_size == voxel, axis=1)
#                 # 这一步得到的会是所有在体素内的点的索引的一个数组
#                 indexes_in_pc = np.arange(start_idx, end_idx)[idxs]
#                 point_map[tuple(voxel)] = indexes_in_pc
#
#         all_voxel_centers = np.vstack(all_voxel_centers)
#         kd_tree = cKDTree(all_voxel_centers)
#
#         return point_map, kd_tree
#
#     def query_points_around_voxel(self, kd_tree, point_map, target_voxel, pcd, k):
#         """
#         查询给定体素坐标周围k个最近的点的信息。
#         TODO 这里的实现还是通过体素来找体素再找点，后续若效果不好可以看看这里是否可以优化
#         """
#         distances, indices = kd_tree.query(target_voxel, k=k)
#         resulting_points = []
#
#         for idx in indices:
#             voxel_center = kd_tree.data[idx]
#             point_indices = point_map.get(tuple(voxel_center), [])
#             for point_idx in point_indices:
#                 resulting_points.append({
#                     "xyz": pcd["xyz"][point_idx].cpu().detach().numpy(),
#                     "features_dc": pcd["features_dc"][point_idx].cpu().detach().numpy(),
#                     "features_rest": pcd["features_rest"][point_idx].cpu().detach().numpy(),
#                     "opacity": pcd["opacity"][point_idx].cpu().detach().numpy(),
#                     "scaling": pcd["scaling"][point_idx].cpu().detach().numpy(),
#                     "rotation": pcd["rotation"][point_idx].cpu().detach().numpy()
#                 })
#
#             if len(resulting_points) >= k:
#                 break
#
#         return resulting_points[:k]
#
#     def construct_normal_distribution(self,points,cube_center):
#         """
#         使用点的属性构建多维正态分布。
#         """
#         # TODO 这里将xyz的均值设置成这个grid的center
#         for p in points:
#             p["xyz"] = cube_center
#         data = np.array([[p[key] for key in points[0].keys()] for p in points])
#         mean = np.mean(data, axis=0)
#         cov = np.cov(data, rowvar=False)
#         return multivariate_normal(mean=mean, cov=cov)
#
#     def sample_points_from_distribution(self,dist, n, keys):
#         """
#         从给定的正态分布中采样n个新的点，并封装属性值。
#         """
#         samples = dist.rvs(size=n)
#         sampled_points = [{key: sample for key, sample in zip(keys, point)} for point in samples]
#         return sampled_points
#

# class TrainHelper:
#
#     def __init__(self, args, runner):
#         self.args = args
#         self.runner = runner
#     def udf_guide_gs_global_densification(
#         self,
#         iteration,
#         cube_size=1,
#         gaussians = None
#     ):
#         # UDF-guided Global Densification
#         '''
#         将三维空间分为N^3的cubes，然后计算每个cube的中心点处的UDF值，然后以这个中心点作为Threshold的判断条件
#         If the value falls below the threshold (Sc < τs), it indicates that the grid is in proximity to the scene surface.
#         Subsequently, we enumerate the existing Gaussian primitives within each grid. In cases where the number of
#         Gaussian primitives is insufficient (Ng < τn),
#         we select the K Gaussian neighbors of the grid’s center point and generate K new Gaussian primitives within the grid.
#         The initial attributes of these newly generated Gaussian primitives are sampled from a normal distribution
#         defined by the mean and variance of the K neighboring Gaussians.
#         '''
#
#         # 首先要将3D空间分为N^3的cubes,并且得到每个cube的中心点以及其所包含的 gs 的数量与相关信息
#         # todo 因此在调用这个函数之前，需要将其存入.ply文件中
#         ply_path = "output/ggbond/point_cloud/iteration_" + str(iteration) + "/point_cloud.ply"
#         voxel_point_map, kd_tree = self.devide_space_into_cubes(ply_path, cube_size)
#         for voxel_point in voxel_point_map:
#             # 计算每个cube的中心点处的UDF值
#             # TODO 这里传入当前runner的UDF网络从而计算每个cube的中心点的UDF值
#             udf = self.runner.compute_udf(voxel_point[0])
#             # 以这个中心点作为Threshold的判断条件
#             if udf < self.args.threshold_u:
#                 # 枚举该cube内的所有Gaussian
#                 # for gaussian in cube.gaussians:
#                 #     # 如果该Gaussian的数量小于τn，则选择K个Gaussian的邻域，生成K个新的Gaussian
#                 #     if len(gaussian) < self.args.threshold_n:
#                 #         # 采样K个Gaussian的邻域
#                 #         pass
#                 # 若当前位置的UDF值小于τs，则将当前位置可能需要产生高斯，就看看该位置的高斯数量是否足够
#                 gs_nums = len(voxel_point[1])
#                 if gs_nums < self.args.threshold_n:
#                     # 若数量不够，就需要sample k个近邻的gs创建新的gs放在该位置
#                     # 查询给定体素坐标周围的最近的k个点的信息
#                     target_voxel = np.array(voxel_point[0])
#                     nearest_points = self.processor.query_points_around_voxel(kd_tree, voxel_point_map, target_voxel, self.pcd, k=5)
#
#                     # 打印查询结果的详细信息
#                     for point in nearest_points:
#                         print(point)
#                     keys = list(self.pcd.keys())
#                     dist = self.processor.construct_normal_distribution(nearest_points, voxel_point[0])
#                     sampled_points = self.processor.sample_points_from_distribution(dist, n=5, keys=keys)
#                     for point in sampled_points:
#                         # 添加新的sample出来的点加入到当前的gaussian model中
#                         gaussians.densification_postfix(point["xyz"], point["features_dc"], point["features_rest"], point["opacity"], point["scaling"], point["rotation"])
#
#
#     # TODO 将PointPreprocesser的逻辑加入这里
#     # 用
#     def devide_space_into_cubes(self,ply, cube_size=1):
#         self.processor = PointCloudProcessor()
#         self.pcd = self.processor.load_ply("output/ggbond/point_cloud/iteration_7000/point_cloud.ply")
#         # pcd = processor.load_ply(ply)
#         return self.processor.voxelize_and_structure(self.pcd, voxel_size=1, batch_size=10000)
#
#     def udf_guide_gs_densification_pruning(
#             self,
#             iteration,
#             cube_size=1
#     ):
#         # TODO UDF-guided Densification and Pruning
#         '''
#         （增强版Densification and Pruning) : 对于已经有了足够的GS的Cubes，我们开始做这一步（应该说，做完上一步就继续做这一步），
#         对于每个GS，我们通过其三维坐标 x 计算对应UDF值（直接丢入当前的UDF网络即可），然后通过下式（如何推导的？）计算每个GS对应的这个值，
#         it signifies that the Gaussian is either far from the SDF zero-level set or possesses low opacity.
#         In such instances, if η < τp, the Gaussian primitive will be pruned. Conversely,
#         when η > τd and the gradient of the Gaussian satisfies ∇g > τg, the Gaussian primitive will be densified.
#         '''
#         pass
#     def gs_guide_udf(
#         self,
#         iteration,
#         cube_size=1
#     ):
#         pass

# # 使用示例
# processor = PointCloudProcessor()
# pcd = processor.load_ply("output/ggbond/point_cloud/iteration_7000/point_cloud.ply")
# point_map, kd_tree = processor.voxelize_and_structure(pcd, voxel_size=1, batch_size=10000)
#
# # 查询给定体素坐标周围的最近的k个点的信息
# target_voxel = np.array([0.5, 0.5, 0.5])
# nearest_points = processor.query_points_around_voxel(kd_tree, point_map, target_voxel, pcd, k=5)
#
# # 打印查询结果的详细信息
# for point in nearest_points:
#     print(point)
# keys = list(pcd.keys())
# dist = processor.construct_normal_distribution(nearest_points)
# sampled_points = processor.sample_points_from_distribution(dist, n=2, keys=keys)

'''
TODO 
上面的划分操作是类似GSDF与Scaffold-GS的，是从初始点云中划分出体素，然后根据体素中心点计算UDF值，然后根据UDF值判断是否需要densification或者pruning
而下面这个的划分实现是GaussianRoom的方式，是将整个Scene划分成Grid，然后计算每个Grid的中心点处的UDF值，然后根据UDF值判断是否需要densification或者pruning
'''

class TrainHelper:

    def __init__(self, args, runner,ply):
        self.args = args
        self.runner = runner
        self.ply = ply
    def udf_guide_gs_global_densification(
        self,
        iteration=7000,
        cube_size=1,
        gaussians = None
    ):
        # UDF-guided Global Densification
        '''
        将三维空间分为N^3的cubes，然后计算每个cube的中心点处的UDF值，然后以这个中心点作为Threshold的判断条件
        If the value falls below the threshold (Sc < τs), it indicates that the grid is in proximity to the scene surface.
        Subsequently, we enumerate the existing Gaussian primitives within each grid. In cases where the number of
        Gaussian primitives is insufficient (Ng < τn),
        we select the K Gaussian neighbors of the grid’s center point and generate K new Gaussian primitives within the grid.
        The initial attributes of these newly generated Gaussian primitives are sampled from a normal distribution
        defined by the mean and variance of the K neighboring Gaussians.
        '''

        # 首先要将3D空间分为N^3的cubes,并且得到每个cube的中心点以及其所包含的 gs 的数量与相关信息
        # 因此在调用这个函数之前，需要将其存入.ply文件中

        cubic_grid = self.devide_space_into_cubes()

        # TODO 这里或许可以通过改变compute_udf变成批量计算udf值，然后批量执行操作
        for point, point_count in cubic_grid:
            # 计算每个cube的中心点处的UDF值
            point = np.array(point)
            udf = self.runner.compute_udf(point)
            # 以这个中心点作为Threshold的判断条件
            if udf < self.args.threshold_u:
                #     # 如果该Gaussian的数量小于τn，则选择K个Gaussian的邻域，生成K个新的Gaussian
                #      采样K个Gaussian的邻域
                # 若当前位置的UDF值小于τs，则将当前位置可能需要产生高斯，就看看该位置的高斯数量是否足
                if point_count < self.args.threshold_n:
                    # 若数量不够，就需要sample k个近邻的gs创建新的gs放在该位置
                    # 查询给定体素坐标周围的最近的k个点的信息
                    point = torch.tensor(point)
                    nearest_points = self.processor.find_nearest_points(point, k=5)

                    # 打印查询结果的详细信息
                    for neighbour in nearest_points:
                        print(neighbour)
                    sampled_points = self.processor.sample_from_distribution(nearest_points, n=10)
                    for new_point in sampled_points:
                        # 添加新的sample出来的点加入到当前的gaussian model中
                        gaussians.densification_postfix(new_point["xyz"], new_point["features_dc"], new_point["features_rest"],
                                                        new_point["opacity"], new_point["scaling"], new_point["rotation"])


    def devide_space_into_cubes(self):

        # self.ply = "output/ggbond/point_cloud/iteration_7000/point_cloud.ply"
        self.processor = PointCloudProcessor(self.ply)
        self.pcd = self.processor.point_cloud
        self.processor.create_cubic_grid()
        cubic_grid = self.processor.cubic_grid
        # self.ply = ply
        # self.pcd = processor.load_ply(ply)
        return cubic_grid

    def udf_guide_gs_densification_pruning(
            self,
            iteration,
            gaussians,
            runner
    ):
        # TODO UDF-guided Densification and Pruning
        '''
        （增强版Densification and Pruning) : 对于已经有了足够的GS的Cubes，我们开始做这一步（应该说，做完上一步就继续做这一步），
        对于每个GS，我们通过其三维坐标 x 计算对应UDF值（直接丢入当前的UDF网络即可），然后通过下式计算每个GS对应的UDF值，
        it signifies that the Gaussian is either far from the SDF zero-level set or possesses low opacity.
        In such instances, if η < τp, the Gaussian primitive will be pruned. Conversely,
        when η > τd and the gradient of the Gaussian satisfies ∇g > τg, the Gaussian primitive will be densified.
        '''
        # TODO 利用下文的Gaussian Processor类来进行操作
        # Example usage:
        udf_renderer = runner.udf_renderer  # Initialize your UDF renderer here
        processor = GaussianProcessor(udf_renderer,gaussians,self)
        pcd = load_ply(self.ply)

        processor.process_gaussians(pcd, self.args.lambda_sigma, self.args.threshold_p, self.args.threshold_d)

    def gs_guide_udf(
        self,
        iteration,
        gaussians,
        image_perm
    ):
        '''

        :param iteration:
        :param gaussians:
        :return:
        '''
        # TODO

        """
        训练用户定义函数（UDF）的函数。
        这个函数包含了训练过程中的所有步骤，例如更新学习率、调整颜色损失权重、渲染图像等。
        它使用了PyTorch和一些自定义的损失函数和渲染器来逐步优化模型。
        需要注意：已经被改成了训练的
        参数:
        self: 类的实例，包含了训练所需的所有属性，如优化器、学习率、数据集等。
        image_perm: 图像序列
        """

        # 每轮训练过程（注意只是一轮，外部需要套上训练循环）
        # for iter_i in tqdm(range(res_step)):
            # 根据是否使用相同的学习率更新学习率
        # 就是更新学习率的步骤
        if self.runner.udf_same_lr:
            self.runner.udf_update_learning_rate(start_g_id=0)
        else:
            self.runner.udf_update_learning_rate(start_g_id=1)
            self.runner.udf_update_learning_rate_geo()

        # 调整颜色损失权重
        color_base_weight, color_weight, color_pixel_weight, color_patch_weight = self.runner.udf_adjust_color_loss_weights()

        # 根据当前迭代步数获取图像索引，并准备随机射线和对应的Ground truth
        # 也就是每次训练只取一个图像
        img_idx = image_perm[self.runner.udf_iter_step % len(image_perm)]
        # 随机获取一些随机射线和对应的Ground truth用于训练
        sample = self.runner.udf_dataset.runner.gen_random_rays_patches_at_return_pixels(
            img_idx, self.runner.batch_size,
            crop_patch=color_patch_weight > 0.0, h_patch_size=self.runner.udf_color_loss_func.h_patch_size)

        # 解析样本数据
        data = sample['rays']
        rays_uv = sample['rays_ndc_uv']
        gt_patch_colors = sample['rays_patch_color']
        gt_patch_mask = sample['rays_patch_mask']
        pixels_x = sample['pixels_x']
        pixels_y = sample['pixels_y']

        # 根据颜色像素权重和颜色块权重决定是否加载参考和源图像信息
        if color_pixel_weight > 0. or color_patch_weight > 0.:
            # todo: this load is very slow
            ref_c2w, src_c2ws, src_intrinsics, src_images, img_wh = self.runner.udf_dataset.get_ref_src_info(img_idx)
            src_w2cs = torch.inverse(src_c2ws)
        else:
            ref_c2w, src_c2ws, src_w2cs, src_intrinsics, src_images = None, None, None, None, None

        # todo load supporting images

        # 提取射线的起点、方向、真实RGB值和mask
        rays_o, rays_d, true_rgb, mask = data[:, :3], data[:, 3: 6], data[:, 6: 9], data[:, 9: 10]

        # 计算射线的近远裁剪距离
        near, far = self.runner.udf_dataset.near_far_from_sphere(rays_o, rays_d)

        # 将遮罩转换为浮点格式
        mask = (mask > 0.5).float()

        # 避免除以零
        mask_sum = mask.sum() + 1e-5

        # 使用渲染器渲染图像
        render_out = self.runner.udf_renderer.render(rays_o, rays_d, near, far,
                                          flip_saturation=self.runner.udf_get_flip_saturation(),
                                          color_maps=src_images if color_pixel_weight > 0. else None,
                                          w2cs=src_w2cs,
                                          intrinsics=src_intrinsics,
                                          query_c2w=ref_c2w,
                                          img_index=None,
                                          rays_uv=rays_uv if color_patch_weight > 0 else None,
                                          cos_anneal_ratio=self.runner.udf_get_cos_anneal_ratio())

        # 提取渲染结果中的各项信息
        weight_sum = render_out['weight_sum']
        color_base = render_out['color_base']
        color = render_out['color']
        color_pixel = render_out['color_pixel']
        patch_colors = render_out['patch_colors']
        # 这里也就说明了Mask是optional的，这也是我们选用NeuralUDF的原因
        patch_mask = (render_out['patch_mask'].float()[:, None] * (weight_sum > 0.5).float()) > 0. \
            if render_out['patch_mask'] is not None else None
        pixel_mask = mask if self.runner.udf_mask_weight > 0 else None

        variance = render_out['variance']
        beta = render_out['beta']
        gamma = render_out['gamma']

        gradient_error = render_out['gradient_error']
        gradient_error_near_surface = render_out['gradient_error_near_surface']
        sparse_error = render_out['sparse_error']

        udf = render_out['udf']
        udf_min = udf.min(dim=1)[0][mask[:, 0] > 0.5].mean()

        # 计算颜色损失
        color_losses = self.runner.udf_color_loss_func(
            color_base, color, true_rgb, color_pixel,
            pixel_mask, patch_colors, gt_patch_colors, patch_mask
        )

        # 提取各项颜色损失
        color_total_loss = color_losses['loss']
        color_base_loss = color_losses['color_base_loss']
        color_loss = color_losses['color_loss']
        color_pixel_loss = color_losses['color_pixel_loss']
        color_patch_loss = color_losses['color_patch_loss']

        # 计算PSNR（峰值信噪比）
        psnr = 20.0 * torch.log10(
            1.0 / (((color - true_rgb) ** 2 * mask).sum() / (mask_sum * 3.0)).sqrt())

        # 计算遮罩损失
        # mask_loss = (weight_sum - mask).abs().mean()
        mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask)

        # 计算Eikonal损失
        gradient_error_loss = gradient_error

        # 设置遮罩权重
        mask_weight = self.runner.udf_mask_weight

        # 根据条件判断是否使beta可训练
        if variance.mean() < 2 * beta.item() and variance.mean() < 0.01 and self.runner.beta_flag and self.runner.udf_variance_network_fine.variance.requires_grad:
            print("make beta trainable")
            self.runner.udf_beta_network.set_beta_trainable()
            self.runner.beta_flag = False

        # 根据迭代步数决定是否使变差网络可训练
        if self.runner.udf_variance_network_fine.variance.requires_grad is False and self.runner.udf_iter_step > 20000:
            self.runner.udf_variance_network_fine.set_trainable()

        # 根据是否启用权重调度决定常规权重
        if not self.runner.udf_reg_weights_schedule:
            igr_ns_weight = self.runner.udf_igr_ns_weight
            sparse_weight = self.runner.udf_sparse_weight
        else:
            igr_ns_weight, sparse_weight = self.runner.udf_regularization_weights_schedule()

        loss = color_total_loss + \
               mask_loss * mask_weight + \
               gradient_error_near_surface * igr_ns_weight + \
               sparse_error * sparse_weight + \
               gradient_error_loss * self.runner.udf_igr_weight

        self.runner.udf_optimizer.zero_grad()
        loss.backward()
        self.runner.udf_optimizer.step()

        self.runner.udf_iter_step += 1

        # self.writer.add_scalar('Loss/loss', loss, self.udf_iter_step)
        # self.writer.add_scalar('Loss/mask_loss', mask_loss, self.udf_iter_step)
        # self.writer.add_scalar('Loss/gradient_error_loss', gradient_error_loss, self.udf_iter_step)
        # self.writer.add_scalar('Sta/variance', variance.mean(), self.udf_iter_step)
        # self.writer.add_scalar('Sta/beta', beta.item(), self.udf_iter_step)
        # self.writer.add_scalar('Sta/psnr', psnr, self.udf_iter_step)
        if self.runner.udf_iter_step % self.runner.report_freq == 0:
            print(self.runner.udf_base_exp_dir)
            print('iter:{:8>d} loss = {:.4f} '
                  'color_total_loss = {:.4f} '
                  'eki_loss = {:.4f} '
                  'eki_ns_loss = {:.4f} '
                  'mask_loss = {:.4f} '
                  'sparse_loss = {:.4f} '.format(self.runner.udf_iter_step, loss, color_total_loss, gradient_error_loss,
                                                 gradient_error_near_surface,
                                                 mask_loss,
                                                 sparse_error))
            print('iter:{:8>d} c_base_loss = {:.4f} '
                  'color_loss = {:.4f} '
                  'c_pixel_loss = {:.4f} '
                  'c_patch_loss = {:.4f} '.format(self.runner.udf_iter_step, color_base_loss, color_loss, color_pixel_loss,
                                                  color_patch_loss))
            print('iter:{:8>d} '
                  'variance = {:.6f} '
                  'beta = {:.6f} '
                  'gamma = {:.4f} '
                  'lr_geo={:.8f} lr={:.8f} '.format(self.runner.udf_iter_step,
                                                    variance.mean(), beta.item(), gamma.item(),
                                                    self.runner.udf_optimizer.param_groups[0]['lr'],
                                                    self.runner.udf_optimizer.param_groups[1]['lr']))

            print(colored('psnr = {:.4f} '
                          'weight_sum = {:.4f} '
                          'weight_sum_fg_bg = {:.4f} '
                          'udf_min = {:.8f} '
                          'udf_mean = {:.4f} '
                          'mask_weight = {:.4f} '
                          'sparse_weight = {:.4f} '
                          'igr_ns_weight = {:.4f} '
                          'igr_weight = {:.4f} '.format(psnr, (render_out['weight_sum'] * mask).sum() / mask_sum,
                                                        (render_out['weight_sum_fg_bg'] * mask).sum() / mask_sum,
                                                        udf_min, udf.mean(), mask_weight, sparse_weight,
                                                        igr_ns_weight,
                                                        self.runner.udf_igr_weight,
                                                        ), 'green'))

            ic(self.runner.udf_get_flip_saturation())

        if self.runner.udf_iter_step % self.runner.save_freq == 0:
            self.runner.udf_save_checkpoint()

        if self.runner.udf_dataset_name == 'general':
            if self.runner.udf_iter_step % self.runner.val_freq == 0:
                self.runner.udf_validate()

            if self.runner.udf_iter_step % (self.runner.val_mesh_freq * 2) == 0 and self.runner.udf_vis_ray:
                for i in range(-self.runner.udf_dataset.H // 4, self.runner.udf_dataset.H // 4, 20):
                    self.runner.udf_visualize_one_ray(img_idx=33, px=self.runner.udf_dataset.W // 2, py=self.runner.udf_dataset.H // 2 + i)

        # Python版本与marching cubes的版本有冲突，mesh能否不用？？
        # 尝试一下用0.1.3版本的pymcubes
        if self.runner.udf_iter_step % self.runner.val_mesh_freq == 0:
            self.runner.udf_validate_mesh(threshold=self.runner.args.udf_threshold)
            try:
                self.runner.extract_udf_mesh(world_space=True, dist_threshold_ratio=2.0)
            except:
                print("extract udf mesh fails")

        if self.runner.udf_iter_step % len(image_perm) == 0:
            image_perm = torch.randperm(self.runner.udf_dataset.n_images)

# 这个class用于Global Densification
class PointCloudProcessor:
    def __init__(self, path, cube_size=1, batch_size=10000):
        self.cube_size = cube_size
        self.batch_size = batch_size
        self.point_cloud = load_ply(path)

    def cubic_grid(self):
        self.cubic_grid = self.create_cubic_grid()


    def create_cubic_grid(self):
        min_xyz = torch.min(self.point_cloud['xyz'], dim=0).values
        max_xyz = torch.max(self.point_cloud['xyz'], dim=0).values
        grid_shape = torch.ceil((max_xyz - min_xyz) / self.cube_size).to(torch.int)

        grid = {}

        # Create all possible grid centers
        grid_indices = torch.stack(torch.meshgrid(
            torch.arange(grid_shape[0], device="cuda"),
            torch.arange(grid_shape[1], device="cuda"),
            torch.arange(grid_shape[2], device="cuda")
        ), dim=-1).reshape(-1, 3)

        centers = min_xyz + grid_indices.float() * self.cube_size + self.cube_size / 2
        centers = centers.cpu().detach().numpy()

        # Initialize all grid cells with 0 points
        for center in centers:
            grid[tuple(center)] = 0

        # Iterate through batches
        num_points = self.point_cloud['xyz'].shape[0]
        for batch_start in range(0, num_points, self.batch_size):
            batch_end = min(batch_start + self.batch_size, num_points)
            batch_xyz = self.point_cloud['xyz'][batch_start:batch_end]

            # Compute the voxel indices for the batch
            indices = torch.floor((batch_xyz - min_xyz) / self.cube_size).to(torch.int)

            # Filter out-of-bound indices
            valid_mask = (indices >= 0) & (indices < grid_shape)
            valid_mask = valid_mask.all(dim=1)
            indices = indices[valid_mask]

            # Compute voxel centers
            centers = (indices.float() * self.cube_size) + min_xyz + self.cube_size / 2
            centers = centers.cpu().detach().numpy()

            # Count points in each voxel
            unique_centers, counts = np.unique(centers, axis=0, return_counts=True)
            for center, count in zip(unique_centers, counts):
                center_tuple = tuple(center)
                grid[center_tuple] += count

        return grid
    def find_nearest_points(self, center, k):
        dists = torch.norm(self.point_cloud['xyz'] - center, dim=1)
        nearest_indices = torch.topk(dists, k, largest=False).indices
        nearest_points = {
            "xyz": self.point_cloud['xyz'][nearest_indices],
            "features_dc": self.point_cloud['features_dc'][nearest_indices],
            "features_rest": self.point_cloud['features_rest'][nearest_indices],
            "opacity": self.point_cloud['opacity'][nearest_indices],
            "scaling": self.point_cloud['scaling'][nearest_indices],
            "rotation": self.point_cloud['rotation'][nearest_indices]
        }
        return nearest_points

    def sample_from_distribution(self, nearest_points, n):
        sampled_points = {}

        for key, value in nearest_points.items():
            mean = value.mean(dim=0)
            std = value.std(dim=0)
            sampled_points[key] = torch.normal(mean, std, size=(n, *mean.shape), device="cuda")

        return sampled_points

# # Example usage
# path = "output/ggbond/point_cloud/iteration_7000/point_cloud.ply"
# pc_processor = PointCloudProcessor(path)
#
# # Get the cubic grid with centers and point counts
# cubic_grid = pc_processor.cubic_grid
#
# # Find nearest points to a specific center
# # center = torch.tensor(list(cubic_grid.keys())[0], dtype=torch.float, device="cuda")
# center = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float, device="cuda")
# nearest_points = pc_processor.find_nearest_points(center, k=10)
# print(nearest_points)

# 这是用于UDF-guide-GS densificaiton & pruning的类与函数
class GaussianProcessor:
    def __init__(self, runner, gaussians, train_helper, batch_size=10000):
        self.runner = runner
        self.gaussians = gaussians
        self.train_helper = train_helper
        self.batch_size = batch_size


    def process_gaussians(self, pcd, lambda_sigma, tau_p, tau_d):
        '''
        这个函数用于计算每个gaussian处的udf值，然后进行densify与prune
        :param pcd:
        :param lambda_sigma:
        :param tau_p:
        :param tau_d:
        :return:
        '''
        # TODO 这里需要考虑他们的shape是否是相同的
        opacities = pcd['opacity']
        xyz = pcd['xyz']
        # TODO 若这里梯度shape不对的话，可以看看 densify_and_split函数中计算梯度的方式
        grads = self.gaussians.xyz_gradient_accum / self.gaussians.denom
        grads[grads.isnan()] = 0.0  # 将nan的梯度设置成0

        num_points = xyz.shape[0]
        num_batches = (num_points + self.batch_size - 1) // self.batch_size

        etas = []

        for i in range(num_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, num_points)
            batch_centers = xyz[start_idx:end_idx]
            batch_opacities = opacities[start_idx:end_idx]

            # 批量计算 UDF 值
            batch_udfs = self.runner.compute_udf_batch(batch_centers)

            # 计算 eta
            batch_etas = torch.exp(-batch_udfs ** 2 / (lambda_sigma * batch_opacities ** 2))
            etas.append(batch_etas)

        etas = torch.cat(etas, dim=0)

        prune_mask = etas < tau_p
        densify_mask = torch.logical_and(etas > tau_d, torch.norm(grads, dim=-1) > self.runner.args.threshold_g)

        self.densify_and_prune(etas, grads, prune_mask, densify_mask)
    def densify_and_prune(self, etas, grads, prune_mask, densify_mask):
        # Apply pruning
        prune_indices = torch.where(prune_mask)[0]
        self.prune_points(prune_indices)

        # Apply densification
        densify_indices = torch.where(densify_mask)[0]
        self.densify_points(densify_indices, grads)

    def prune_points(self, indices):
        # Prune the points at the given indices
        # print(f"Pruning points at indices: {indices}")
        # Implement pruning logic here
        self.gaussians.prune_points(indices)

    def densify_points(self, indices, grads):
        # Densify the points at the given indices
        # print(f"Densifying points at indices: {indices}")
        # Implement densification logic here

        # Densification & Clone 的逻辑
        new_xyz = self.gaussians._xyz[indices]
        new_features_dc = self.gaussians._features_dc[indices]
        new_features_rest = self.gaussians._features_rest[indices]
        new_opacities = self.gaussians._opacity[indices]
        new_scaling = self.gaussians._scaling[indices]
        new_rotation = self.gaussians._rotation[indices]

        self.gaussians.densification_postfix(new_xyz, new_features_dc,
                                             new_features_rest, new_opacities, new_scaling,new_rotation)
        # TODO Densification & Split 的逻辑 是否需要呢？


def load_ply(path):
    print("Loading ply")
    plydata = PlyData.read(path)

    xyz = np.stack((np.asarray(plydata.elements[0]['x']),
                    np.asarray(plydata.elements[0]['y']),
                    np.asarray(plydata.elements[0]['z'])), axis=1)
    opacities = np.asarray(plydata.elements[0]['opacity'])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]['f_dc_0'])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]['f_dc_1'])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]['f_dc_2'])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith('f_rest_')]
    extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    features_extra = features_extra.reshape(
        (features_extra.shape[0], 3, (4 ** 2 - 1)))  # Assuming max_sh_degree is 3

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith('scale_')]
    scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith('rot')]
    rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
    print("Finishing loading ply")
    pcd = {
        "xyz": torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True),
        "features_dc": torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1,
                                                                                             2).contiguous().requires_grad_(
            True),
        "features_rest": torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1,
                                                                                                  2).contiguous().requires_grad_(
            True),
        "opacity": torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True),
        "scaling": torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True),
        "rotation": torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True),
        "active_sh_degree": 3
    }
    return pcd