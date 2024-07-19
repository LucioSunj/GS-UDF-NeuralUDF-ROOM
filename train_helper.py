#
# import numpy as np
# import torch
# from torch import nn
# # from data import dataset_readers
# from gsBranch.scene import dataset_readers
# from collections import defaultdict
# from scipy.spatial import cKDTree as KDTree
# from plyfile import PlyData
#
# class TrainHelper:
#     def test_main(self,ply_path,voxel_size=1,k=10):
#         # basic_point_cloud = dataset_readers.fetchPly(ply_path)
#         basic_point_cloud = self.load_ply(ply_path)
#         # 我们的需求是：
#         # 能够拿到所有voxel的中心点坐标（用于计算UDF值）以及其所包含的Points信息
#         voxel_centers, voxel_stats = TrainHelper.voxelize_point_cloud(basic_point_cloud, voxel_size)
#         processor = TrainHelper.PointCloudProcessor(voxel_centers, voxel_stats,voxel_size,k)
#
#         query_point = voxel_centers[0] # 替换为您感兴趣的查询点
#         neighbor_info = processor.get_k_nearest_points(query_point)
#         print(neighbor_info)
#     def __init__(self, args=None, runner=None):
#         self.args = args
#         self.runner = runner
#     def udf_guide_gs_global_densification(
#         self,
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
#         cubes = self.devide_space_into_cubes()
#         for cube in cubes:
#             # 计算每个cube的中心点处的UDF值
#             # TODO 这里传入当前runner的UDF网络从而计算每个cube的中心点的UDF值
#             udf = self.runner.compute_udf(cube)
#             # 以这个中心点作为Threshold的判断条件
#             if udf < self.args.threshold_u:
#                 # 枚举该cube内的所有Gaussian
#                 for gaussian in cube.gaussians:
#                     # 如果该Gaussian的数量小于τn，则选择K个Gaussian的邻域，生成K个新的Gaussian
#                     if len(gaussian) < self.args.threshold_n:
#                         # 采样K个Gaussian的邻域
#                         pass
#
#     # TODO 测试这个分割代码是否有用
#     # 用
#     def devide_space_into_cubes(self,ply, cube_size):
#         try:
#             pcd = dataset_readers.fetchPly(ply)
#         except:
#             pcd = None
#         # points = pcd.points[::1] # 这是每个点的位置信息
#
#         # 检查cube_size是否合理
#         # if cube_size <= 0:
#         #     init_points = torch.tensor(points).float().cuda()
#         #     init_dist = distCUDA2(init_points).float().cuda()
#         #     median_dist, _ = torch.kthvalue(init_dist, int(init_dist.shape[0] * 0.5))
#         #     cube_size = median_dist.item()
#         #     del init_dist
#         #     del init_points
#         #     torch.cuda.empty_cache()
#         # 产生cubes
#         # points = self.voxelize_sample(points, voxel_size=cube_size)
#         voxel_centers, voxel_stats = TrainHelper.voxelize_point_cloud(pcd, cube_size)
#
#         result = {
#
#         }
#         return result
#
#     # def voxelize_sample(self, data=None, voxel_size=0.01):
#     #     """
#     #     对给定的数据进行体素化处理。
#     #
#     #     体素化是将三维空间中的数据点转换成一定大小的体素(三维像素)的过程。这个函数首先对数据点进行随机打乱，
#     #     然后根据指定的体素大小进行体素化处理，确保结果中的每个数据点代表一个独特的体素中心。
#     #
#     #     参数:
#     #     data: numpy数组，包含需要进行体素化处理的数据点坐标。默认为None。
#     #     voxel_size: float，表示体素的大小。默认为0.01。
#     #
#     #     返回:
#     #     numpy数组，经过体素化处理后的数据点坐标。
#     #     """
#     #     # 对数据点进行随机打乱，以便后续处理
#     #     np.random.shuffle(data)
#     #
#     #     # 将数据点按体素大小进行量化，并去除重复的体素中心
#     #     data = np.unique(np.round(data / voxel_size), axis=0) * voxel_size
#     #
#     #     # 返回体素化处理后的数据
#     #     return data
#     def load_ply(self, path):
#         plydata = PlyData.read(path)
#
#         xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
#                         np.asarray(plydata.elements[0]["y"]),
#                         np.asarray(plydata.elements[0]["z"])),  axis=1)
#         opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
#
#         features_dc = np.zeros((xyz.shape[0], 3, 1))
#         features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
#         features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
#         features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
#
#         extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
#         extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
#         assert len(extra_f_names)==3*(self.args.max_sh_degree + 1) ** 2 - 3
#         features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
#         for idx, attr_name in enumerate(extra_f_names):
#             features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
#         # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
#         features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.args.max_sh_degree + 1) ** 2 - 1))
#
#         scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
#         scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
#         scales = np.zeros((xyz.shape[0], len(scale_names)))
#         for idx, attr_name in enumerate(scale_names):
#             scales[:, idx] = np.asarray(plydata.elements[0][attr_name])
#
#         rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
#         rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
#         rots = np.zeros((xyz.shape[0], len(rot_names)))
#         for idx, attr_name in enumerate(rot_names):
#             rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
#
#         pcd = {}
#         pcd.xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
#         pcd.features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
#         pcd.features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
#         pcd.opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
#         pcd.scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
#         pcd.rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
#         # TODO pcd.active_sh_degree = self.args.max_sh_degree
#         pcd.active_sh_degree = 3
#         return pcd
#
#
#     class VoxelGrid:
#         def __init__(self, voxel_size: float):
#             self.voxel_size = voxel_size
#             self.voxels = defaultdict(list)
#
#         def add_point(self, point, color, normal):
#             voxel_index = tuple(np.floor(point / self.voxel_size).astype(int))
#             self.voxels[voxel_index].append((point, color, normal))
#
#         # TODO 能跑之后可以用numpy进行批量添加的计算
#         # def add_points(self, points, colors, normals):
#         #     # 用 NumPy 进行批量计算
#         #     voxel_indices = np.floor(points / self.voxel_size).astype(int)
#         #
#         #     for index, (point, color, normal) in enumerate(zip(points, colors, normals)):
#         #         voxel_index = tuple(voxel_indices[index])
#         #         self.voxels[voxel_index].append((point, color, normal))
#
#         def get_voxel_centers_and_stats(self):
#             voxel_centers = []
#             voxel_stats = []
#
#             for voxel_index, points in self.voxels.items():
#                 positions = np.array([p[0] for p in points])
#                 colors = np.array([p[1] for p in points])
#                 normals = np.array([p[2] for p in points])
#
#                 center = (np.array(voxel_index) + 0.5) * self.voxel_size
#                 voxel_centers.append(center)
#                 voxel_stats.append({
#                     'num_points': len(points),
#                     'points': positions,
#                     'colors': colors,
#                     'normals': normals
#                 })
#
#             return np.array(voxel_centers), voxel_stats
#
#
#     def voxelize_point_cloud(self,pcd, voxel_size: float):
#         grid = self.VoxelGrid(voxel_size=voxel_size)
#
#         # 将点云数据添加到体素网格中
#         for i in range(len(pcd.xyz)):
#             grid.add_point(pcd.xyz[i], pcd.colors[i], pcd.normals[i])
#
#         return grid.get_voxel_centers_and_stats()
#
#
#     class PointCloudProcessor:
#         def __init__(self, voxel_centers,voxel_stats, k):
#             self.k = k
#             self.voxel_stats = voxel_stats
#             self.kd_tree = KDTree(voxel_centers)  # 使用KD-Tree来加速邻近点查询
#
#         # 这一步用于获取k近邻来生成当前新的gs
#         def get_k_nearest_points(self, query_point):
#             distances, nearest_indices = self.kd_tree.query(query_point, k=self.k)
#
#             neighbor_points = []
#             neighbor_colors = []
#             neighbor_normals = []
#
#             for index in nearest_indices:
#                 stats = self.voxel_stats[index]
#                 neighbor_points.append(stats['points'])
#                 neighbor_colors.append(stats['colors'])
#                 neighbor_normals.append(stats['normals'])
#
#             return {
#                 'points': np.vstack(neighbor_points),
#                 'colors': np.vstack(neighbor_colors),
#                 'normals': np.vstack(neighbor_normals)
#             }
#     def udf_guide_gs_densification_pruning(self):
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
#         self
#     ):
#         pass
#
#
#
# if __name__ == '__main__':
#     ply_file = 'output/ggbond/point_cloud/iteration_7000/point_cloud.ply'
#     cube_size = 1  # 假设每个小立方体的边长为 1
#     # result = TrainHelper.devide_space_into_cubes(ply_file, cube_size)
#     trainhelper = TrainHelper()
#     trainhelper.test_main(ply_path=ply_file, voxel_size=cube_size,k=10)
#     # # 打印结果或者进一步处理
#     # for cube in result:
#     #     if len(cube['points']) > 0:  # 仅打印包含点的立方体
#     #         print(f"Cube Center: {cube['center']}")
#     #         print(f"Contained Points: {cube['points']}")
#
#
#
import math

import numpy as np
from plyfile import PlyData
import torch
from scipy.spatial import cKDTree


class PointCloudProcessor:

    def voxelize_sample(self, data=None, voxel_size=0.01):
        """
        对给定的数据进行体素化处理。

        体素化是将三维空间中的数据点转换成一定大小的体素(三维像素)的过程。这个函数首先对数据点进行随机打乱，
        然后根据指定的体素大小进行体素化处理，确保结果中的每个数据点代表一个独特的体素中心。

        参数:
        data: numpy数组，包含需要进行体素化处理的数据点坐标。默认为None。
        voxel_size: float，表示体素的大小。默认为0.01。

        返回:
        numpy数组，经过体素化处理后的数据点坐标。
        """
        # 对数据点进行随机打乱，以便后续处理
        np.random.shuffle(data)

        # 将数据点按体素大小进行量化，并去除重复的体素中心
        print("将数据点按体素大小进行量化，并去除重复的体素中心")
        data = np.unique(np.round(data / voxel_size), axis=0) * voxel_size
        print("返回体素化处理后的数据")
        # 返回体素化处理后的数据
        return data

    def load_ply(self, path):
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

    def voxelize_and_structure(self, pcd, voxel_size=1, batch_size=10000):
        """
        对点云进行体素化并分批处理，以防内存耗尽。
        """
        xyz = pcd["xyz"].cpu().detach().numpy()
        num_batches = math.ceil(xyz.shape[0] / batch_size)

        voxel_grid = {}
        all_voxel_centers = []
        print("Starting voxelization and structure")
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, xyz.shape[0])

            batch_xyz = xyz[start_idx:end_idx]
            voxelized_batch = self.voxelize_sample(batch_xyz, voxel_size)
            all_voxel_centers.append(voxelized_batch)

            for voxel in voxelized_batch:
                idxs = np.all(np.round(batch_xyz / voxel_size) * voxel_size == voxel, axis=1)
                indexes_in_pc = np.arange(start_idx, end_idx)[idxs]
                voxel_grid[tuple(voxel)] = {
                    "center": voxel,
                    "count": len(indexes_in_pc),
                    "xyz": pcd["xyz"][indexes_in_pc],
                    "features_dc": pcd["features_dc"][indexes_in_pc],
                    "features_rest": pcd["features_rest"][indexes_in_pc],
                    "opacity": pcd["opacity"][indexes_in_pc],
                    "scaling": pcd["scaling"][indexes_in_pc],
                    "rotation": pcd["rotation"][indexes_in_pc]
                }

        all_voxel_centers = np.vstack(all_voxel_centers)
        kd_tree = cKDTree(all_voxel_centers)

        return voxel_grid, kd_tree

    def query_nearest_voxels(self, kd_tree, voxel_grid, target_point, k):
        """
        查询给定坐标周围k个最近的体素

        参数:
        kd_tree: 用于查询的cKDTree.
        voxel_grid: 包含体素化点云数据的字典.
        target_point: 目标坐标.
        k: 要查询的最近体素数量.

        返回:
        nearest_voxels: 最近体素的详细信息
        """
        print("Starting query k neighbous")
        distances, indices = kd_tree.query(target_point, k=k)
        nearest_voxels = [voxel_grid[tuple(kd_tree.data[idx])] for idx in indices]
        print("Finishing query k neighbous")
        return nearest_voxels


class TrainHelper:
    def test(self,ply_path,voxel_size=1,k=10):
        # basic_point_cloud = dataset_readers.fetchPly(ply_path)
        basic_point_cloud = self.load_ply(ply_path)
        # 我们的需求是：
        # 能够拿到所有voxel的中心点坐标（用于计算UDF值）以及其所包含的Points信息
        voxel_centers, voxel_stats = TrainHelper.voxelize_point_cloud(basic_point_cloud, voxel_size)
        processor = TrainHelper.PointCloudProcessor(voxel_centers, voxel_stats,voxel_size,k)

        query_point = voxel_centers[0] # 替换为您感兴趣的查询点
        neighbor_info = processor.get_k_nearest_points(query_point)
        print(neighbor_info)
    def __init__(self, args=None, runner=None):
        self.args = args
        self.runner = runner
    def udf_guide_gs_global_densification(
        self,
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
        cubes = self.devide_space_into_cubes()
        for cube in cubes:
            # 计算每个cube的中心点处的UDF值
            # TODO 这里传入当前runner的UDF网络从而计算每个cube的中心点的UDF值
            udf = self.runner.compute_udf(cube)
            # 以这个中心点作为Threshold的判断条件
            if udf < self.args.threshold_u:
                # 枚举该cube内的所有Gaussian
                for gaussian in cube.gaussians:
                    # 如果该Gaussian的数量小于τn，则选择K个Gaussian的邻域，生成K个新的Gaussian
                    if len(gaussian) < self.args.threshold_n:
                        # 采样K个Gaussian的邻域
                        nearest_voxels = processor.query_nearest_voxels(kd_tree, voxel_grid, target_point, k=5)

    # TODO 将PointPreprocesser的逻辑加入这里
    # 用
    def devide_space_into_cubes(self,ply, cube_size):
        self.processor = PointCloudProcessor()
        pcd = processor.load_ply("output/ggbond/point_cloud/iteration_7000/point_cloud.ply")
        voxel_grid, kd_tree = processor.voxelize_and_structure(pcd, voxel_size=1)
        result = {

        }
        return result

    def udf_guide_gs_densification_pruning(self):
        # TODO UDF-guided Densification and Pruning
        '''
        （增强版Densification and Pruning) : 对于已经有了足够的GS的Cubes，我们开始做这一步（应该说，做完上一步就继续做这一步），
        对于每个GS，我们通过其三维坐标 x 计算对应UDF值（直接丢入当前的UDF网络即可），然后通过下式（如何推导的？）计算每个GS对应的这个值，
        it signifies that the Gaussian is either far from the SDF zero-level set or possesses low opacity.
        In such instances, if η < τp, the Gaussian primitive will be pruned. Conversely,
        when η > τd and the gradient of the Gaussian satisfies ∇g > τg, the Gaussian primitive will be densified.
        '''
        pass
    def gs_guide_udf(
        self
    ):
        pass

# 使用示例
processor = PointCloudProcessor()
pcd = processor.load_ply("output/ggbond/point_cloud/iteration_7000/point_cloud.ply")
voxel_grid, kd_tree = processor.voxelize_and_structure(pcd, voxel_size=0.01)

# 查询给定坐标周围的最近的k个体素
target_point = np.array([0.5, 0.5, 0.5])
nearest_voxels = processor.query_nearest_voxels(kd_tree, voxel_grid, target_point, k=5)

# 打印查询结果的详细信息
for voxel in nearest_voxels:
    print(voxel)
