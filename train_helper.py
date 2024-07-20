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
而下面这个的划分实现是GaussianRoom的方式，是将整个Scene划分成Grid，然后计算每个Grid的中心点处的UDF值，然后根据UDF值判断是否需要densification或者pruning，会慢很多
'''

class TrainHelper:

    def __init__(self, args, runner):
        self.args = args
        self.runner = runner
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
        # todo 因此在调用这个函数之前，需要将其存入.ply文件中
        ply_path = "output/ggbond/point_cloud/iteration_" + str(iteration) + "/point_cloud.ply"
        cubic_grid = self.devide_space_into_cubes(ply_path, cube_size)
        for point, point_count in cubic_grid:
            # 计算每个cube的中心点处的UDF值
            # TODO 这里传入当前runner的UDF网络从而计算每个cube的中心点的UDF值
            point = np.array(point)
            udf = self.runner.compute_udf(point)
            # 以这个中心点作为Threshold的判断条件
            if udf < self.args.threshold_u:
                # 枚举该cube内的所有Gaussian
                # for gaussian in cube.gaussians:
                #     # 如果该Gaussian的数量小于τn，则选择K个Gaussian的邻域，生成K个新的Gaussian
                #     if len(gaussian) < self.args.threshold_n:
                #         # 采样K个Gaussian的邻域
                #         pass
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


    # TODO 将PointPreprocesser的逻辑加入这里
    #
    def devide_space_into_cubes(self,ply, cube_size=1):

        self.ply = "output/ggbond/point_cloud/iteration_7000/point_cloud.ply"
        self.processor = PointCloudProcessor(self.ply)
        self.pcd = self.processor.point_cloud
        cubic_grid = self.processor.cubic_grid
        # self.ply = ply
        # self.pcd = processor.load_ply(ply)
        return cubic_grid

    def udf_guide_gs_densification_pruning(
            self,
            iteration,
            cube_size=1
    ):
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
        self,
        iteration,
        cube_size=1
    ):
        pass




class PointCloudProcessor:
    def __init__(self, path, cube_size=1, batch_size=10000):
        self.cube_size = cube_size
        self.batch_size = batch_size
        self.point_cloud = self.load_ply(path)
        self.cubic_grid = self.create_cubic_grid()

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

    def create_cubic_grid(self):
        min_xyz = torch.min(self.point_cloud['xyz'], dim=0).values
        max_xyz = torch.max(self.point_cloud['xyz'], dim=0).values
        grid_shape = torch.ceil((max_xyz - min_xyz) / self.cube_size).to(torch.int)

        grid = {}

        # Iterate through batches
        num_points = self.point_cloud['xyz'].shape[0]
        for batch_start in range(0, num_points, self.batch_size):
            batch_end = min(batch_start + self.batch_size, num_points)
            batch_xyz = self.point_cloud['xyz'][batch_start:batch_end]

            for i in range(grid_shape[0]):
                for j in range(grid_shape[1]):
                    for k in range(grid_shape[2]):
                        center = min_xyz + torch.tensor([i, j, k], dtype=torch.float,
                                                        device="cuda") * self.cube_size + self.cube_size / 2
                        center_tuple = tuple(center.cpu().detach().numpy())  # Convert to tuple for dictionary key

                        # Calculate points in this cube
                        low_bound = min_xyz + torch.tensor([i, j, k], dtype=torch.float, device="cuda") * self.cube_size
                        high_bound = low_bound + self.cube_size
                        points_in_cube = ((batch_xyz >= low_bound) & (batch_xyz < high_bound)).all(dim=1)
                        point_count = points_in_cube.sum().item()

                        if center_tuple in grid:
                            grid[center_tuple] += point_count
                        else:
                            grid[center_tuple] = point_count

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

# Example usage
path = "output/ggbond/point_cloud/iteration_7000/point_cloud.ply"
pc_processor = PointCloudProcessor(path)

# Get the cubic grid with centers and point counts
cubic_grid = pc_processor.cubic_grid

# Find nearest points to a specific center
# center = torch.tensor(list(cubic_grid.keys())[0], dtype=torch.float, device="cuda")
center = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float, device="cuda")
nearest_points = pc_processor.find_nearest_points(center, k=10)
print(nearest_points)
