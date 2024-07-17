
import numpy as np
import torch
from plyfile import PlyData



# def udf_guide_gs(
#     runner,
#
# ):
#     # UDF-guided Global Densification
#     '''
#     将三维空间分为N^3的cubes，然后计算每个cube的中心点处的UDF值，然后以这个中心点作为Threshold的判断条件
#     If the value falls below the threshold (Sc < τs), it indicates that the grid is in proximity to the scene surface.
#     Subsequently, we enumerate the existing Gaussian primitives within each grid. In cases where the number of
#     Gaussian primitives is insufficient (Ng < τn),
#     we select the K Gaussian neighbors of the grid’s center point and generate K new Gaussian primitives within the grid.
#     The initial attributes of these newly generated Gaussian primitives are sampled from a normal distribution
#     defined by the mean and variance of the K neighboring Gaussians.
#     '''
#
#     # 首先要将3D空间分为N^3的cubes
#     cubes = devide_space_into_cubes()
#     for cube in cubes:
#         # 计算每个cube的中心点处的UDF值
#         # TODO 这里传入当前runner的UDF网络从而计算每个cube的中心点的UDF值
#         udf = runner.compute_udf(cube)
#         # 以这个中心点作为Threshold的判断条件
#         if udf < threshold:
#             # 枚举该cube内的所有Gaussian
#             for gaussian in cube.gaussians:
#                 # 如果该Gaussian的数量小于τn，则选择K个Gaussian的邻域，生成K个新的Gaussian
#                 if len(gaussian) < threshold_n:
#                     # 采样K个Gaussian的邻域
#
#
#
#
#     # TODO UDF-guided Densification and Pruning
#     '''
#     （增强版Densification and Pruning) : 对于已经有了足够的GS的Cubes，我们开始做这一步（应该说，做完上一步就继续做这一步），
#     对于每个GS，我们通过其三维坐标 x 计算对应UDF值（直接丢入当前的UDF网络即可），然后通过下式（如何推导的？）计算每个GS对应的这个值，
#     it signifies that the Gaussian is either far from the SDF zero-level set or possesses low opacity.
#     In such instances, if η < τp, the Gaussian primitive will be pruned. Conversely,
#     when η > τd and the gradient of the Gaussian satisfies ∇g > τg, the Gaussian primitive will be densified.
#     '''


def gs_guide_udf(

):
    pass



'''
read_ply: 读取 .ply 文件并提取点的位置信息、颜色和高斯分布参数。
get_bounding_box: 获取点云的包围盒，以确定空间分割的范围。
generate_cubes: 在包围盒范围内生成多个小立方体的中心坐标。
points_in_cube: 检查哪些点落在某个立方体内，并返回这些点的索引。
main: 主函数执行上述步骤，并返回每个小立方体的中心坐标及包含的点及参数。
'''
# TODO 测试这个分割代码是否有用
# 用
def devide_space_into_cubes(ply_file, cube_size):
    points = read_ply(ply_file)
    min_corner, max_corner = get_bounding_box(points)
    centers = generate_cubes(min_corner, max_corner, cube_size)
    indices_list = points_in_cube(points, centers, cube_size)

    result = []
    for center, indices in zip(centers, indices_list):
        contained_points = points[indices]

        result.append({
            'center': center,
            'points': contained_points,
        })

    return result


def read_ply(ply_file):
    ply_data = PlyData.read(ply_file)
    vertices = ply_data['vertex']

    points = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T

    return points


def get_bounding_box(points):
    min_corner = np.min(points, axis=0)
    max_corner = np.max(points, axis=0)
    return min_corner, max_corner


def generate_cubes(min_corner, max_corner, cube_size):
    x_centers = np.arange(min_corner[0] + cube_size / 2, max_corner[0], cube_size)
    y_centers = np.arange(min_corner[1] + cube_size / 2, max_corner[1], cube_size)
    z_centers = np.arange(min_corner[2] + cube_size / 2, max_corner[2], cube_size)

    grid_points = np.meshgrid(x_centers, y_centers, z_centers, indexing='ij')
    grid_points = np.stack(grid_points, axis=-1).reshape(-1, 3)
    return grid_points


def points_in_cube(points, centers, cube_size, chunk_size=3000):
    half_size = cube_size / 2
    indices_list = [[] for _ in range(len(centers))]

    min_corners = centers - half_size
    max_corners = centers + half_size

    for i in range(0, len(points), chunk_size):
        points_chunk = points[i:i + chunk_size]
        conditions = np.all(
            (points_chunk[:, np.newaxis] >= min_corners[np.newaxis]) &
            (points_chunk[:, np.newaxis] <= max_corners[np.newaxis]),
            axis=-1
        )
        chunk_indices_list = [np.flatnonzero(cond) for cond in conditions.T]

        for j, chunk_indices in enumerate(chunk_indices_list):
            if len(chunk_indices) > 0:
                indices_list[j].extend(chunk_indices + i)

    return indices_list


if __name__ == '__main__':
    ply_file = 'output/ggbond/point_cloud/iteration_7000/point_cloud.ply'
    cube_size = 1  # 假设每个小立方体的边长为 1
    result = devide_space_into_cubes(ply_file, cube_size)

    # 打印结果或者进一步处理
    for cube in result:
        if len(cube['points']) > 0:  # 仅打印包含点的立方体
            print(f"Cube Center: {cube['center']}")
            print(f"Contained Points: {cube['points']}")


# def devide_space_into_cubes(ply_file, cube_size):
#     points = read_ply(ply_file)
#     min_corner, max_corner = get_bounding_box(points)
#     centers = generate_cubes(min_corner, max_corner, cube_size)
#     indices_list = points_in_cube(points, centers, cube_size)
#
#     result = []
#     for center, indices in zip(centers, indices_list):
#         contained_points = points[indices]
#
#         result.append({
#             'center': center,
#             'points': contained_points,
#         })
#
#     return result
#
# def read_ply(ply_file):
#     ply_data = PlyData.read(ply_file)
#     vertices = ply_data['vertex']
#
#     points = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
#
#     return points
#
# def get_bounding_box(points):
#     min_corner = np.min(points, axis=0)
#     max_corner = np.max(points, axis=0)
#     return min_corner, max_corner
#
# def generate_cubes(min_corner, max_corner, cube_size):
#     x_centers = np.arange(min_corner[0] + cube_size / 2, max_corner[0], cube_size)
#     y_centers = np.arange(min_corner[1] + cube_size / 2, max_corner[1], cube_size)
#     z_centers = np.arange(min_corner[2] + cube_size / 2, max_corner[2], cube_size)
#
#     grid_points = np.meshgrid(x_centers, y_centers, z_centers, indexing='ij')
#     grid_points = np.stack(grid_points, axis=-1).reshape(-1, 3)
#     return grid_points
#
# def points_in_cube(points, centers, cube_size, chunk_size=1000):
#     points = torch.tensor(points, device='cuda')
#     centers = torch.tensor(centers, device='cuda')
#     half_size = cube_size / 2
#
#     min_corners = centers - half_size
#     max_corners = centers + half_size
#
#     indices_list = [[] for _ in range(len(centers))]
#
#     for i in range(0, len(points), chunk_size):
#         points_chunk = points[i:i + chunk_size]
#         conditions = (points_chunk[:, None] >= min_corners[None]) & (points_chunk[:, None] <= max_corners[None])
#         conditions = conditions.all(dim=-1)
#
#         for j in range(len(centers)):
#             chunk_indices = torch.nonzero(conditions[:, j]).squeeze().cpu().numpy()
#             if len(chunk_indices) > 0:
#                 indices_list[j].extend(chunk_indices + i)
#
#     return indices_list
#
# if __name__ == '__main__':
#     ply_file = 'output/ggbond/point_cloud/iteration_7000/point_cloud.ply'
#     cube_size = 1  # 假设每个小立方体的边长为 1
#     result = devide_space_into_cubes(ply_file, cube_size)
#
#     # 打印结果或者进一步处理
#     for cube in result:
#         if len(cube['points']) > 0:  # 仅打印包含点的立方体
#             print(f"Cube Center: {cube['center']}")
#             print(f"Contained Points: {cube['points']}")