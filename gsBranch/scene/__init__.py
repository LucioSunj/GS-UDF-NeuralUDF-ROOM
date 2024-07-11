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

import os
import random
import json
from gsBranch.utils.system_utils import searchForMaxIteration
from gsBranch.scene.dataset_readers import sceneLoadTypeCallbacks
from gsBranch.scene.gaussian_model import GaussianModel
from gsBranch.arguments import ModelParams
from gsBranch.utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    def __init__(self, args: ModelParams, gaussians: GaussianModel, load_iteration=None, shuffle=True,
                 resolution_scales=[1.0]):
        """
       Initializes the class with parameters and sets up the Gaussian model.

       Parameters:
       - args (ModelParams): An object containing model parameters including the model path.
       - gaussians (GaussianModel): A GaussianModel object representing Gaussian distributions.
       - load_iteration (int, optional): Specifies the iteration of the trained model to load. If None, no model is loaded.
       - shuffle (bool): Determines whether to shuffle the training and test cameras.
       - resolution_scales (list): A list of resolution scaling factors.

       - args: 模型参数对象，包含模型路径等信息。
        - gaussians: 高斯模型对象，用于表示高斯分布。
        - load_iteration: 可选参数，指定加载的训练迭代次数。如果为None，则不加载。
        - shuffle: 是否对训练和测试相机进行随机打乱。
        - resolution_scales: 一个列表，包含不同分辨率的缩放因子。
        """
        self.model_path = args.model_path
        self.loaded_iter = None # 在这里的时候，一直都是
        self.gaussians = gaussians

        # Determines whether to load a trained model based on load_iteration's value
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print(f"Loading trained model at iteration {self.loaded_iter}")

        self.train_cameras = {}
        self.test_cameras = {}

        # Identifies the scene type based on source path files and loads scene information
        # 根据源路径中的文件判断场景类型，并加载场景信息
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        # If no iteration model is loaded, copies point cloud and camera information from scene info
        # 如果没有加载迭代模型，则从场景信息中复制点云文件和相机信息
        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply"),'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        # Shuffles the training and test cameras if required
        # 如果需要，对训练和测试相机进行随机打乱
        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-resolution consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-resolution consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        # Loads training and test camera information for different resolution scales
        # 根据不同的分辨率缩放因子，加载训练和测试相机信息
        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras,
                                                                            resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras,
                                                                           resolution_scale, args)

        # Initializes the Gaussian model based on whether an iteration model was loaded
        # 根据是否加载了迭代模型，决定如何初始化高斯模型
        # 一个从文件中得到初始化数据，一个从object中得到初始化数据
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                 "point_cloud",
                                                 f"iteration_{self.loaded_iter}",
                                                 "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]


