import argparseimport loggingimport torch.nn.functional as Fimport numpy as npimport torch# from torch.utils.tensorboard import SummaryWriterfrom gsBranch import arguments, scenefrom gsBranch.arguments import ModelParams, PipelineParams, OptimizationParamsfrom gsBranch.scene import Scene,GaussianModelfrom gsBranch.utils.general_utils import safe_statefrom gsBranch.arguments import ModelParams, PipelineParams, OptimizationParamsimport uuidfrom tqdm import tqdmfrom gsBranch.utils.image_utils import psnrimport sysimport osimport torchfrom random import randintfrom gsBranch.utils.loss_utils import l1_loss, ssimfrom gsBranch.gaussian_renderer import render, network_guifrom pyhocon import ConfigFactory, HOCONConverterimport cv2 as cvfrom udfBranch.dataset.dataset import Datasetfrom udfBranch.loss.loss import ColorLossfrom udfBranch.models.fields import NeRF, UDFNetwork, SingleVarianceNetwork, ResidualRenderingNetwork, BetaNetworkfrom udfBranch.models.udf_renderer_blending import UDFRendererBlending# GS Branch的东西try:    from torch.utils.tensorboard import SummaryWriter    TENSORBOARD_FOUND = Trueexcept ImportError:    TENSORBOARD_FOUND = Falsedef parse_arguments():    """       Parses command line arguments.       This function uses the argparse library to define and parse command line arguments. It sets up several arguments,       including model parameters, optimization parameters, and pipeline parameters, as well as an argument for specifying       the data directory. After parsing the arguments, the function returns a namespace object containing all specified       arguments.       Returns:           args: A namespace object containing command line arguments.       """    # Initialize the command line argument parser    parser = argparse.ArgumentParser()    # Add model parameter configuration    lp = ModelParams(parser)    # Add optimization parameter configuration    op = OptimizationParams(parser)    # Add pipeline parameter configuration    pp = PipelineParams(parser)    # Define a command line argument for specifying the location of the data directory    parser.add_argument("--data_dir", type=str, default="./data/", help="data directory")    # 以下是3dgs中照搬的参数    parser.add_argument('--ip', type=str, default="127.0.0.1")    parser.add_argument('--port', type=int, default=6009)    parser.add_argument('--debug_from', type=int, default=-1)    parser.add_argument('--detect_anomaly', action='store_true', default=False)    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])    parser.add_argument("--quiet", action="store_true")    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])    parser.add_argument("--start_checkpoint", type=str, default=None)    # UDF中的参数    parser.add_argument('--udf_conf', type=str, default='udfBranch/confs/base.conf')    parser.add_argument('--udf_mode', type=str, default='train')    parser.add_argument('--udf_model_type', type=str, default='')    parser.add_argument('--udf_threshold', type=float, default=0.005)    parser.add_argument('--udf_is_continue', default=False, action="store_true")    parser.add_argument('--udf_is_finetune', default=False, action="store_true")    parser.add_argument('--udf_reg_weights_schedule', default=False, action="store_true",                        help='the schedule of regularization weights')    parser.add_argument('--udf_vis_ray', default=False, action="store_true", help='visualize the udf of a ray for debug')    parser.add_argument('--udf_gpu', type=int, default=0)    parser.add_argument('--udf_resolution', type=int, default=128)    # parser.add_argument('--r', type=int, default=128)    parser.add_argument('--udf_case', type=str, default='', help='the object name or index of a dataset')    parser.add_argument('--udf_learning_rate', type=float, default=0)    parser.add_argument('--udf_learning_rate_geo', type=float, default=0,                        help='the learning rate of udf network, if do not use the global learning rate')    parser.add_argument('--udf_sparse_weight', type=float, default=0, help='the weight of geo regularizer')    # Parse the command line arguments and return    args = parser.parse_args(sys.argv[1:])    return args,lp,op,ppclass Trainer:    def __init__(self, args, lp, op, pp, conf_path, mode='train', case='CASE_NAME', model_type='', is_continue=False):        self.args = args        self.lp = lp        self.op = op        self.pp = pp        self.device = torch.device('cuda')        '''        接下来进行UDF branch的初始化        '''        self.udf_conf_path = conf_path        f = open(self.udf_conf_path)        conf_text = f.read()        conf_text = conf_text.replace('CASE_NAME', case)        f.close()        # 加载Configurations并且将配置信息存入self.conf中        self.udf_conf = ConfigFactory.parse_string(conf_text)        self.udf_conf['dataset.data_dir'] = self.udf_conf['dataset.data_dir'].replace('CASE_NAME', case)        # modify the setting based on input        if args.udf_learning_rate > 0:            self.udf_conf['train']['learning_rate'] = args.udf_learning_rate        if args.udf_learning_rate_geo > 0:            self.udf_conf['train']['learning_rate_geo'] = args.udf_learning_rate_geo        if args.udf_sparse_weight > 0:            self.udf_conf['train']['sparse_weight'] = args.udf_sparse_weight        # 设置 exp 的地址        self.udf_base_exp_dir = os.path.join(self.udf_conf['general.base_exp_dir'], self.udf_conf['general.expname'])        os.makedirs(self.udf_base_exp_dir, exist_ok=True)        # 加载dataset        self.udf_dataset_name = self.udf_conf.get_string('dataset.dataset_name', default='general')        self.udf_dataset = Dataset(self.udf_conf['dataset'])        self.iter_step = 0        # 加载参数        # trainning parameters        self.end_iter = self.udf_conf.get_int('train.end_iter')        self.save_freq = self.udf_conf.get_int('train.save_freq')        self.report_freq = self.udf_conf.get_int('train.report_freq')        self.val_freq = self.udf_conf.get_int('train.val_freq')        self.val_mesh_freq = self.udf_conf.get_int('train.val_mesh_freq')        self.batch_size = self.udf_conf.get_int('train.batch_size')        self.validate_resolution_level = self.udf_conf.get_int('train.validate_resolution_level')        self.use_white_bkgd = self.udf_conf.get_bool('train.use_white_bkgd')        # setting about learning rate schedule        self.udf_learning_rate = self.udf_conf.get_float('train.learning_rate')        self.udf_learning_rate_geo = self.udf_conf.get_float('train.learning_rate_geo')        self.udf_learning_rate_alpha = self.udf_conf.get_float('train.learning_rate_alpha')        self.udf_warm_up_end = self.udf_conf.get_float('train.warm_up_end', default=0.0)        self.udf_anneal_end = self.udf_conf.get_float('train.anneal_end', default=0.0)        # don't train the udf network in the early steps        self.udf_fix_geo_end = self.udf_conf.get_float('train.fix_geo_end', default=500)        self.udf_reg_weights_schedule = args.udf_reg_weights_schedule        self.udf_warmup_sample = self.udf_conf.get_bool('train.warmup_sample', default=False)  # * training schedule        # whether the udf network and appearance network share the same learning rate        self.udf_same_lr = self.udf_conf.get_bool('train.same_lr', default=False)        # weights        self.udf_igr_weight = self.udf_conf.get_float('train.igr_weight')        self.udf_igr_ns_weight = self.udf_conf.get_float('train.igr_ns_weight', default=0.0)        self.udf_mask_weight = self.udf_conf.get_float('train.mask_weight')        self.udf_sparse_weight = self.udf_conf.get_float('train.sparse_weight', default=0.0)        # loss functions        self.udf_color_loss_func = ColorLoss(**self.udf_conf['color_loss'])        self.udf_color_base_weight = self.udf_conf.get_float('color_loss.color_base_weight', 0.0)        self.udf_color_weight = self.udf_conf.get_float('color_loss.color_weight', 0.0)        self.udf_color_pixel_weight = self.udf_conf.get_float('color_loss.color_pixel_weight', 0.0)        self.udf_color_patch_weight = self.udf_conf.get_float('color_loss.color_patch_weight', 0.0)        self.udf_is_continue = is_continue        self.udf_is_finetune = args.udf_is_finetune        self.udf_vis_ray = args.udf_vis_ray  # visualize a ray for debug        self.udf_mode = mode        self.udf_model_type = self.udf_conf['general.model_type']        if model_type != '':  # overwrite            self.udf_model_type = model_type        self.udf_model_list = []        self.udf_writer = None        # Networks        params_to_train = []        params_to_train_nerf = []        params_to_train_geo = []        self.udf_nerf_outside = None        self.udf_nerf_coarse = None        self.udf_nerf_fine = None        self.udf_sdf_network_fine = None        self.udf_udf_network_fine = None        self.udf_variance_network_fine = None        self.udf_color_network_coarse = None        self.udf_color_network_fine = None        # 这NeRF用在这里干啥的？        self.udf_nerf_outside = NeRF(**self.udf_conf['model.nerf']).to(self.device)        self.udf_udf_network_fine = UDFNetwork(**self.udf_conf['model.udf_network']).to(self.device)        # 这俩网络干啥的？？        self.udf_variance_network_fine = SingleVarianceNetwork(**self.udf_conf['model.variance_network']).to(self.device)        self.udf_color_network_fine = ResidualRenderingNetwork(**self.udf_conf['model.rendering_network']).to(self.device)        self.udf_beta_network = BetaNetwork(**self.udf_conf['model.beta_network']).to(self.device)        params_to_train_nerf += list(self.udf_nerf_outside.parameters())        params_to_train_geo += list(self.udf_udf_network_fine.parameters())        params_to_train += list(self.udf_variance_network_fine.parameters())        params_to_train += list(self.udf_color_network_fine.parameters())        params_to_train += list(self.udf_beta_network.parameters())        # 设置 optimizer        self.udf_optimizer = torch.optim.Adam(            [{'params': params_to_train_geo, 'lr': self.udf_learning_rate_geo}, {'params': params_to_train},             {'params': params_to_train_nerf}],            lr=self.udf_learning_rate)        # 渲染对象        self.udf_renderer = UDFRendererBlending(self.udf_nerf_outside,                                            self.udf_udf_network_fine,                                            self.udf_variance_network_fine,                                            self.udf_color_network_fine,                                            self.udf_beta_network,                                            **self.udf_conf['model.udf_renderer'])        # TODO Load checkpoint 这里需要加入原函数中很多东西，而我们的checkpoint是不是应该重新写？因为需要两个模型同时的checkpoint吧        # latest_model_name = None        # if is_continue:        #     model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))        #     model_list = []        #     for model_name in model_list_raw:        #         if model_name[-3:] == 'pth':        #             # if model_name[-3:] == 'pth' and int(model_name[5:-4]) <= self.end_iter:        #             model_list.append(model_name)        #     model_list.sort()        #     latest_model_name = model_list[-1]        #        # if latest_model_name is not None:        #     logging.info('Find checkpoint: {}'.format(latest_model_name))        #     self.load_checkpoint(latest_model_name)        #        # if self.mode[:5] == 'train':        #     self.file_backup()    def training(        self,        dataset,        opt,        pipe,        testing_iterations,        saving_iterations,        checkpoint_iterations,        checkpoint,        debug_from,    ):        # 应该通过sfm首先初始化点云，这个点云用作3dgs的训练        # TODO 或许这里可以换成SLAM来初始化点云（ https://github.com/yanyan-li/gaussian-splatting-using-PlanarSLAM?tab=readme-ov-file ），借鉴一下GeoGaussian        # TODO 也可以考虑用 COLMAP-FREE GS 来进行渲染？        first_iter = 0        # 初始化        # Gaussian model Initialization        gaussians = GaussianModel(dataset.sh_degree)        # 场景的数据加载        # 就是利用COLMAP的初始化        # 已经完成了点云初始化：            # 如果没有加载迭代模型，点云数据是从场景信息中复制并创建高斯模型            # 如果加载了迭代模型，则直接从模型路径加载点云文件初始化高斯模型        # 对象中存储了相机数据        scene = Scene(dataset, gaussians)        # 设置训练过程的初始化参数和优化器。        # 此方法根据传入的训练参数初始化对象的某些属性，并配置优化器以适应特定的学习率策略。        gaussians.training_setup(opt)        # 看本次运行的时候是不是从checkpoint来开始的，如果是，则加载checkpoint的时候的模型参数以及训练次数        if checkpoint:            (model_params, first_iter) = torch.load(checkpoint)            gaussians.restore(model_params, opt)        # 根据背景设置初始化背景色        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")        # 初始化渲染的开始和结束事件，用于计算渲染时间        iter_start = torch.cuda.Event(enable_timing=True)        iter_end = torch.cuda.Event(enable_timing=True)        # 初始化 视图堆栈 和 训练日志的累积损失        self.viewpoint_stack = None        self.ema_loss_for_log = 0.0        # 这个是 Gaussian 的 Logger Writer        tb_writer = self.gs_prepare_output_and_logger(dataset)        ## UDF 训练初始化        # 随机重排图像顺序，以增加训练的多样性        image_perm = torch.randperm(self.udf_dataset.n_images)        # 初始化TensorBoard日志记录器        # TODO 这里由于 torch.util中找不到者tensorbored，由于这是日志相关信息，我们先不管吧        if TENSORBOARD_FOUND:            self.writer = SummaryWriter(log_dir=os.path.join(str(self.udf_base_exp_dir), 'logs'))        # 计算剩余的迭代步数        res_step = self.end_iter - self.iter_step        # 设置udf的优化器的学习率        for g in self.udf_optimizer.param_groups:            g['lr'] = self.udf_learning_rate        # 初始化beta标志        self.beta_flag = True        # 设置进度条        progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")        first_iter += 1        # # TODO 两个网络需要在同一个训练循环中进行训练，因此需要整合两者的训练iters        # 两个网络先分开训练，然后在一些时候进行一个对另一个的指导，然后再在某个时候，反过来指导一次，后面再分开训练得到最终结果        for iteration in range(first_iter, opt.iterations + 1):            print(iteration)            self.gs_process(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint,debug_from, scene=scene,gaussians=gaussians, background=background, iter_start=iter_start, iter_end=iter_end, iteration=iteration, tb_writer = tb_writer)            self.udf_process(image_perm)            with torch.no_grad():                # TODO 进度条的变化                if iteration % 10 == 0:                    progress_bar.set_postfix({"Loss": f"{self.ema_loss_for_log:.7f}"})                    progress_bar.update(10) # 每10个iteration迭代一次                if iteration == opt.iterations:                    progress_bar.close()    '''    以下是 gs 训练相关函数    '''    def gs_process(self,                   dataset,                   opt,                   pipe,                   testing_iterations,                   saving_iterations,                   checkpoint_iterations,                   checkpoint,                   debug_from,                   gaussians,                   scene,                   background,                   iter_start,                   iter_end,                   iteration, # 当前迭代到哪里了                   tb_writer                   ):        """            主训练函数，负责执行整个训练流程。            :param dataset: 数据集对象，提供训练所需数据            :param opt: 选项对象，包含训练相关的各种配置参数            :param pipe: 管道对象，用于渲染和处理图像            :param testing_iterations: 测试迭代次数列表，指定在哪些迭代进行测试            :param saving_iterations: 保存模型的迭代次数列表，指定在哪些迭代保存模型            :param checkpoint_iterations: 检查点的迭代次数列表，用于模型的断点续训            :param checkpoint: 检查点文件路径，用于加载预训练模型            :param debug_from: 调试开始的迭代次数，用于开启调试模式            """        # 主训练循环        # for iteration in range(first_iter, opt.iterations + 1):            # 尝试与GUI客户端建立连接        if network_gui.conn == None:            network_gui.try_connect()        while network_gui.conn != None:            try:                # 接收客户端的指令和图像数据                net_image_bytes = None                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()                if custom_cam != None:                    # 根据客户端的摄像头参数渲染图像                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2,                                                                                                               0).contiguous().cpu().numpy())                # 发送渲染的图像数据给客户端                network_gui.send(net_image_bytes, dataset.source_path)                # 如果客户端请求训练，且当前迭代未超过预定的最大迭代次数                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):                    break            except Exception as e:                network_gui.conn = None        # 记录渲染开始时间        iter_start.record()        # 更新高斯模型的学习率        gaussians.update_learning_rate(iteration)        # 每隔1000次迭代，提升SH的阶数        if iteration % 1000 == 0:            gaussians.oneupSHdegree()        # 随机选择一个训练用的摄像头        if not self.viewpoint_stack:            self.viewpoint_stack = scene.getTrainCameras().copy()        viewpoint_cam = self.viewpoint_stack.pop(randint(0, len(self.viewpoint_stack) - 1))        # 根据迭代次数决定是否开启调试模式        if (iteration - 1) == debug_from:            pipe.debug = True        # 根据配置生成随机背景或使用固定背景        bg = torch.rand((3), device="cuda") if opt.random_background else background        # 执行渲染        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg[            "viewspace_points"], \            render_pkg["visibility_filter"], render_pkg["radii"]        # 计算损失函数        gt_image = viewpoint_cam.original_image.cuda()        Ll1 = l1_loss(image, gt_image)        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))        loss.backward()        # 记录渲染结束时间        iter_end.record()        # 更新进度条和日志        with torch.no_grad():            self.ema_loss_for_log = 0.4 * loss.item() + 0.6 * self.ema_loss_for_log            # if iteration % 10 == 0:            #     progress_bar.set_postfix({"Loss": f"{self.ema_loss_for_log:.7f}"})            #     progress_bar.update(10)            # if iteration == opt.iterations:            #     progress_bar.close()            # 记录训练日志和保存模型            self.gs_training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),                            testing_iterations, scene, render, (pipe, background))            if iteration in saving_iterations:                print(f"\n[ITER {iteration}] Saving Gaussians")                scene.save(iteration)            # 根据迭代次数进行密度增加和不透明度重置            if iteration < opt.densify_until_iter:                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],                                                                     radii[visibility_filter])                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent,                                                size_threshold)                if iteration % opt.opacity_reset_interval == 0 or (                        dataset.white_background and iteration == opt.densify_from_iter):                    gaussians.reset_opacity()            # 进行优化器的一步更新            if iteration < opt.iterations:                gaussians.optimizer.step()                gaussians.optimizer.zero_grad(set_to_none=True)            # 保存检查点            if iteration in checkpoint_iterations:                print(f"\n[ITER {iteration}] Saving Checkpoint")                torch.save((gaussians.capture(), iteration), scene.model_path + f"/chkpnt{iteration}.pth")    def gs_training_report(self,tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,                        renderArgs):        if tb_writer:            tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)            tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)            tb_writer.add_scalar('iter_time', elapsed, iteration)        # Report test and samples of training set        if iteration in testing_iterations:            torch.cuda.empty_cache()            validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},                                  {'name': 'train',                                   'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in                                               range(5, 30, 5)]})            for config in validation_configs:                if config['cameras'] and len(config['cameras']) > 0:                    l1_test = 0.0                    psnr_test = 0.0                    for idx, viewpoint in enumerate(config['cameras']):                        image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)                        if tb_writer and (idx < 5):                            tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),                                                 image[None], global_step=iteration)                            if iteration == testing_iterations[0]:                                tb_writer.add_images(                                    config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),                                    gt_image[None], global_step=iteration)                        l1_test += l1_loss(image, gt_image).mean().double()                        psnr_test += psnr(image, gt_image).mean().double()                    psnr_test /= len(config['cameras'])                    l1_test /= len(config['cameras'])                    print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test,                                                                            psnr_test))                    if tb_writer:                        tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)                        tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)            if tb_writer:                tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)                tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)            torch.cuda.empty_cache()    def gs_prepare_output_and_logger(self,args):        if not args.model_path:            if os.getenv('OAR_JOB_ID'):                unique_str = os.getenv('OAR_JOB_ID')            else:                unique_str = str(uuid.uuid4())            args.model_path = os.path.join("./output/", unique_str[0:10])        # Set up output folder        print("Output folder: {}".format(args.model_path))        os.makedirs(args.model_path, exist_ok=True)        with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:            cfg_log_f.write(str(argparse.Namespace(**vars(args))))        # Create Tensorboard writer        tb_writer = None        if TENSORBOARD_FOUND:            tb_writer = SummaryWriter(args.model_path)        else:            print("Tensorboard not available: not logging progress")        return tb_writer    '''    以下是与udf相关函数    '''    def udf_process(self, image_perm):        """        训练用户定义函数（UDF）的函数。        这个函数包含了训练过程中的所有步骤，例如更新学习率、调整颜色损失权重、渲染图像等。        它使用了PyTorch和一些自定义的损失函数和渲染器来逐步优化模型。        需要注意：已经被改成了训练的        参数:        self: 类的实例，包含了训练所需的所有属性，如优化器、学习率、数据集等。        image_perm: 图像序列        """        # 每轮训练过程（注意只是一轮，外部需要套上训练循环）        # for iter_i in tqdm(range(res_step)):            # 根据是否使用相同的学习率更新学习率        if self.udf_same_lr:            self.udf_update_learning_rate(start_g_id=0)        else:            self.udf_update_learning_rate(start_g_id=1)            self.udf_update_learning_rate_geo()        # 调整颜色损失权重        color_base_weight, color_weight, color_pixel_weight, color_patch_weight = self.udf_adjust_color_loss_weights()        # 根据当前迭代步数获取图像索引，并准备随机射线和对应的地面 truth        img_idx = image_perm[self.iter_step % len(image_perm)]        sample = self.udf_dataset.gen_random_rays_patches_at(            img_idx, self.batch_size,            crop_patch=color_patch_weight > 0.0, h_patch_size=self.udf_color_loss_func.h_patch_size)        # 解析样本数据        data = sample['rays']        rays_uv = sample['rays_ndc_uv']        gt_patch_colors = sample['rays_patch_color']        gt_patch_mask = sample['rays_patch_mask']        # 根据颜色像素权重和颜色块权重决定是否加载参考和源图像信息        if color_pixel_weight > 0. or color_patch_weight > 0.:            # todo: this load is very slow            ref_c2w, src_c2ws, src_intrinsics, src_images, img_wh = self.udf_dataset.get_ref_src_info(img_idx)            src_w2cs = torch.inverse(src_c2ws)        else:            ref_c2w, src_c2ws, src_w2cs, src_intrinsics, src_images = None, None, None, None, None        # todo load supporting images        # 提取射线的起点、方向、真实RGB值和遮罩        rays_o, rays_d, true_rgb, mask = data[:, :3], data[:, 3: 6], data[:, 6: 9], data[:, 9: 10]        # 计算射线的近远裁剪距离        near, far = self.udf_dataset.near_far_from_sphere(rays_o, rays_d)        # 将遮罩转换为浮点格式        mask = (mask > 0.5).float()        # 避免除以零        mask_sum = mask.sum() + 1e-5        # 使用渲染器渲染图像        render_out = self.udf_renderer.render(rays_o, rays_d, near, far,                                          flip_saturation=self.udf_get_flip_saturation(),                                          color_maps=src_images if color_pixel_weight > 0. else None,                                          w2cs=src_w2cs,                                          intrinsics=src_intrinsics,                                          query_c2w=ref_c2w,                                          img_index=None,                                          rays_uv=rays_uv if color_patch_weight > 0 else None,                                          cos_anneal_ratio=self.udf_get_cos_anneal_ratio())        # 提取渲染结果中的各项信息        weight_sum = render_out['weight_sum']        color_base = render_out['color_base']        color = render_out['color']        color_pixel = render_out['color_pixel']        patch_colors = render_out['patch_colors']        # 这里也就说明了Mask是optional的，这也是我们选用NeuralUDF的原因        patch_mask = (render_out['patch_mask'].float()[:, None] * (weight_sum > 0.5).float()) > 0. \            if render_out['patch_mask'] is not None else None        pixel_mask = mask if self.udf_mask_weight > 0 else None        variance = render_out['variance']        beta = render_out['beta']        gamma = render_out['gamma']        gradient_error = render_out['gradient_error']        gradient_error_near_surface = render_out['gradient_error_near_surface']        sparse_error = render_out['sparse_error']        udf = render_out['udf']        udf_min = udf.min(dim=1)[0][mask[:, 0] > 0.5].mean()        # 计算颜色损失        color_losses = self.udf_color_loss_func(            color_base, color, true_rgb, color_pixel,            pixel_mask, patch_colors, gt_patch_colors, patch_mask        )        # 提取各项颜色损失        color_total_loss = color_losses['loss']        color_base_loss = color_losses['color_base_loss']        color_loss = color_losses['color_loss']        color_pixel_loss = color_losses['color_pixel_loss']        color_patch_loss = color_losses['color_patch_loss']        # 计算PSNR（峰值信噪比）        psnr = 20.0 * torch.log10(            1.0 / (((color - true_rgb) ** 2 * mask).sum() / (mask_sum * 3.0)).sqrt())        # 计算遮罩损失        # mask_loss = (weight_sum - mask).abs().mean()        mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask)        # 计算Eikonal损失        gradient_error_loss = gradient_error        # 设置遮罩权重        mask_weight = self.udf_mask_weight        # 根据条件判断是否使beta可训练        if variance.mean() < 2 * beta.item() and variance.mean() < 0.01 and self.beta_flag and self.udf_variance_network_fine.variance.requires_grad:            print("make beta trainable")            self.udf_beta_network.set_beta_trainable()            self.beta_flag = False        # 根据迭代步数决定是否使变差网络可训练        if self.udf_variance_network_fine.variance.requires_grad is False and self.iter_step > 20000:            self.udf_variance_network_fine.set_trainable()        # 根据是否启用权重调度决定常规权重        if not self.udf_reg_weights_schedule:            igr_ns_weight = self.udf_igr_ns_weight            sparse_weight = self.udf_sparse_weight        else:            igr_ns_weight, sparse_weight = self.udf_regularization_weights_schedule()    def udf_update_learning_rate(self, start_g_id=0):        """        更新学习率的函数。根据迭代步骤数动态调整学习率。        在热身阶段（warm_up_end之前），学习率线性增加。热身阶段后，学习率根据余弦退火（Cosine Annealing）策略调整。        参数:        - start_g_id: int, 默认值为0。优化器参数组的起始索引，用于指定从哪个参数组开始更新学习率。        """        # 检查是否处于热身阶段        if self.iter_step < self.udf_warm_up_end:            # 在热身阶段，学习率线性增加            learning_factor = self.iter_step / self.udf_warm_up_end        else:            # 热身阶段后，应用余弦退火策略            alpha = self.udf_learning_rate_alpha            progress = (self.iter_step - self.udf_warm_up_end) / (self.end_iter - self.udf_warm_up_end)            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha        # 更新优化器中指定参数组的学习率        for g in self.udf_optimizer.param_groups[start_g_id:]:            g['lr'] = self.udf_learning_rate * learning_factor    def udf_update_learning_rate_geo(self):        """        根据迭代步骤更新学习率，采用地理学习率策略。        在不同的迭代阶段，学习率的更新方式不同，旨在在训练的初期和后期都能保持有效的学习。        - 初始阶段：学习率固定为0，以便模型先学习基本的几何结构。        - 预热阶段：学习率线性增加，逐渐增加模型的学习能力。        - 快速训练阶段：学习率保持在1，模型进行快速学习。        - 梯度下降阶段：学习率根据余弦退火策略递减，以便模型能够更细致地调整参数。        """        # 判断当前迭代步骤是否在固定几何结构学习率的阶段        if self.iter_step < self.udf_fix_geo_end:  # * make bg nerf learn first            learning_factor = 0.0        # 判断当前迭代步骤是否在预热阶段        elif self.iter_step < self.udf_warm_up_end * 2:            learning_factor = self.iter_step / (self.udf_warm_up_end * 2)        # 判断当前迭代步骤是否在快速训练阶段        elif self.iter_step < self.end_iter * 0.5:            learning_factor = 1.0        else:            # 计算当前迭代步骤的学习率衰减因子            alpha = self.udf_learning_rate_alpha            progress = (self.iter_step - self.end_iter * 0.5) / (self.end_iter - self.end_iter * 0.5)            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha        # 更新优化器中第一个参数组的学习率        for g in self.udf_optimizer.param_groups[:1]:            g['lr'] = self.udf_learning_rate_geo * learning_factor    def udf_adjust_color_loss_weights(self):        """        调整颜色损失权重的函数。        根据训练的阶段（fine-tune与否以及迭代步数）动态调整颜色基础权重、颜色权重、颜色像素权重和颜色块权重。        这有助于在训练过程中平衡颜色恢复和细节保留之间的关系。        返回:        - color_base_weight: 颜色基础权重，用于整体颜色恢复。        - color_weight: 颜色权重，用于局部颜色恢复。        - color_pixel_weight: 颜色像素权重，用于像素级颜色恢复。        - color_patch_weight: 颜色块权重，用于区域级颜色恢复。        """        # 根据是否进行fine-tune调整权重因子        if self.udf_is_finetune:            factor = 1.0        else:            # 根据迭代步数分阶段调整权重因子            if self.iter_step < 10000:                factor = 0            elif self.iter_step < 20000:                factor = np.clip((self.iter_step - 10000) / 10000, 0, 1)            else:                factor = 1.        # 根据颜色基础权重和当前因子调整颜色权重        if self.udf_color_base_weight < self.udf_color_weight:            color_base_weight = self.udf_color_base_weight * factor        else:            color_base_weight = self.udf_color_base_weight        # 固定颜色权重        color_weight = self.udf_color_weight        # 根据因子调整颜色像素权重和颜色块权重        color_pixel_weight = self.udf_color_pixel_weight * factor        color_patch_weight = self.udf_color_patch_weight * factor        # 设置调整后的颜色损失权重        self.udf_color_loss_func.set_color_weights(color_base_weight, color_weight, color_pixel_weight, color_patch_weight)        return color_base_weight, color_weight, color_pixel_weight, color_patch_weight    def udf_regularization_weights_schedule(self):        """        规则化权重调度函数。        根据训练进程动态调整不规则噪声权重和稀疏权重，以平衡模型复杂度和训练效果。        返回:        - igr_ns_weight: 不规则噪声权重。        - sparse_weight: 稀疏权重。        """        # 初始化不规则噪声权重和稀疏权重为0        igr_ns_weight = 0.0        sparse_weight = 0.0        # 计算权重调整的两个关键迭代步数        end1 = self.end_iter // 5        end2 = self.end_iter // 2        # 根据当前迭代步数调整不规则噪声权重        if self.iter_step >= end1:            igr_ns_weight = self.udf_igr_ns_weight * np.clip((self.iter_step - end1) / end1, 0.0, 1.0)        # 根据当前迭代步数决定是否启用稀疏权重        if self.iter_step >= end2:            sparse_weight = self.udf_sparse_weight        return igr_ns_weight, sparse_weight    def udf_get_flip_saturation(self, flip_saturation_max=0.9):        """        获取饱和度翻转概率的函数。        根据训练进程动态调整饱和度翻转的概率，以探索不同的色彩空间。        参数:        - flip_saturation_max: 饱和度翻转的最大概率。        返回:        - flip_saturation: 饱和度翻转的概率。        """        # 设置饱和度翻转概率的起始迭代步数        start = 10000        # 根据当前迭代步数决定饱和度翻转概率        if self.iter_step < start:            flip_saturation = 0.0        elif self.iter_step < self.end_iter * 0.5:            flip_saturation = flip_saturation_max        else:            flip_saturation = 1.0        # 在fine-tune过程中不进行饱和度翻转        if self.udf_is_finetune:            flip_saturation = 1.0        return flip_saturation    def udf_get_cos_anneal_ratio(self):        """        获取余弦退火比例的函数。        根据训练进程动态调整学习率，采用余弦退火策略。        返回:        - cos_anneal_ratio: 余弦退火的比例。        """        # 如果没有设置退火结束点，则不进行退火        if self.udf_anneal_end == 0.0:            return 1.0        else:            # 计算当前迭代步数相对于退火结束点的比例，并限制在[0, 1]之间            return np.min([1.0, self.iter_step / self.udf_anneal_end])if __name__ == "__main__":    torch.set_default_tensor_type('torch.cuda.FloatTensor')    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"    logging.basicConfig(level=logging.DEBUG, format=FORMAT)    # 首先读取argumentss    args,lp,op,pp = parse_arguments()    args.save_iterations.append(args.iterations)    print("Optimizing " + args.model_path)    torch.cuda.set_device(args.udf_gpu)    # Initialize system state (RNG)    safe_state(args.quiet)    torch.autograd.set_detect_anomaly(args.detect_anomaly)    # TODO Start GUI server, configure and run training    # network_gui.init(args.ip, args.port)    # 初始化trainer    # 后五个都是UDF的数据    trainer = Trainer(args, lp, op, pp, args.udf_conf, args.udf_mode, args.udf_case, args.udf_model_type, args.udf_is_continue)    # 前8个参数都是在3dgs中设置的    # UDF 的参数已经在__init__函数中初始化完成了，它会直接从self对象当中提取，而无需从function传入了    trainer.training(lp.extract(args),                     op.extract(args),                     pp.extract(args),                     args.test_iterations,                     args.save_iterations,                     args.checkpoint_iterations,                     args.start_checkpoint,                     args.debug_from)