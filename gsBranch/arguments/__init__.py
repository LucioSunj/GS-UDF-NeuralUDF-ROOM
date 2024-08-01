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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    # 这是一个空类，作为参数组的容器。
    pass

class ParamGroup:
    '''
    ParamGroup类用于将参数添加到ArgumentParser的一个参数组中。
    __init__方法会根据类的属性自动生成命令行参数，属性名以_开头的会生成简写参数。
    extract方法则将解析后的参数提取到GroupParams实例中。
    '''
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup):
    '''
    ModelParams类继承自ParamGroup，定义了一组与模型加载相关的参数。
    __init__方法初始化了这些参数并调用父类的构造函数生成命令行参数。
    extract方法在提取参数后，将source_path转换为绝对路径。
    '''
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        # self._model_path = ""
        self._model_path = "./output"
        self._images = "images"
        # self._resolution = -1
        self._GSresolution = 1 # 这里我们直接保证了不会进行缩放
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        # 接下来还有一些RaDe-GS相关的参数
        self._dataset = ""
        self.use_decoupled_appearance = False
        self.disable_filter3D = False
        self.kernel_size = 0.0 # Size of 2D filter in mip-splatting
        self.use_coord_map = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    '''
    PipelineParams类定义了一组与流水线处理相关的参数。
    '''
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    '''
    OptimizationParams类定义了一组与优化相关的参数。
    '''
    def __init__(self, parser):
        self.iterations = 30_000
        # self.gs_iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.random_background = False
        # 一些RaDe-GS相关参数
        self.appearance_embeddings_lr = 0.001
        self.appearance_network_lr = 0.001
        self.lambda_distortion = 100
        self.lambda_depth_normal = 0.05
        self.regularization_from_iter = 15_000
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    '''
    get_combined_args函数从命令行和配置文件中读取参数并合并。
    它首先解析命令行参数，然后尝试从指定路径读取配置文件中的参数，
    最后将命令行参数覆盖配置文件中的参数，返回合并后的参数集合。
    :param parser:
    :return:
    '''
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    '''
    argparse.Namespace 是 argparse 模块中的一个类，用于存储解析后的命令行参数。
    它类似于一个简单的对象，可以通过属性访问存储的参数值。

    在 argparse 模块中，当你调用 parse_args 方法解析命令行参数时，
    它会返回一个 Namespace 对象，里面包含所有解析后的参数
    '''
    return Namespace(**merged_dict)
