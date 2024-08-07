# Notice
Please DONNOT push the new commit to master branch. We could first push them to the develop/1.0 branch. After discussing and the meeting, we will update the master branch then. 

---
# GS-UDFRoom 流程

## 初始化过程

### 读取参数配置



### 读取、准备数据
- 将图片通过3dgs的convertor.py的类似方式转换成GS部分可用部分, 然后将这个数据放在 data/GS-Branch/CASE NAME 目录下
  - （CASE NAME指的是本次数据的命名）
- 将图片根据 [教程](https://blog.csdn.net/weixin_59961223/article/details/135429437) 转换成最终格式，放在 data/UDF-Branch/CustomData/CASE NAME/n 目录下 
  - （CASE NAME指的是本次数据的命名）
  - （n) 指的是数字 case 

- 写udf的.conf 文件放在udfBranch/confs下

  - 首先是一个：模板按照他们的无_ft 的, 需要改的部分：

    - ```
      base_exp_dir = ./exp/udf/CASE_NAME/n/
      expname = udf_CASE_NAME
      model_type = udf
      recording = [
        ./udfBranch,
        ./udfBranch/models,
        ./udfBranch/dataset,
      ]
      ```

    - ```
      data_dir = data/UDF-Branch/CustomData/CASE_NAME/n/
      ```

  - 再写一个有ft的：
    - 要改的内容与上面一样
#### GS-Branch -> 准备COLMAP数据
- copied from 3dgs
Our COLMAP loaders expect the following dataset structure in the source path location:

```
<location>
|---images
|   |---<image 0>
|   |---<image 1>
|   |---...
|---sparse
    |---0
        |---cameras.bin
        |---images.bin
        |---points3D.bin
```

For rasterization, the camera models must be either a SIMPLE_PINHOLE or PINHOLE camera. We provide a converter script ```convert.py```, to extract undistorted images and SfM information from input images. Optionally, you can use ImageMagick to resize the undistorted images. This rescaling is similar to MipNeRF360, i.e., it creates images with 1/2, 1/4 and 1/8 the original resolution in corresponding folders. To use them, please first install a recent version of COLMAP (ideally CUDA-powered) and ImageMagick. Put the images you want to use in a directory ```<location>/input```.
```
<location>
|---input
    |---<image 0>
    |---<image 1>
    |---...
```
 If you have COLMAP and ImageMagick on your system path, you can simply run 
```shell
python convert.py -s <location> [--resize] #If not resizing, ImageMagick is not needed
```
Alternatively, you can use the optional parameters ```--colmap_executable``` and ```--magick_executable``` to point to the respective paths. Please note that on Windows, the executable should point to the COLMAP ```.bat``` file that takes care of setting the execution environment. Once done, ```<location>``` will contain the expected COLMAP data set structure with undistorted, resized input images, in addition to your original images and some temporary (distorted) data in the directory ```distorted```.

If you have your own COLMAP dataset without undistortion (e.g., using ```OPENCV``` camera), you can try to just run the last part of the script: Put the images in ```input``` and the COLMAP info in a subdirectory ```distorted```:
```
<location>
|---input
|   |---<image 0>
|   |---<image 1>
|   |---...
|---distorted
    |---database.db
    |---sparse
        |---0
            |---...
```
Then run 
```shell
python convert.py -s <location> --skip_matching [--resize] #If not resizing, ImageMagick is not needed
```

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for convert.py</span></summary>

  #### --no_gpu
  Flag to avoid using GPU in COLMAP.
  #### --skip_matching
  Flag to indicate that COLMAP info is available for images.
  #### --source_path / -s
  Location of the inputs.
  #### --camera 
  Which camera model to use for the early matching steps, ```OPENCV``` by default.
  #### --resize
  Flag for creating resized versions of input images.
  #### --colmap_executable
  Path to the COLMAP executable (```.bat``` on Windows).
  #### --magick_executable
  Path to the ImageMagick executable.
</details>
<br>

#### UDF-Branch
- Custom data: [Same with NeuS](https://github.com/Totoro97/NeuS/tree/main/preprocess_custom_data)
  - 首先我们先学习一下它用ColMap创建的数据集的格式，这样两个就都是统一的方式了
    - [ColMap Custom Dataset](https://blog.csdn.net/weixin_59961223/article/details/135429437)
  - 但我们后面还是要再试试另一种，因为说不定多样性可以让结果表现更好
### 准备训练



## 训练过程

### Monocular Normals and Depth
- We use the pretrained [Metric3D](https://github.com/yvanyin/metric3d?tab=readme-ov-file) 
```python
import torch
model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_small', pretrain=True)
pred_depth, confidence, output_dict = model.inference({'input': rgb})
pred_normal = output_dict['prediction_normal'][:, :3, :, :] # only available for Metric3Dv2 i.e., ViT models
normal_confidence = output_dict['prediction_normal'][:, 3, :, :] # see https://arxiv.org/abs/2109.09881 for details
```
Supported models: metric3d_convnext_tiny, metric3d_convnext_large, 
metric3d_vit_small, metric3d_vit_large, metric3d_vit_giant2.


### Mutual Learning



### Backward Propagation
