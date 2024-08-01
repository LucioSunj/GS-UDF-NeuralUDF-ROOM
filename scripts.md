# 运行命令
- 首先进行一次这个
```ssh
python train.py -s <path to COLMAP or NeRF Synthetic dataset> --eval --udf_gpu 0 --udf_case <case number> --udf_conf udfBranch/confs/udf_scene0_blending.conf --udf_threshold 0.005 --udf_resolution 128 --udf_vis_ray --udf_reg_weights_schedule --udf_sparse_weight 0.001
```
- 再来一个这个：
```ssh
python train.py -s <path to COLMAP or NeRF Synthetic dataset> --eval --udf_gpu 0 --udf_case <case number> --udf_conf udfBranch/confs/udf_scene0_blending_ft.conf --udf_threshold 0.005 --udf_resolution 128 --udf_vis_ray --udf_reg_weights_schedule --udf_sparse_weight 0.01
```

- 然后进行rendering evaluation
```ssh
python render.py -m <path to pre-trained model> -s <path to dataset>
python metrics.py -m <path to trained model> # Compute error metrics on renderings
```
