import subprocess

commands = [
    'python train.py -s <path to COLMAP or NeRF Synthetic dataset> --eval --udf_gpu 0 --udf_case 0 --udf_conf udfBranch/confs/udf_scene0_blending.conf --udf_threshold 0.005 --udf_resolution 128 --udf_vis_ray --udf_reg_weights_schedule --udf_sparse_weight 0.001',
    'python train.py -s <path to COLMAP or NeRF Synthetic dataset> --eval --udf_gpu 0 --udf_case 0 --udf_conf udfBranch/confs/udf_scene0_blending_ft.conf --udf_threshold 0.005 --udf_resolution 128 --udf_vis_ray --udf_reg_weights_schedule --udf_sparse_weight 0.01',
    'python render.py -m <path to pre-trained model> -s <path to dataset>',
    'python metrics.py -m <path to trained model> # Compute error metrics on renderings'
]

for command in commands:
    try:
        print("Executing: ",command)
        subprocess.run(command, check=True, shell=True)
        print("Successfully executed: ",command)
    except subprocess.CalledProcessError as e:
        print("Error occurred while executing: ",command)
        print(e)