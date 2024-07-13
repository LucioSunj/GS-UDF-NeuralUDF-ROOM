import subprocess
import argparse
if __name__ == '__main__':
    # TODO 改为可以通过不同的命令行参数来切换运行模式
    # parser = argparse.ArgumentParser()

    # 编译Cython程序
    # 不能这样做，因为编译过程需要当前目录处于custom_mc, 因此请在命令行中输入以下代码
    # subprocess.run('cd udfBranch/custom_mc')
    # subprocess.run('python setup.py build_ext --inplace')
    # subprocess.run('cd ..')
    # subprocess.run('cd ..')

    # closed objects for udf
    # subprocess.run('bash bashes/running_gs_ggbond-ne-udf_dtu.sh --gpu 0 --case scan118')
    # subprocess.run('bash bashes/running_gs_ggbond-ne-udf_dtu_ft.sh --gpu 0 --case scan118')

    # unclosed objects for udf
    subprocess.run('bash bashes/running_gs_ggbond-ne-udf_garment.sh --gpu 0 --case scan320 -s 0.001',text=True)
    subprocess.run('bash bashes/running_gs_ggbond-ne-udf_garment_ft.sh --gpu 0 --case scan320 -s 0.001',text=True)