#!/bin/bash
# usage函数是一个辅助函数，主要用于在用户提供的命令行参数不正确或不符合预期时，向用户显示脚本的正确用法。
# 这个函数帮助用户理解如何正确使用脚本，并确保脚本在参数错误的情况下不会继续执行，从而避免可能的错误或意外行为
usage() {
  echo "Usage: ${0} [-g|--gpu] [-c|--case]  [-lr|--learning_rate]  [-lr_geo|--learning_rate_geo]"  1>&2
  exit 1
}
while [[ $# -gt 0 ]]; # 当所有参数都被处理完后，$#将变为0，while循环结束
  do
  key=${1}
  case ${key} in
    -c|--case)
      CASE=${2}
      shift 2 # 这里表示开始处理下一个参数
      ;;
    -g|--gpu)
      GPU=${2}
      shift 2
      ;;
    -lr|--learning_rate)
      LR=${2}
      shift 2
      ;;
    -lr_geo|--learning_rate_geo)
      LR_GEO=${2}
      shift 2
      ;;
    *)
      usage
      shift
      ;;
  esac
done

CUDA_VISIBLE_DEVICES=${GPU} python exp_runner_blending.py --conf ./confs/udf_dtu_blending.conf \
--case ${CASE} --threshold 0.005 --resolution 128