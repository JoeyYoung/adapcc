#!/bin/bash

horovod_home="/home/xyzhao/horovod"
nccl_home="/home/xyzhao/dynamic-nccl/build"
gpu_op="NCCL"

cd $horovod_home
python setup.py sdist
HOROVOD_NCCL_HOME=$nccl_home HOROVOD_GPU_OPERATIONS=$gpu_op /home/xyzhao/anaconda3/envs/torch/bin/pip install --no-cache-dir dist/horovod-0.22.0.tar.gz

echo "Horovod Rebuilt"
