/home/xyzhao/openmpi/bin/mpirun \
        -np 4 \
        -H 10.28.1.30:4 \
        -mca pml ucx -x UCX_NET_DEVICES=mlx5_0:1 \
        -x NCCL_DEBUG=INFO \
        -x LD_LIBRARY_PATH \
        -x PATH \
        ./build/all_reduce
