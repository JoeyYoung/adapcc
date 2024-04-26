/home/xyzhao/openmpi/bin/mpirun \
        -np 2 \
        -H 127.0.0.1:1,127.0.0.1:1 \
        -mca pml ucx -x UCX_NET_DEVICES=mlx5_0:1 \
        -x MASTER_ADDR=127.0.0.1 \
        -x MASTER_PORT=1234 \
        -x LD_LIBRARY_PATH \
        -x PATH \
        ./check_mpi_p2p

# -mca pml ucx -x UCX_NET_DEVICES=mlx5_0:1 \