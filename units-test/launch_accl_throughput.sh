/home/xyzhao/openmpi/bin/mpirun \
	-np 4 \
	-H 127.0.0.1:4 \
	-x MASTER_ADDR=127.0.0.1 \
	-x MASTER_PORT=1234 \
	-x LD_LIBRARY_PATH \
	-x PATH \
	python throughput.py \
    --backend accl --coordinator 127.0.0.1 --parallel_degree 4 --chunk_size 2 --collective alltoall --stack rdma --nic mlx5:0 --steps 10
