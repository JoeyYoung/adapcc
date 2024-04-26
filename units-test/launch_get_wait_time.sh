/home/xyzhao/openmpi/bin/mpirun \
	-np 8 \
	-H 127.0.0.1:4,127.0.0.1:4 \
	-x MASTER_ADDR=127.0.0.1 \
	-x MASTER_PORT=1234 \
	-x LD_LIBRARY_PATH \
	-x PATH \
	python get_wait_time.py \
    --backend nccl --batch 128 --steps 10000 --coordinator 127.0.0.1 --output wait_time_heter_bc128_8g.csv --heter_alpha 2.7
