/home/xyzhao/openmpi/bin/mpirun -np 1 \
    -H 10.28.1.30:1 \
    -x MASTER_ADDR=10.28.1.30 \
    -x MASTER_PORT=1234 \
    -x PATH \
    -x LD_LIBRARY_PATH \
    python train_gpt2_ddp.py \
    --train_batch_size 2
