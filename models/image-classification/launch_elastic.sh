torchrun \
    --nnodes=1:3 \
    --nproc_per_node=4 \
    --max_restarts=3\
    --rdzv_id=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint="net-g16:1234" \
    main_elastic.py \
    --arch resnet18 \
    --epochs 100 \
    --batch-size 256 \
    /data2/xyzhao/imagenet