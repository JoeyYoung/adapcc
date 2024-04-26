#include "helper.h"

int post_recv(int size){
    struct ibv_sge list = {
        .addr	= (uint64_t)ib_res.buf,
        .length = size,
        .lkey	= ib_res.mr->lkey
    };

    struct ibv_recv_wr *bad_wr, wr = {
        .wr_id	    = RECV_WRID,
        .sg_list    = &list,
        .num_sge    = 1,
        .next       = NULL
    };

    ibv_post_recv(ib_res.qp, &wr, &bad_wr);
}

int post_send(int size){
    struct ibv_sge list = {
        .addr	= (uint64_t)ib_res.buf,
        .length = size,
        .lkey	= ib_res.mr->lkey
    };

    struct ibv_send_wr *bad_wr, wr = {
        .wr_id	    = SEND_WRID,
        .sg_list    = &list,
        .num_sge    = 1,
        .opcode     = IBV_WR_SEND,
        .send_flags = IBV_SEND_SIGNALED,
        .next       = NULL
    };

    return ibv_post_send(ib_res.qp, &wr, &bad_wr);
}

int post_send_read(struct RemoteMR remote_mr, int size){
    struct ibv_sge list = {
        .addr	= (uint64_t)ib_res.buf,
        .length = size,
        .lkey	= ib_res.mr->lkey
    };

    struct ibv_send_wr *bad_wr, wr = {
        .wr_id	    = READ_WRID,
        .sg_list    = &list,
        .num_sge    = 1,
        .opcode     = IBV_WR_RDMA_READ,
        .send_flags = IBV_SEND_SIGNALED,
        .wr.rdma.remote_addr = remote_mr.remote_addr,
        .wr.rdma.rkey = remote_mr.rkey,
        .next       = NULL
    };

    ibv_post_send(ib_res.qp, &wr, &bad_wr);
}

int post_send_write(struct RemoteMR remote_mr, int size){
    struct ibv_sge list = {
        .addr	= (uint64_t)ib_res.buf,
        .length = size,
        .lkey	= ib_res.mr->lkey
    };

    struct ibv_send_wr *bad_wr, wr = {
        .wr_id	    = WRITE_WRID,
        .sg_list    = &list,
        .num_sge    = 1,
        .opcode     = IBV_WR_RDMA_WRITE,
        .send_flags = IBV_SEND_SIGNALED,
        .wr.rdma.remote_addr = remote_mr.remote_addr,
        .wr.rdma.rkey = remote_mr.rkey,
        .next       = NULL
    };

    ibv_post_send(ib_res.qp, &wr, &bad_wr);
}

int wait_completions(int wr_id){
    int finished = 0, count = 1;

    while (finished < count){
        struct ibv_wc wc[WC_BATCH];
        int n;
        do {
            n = ibv_poll_cq(ib_res.cq, WC_BATCH, wc);
            if (n < 0){
                fprintf(stderr, "Poll CQ failed %d\n", n);
                return 1;
            }
        } while (n < 1);

        for (int i = 0; i < n; i++){
            if (wc[i].status != IBV_WC_SUCCESS){
                fprintf(
                    stderr, 
                    "Failed status %s (%d) for wr_id %d\n",
                    ibv_wc_status_str(wc[i].status), 
                    wc[i].status, 
                    (int)wc[i].wr_id);
                return 1;
            }

            if (wc[i].wr_id == wr_id)
                finished++;
        }
    }
    return 0;
}
