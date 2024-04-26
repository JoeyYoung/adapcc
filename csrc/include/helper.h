#ifndef HELPER_H
#define HELPER_H

#include <infiniband/verbs.h>
#include <stdio.h>
#include "setup_ib.h"

#define VB_PORT 3000
#define VB_PORT_STR "3000"
#define IB_PORT 1
#define CQ_COUNT 1000
#define WC_BATCH 100
#define MAX_BUF_SIZE 400*1024*1024

enum {
    RECV_WRID = 1,
    SEND_WRID = 2,
    READ_WRID = 4,
    WRITE_WRID = 8,
};

struct RemoteMR {
    uint64_t remote_addr;
    uint32_t rkey;
};

// Queue Pairs message exchanges
int post_recv(int size);
int post_send(int size);
int post_send_read(struct RemoteMR remote_mr, int size);
int post_send_write(struct RemoteMR remote_mr, int size);
int wait_completions(int wr_id);

#endif
