#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/param.h>
#include <sys/time.h>
#include <arpa/inet.h>
#include <netdb.h>

#include "setup_ib.h"

struct IBRes ib_res;
char *server_name;
struct IBDest local_dest;
struct IBDest remote_dest;

int get_server_dest(){
    int sockfd;
    struct addrinfo hints, *result, *rp;

    memset(&hints, 0, sizeof(struct addrinfo));
    hints.ai_family = AF_UNSPEC; // Allow IPv4 or IPv6.
    hints.ai_socktype = SOCK_STREAM; // TCP

    if(getaddrinfo(server_name, VB_PORT_STR, &hints, &result)){
        perror("Found address failed!\n");
        return 1;
    }

    // Start connection:
    for(rp = result; rp; rp = rp->ai_next){
        sockfd = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
        if (sockfd == -1)
            continue;
        if (!connect(sockfd, rp->ai_addr, rp->ai_addrlen))
            break; // Success.
        close(sockfd);
    }

    if(!rp){
        perror("Connection with the server failed.\n");
        return 1;
    }

    freeaddrinfo(result);
    char *buf;
    buf = (char *)malloc(sizeof(struct IBDest));
    int offset = 0;
    memcpy(buf, &local_dest, sizeof(struct IBDest));
    write(sockfd, buf, sizeof(struct IBDest));
    memset(buf, 0, sizeof(struct IBDest));
    offset = 0;
    while (offset < sizeof(struct IBDest))
        offset += read(sockfd, buf + offset, sizeof(struct IBDest) - offset);
    memcpy(&remote_dest, buf, sizeof(struct IBDest));

    // Finish connection:
    close(sockfd);
    return 0;
}

int get_client_dest(){
    int sockfd, connfd, len;
    struct sockaddr_in server_address, client_address;

    // Start connection:
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if(sockfd == -1){
        perror("Socket creation failed!\n");
        return 1;
    }

    memset(&server_address, 0, sizeof(server_address));
    server_address.sin_family = AF_INET;
    server_address.sin_addr.s_addr = htonl(INADDR_ANY);
    server_address.sin_port = htons(VB_PORT);

    if(bind(sockfd, (struct sockaddr*)&server_address, sizeof(server_address))){
        perror("Socket bind failed!\n");
        return 1;
    }

    if(listen(sockfd, 5)){
        perror("Listen failed...\n");
        return 1;
    }

    len = sizeof(client_address);
    connfd = accept(sockfd, (struct sockaddr*)&client_address, &len);
    if (connfd < 0){
        perror("Server accept failed!\n");
        return 1;
    }

    char *buf;
    buf = (char*)malloc(sizeof(struct IBDest));
    int offset = 0;

    while (offset < sizeof(struct IBDest))
        offset += read(connfd, buf + offset, sizeof(struct IBDest) - offset);

    memcpy(&remote_dest, buf, sizeof(struct IBDest));
    memset(buf, 0, sizeof(struct IBDest));
    offset = 0;
    memcpy(buf, &local_dest, sizeof(struct IBDest));
    write(connfd, buf, sizeof(struct IBDest));
    close(sockfd);

    return 0;
}

int connect_between_qps(){
    struct ibv_qp_attr qp_attr = {
        .qp_state		= IBV_QPS_RTR,
        .path_mtu		= IBV_MTU_1024,
        .dest_qp_num	= remote_dest.qpn,
        .rq_psn			= remote_dest.psn,
        .max_dest_rd_atomic	= 1,
        .min_rnr_timer		= 12,
        .ah_attr		= {
            .is_global	= 0,
            .dlid		= remote_dest.lid,
            .sl		    = 0,
            .src_path_bits	= 0,
            .port_num	= IB_PORT
        }
    };

    if(ibv_modify_qp(
        ib_res.qp, &qp_attr, 
        IBV_QP_STATE | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | \
        IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | 
        IBV_QP_MIN_RNR_TIMER | IBV_QP_AV
    )){
        perror("Failed to modify QP to RTR.\n");
        return 1;
    }

    qp_attr.qp_state = IBV_QPS_RTS;
    qp_attr.timeout = 14;
    qp_attr.retry_cnt = 7;
    qp_attr.rnr_retry = 7;
    qp_attr.sq_psn = local_dest.psn;
    qp_attr.max_rd_atomic = 1;
    if (ibv_modify_qp(
        ib_res.qp, &qp_attr, IBV_QP_STATE | 
        IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | \
        IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | 
        IBV_QP_MAX_QP_RD_ATOMIC
    )){
        perror("Failed to modify QP to RTS.\n");
        return 1;
    }

    return 0;
}

int setup_ib(){
    struct ibv_device **dev_list, *ib_dev;

    srand48(getpid() * time(NULL));

    memset(&ib_res, 0, sizeof(struct IBRes));
    ib_res.size = MAX_BUF_SIZE;
    ib_res.buf = malloc(roundup(MAX_BUF_SIZE, sysconf(_SC_PAGESIZE)));
    if(!ib_res.buf){
        perror("Couldn't allocate work buf.\n");
        return 1;
    }
    memset(ib_res.buf, 0x7b + (server_name ? 0 : 1), MAX_BUF_SIZE);

    // Get an IB devices list:
    dev_list = ibv_get_device_list(NULL);
    if(!dev_list){
        perror("Failed to get IB devices list.\n");
        return 1;
    }
    ib_dev = *dev_list;

    if (!ib_dev){
        perror("No IB devices found.\n");
        return 1;
    }

    // Open an IB device context:
    ib_res.ctx = ibv_open_device(ib_dev);
    printf("Device name: %s\n", ibv_get_device_name(ib_dev));
    if (!ib_res.ctx){
        fprintf(stderr, "Couldn't get context for %s.\n", ibv_get_device_name(ib_dev));
        return 1;
    }

    // Allocate a Protection Domain:
    ib_res.pd = ibv_alloc_pd(ib_res.ctx);
    if (!ib_res.pd){
        perror("Failed to allocate Protection Domain.\n");
        return 1;
    }

    // Query IB port attribute:
    if (ibv_query_port(ib_res.ctx, IB_PORT, &ib_res.port_info)){
        perror("Failed to query IB port information.\n");
        return 1;
    }

    // Register a Memory Region:
    ib_res.mr = ibv_reg_mr(ib_res.pd, ib_res.buf, ib_res.size,
                           IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE);
    if (!ib_res.mr){
        perror("Couldn't register Memory Region.\n");
        return 1;
    }

    // Create a Completion Queue:
    ib_res.cq = ibv_create_cq(ib_res.ctx, 2 * CQ_COUNT, NULL, NULL, 0);
    if (!ib_res.cq){
        perror("Couldn't create Completion Queue.\n");
        return 1;
    }

    struct ibv_qp_init_attr qp_init_attr = {
            .send_cq = ib_res.cq,
            .recv_cq = ib_res.cq,
            .cap = {
                .max_send_wr = CQ_COUNT,
                .max_recv_wr = CQ_COUNT,
                .max_send_sge = 1,
                .max_recv_sge = 1,
            },
            .qp_type = IBV_QPT_RC,
    };

    ib_res.qp = ibv_create_qp(ib_res.pd, &qp_init_attr);
    if (!ib_res.qp) {
        perror("Couldn't create Queue Pair.\n");
        return 1;
    }

    struct ibv_qp_attr qp_attr = {
        .qp_state        = IBV_QPS_INIT,
        .pkey_index      = 0,
        .port_num        = IB_PORT,
        .qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ
    };

    if (ibv_modify_qp(
        ib_res.qp, &qp_attr, 
        IBV_QP_STATE | IBV_QP_PKEY_INDEX | 
        IBV_QP_PORT | IBV_QP_ACCESS_FLAGS
    )) {
        perror("Failed to modify QP to INIT.\n");
        return 1;
    }

    local_dest.lid = ib_res.port_info.lid;
    if (ib_res.port_info.link_layer == IBV_LINK_LAYER_INFINIBAND 
            && !local_dest.lid){
        perror("Couldn't get LID.\n");
        return 1;
    }

    local_dest.qpn = ib_res.qp->qp_num;
    local_dest.psn = lrand48() & 0xffffff;
    if (server_name){
        if (get_server_dest()){
            perror("Couldn't get server's details.\n");
            return 1;
        }
    }
    else{
        if(get_client_dest()){
            perror("Couldn't get client's details.\n");
            return 1;
        }
    }

    // Connect Queue Pair:
    if (connect_between_qps()){
        perror("Couldn't connect between client and server.\n");
        return 1;
    }

    ibv_free_device_list(dev_list);
    return 0;
}

void close_ib_connection(){
    if(ib_res.qp)
        ibv_destroy_qp(ib_res.qp);

    if(ib_res.cq)
        ibv_destroy_cq(ib_res.cq);

    if(ib_res.mr)
        ibv_dereg_mr(ib_res.mr);

    if(ib_res.pd)
        ibv_dealloc_pd(ib_res.pd);

    if(ib_res.ctx)
        ibv_close_device(ib_res.ctx);

    if(ib_res.buf)
        free(ib_res.buf);
}

void ib_send(float* buffer){
    struct RemoteMR remote_mr = {
        .remote_addr = (uint64_t)ib_res.buf,
        .rkey = ib_res.mr->rkey
    };

    // Get remote address and rkey:
    post_recv(sizeof(struct RemoteMR));
    wait_completions(RECV_WRID);
    memcpy(&remote_mr, ib_res.buf, sizeof(struct RemoteMR));

    // Send message via RDMA Write:
    memcpy(ib_res.buf, buffer, sizeof(buffer));
    post_send_write(remote_mr, MAX_BUF_SIZE);
    wait_completions(WRITE_WRID);

    post_send(1); // Notify the receiver for continue.
    wait_completions(SEND_WRID);
}

float* ib_recv(){
    struct RemoteMR remote_mr;

    // Send remote address and rkey:
    remote_mr.remote_addr = (uint64_t)ib_res.buf;
    remote_mr.rkey = ib_res.mr->rkey;
    memcpy(ib_res.buf, &remote_mr, sizeof(struct RemoteMR));
    post_send(sizeof(struct RemoteMR));
    wait_completions(SEND_WRID);

    // Wait to the sender for send message via RDMA Write:
    post_recv(1);
    wait_completions(RECV_WRID);

    return (float*)ib_res.buf;
}
