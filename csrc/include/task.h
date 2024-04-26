#ifndef TASK_H
#define TASK_H

#include "cuda_runtime.h"
#include "init.h"
#include "trans.h"
#include "time.h"
#include <map>
#include <sys/time.h>

enum prof_task_type{
    LATENCY,
    BANDWIDTH
};

typedef struct prof_task_st{
    enum prof_task_type type;
    char src_ip[20];
    char dst_ip[20];
    int src_local_rank;
    int src_world_rank;
    int dst_local_rank;
    int dst_world_rank;
} prof_task;

typedef struct round_task_st {
    prof_task_st* send_lc_task;
    prof_task_st* send_bw_task;
    prof_task_st* recv_lc_task;
    prof_task_st* recv_bw_task;
} round_task;

void check_intra_task_list(vector<prof_task*> task_list);
void check_inter_task_list(vector<round_task*> task_list);

prof_task* pack_task(enum prof_task_type type, char* src_ip, char* dst_ip, int src_rank, int dst_rank, map<int, int> lookup);

int find_pos_vector(vector<char*> list, char* ip);

int get_duration_us(struct timeval start, struct timeval end);

float get_bandwidth_GBps(int float_num, int duration_us);

void print_matrix(vector<vector<float>> matrix, int m, int n);

#endif
