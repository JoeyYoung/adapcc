/*
    Wrapped tasks for profiling context
*/

#include "task.h"

using namespace std;

void check_intra_task_list(vector<prof_task*> task_list){
    prof_task* task;
    for(int i = 0; i < task_list.size(); i++){
        task = task_list[i];
        printf(
            "\tip%s:src%d -> ip%s:dst%d : type %d\n", 
            task->src_ip, task->src_world_rank, 
            task->dst_ip, task->dst_world_rank, 
            task->type
        );
    }
}

void check_inter_task_list(vector<round_task*> task_list){
    prof_task* send_task;
    prof_task* recv_task;
    for(int i = 0; i < task_list.size(); i++){
        send_task = task_list[i]->send_lc_task;
        recv_task = task_list[i]->recv_lc_task;
        printf(
            "\tround send: ip%s:src%d -> ip%s:dst%d : type %d\n", 
            send_task->src_ip, send_task->src_world_rank, 
            send_task->dst_ip, send_task->dst_world_rank, 
            send_task->type
        );
        printf("\tround recv: ip%s:src%d -> ip%s:dst%d : type %d\n", 
            recv_task->src_ip, recv_task->src_world_rank, 
            recv_task->dst_ip, recv_task->dst_world_rank, 
            recv_task->type
        );
    }
}

prof_task* pack_task(enum prof_task_type type, char* src_ip, 
                        char* dst_ip, int src_rank, int dst_rank, 
                            map<int, int> lookup){
    prof_task* task = (prof_task*)malloc(sizeof(struct prof_task_st));
    task->type = type;

    strcpy(task->src_ip, src_ip);
    strcpy(task->dst_ip, dst_ip);
    
    task->src_world_rank = src_rank;
    task->src_local_rank = lookup[src_rank];    
    task->dst_world_rank = dst_rank;
    task->dst_local_rank = lookup[dst_rank];
    
    return task;
}

int find_pos_vector(vector<char*> list, char* ip){
    int pos = 0;
    vector<char*>::iterator iter;
    for(iter = list.begin(); iter != list.end(); iter++){
        char* t_ip = *iter;
        if(strcmp(t_ip, ip) == 0)
            return pos;
        pos++;
    }
    return -1;
}

int get_duration_us(struct timeval start, struct timeval end){
    return (end.tv_sec * 1000000 + end.tv_usec) 
            - (start.tv_sec * 1000000 + start.tv_usec);
}

float get_bandwidth_GBps(int float_num, int duration_us){
    return (float_num * sizeof(float) / 
            (1024.0 * 1024 * 1024)) / (duration_us / 1000000.0);
}

void print_matrix(vector<vector<float>> matrix, int m, int n){
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            printf("%d -> %d: %f \n", i, j, matrix[i][j]);
        }
    }
}
