/*
    Two phases profling
    Phase 1: each node profiles intra-connections, nvlinks
    Phase 2: node-node inter-connections

    Phase 2 executes in a round manner, within each round every probes executed in parallel,
    for N nodes, there are N-1 rounds in totoal
*/

#include "control.h"
#include "trans.h"
#include "task.h"
#include <map>

static MPIInfo* myMPIInfo;
static char* ipTable[MAX_DEVICES];
static vector<char*> ip_list;
static map<char*, vector<int>> ip_ranks_map;

static socketFds* globalIpcFds;
static vector<prof_task*> intra_task_list;
static vector<round_task*> inter_task_list;

static vector<vector<float>> latency_matrix(
    MAX_DEVICES, vector<float>(MAX_DEVICES)
);
static vector<vector<float>> bandwidth_matrix(
    MAX_DEVICES, vector<float>(MAX_DEVICES)
);

static void globalBarrier(){
    char init_barrier = 'P';
    for(int i = 0; i < myMPIInfo->nRanks; i++){
        if(i == myMPIInfo->myRank) continue;
        socketIpcSend(globalIpcFds->sendFd[i], &init_barrier, 1);
    }
    for(int i = 0; i < myMPIInfo->nRanks; i++){
        if(i == myMPIInfo->myRank) continue;
        socketIpcRecv(globalIpcFds->recvFd[i], &init_barrier, 1);
    }
}

static void roundBarrier(vector<int> ranks){
    char round_barrier = 'r';
    for(int i = 0; i < ranks.size(); i++){
        if(ranks[i] == myMPIInfo->myRank) continue;
        socketIpcSend(globalIpcFds->sendFd[ranks[i]], &round_barrier, 1);
    }
    for(int i = 0; i < ranks.size(); i++){
        if(ranks[i] == myMPIInfo->myRank) continue;
        socketIpcRecv(globalIpcFds->recvFd[ranks[i]], &round_barrier, 1);
    }
}

// fill in ip - ranks info, pack tasks, consensus
static void read_logic_graph(char* filename){
    XMLDocument xml;
    xml.LoadFile(filename);
    XMLElement *graph = xml.RootElement();
    
    // fill in info
    map<int, int> local_rank_mapping;
    XMLElement *server = graph->FirstChildElement("server");
    while(server != NULL){
        int sid = atoi(server->Attribute("id"));
        char* ip = (char*)server->Attribute("ip");
        ip_list.push_back(ip);

        int local_rank = 0;
        vector<int> ip_ranks;
        ip_ranks.clear();
        XMLElement *pcie_group = server->FirstChildElement("nic");
        while(pcie_group != NULL){
            XMLElement *gpu = pcie_group->FirstChildElement("gpu");
            while(gpu != NULL){                
                int gid = atoi(gpu->Attribute("id"));
                ipTable[gid] = ip;
                local_rank_mapping[gid] = local_rank;

                gpu = gpu->NextSiblingElement("gpu");
                local_rank += 1;
                ip_ranks.push_back(gid);
                ip_ranks.push_back(gid);
            }
            pcie_group = pcie_group->NextSiblingElement("nic");
        }
                
        ip_ranks_map[ip] = ip_ranks;
        server = server->NextSiblingElement("server");
    }
    
    // profiling only executed by local rank 0
    if(myMPIInfo->localRank != 0)   return;

    // pack intra tasks
    char local_ip[256];
    strcpy(local_ip, ipTable[myMPIInfo->myRank]);
    intra_task_list.clear();
    for(int i = 0; i < myMPIInfo->nRanks; i++){
        for(int j = i+1; j < myMPIInfo->nRanks; j++){
            if(strcmp(local_ip, ipTable[i])!= 0 || 
                strcmp(local_ip, ipTable[j]) != 0) continue;

            prof_task* lc_task = pack_task(
                LATENCY, ipTable[i], ipTable[j], 
                i, j, local_rank_mapping
            );
            prof_task* bw_task = pack_task(
                BANDWIDTH, ipTable[i], ipTable[j], 
                i, j, local_rank_mapping
            );

            intra_task_list.push_back(lc_task);
            intra_task_list.push_back(bw_task);
        }
    }

    // pack inter tasks by rounds
    inter_task_list.clear();
    int prof_rounds = ip_list.size() - 1;
    int local_pos = find_pos_vector(ip_list, local_ip);
    for(int i = 1; i <= prof_rounds; i++){
        round_task* task = (round_task*)malloc(sizeof(struct round_task_st));

        // send probes to the next node
        char* dst_ip = ip_list[(local_pos + i) % ip_list.size()];
        prof_task* lc_task = pack_task(
            LATENCY, local_ip, dst_ip, 
            myMPIInfo->myRank, ip_ranks_map[dst_ip][0], 
            local_rank_mapping
        );
        prof_task* bw_task = pack_task(
            BANDWIDTH, local_ip, dst_ip, 
            myMPIInfo->myRank, ip_ranks_map[dst_ip][0], 
            local_rank_mapping
        );
        task->send_lc_task = lc_task;
        task->send_bw_task = bw_task;

        // recv probes from the previous node
        char* src_ip = ip_list[
            (ip_list.size() + local_pos - i) % ip_list.size()
        ];
        lc_task = pack_task(
            LATENCY, src_ip, local_ip, 
            ip_ranks_map[src_ip][0], 
            myMPIInfo->myRank, local_rank_mapping
        );
        bw_task = pack_task(
            BANDWIDTH, src_ip, local_ip, 
            ip_ranks_map[src_ip][0], 
            myMPIInfo->myRank, local_rank_mapping
        );
        task->recv_lc_task = lc_task;
        task->recv_bw_task = bw_task;
        
        inter_task_list.push_back(task);
    }

    printf("[Rank %d] Done packing tasks.\n", myMPIInfo->myRank);
}

static void execute_intra_task(prof_task* task){
    int bw_float_num = 20 * 1024 * 1024;
    int lc_float_num = 64;
    int transfer_num = 1;
    int float_num = 0;

    if(task->type == 0) float_num = lc_float_num;
    else if(task->type == 1) float_num = bw_float_num;
    else printf("wrong task type.\n");

    cudaSetDevice(task->src_local_rank);
    float* sendDevBuffer;
    cudaMalloc(&sendDevBuffer, sizeof(float)*bw_float_num);

    cudaSetDevice(task->dst_local_rank);
    float* recvDevBuffer;
    cudaMalloc(&recvDevBuffer, sizeof(float)*bw_float_num);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    struct timeval start, end;
    float sum_duration = 0.0;
    float sum_bandwidth = 0.0;
    float avg_duration = 0.0;
    float avg_bandwidth = 0.0;
    for(int i = 0; i < transfer_num; i++){
        gettimeofday(&start, NULL);   
        cudaDeviceEnablePeerAccess(task->dst_local_rank, 0);
        cudaDeviceEnablePeerAccess(task->src_local_rank, 0);
        cudaMemcpyPeerAsync(
            recvDevBuffer, task->dst_local_rank, 
            sendDevBuffer, task->src_local_rank, 
            sizeof(float) * float_num, stream
        );
        cudaStreamSynchronize(stream);
        gettimeofday(&end, NULL);

        int duration_us = get_duration_us(start, end);
        sum_duration += duration_us;
        float bandwidth = get_bandwidth_GBps(float_num, duration_us);
        sum_bandwidth += bandwidth;
    }
    avg_duration = sum_duration / transfer_num;
    avg_bandwidth = sum_bandwidth / transfer_num;
    
    if(task->type == 0){
        // store latency
        latency_matrix[task->src_world_rank][task->dst_world_rank] = avg_duration;
        latency_matrix[task->dst_world_rank][task->src_world_rank] = avg_duration;
    }else{
        // store bandwidth
        bandwidth_matrix[task->src_world_rank][task->dst_world_rank] = avg_bandwidth;
        bandwidth_matrix[task->dst_world_rank][task->src_world_rank] = avg_bandwidth;
    }
}

static void execute_inter_task(round_task* task){
    int bw_float_num = 1024 * 1024;
    int lc_float_num = 64;

    // do bw profiling
    prof_task_st* send_bw_task = task->send_bw_task;
    prof_task_st* recv_bw_task = task->recv_bw_task;

    cudaSetDevice(send_bw_task->src_local_rank);
    float* sendDevBuffer;
    cudaMalloc(&sendDevBuffer, sizeof(float)*bw_float_num);

    cudaSetDevice(recv_bw_task->dst_local_rank);
    float* recvDevBuffer;
    cudaMalloc(&recvDevBuffer, sizeof(float)*bw_float_num);

    struct timeval start, end;
    int duration_us;
    float bandwidth;

    MPI_Request send_request, recv_request;
    MPI_Status send_status, recv_status;

    if(myMPIInfo->myRank == 0){
        // only world rank 0 first send then recv
        MPI_Isend(
            sendDevBuffer, bw_float_num, MPI_FLOAT, 
            send_bw_task->dst_world_rank, 0, 
            MPI_COMM_WORLD, &send_request
        );
        MPI_Irecv(
            recvDevBuffer, bw_float_num, MPI_FLOAT, 
            recv_bw_task->src_world_rank, 0, 
            MPI_COMM_WORLD, &recv_request
        );

        gettimeofday(&start, NULL);
        MPI_Wait(&send_request, &send_status);
        gettimeofday(&end, NULL);
        duration_us = get_duration_us(start, end);
        bandwidth = get_bandwidth_GBps(bw_float_num, duration_us);

        MPI_Wait(&recv_request, &recv_status);
    }else{
        MPI_Irecv(
            recvDevBuffer, bw_float_num, MPI_FLOAT, 
            recv_bw_task->src_world_rank, 0, 
            MPI_COMM_WORLD, &recv_request
        );
        MPI_Wait(&recv_request, &recv_status);
        MPI_Isend(
            sendDevBuffer, bw_float_num, MPI_FLOAT, 
            send_bw_task->dst_world_rank, 0, 
            MPI_COMM_WORLD, &send_request
        );
        gettimeofday(&start, NULL);
        MPI_Wait(&send_request, &send_status);
        gettimeofday(&end, NULL);
        duration_us = get_duration_us(start, end);
        bandwidth = get_bandwidth_GBps(bw_float_num, duration_us);
    }

    bandwidth_matrix[send_bw_task->src_world_rank][send_bw_task->dst_world_rank] = bandwidth;
    // split in/out port, only record sender info, then gather

    // do lc profiling
    prof_task_st* send_lc_task = task->send_lc_task;
    prof_task_st* recv_lc_task = task->recv_lc_task;
    
    cudaSetDevice(send_lc_task->src_local_rank);
    cudaMalloc(&sendDevBuffer, sizeof(float)*lc_float_num);

    cudaSetDevice(recv_lc_task->dst_local_rank);
    cudaMalloc(&recvDevBuffer, sizeof(float)*lc_float_num);

    if(myMPIInfo->myRank == 0){
        // only world rank 0 first send then recv
        MPI_Isend(
            sendDevBuffer, lc_float_num, MPI_FLOAT, 
            send_lc_task->dst_world_rank, 0, 
            MPI_COMM_WORLD, &send_request
        );
        MPI_Irecv(
            recvDevBuffer, lc_float_num, MPI_FLOAT, 
            recv_lc_task->src_world_rank, 0, 
            MPI_COMM_WORLD, &recv_request
        );

        gettimeofday(&start, NULL);
        MPI_Wait(&send_request, &send_status);
        gettimeofday(&end, NULL);
        duration_us = get_duration_us(start, end);

        MPI_Wait(&recv_request, &recv_status);
    }else{
        MPI_Irecv(
            recvDevBuffer, lc_float_num, MPI_FLOAT, 
            recv_lc_task->src_world_rank, 0, 
            MPI_COMM_WORLD, &recv_request
        );
        MPI_Wait(&recv_request, &recv_status);

        MPI_Isend(
            sendDevBuffer, lc_float_num, MPI_FLOAT, 
            send_lc_task->dst_world_rank, 0, 
            MPI_COMM_WORLD, &send_request
        );
        gettimeofday(&start, NULL);
        MPI_Wait(&send_request, &send_status);
        gettimeofday(&end, NULL);
        duration_us = get_duration_us(start, end);
    }
    
    latency_matrix[send_lc_task->src_world_rank][send_lc_task->dst_world_rank] = duration_us;
}

static void dump_profile_result(char* filename){
    // only local rank 0 dump latency/bandwidth matrix: src_rank, dst_rank, type, value
    FILE* file = fopen(filename, "w");
    if(file == NULL){
        printf("Dump profile file failed.\n");
        return;
    }

    for(int i = 0; i < myMPIInfo->nRanks; i++){
        for(int j = 0; j < myMPIInfo->nRanks; j++){
            fprintf(
                file, "%d, %d, %d, %0.3f\n", i, j, 
                BANDWIDTH, bandwidth_matrix[i][j]
            );
            fprintf(
                file, "%d, %d, %d, %0.3f\n", i, j, 
                LATENCY, latency_matrix[i][j]
            );
        }
    }
    fclose(file);
}


/* Profile API */
namespace profileContext{

void getMPIInfo(MPIInfo* info){
    myMPIInfo = info;
}

// must ensure the bootstrap file is given or generated
void bootstrapFromLogicGraph(char* filename){
    struct timespec start = {0, 0};
    struct timespec end = {0, 0};
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &start);

    read_logic_graph(filename);
    if(myMPIInfo->localRank == 0){
        printf("check intra task list:\n");
        check_intra_task_list(intra_task_list);
        printf("check inter task list:\n");
        check_inter_task_list(inter_task_list);
    }

    socketFds** channels = buildIpcChannels(ipTable, 
                            myMPIInfo->myRank, myMPIInfo->nRanks);
    globalIpcFds = channels[0];

    // only local rank 0 has intra tasks to execute
    if(myMPIInfo->localRank == 0){
        for(int t = 0; t < intra_task_list.size(); t++){
            prof_task* task = intra_task_list[t];
            execute_intra_task(task);
        }
    }
    globalBarrier();
    printf("[Rank %d] Done intra tasks, start inter tasks.\n", myMPIInfo->myRank);

    // get local rank 0 list
    vector<int> round_ranks;
    for(int i = 0; i < ip_list.size(); i++){
        char* ip = ip_list[i];
        round_ranks.push_back(ip_ranks_map[ip][0]);
    }
    
    // only local rank 0 has inter tasks to execute
    if(myMPIInfo->localRank == 0){
        for(int t = 0; t < inter_task_list.size(); t++){
            round_task* task = inter_task_list[t];
            execute_inter_task(task);

            roundBarrier(round_ranks);
            printf("[Rank %d] Done inter round %d.\n", myMPIInfo->myRank, t);
        }
    }
    
    globalBarrier();

    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &end);
    double duration = (end.tv_sec - start.tv_sec) * 1000 
                        + (end.tv_nsec - start.tv_nsec) / 1000000;
    printf("[Rank %d]profiling time=%6.2f(ms)\n", myMPIInfo->myRank, duration);

    if(myMPIInfo->localRank == 0){
        char dump_file[100];
        sprintf(dump_file, "./topology/topo_profile_%d", 
                    myMPIInfo->myRank);
        dump_profile_result(dump_file);
    }
}

} // end namespace
