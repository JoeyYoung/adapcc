/*
    Detection module with 3 Tasks
    Task 1: determine the cpu affinity numa node
    Task 2: determine the gpu pair under the same pcie switch
    Task 3: determine the nic afffinity to the numa node
*/

#include "control.h"
#include "trans.h"
#include "time.h"
#include <arpa/inet.h>
#include <errno.h>
#include <sys/time.h>
#include <unistd.h>
#include <pthread.h>
#include <numa.h>

using namespace std;

#define PORT 0
#define BACKLOG 5

// external info
struct sockaddr_in sock_sin;
struct timeval start;
static MPIInfo* myMPIInfo;
// static int numTrans;
static socketFds* globalIpcFds;
static char* ipTable[MAX_DEVICES];

static char *my_itoa(int num, char *str){
    if(str == NULL){
        return NULL;
    }
    sprintf(str, "%d", num);
    return str;
}

static void globalBarrier(){
    char initBarrier = 'I';
    for(int i = 0; i < myMPIInfo->nRanks; i++){
        if(i == myMPIInfo->myRank) continue;
        socketIpcSend(globalIpcFds->sendFd[i], &initBarrier, 1);
    }
    for(int i = 0; i < myMPIInfo->nRanks; i++){
        if(i == myMPIInfo->myRank) continue;
        socketIpcRecv(globalIpcFds->recvFd[i], &initBarrier, 1);
    }
}

static void *connect_as_client(void *vargp){
    int iSocketFD = 0;
	iSocketFD = socket(AF_INET, SOCK_STREAM, 0);
    char *str = (char*)malloc(5 * 1024);
    memset(str, 'a', 5 * 1024);
	
    gettimeofday(&start, NULL);
	if(0 > connect(
        iSocketFD, (const struct sockaddr *)&sock_sin, sizeof(sock_sin)
    )){
		printf("Connection failed.\n");
	}else{
		send(iSocketFD, str, sizeof(str), 0);
	}
	close(iSocketFD);

    return NULL;
}

static int loopback() {
    int iSocketFD = 0;
	int new_fd = 0;
    unsigned int iLocalAddr = 0;
	struct sockaddr_in stLocalAddr = {0};
	struct sockaddr_in stRemoteAddr = {0};
	socklen_t stRemoteLen = 0;
    socklen_t stLocalLen = sizeof(sock_sin);
    struct timeval stop;
    int elapsed_time = 0;
 
	iSocketFD = socket(AF_INET, SOCK_STREAM, 0);
	if(0 > iSocketFD){
		printf("Failed to build the socket.\n");
		return 0;
	}	
 
	stLocalAddr.sin_family = AF_INET;
	stLocalAddr.sin_port = htons(PORT);
    inet_pton(AF_INET, "127.0.0.1", &iLocalAddr);
	stLocalAddr.sin_addr.s_addr = iLocalAddr;
 
	if(0 > bind(
        iSocketFD, (const struct sockaddr *)&stLocalAddr, sizeof(stLocalAddr)
    )){
		printf("Binding failed.\n");
		return 0;
	}
	if(0 > listen(iSocketFD, BACKLOG)){
		printf("Listening failed.\n");
		return 0;
	}
    if(getsockname(iSocketFD, (struct sockaddr *)&sock_sin, &stLocalLen) == -1){
        perror("getsockname");
        return 0;
    }

    pthread_t thread_client;
    pthread_create(&thread_client, NULL, connect_as_client, NULL);
    new_fd = accept(iSocketFD, (struct sockaddr *)&stRemoteAddr, &stRemoteLen);
	if(0 > new_fd){
		printf("Failed to receive.\n");
		return 0;
	}else{
        gettimeofday(&stop, NULL);
        elapsed_time = (stop.tv_sec * 1000000 + stop.tv_usec) 
                        - (start.tv_sec * 1000000 + start.tv_usec);
	}
    pthread_join(thread_client, NULL);
    close(iSocketFD);

    return elapsed_time;
}

static int run_loopback_n_times(int n){
    int avg_time = 0;
    for (int i=0; i<n; i++){
        avg_time += loopback();
    }
    return avg_time/n;
}

/*
    Detect whether two gpus are under the same pcie switch

    Input: device no. of the GPU pair for simutanious copy bandwidth test
    Output: have contention or not (true or false)
*/
static bool get_bandwidth(int ref_gpu, int tar_gpu){
    // create streams and malloc addresses on different devices
    cudaSetDevice(tar_gpu);
    cudaStream_t stream_tar;
    cudaStreamCreate(&stream_tar);
    
    unsigned int nElements = 100 * 1024 * 1024;
    unsigned int bytes = nElements * sizeof(float);
    float *h_a_tar = (float*)malloc(bytes);
    float *d_a_tar;
    cudaMalloc((float**)&d_a_tar, bytes);
    memset(h_a_tar, 0, bytes);
    
    cudaSetDevice(ref_gpu);
    int nStreams = 6;
    cudaStream_t streams[nStreams];
    float* h_a_refs[nStreams];
    float* d_a_refs[nStreams];
    for (int i = 0; i < nStreams; i++) {
        cudaStreamCreate(&streams[i]);
        h_a_refs[i] = (float*)malloc(bytes);
        cudaMalloc((float**)&d_a_refs[i], bytes);
        memset(h_a_refs[i], 0, bytes);
    }
  
    // timing parameters
    struct timeval start_org, stop_org, start_sim, stop_sim;
  
    // output device info and transfer size
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, tar_gpu);
  
    // record bandwidth with contention
    for (int i = 0; i < nStreams; i++) {
        cudaMemcpyAsync(
            h_a_refs[i], d_a_refs[i], bytes, cudaMemcpyDeviceToHost, streams[i]
        );
    }
    gettimeofday(&start_sim, NULL);
    cudaMemcpyAsync(h_a_tar, d_a_tar, bytes, cudaMemcpyDeviceToHost, stream_tar);
    cudaDeviceSynchronize();
    gettimeofday(&stop_sim, NULL);
  
    // record original bandwidth
    gettimeofday(&start_org, NULL);
    cudaMemcpyAsync(h_a_tar, d_a_tar, bytes, cudaMemcpyDeviceToHost, stream_tar);
    cudaDeviceSynchronize();
    gettimeofday(&stop_org, NULL);
    int time_with_contention = (stop_sim.tv_sec * 1000000 + stop_sim.tv_usec) 
                                    - (start_sim.tv_sec * 1000000 + start_sim.tv_usec);
    int time_without_contention = (stop_org.tv_sec * 1000000 + stop_org.tv_usec)
                                    - (start_org.tv_sec * 1000000 + start_org.tv_usec);

    // calculate and print the bandwidths
    double org_bandwidth = (bytes / (1024.0 * 1024 * 1024)) 
                                / (time_without_contention / 1000000.0);
    double sim_bandwidth = (bytes / (1024.0 * 1024 * 1024)) 
                                / (time_with_contention / 1000000.0);
    cudaFree(d_a_tar);
    free(h_a_tar);
    for (int i = 0; i < nStreams; i++){
        cudaFree(d_a_refs[i]);
        free(h_a_refs[i]);
    }
  
    if (sim_bandwidth < org_bandwidth * 0.9){
        return true;
    }
    return false;
}

static void inferTask(){
    if(numa_available() < 0){
        printf("System does not support NUMA API!\n");
        return;
    }

    unsigned int nElements = 500*1024*1024;
    unsigned int bytes = nElements * sizeof(float);

    // open streams for all GPUs and set timing variables
    int device_count;
    cudaGetDeviceCount(&device_count);
    cudaStream_t streams[device_count];
    float* h_a[device_count];
    float* d_a[device_count];
    for (int i = 0; i < device_count; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        h_a[i] = (float*)malloc(bytes);
        cudaMalloc((float**)&d_a[i], bytes);
        memset(h_a[i], 0, bytes);
    }

    // Find out rank, size
    int world_rank = myMPIInfo->myRank;
    int world_size = myMPIInfo->nRanks;
    int local_rank = myMPIInfo->localRank;

    if (world_size < 2) {
        fprintf(stderr, "World size must be greater than 1");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    struct timeval starts[device_count];
    struct timeval stop;

    // Task 1: each local rank 0 finds which CPU is closer to NIC
    int cpu0 = 0;
    int cpu1 = 0;
    int closest_cpu_to_nic = 0;

    if(local_rank == 0){
        numa_run_on_node(0);
        numa_set_preferred(0);
        cpu0 = run_loopback_n_times(10);
        numa_run_on_node_mask(numa_all_nodes_ptr);

        numa_run_on_node(1);
        numa_set_preferred(1);
        cpu1 = run_loopback_n_times(10);

        if(cpu0 > cpu1)
            closest_cpu_to_nic = 1;
        numa_run_on_node_mask(numa_all_nodes_ptr);
    }

    printf("[Rank %d] done infer task 1.\n", world_rank);
    globalBarrier();

    // Task 2: find which GPUs share a PCIe switch
    int gpu_pairs[device_count/2][2];
    if(local_rank == 0) {
        int pair_id = 0;
        for(int i = 0; i < device_count; i++){
            for(int j = i + 1; j < device_count; j++){
                bool share_pcie = get_bandwidth(i, j);
                if(share_pcie == true){
                    bool add = true;
                    for(int k = 0; k < device_count / 2; k++){
                        if(
                            gpu_pairs[k][0] == i || 
                            gpu_pairs[k][1] == i || 
                            gpu_pairs[k][0] == j || 
                            gpu_pairs[k][1] == j
                        ){
                            add = false;
                        }
                    }
                    if(add == true){
                        gpu_pairs[pair_id][0] = i;
                        gpu_pairs[pair_id][1] = j;
                        pair_id += 1;
                    }
                }
            }
        }
    }
    
    printf("[Rank %d] done infer task 2.\n", world_rank);
    globalBarrier();

    // Task 3: find which GPUs share a PCIe switch with NIC 
    int flag = 0;
    int closest_gpu_to_nic = 0;
    if(local_rank == 0){
        if(closest_cpu_to_nic == 0){
            numa_run_on_node(0);
            numa_set_preferred(0);
        }else{
            numa_run_on_node(1);
            numa_set_preferred(1);
        }

        flag = 1;
        MPI_Send(
        /* data         = */ &flag, 
        /* count        = */ 1, 
        /* datatype     = */ MPI_INT, 
        /* destination  = */ world_rank + 1, 
        /* tag          = */ 0, 
        /* communicator = */ MPI_COMM_WORLD);

        run_loopback_n_times(15);
        numa_run_on_node_mask(numa_all_nodes_ptr);
    }else if(local_rank == 1){
        if(closest_cpu_to_nic == 0){
            numa_run_on_node(0);
            numa_set_preferred(0);
        }else{
            numa_run_on_node(1);
            numa_set_preferred(1);
        }
        MPI_Recv(
        /* data         = */ &flag, 
        /* count        = */ 1, 
        /* datatype     = */ MPI_INT, 
        /* source       = */ world_rank - 1, 
        /* tag          = */ 0, 
        /* communicator = */ MPI_COMM_WORLD, 
        /* status       = */ MPI_STATUS_IGNORE);

        for(int i = 0; i < device_count; i++){
            gettimeofday(&starts[i], NULL);
            cudaMemcpyAsync(d_a[i], h_a[i], bytes, 
                cudaMemcpyHostToDevice, streams[i]);
        }
        cudaDeviceSynchronize();
        
        gettimeofday(&stop, NULL);

        int time_elapsed[device_count];
        double bandwidths[device_count];
        for (int i = 0; i < device_count; i++){
            time_elapsed[i] = (stop.tv_sec * 1000000 + stop.tv_usec) 
                                - (starts[i].tv_sec * 1000000 + starts[i].tv_usec);
            bandwidths[i] = (bytes / (1024.0 * 1024 * 1024)) 
                                / (time_elapsed[i] / 1000000.0);
            if(bandwidths[i] < bandwidths[closest_gpu_to_nic]){
                closest_gpu_to_nic = i;
            }
        }
        numa_run_on_node_mask(numa_all_nodes_ptr);
    }

    printf("[Rank %d] done infer task 3.\n", world_rank);
    globalBarrier();

    // Task 4: dump
    if(local_rank == 0){
        char* prefix = "./topology/topo_detect_";
        char* tail = ".xml";
        char name[20];
        my_itoa(world_rank, name);

        char filename[100] = {};
        strcat(filename, prefix);
        strcat(filename, name);
        strcat(filename, tail);
        FILE *output = fopen(filename, "w");
        fprintf(output,"<?xml version=\"1.0\" encoding=\"utf-8\"?>\n");
        fprintf(output, "<cpu>\n");
        fprintf(output, "<pcie>\n");
        
        if(closest_cpu_to_nic == 0){
            fprintf(output, "<nic/>\n");
            for(int i = 0; i < device_count / 2; i++){
                if(closest_gpu_to_nic == gpu_pairs[i][0] 
                    || closest_gpu_to_nic == gpu_pairs[i][1]){
                    fprintf(output, "<gpu id=\"%d\"/>\n", gpu_pairs[i][0]);
                    fprintf(output, "<gpu id=\"%d\"/>\n", gpu_pairs[i][1]);
                }
            }
            fprintf(output, "</pcie>\n");
            fprintf(output, "<pcie>\n");
            for (int i=0; i < device_count/2; i++) {
                if(closest_gpu_to_nic != gpu_pairs[i][0] 
                    && closest_gpu_to_nic != gpu_pairs[i][1]){
                    fprintf(output, "<gpu id=\"%d\"/>\n", gpu_pairs[i][0]);
                    fprintf(output, "<gpu id=\"%d\"/>\n", gpu_pairs[i][1]);
                }
            }
        }else{
            for(int i=0; i < device_count/2; i++){
                if(closest_gpu_to_nic != gpu_pairs[i][0] 
                    && closest_gpu_to_nic != gpu_pairs[i][1]){
                    fprintf(output, "<gpu id=\"%d\"/>\n", gpu_pairs[i][0]);
                    fprintf(output, "<gpu id=\"%d\"/>\n", gpu_pairs[i][1]);
                }
            }
            fprintf(output, "</pcie>\n");
            fprintf(output, "<pcie>\n");
            fprintf(output, "<nic/>\n");
            for(int i=0; i<device_count/2; i++){
                if(closest_gpu_to_nic == gpu_pairs[i][0] 
                    || closest_gpu_to_nic == gpu_pairs[i][1]){
                    fprintf(output, "<gpu id=\"%d\"/>\n", gpu_pairs[i][0]);
                    fprintf(output, "<gpu id=\"%d\"/>\n", gpu_pairs[i][1]);
                }
            }
        }
        fprintf(output, "</pcie>\n");
        fprintf(output, "</cpu>\n");
        fclose(output);

        printf("[Rank %d] save detected topology.\n", world_rank);
    }

    globalBarrier();
}


/* Detect API */
namespace detectContext{

void getMPIInfo(MPIInfo* info){
    myMPIInfo = info;
}

void bootstrapFromSketch(char* filename){
    struct timespec start = {0, 0};
    struct timespec end = {0, 0};
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &start);

    FILE* sketch = fopen(filename, "r");
    int rankCnt = 0;
    char buf[1024];
    while(fgets(buf, sizeof(buf), sketch) != NULL){
        buf[strlen(buf)-1]=0;
        ipTable[rankCnt] = (char*)malloc(sizeof(char)*strlen(buf));
        strcpy(ipTable[rankCnt], buf);
        rankCnt += 1;
    }
    fclose(sketch);

    socketFds** channels = buildIpcChannels(ipTable, 
                            myMPIInfo->myRank, myMPIInfo->nRanks);
    globalIpcFds = channels[0];
    inferTask();

    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &end);
    double duration = (end.tv_sec - start.tv_sec) * 1000 
                        + (end.tv_nsec - start.tv_nsec) / 1000000;
    printf("[Rank %d]detect time=%6.2f(ms)\n", myMPIInfo->myRank, duration);
}

} // end namespace
