#include "mpi.h"
#include "cuda_runtime.h"
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>

int main(int argc, char *argv[]){
    int my_rank, n_rank;
    int mpi_support;
    MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &mpi_support);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_rank);
    printf("my_rank: %d, n_rank: %d \n", my_rank, n_rank);
    
    cudaSetDevice(0);
    int num_float = 10;
    float* send_buffer;
    float* recv_buffer;
    cudaMalloc(&send_buffer, sizeof(float)*num_float);
    cudaMalloc(&recv_buffer, sizeof(float)*num_float);

    MPI_Request send_request, recv_request;
    MPI_Status send_status, recv_status;

    if(my_rank == 0){
        // send and recv
        int peer_rank = 1;
        MPI_Isend(send_buffer, num_float, MPI_FLOAT, peer_rank, 0, MPI_COMM_WORLD, &send_request);
        MPI_Irecv(recv_buffer, num_float, MPI_FLOAT, peer_rank, 0, MPI_COMM_WORLD, &recv_request);
        
        printf("[Rank%d] before wait \n", my_rank);
        MPI_Wait(&send_request, &send_status);
        MPI_Wait(&recv_request, &recv_status);
        printf("[Rank%d] after wait \n", my_rank);
    }else{
        // recv and send
        int peer_rank = 0;
        MPI_Irecv(recv_buffer, num_float, MPI_FLOAT, peer_rank, 0, MPI_COMM_WORLD, &recv_request);
        printf("[Rank%d] before recv wait \n", my_rank);
        MPI_Wait(&recv_request, &recv_status);
        
        MPI_Isend(send_buffer, num_float, MPI_FLOAT, peer_rank, 0, MPI_COMM_WORLD, &send_request);
        printf("[Rank%d] before send wait \n", my_rank);
        MPI_Wait(&send_request, &send_status);
        printf("[Rank%d] after send wait \n", my_rank);
    }

    printf("[Rank%d] Completed. \n", my_rank);

    sleep(10);
    cudaFree(send_buffer);
    cudaFree(recv_buffer);
    MPI_Finalize();
    return 0;
}
