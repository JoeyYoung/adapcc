/*
    dynamic lib - each worker process has owned static space
    exposed APIs for the python client:
        initThreads —— Init MPI, receive topo/strategy for management
        allreduce etc. —— pack work elem, set gpu behavior, push into the work queue
        profling / detection
*/

#include <unistd.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <time.h>

#include "control.h"
#include "trans.h"

extern "C"{
    void initThreads(enum Primitive type, char* filename, int sockPort){
        initMPIInfo* initInfo = getInitRanksInfo();
        int myRank = initInfo->myRank;
        int nRanks = initInfo->nRanks;
        int localRank = initInfo->localRank;

        // if init needed
        if(!initInfo->MPI_INIT){
            printf("MPI Init... \n");
            int mpi_support;
            // enable parallel transmissions with multiple streams
            MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &mpi_support);
            if (mpi_support != MPI_THREAD_MULTIPLE){
                printf("MPI_THREAD_MULTIPLE thread support required\n");
            }
                
            MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
            MPI_Comm_size(MPI_COMM_WORLD, &nRanks);
            uint64_t hostHashs[nRanks];
            getHostsAndLocalRank(&localRank, myRank, nRanks, hostHashs);
            setInitRanksInfo(myRank, nRanks, localRank, true);
        }
        
        attachSocketPort(sockPort);
        
        MPIInfo* info = (struct MPIInfo_st*)malloc(sizeof(struct MPIInfo_st));
        info->myRank = myRank;
        info->nRanks = nRanks;
        info->localRank = localRank;
        cudaSetDevice(localRank);
        printf("[Rank %d]World Size %d, Local Size %d.\n", myRank, nRanks, localRank);

        // persistent threads waiting for terminate signals
        // multiple requests are pused into the work queues
        if(type == ALLREDUCE){
            printf("[Rank %d]Allreduce threads starts.\n", myRank);
            allreduceContext::getMPIInfo(info);
            allreduceContext::bootstrapFromXMl(filename);
            allreduceContext::clear();
        }else if(type == REDUCE){
            printf("[Rank %d]Reduce threads starts.\n", myRank);
            reduceContext::getMPIInfo(info);
            reduceContext::bootstrapFromXMl(filename);
            reduceContext::clear();
        }else if(type == BOARDCAST){
            printf("[Rank %d]Boardcast threads starts.\n", myRank);
            boardcastContext::getMPIInfo(info);
            boardcastContext::bootstrapFromXMl(filename);
            boardcastContext::clear();
        }

        // non persistent threasd, quit when complete
        else if(type == DETECT){
            printf("[Rank %d]Detecting starts.\n", myRank);
            detectContext::getMPIInfo(info);
            detectContext::bootstrapFromSketch(filename);
        }else if(type == PROFILE){
            printf("[Rank %d]Profiling starts.\n", myRank);
            profileContext::getMPIInfo(info);
            profileContext::bootstrapFromLogicGraph(filename);
        }else{
            printf("non ready primitive");
        }

        if(type != DETECT && type != PROFILE){
            MPI_Finalize();
            printf("[Rank %d] work threads quit.\n", info->myRank);
            setInitRanksInfo(0, 0, 0, false);  
        }
    }
    
    void exitThreads(enum Primitive type){
        if(type == ALLREDUCE)
            allreduceContext::terminate();
        else if(type == REDUCE)
            reduceContext::terminate();
        else if(type == BOARDCAST)
            boardcastContext::terminate();
        else if(type == DETECT || type == PROFILE);
        else
            printf("non ready primitive");
    }

    void allreduce(void* tensor, int size, int chunkBytes, 
                        int* activeGPU, int numGPU){
        // receive cuda tensor, pack into work element
        float* tensor_dev = (float*) tensor;
        workElement* elem = (struct workElement_st*)malloc(
                                sizeof(struct workElement_st));
        elem->tensorBuf = tensor_dev;
        elem->size = size;
        elem->chunkBytes = chunkBytes;

        // prepare active status for control module
        setActiveStatus(activeGPU, numGPU);
        float** bufs = allreduceContext::processWorkElem(elem);
        int numTrans = allreduceContext::getNumTrans();
        int tranSize = size/numTrans;
        for(int i = 0; i < numTrans; i++){
            int offset = i * tranSize;
            cudaMemcpy(elem->tensorBuf + offset, bufs[i], 
                tranSize*sizeof(float), cudaMemcpyDeviceToDevice);
        }

        // save for free，disattach memory and pointer
        free(bufs);
        free(elem);
    }

    void reduce(void* tensor, int size, int chunkBytes, 
                    int* activeGPU, int numGPU){
        float* tensor_dev = (float*) tensor;
        workElement* elem = (struct workElement_st*)malloc(sizeof(struct workElement_st));
        elem->tensorBuf = tensor_dev;
        elem->size = size;
        elem->chunkBytes = chunkBytes;

        setActiveStatus(activeGPU, numGPU);
        float** bufs = reduceContext::processWorkElem(elem);
        int numTrans = reduceContext::getNumTrans();
        int tranSize = size/numTrans;
        for(int i = 0; i < numTrans; i++){
            int offset = i * tranSize;
            cudaMemcpy(elem->tensorBuf + offset, bufs[i], 
                tranSize*sizeof(float), cudaMemcpyDeviceToDevice);
        }   
        free(bufs);
        free(elem);
    }

    void boardcast(void* tensor, int size, int chunkBytes, 
                        int* activeGPU, int numGPU){
        float* tensor_dev = (float*) tensor;
        workElement* elem = (struct workElement_st*)malloc(sizeof(struct workElement_st));
        elem->tensorBuf = tensor_dev;
        elem->size = size;
        elem->chunkBytes = chunkBytes;

        // setActiveStatus(activeGPU, numGPU);
        float** bufs = boardcastContext::processWorkElem(elem);
        int numTrans = boardcastContext::getNumTrans();
        int tranSize = size/numTrans;
        for(int i = 0; i < numTrans; i++){
            int offset = i * tranSize;
            cudaMemcpy(elem->tensorBuf + offset, bufs[i], 
                tranSize*sizeof(float), cudaMemcpyDeviceToDevice);
        }
        free(bufs);
        free(elem);
    }
    
    void updateActive(int myRank){
        updateActiveStatus(myRank);
    }
}
