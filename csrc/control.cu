#include "control.h"
#include "trans.h"
#include "init.h"

static unordered_map<int, bool> activeStatus;

void setActiveStatus(int* activeGPU, int numGPU){
    // init all to inactive first
    for(int i = 0; i < MAX_DEVICES; i++){
        activeStatus[i] = false;
    }
    // set active control
    for(int i = 0; i < numGPU; i++){
        int gpuId = activeGPU[i];
        activeStatus[gpuId] = true;
    }
}

void updateActiveStatus(int myRank){
    activeStatus[myRank] = true;
}

bool isActive(int rank){
    return activeStatus[rank]; 
}

bool checkActiveRecv(RankRoleInfo* roleRanks, int myRank, char type){
    RankRoleInfo role = roleRanks[myRank];
    if(type == 'r'){
        // for reduce phase
        if(role.precedents.size() == 0){
            return false;
        }
        set<int>::iterator it;
        for(it=role.precedents.begin(); it!=role.precedents.end(); it++){
            if(activeStatus[*it] || checkActiveRecv(roleRanks, *it, 'r')){
                return true;
            }
        }
        return false;
    }else{
        printf("wrong type in function::checkActiveRecv\n");
        return false;
    }
}

bool checkKernelLaunch(RankRoleInfo* roleRanks, int myRank){
    RankRoleInfo role = roleRanks[myRank];
    bool hasRecv = checkActiveRecv(roleRanks, myRank, 'r');
    int count = 0;
    set<int>::iterator it;
    for(it=role.precedents.begin(); it!=role.precedents.end(); it++){
        int srcRank = *it;
        if(isActive(srcRank) || checkActiveRecv(roleRanks, srcRank, 'r'))
            count += 1;
    }
    if(!hasRecv || (count == 1 && !isActive(myRank)))
        return false;
    else
        return true;
}

bool checkActiveSend(RankRoleInfo* roleRanks, int myRank){
    RankRoleInfo role = roleRanks[myRank];
    if(!isActive(myRank) && !checkActiveRecv(roleRanks, myRank, 'r'))
        return false;
    if(role.subsequents.size()==0)
        return false;
    return true;
}

void setRelayController(relayController* control, RankRoleInfo* roleRanks, 
                            int myRank, char type){    
    bool hasRecv = checkActiveRecv(roleRanks, myRank, 'r');
    bool hasLocal = isActive(myRank);
    bool hasKernel = checkKernelLaunch(roleRanks, myRank);
    bool hasSend = checkActiveSend(roleRanks, myRank);

    printf("[Rank %d]Relay GPU Control: ", myRank);
    printf("(Recv: %d, Local: %d, Kernel: %d, Send: %d)\n", 
            hasRecv, hasLocal, hasKernel, hasSend);

    control->hasRecv = hasRecv;
    control->hasLocal = hasLocal;
    control->hasKernel = hasKernel;
    control->hasSend = hasSend;
}

set<int> getActiveRecvs(RankRoleInfo* roleRanks, int myRank, char type){
    set<int> recvs;
    RankRoleInfo role = roleRanks[myRank];

    set<int>::iterator it;
    for(it=role.precedents.begin(); it!=role.precedents.end(); it++){
        int srcRank = *it;
        if(isActive(srcRank) || checkActiveRecv(roleRanks, srcRank, 'r')){
            recvs.insert(srcRank);
        }
    }
    return recvs;
}
