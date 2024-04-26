/**
 * define relay control strategy
 * upper level of trans.h，provide interfaces for prims to set the controller behaviors
 * general to leaf / intermediate / root nodes
 */

#ifndef CONTROL_H
#define CONTROL_H

#include <unistd.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unordered_map>
#include <vector>

#include "cuda_runtime.h"
#include "trans.h"

/* Replay GPU policy */
typedef struct relayController_st{
    bool hasRecv; // whether has receive events
    bool hasLocal; // whether is active
    bool hasKernel; // whether launch reduce kernel
    bool hasSend; // whether has send events
}relayController;


/**
 * @brief obtain active gpus list, set local active status
 * 
 * @params: activeGPU   gpu id list
 * @params: numGPU      length of the list
 */
void setActiveStatus(int* activeGPU, int numGPU);


/**
 * @brief partial join the aggregation
 * 
 * @params: rank id to be active
 */
void updateActiveStatus(int myRank);


/**
 * @brief set the behavior of a rank in a transmission context
 * 
 * @params: role  ranks' role in current transmission context
 * @params: type  'r' for prim with aggregation
 * 
 * relay should based on the topology:
 * 1. hasRecv:  
 *      recursively judge whether all precedents are inactive or none, if so, set false; 
 *      leave nodes must be false
 * 2. hasLocal:
 * 3. hasKernel:
 *      set false if hasRecv is false
 *      not active and only one precedent, set false
 * 4. hasSend:
 *      not active and no precedent, set false
 *      root nodes must be false
 */
void setRelayController(relayController* controlSet, RankRoleInfo* role, int myRank, char type);


/**
 * @brief get the precedents set to receive flows
 * 
 * @params: roleRanks   ranks' role in current transmission context
 * @params: type        'r' for prim with aggregation
 */
set<int> getActiveRecvs(RankRoleInfo* roleRanks, int myRank, char type);

#endif
