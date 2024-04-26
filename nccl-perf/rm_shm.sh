#!/bin/bash

for i in `ipcs -m | tail -n +4 | awk {'print $2'}`
do
    ipcrm -m $i;
    echo remove shm = $i
done