/home/xyzhao/openmpi/bin/mpirun \
	-np 2\
	-H 10.28.1.31:1,10.28.1.32:1 \
	-x LD_LIBRARY_PATH \
	-x PATH \
	./flow_dedicate
