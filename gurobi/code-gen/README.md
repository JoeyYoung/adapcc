# CVXPy based OPT problem, fiddlelink

To run this first install cvxpy and numpy using `pip` with a command like
`pip install numpy cvxpy`

To run the program for say 4 GPUs, 3 partitions, with all-to-all connectivity you can run
 
python cvxpy-broadcast-multi-round.py 4 3 all-to-all.txt
