from cvxpy import *
import numpy as np
import sys
from collections import defaultdict
import networkx as nx

def read_arcs(arcs, inflow_idxs, outflow_idxs, file_name, num_nodes):
  arc_id = 0
  with open(file_name) as f:
    for line in f:
      arc = line.split(" ")
      src = int(arc[0])
      dst = int(arc[1])
      assert src < num_nodes and dst < num_nodes
      arcs[(src, dst)] = arc_id
      inflow_idxs[dst].append(arc_id)
      outflow_idxs[src].append(arc_id)
      arc_id = arc_id + 1

def compute_cycles(arcs, cycles, num_nodes):
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(arcs.keys())
    nx_cycs = nx.simple_cycles(G)
    for cyc in nx_cycs:
        cyc_edges = []
        # cyc if of format node1, node2 ..
        first_node = cyc[0]
        for i in xrange(len(cyc)):
            if i == len(cyc) - 1:
                cyc_edges.append(arcs[(cyc[i], first_node)])
            else:
                cyc_edges.append(arcs[(cyc[i], cyc[i + 1])])

        # Add cyc_edges to cycles
        cycles.append(cyc_edges)

def populate_supply(supply, src_node):
  supply[:] = 0.0
  supply[src_node, :] = 1.0

# Nodes can only forward elements they have received in an earlier
# timestep or have supply of
def forwarding_rule(flows, supply, inflow_idxs, outflow_idxs):
  constraints = []
  flow = flows[0]
  # inflow_mat and outflow_mat start as single rows
  # They will become matrices of size num_nodes x num_partitions
  inflow_mat =  sum_entries(flow[inflow_idxs[0], :], axis=0)

  # Note: since we can send out the same data on many links
  # we take max_entries here. i.e. if we send it out on two link
  # we should get a value of 1 and not 2
  outflow_mat = max_entries(flow[outflow_idxs[0], :], axis=0)
  for a in xrange(1, supply.shape[0]):
    # For each incoming arc to 'a', we extract its partitions
    # Then we sum up these rows to get all incoming partitions to 'a'
    inparts_node = sum_entries(flow[inflow_idxs[a], :], axis=0)
    inflow_mat = vstack(inflow_mat, inparts_node)

    outparts_node = max_entries(flow[outflow_idxs[a], :], axis=0)
    outflow_mat = vstack(outflow_mat, outparts_node)
  
  constraints.append((supply + inflow_mat - outflow_mat) >= 0)

  # At each timestep assert that supply + inflow - outflow >= 0
  for f in xrange(1, len(flows)):
    flow = flows[f]
    inflow_mat_f =  sum_entries(flow[inflow_idxs[0], :], axis=0)
    outflow_mat_f = max_entries(flow[outflow_idxs[0], :], axis=0)
    for a in xrange(1, supply.shape[0]):
        inparts_node = sum_entries(flow[inflow_idxs[a], :], axis=0)
        inflow_mat_f = vstack(inflow_mat_f, inparts_node)

        outparts_node = max_entries(flow[outflow_idxs[a], :], axis=0)
        outflow_mat_f = vstack(outflow_mat_f, outparts_node)

    inflow_mat = inflow_mat + inflow_mat_f
    outflow_mat = outflow_mat + outflow_mat_f
  
    constraints.append((supply + inflow_mat - outflow_mat) >= 0)

  return constraints

# All nodes should have all partitions eventually
# We enforce this by constructing matrices of all partitions that are
# supplied and all partitions that are input to a node
#
# The sum of these two matrices should have 1 in all elements
def broadcast_rule(flows, supply, inflow_idxs):
  # Calculate inflow_mat for each timestep.
  # Add these matrices to setup broadcast constraint
  flow = flows[0]
  inflow_mat =  sum_entries(flow[inflow_idxs[0], :], axis=0)
  for a in xrange(1, supply.shape[0]):
    # For each incoming arc to 'a', we extract its partitions
    # Then we sum up these rows to get all incoming partitions to 'a'
    inparts_node = sum_entries(flow[inflow_idxs[a], :], axis=0)
    inflow_mat = vstack(inflow_mat, inparts_node)

  for f in xrange(1, len(flows)):
      flow = flows[f]
      inflow_mat_t = sum_entries(flow[inflow_idxs[0], :], axis=0)
      for a in xrange(1, supply.shape[0]):
        # For each incoming arc to 'a', we extract its partitions
        # Then we sum up these rows to get all incoming partitions to 'a'
        inparts_node = sum_entries(flow[inflow_idxs[a], :], axis=0)
        inflow_mat_t = vstack(inflow_mat_t, inparts_node)

      inflow_mat = inflow_mat + inflow_mat_t

  return ((supply + inflow_mat) == 1)

def one_partition_per_flow_timestep(flows):
    # every row in every flow should only have 1 non-zero entry
    constraints = []
    for flow in flows:
        constraints.append(sum_entries(flow, axis=1) <= 1)
    return constraints
    
def num_active_timesteps(flows):
    # Count the number of timestamps that have some data going through them
    # 
    # Since each flow entry can only be 1 or 0, this will be 1 if timestep is used
    # 0 otherwise.
    running_sum = max_entries(flows[0])
    for f in xrange(1, len(flows)):
        running_sum = running_sum + max_entries(flows[f])

    return running_sum

# Disallows empty timesteps in the middle. i.e. if we have 3 timesteps
# T1, T2, T3 this constraints that T2 cannot be empty while T3 is not empty
def no_empty_timesteps(flows):
    constraints = []
    for i in xrange(0, len(flows) - 1):
        lhs = (flows[i])
        rhs = (flows[i + 1])
        constraints.append(sum_entries((lhs - rhs)) > 0)
    return constraints

# Constraint that there are no cycles within each timestep
def nocycles(flows, cycles):
    constraints = []
    for flow in flows:
        for cyc in cycles:
            # This is a num_edges_in_cycle x num_partitions
            cyc_flow = flow[cyc, :]
            # Compute column sums and constraint the sum to be less than
            # number of edges
            constraints.append(sum_entries(cyc_flow, axis=0) <= (len(cyc) - 1))

    return constraints

def range_of_variables(flows):
    constraints = []
    for flow in flows:
       constraints.append(flow >= 0.0)
       constraints.append(flow <= 1.0)
    return constraints

def bias_towards_zero_one(flows):
    flow = flows[0]
    val = sum_entries(-1.0*abs(2*flow - 1.0) + 1)
    for f in xrange(1, len(flows)):
      # -1*abs(2*x - 1) + 1 is a function that is 1 at x=0.5 and 0 at x=0 and x=1
      # Hence this will bias our solution away from 0.5 and towards 0 and 1
      diff = sum_entries(-1.0*abs(2*flows[f] - 1.0) + 1.0)
      val = val + diff
     
    return val


def main():

  if len(sys.argv) != 4:
    print "Usage cvxpy_test.py <num_nodes> <num_partitions> <arc file>"
    sys.exit(0)

  num_nodes = int(sys.argv[1])
  num_partitions = int(sys.argv[2])
  arc_filename = sys.argv[3]

  source_node = 0
  num_timesteps = 3
  print "num_nodes ", num_nodes, " num_partitions ", num_partitions, " num_timesteps ", num_timesteps

  arcs = {} # Map from (src, dest) -> arc_id
  cycles = [] # List of cycles. Each cycle is of the form [arc_id1, arc_id2, ...]

  inflow_idxs = defaultdict(list)
  outflow_idxs = defaultdict(list)
  supply = np.zeros((num_nodes, num_partitions))

  read_arcs(arcs, inflow_idxs, outflow_idxs, arc_filename, num_nodes)
  compute_cycles(arcs, cycles, num_nodes)
  populate_supply(supply, source_node)

  flows = []
  for i in xrange(num_timesteps):
    flow = Bool(len(arcs), num_partitions)
    flows.append(flow)

  objective = Minimize(num_active_timesteps(flows))
  constraints = []
  constraints.extend(one_partition_per_flow_timestep(flows))
  constraints.append(broadcast_rule(flows, supply, inflow_idxs))
  constraints.extend(nocycles(flows, cycles))

  constraints.extend(forwarding_rule(flows, supply, inflow_idxs, outflow_idxs))
  constraints.extend(no_empty_timesteps(flows))
  #constraints.extend(range_of_variables(flows))

  problem = Problem(objective, constraints)
  problem.solve(verbose=False)
  print "status:", problem.status
  print "solver: ", problem.solver_stats.solver_name
  print "setup_time:", problem.solver_stats.setup_time
  print "solve_time:", problem.solver_stats.solve_time
  print "optimal value", problem.value

  if problem.status.startswith("optimal"):
      final_flows = []
      for flow in flows:
        final_flows.append(flow.value)

      for t in xrange(num_timesteps):
          print "Time step ", t
          print "Source -> Dest : Weights"
          for arc in sorted(arcs.keys()):
            arc_id = arcs[arc]
            print arc[0], "->", arc[1], ":", (np.round(final_flows[t][arc_id, :]))


if __name__ == "__main__":
  main()
