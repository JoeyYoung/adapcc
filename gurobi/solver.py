import os, sys

# User LICENSE
import gurobipy as gp
from gurobipy import GRB

class Solver:
    def __init__(self):
        pass
    
    def optimize(self, prim, parallel_degree, transmission_size,
                    bandwidth_graph, latency_graph, strategy_file):
        num_gpus = len(bandwidth_graph)
        trans = [i for i in range(parallel_degree)]
        gpus = [i for i in range(num_gpus)]
        links_latency = {}
        links_bandwidth = {}
        for i in range(num_gpus):
            for j in range(num_gpus):
                links_latency[(gpus[i], gpus[j])] = latency_graph[gpus[i]][gpus[j]] # us 
                links_bandwidth[(gpus[i], gpus[j])] = bandwidth_graph[gpus[i]][gpus[j]] # GBps

        edges, _ = gp.multidict(links_latency)

        model = gp.Model('collective')
        r_mg = model.addVars(trans, gpus, vtype=GRB.BINARY, name='r_mg')
        s_m = model.addVars(trans, name='s_m')
        root_m = model.addVars(trans, vtype=GRB.INTEGER, name='root_m')

        model.addConstrs((r_mg.sum(m, '*') == 1 for m in trans), "one_tran_one_root")
        model.addConstrs((s_m[m] >= 0 for m in trans), "sm_positive")
        model.addConstr((s_m.sum('*') == transmission_size), "sm_total_size")

        model.addConstrs((root_m[m] == gp.quicksum(r_mg[m, g]*g for g in gpus)
                            for m in trans), "root_gid")

        x_ijf = model.addVars(edges, trans, gpus, vtype=GRB.BINARY, name="x_ijf")
        src_f = model.addVars(trans, gpus, vtype=GRB.INTEGER, name="src_f")
        dst_f = model.addVars(trans, gpus, vtype=GRB.INTEGER, name="dst_f")
        model.addConstrs((src_f[m, g] == g for m in trans for g in gpus), "cons_src")
        model.addConstrs(((r_mg[m, g] == 0) >> (dst_f[m, g] == root_m[m])
                            for m in trans for g in gpus), "cons_dst")

        model.addConstrs((((src_f[m, g] - v) == 0) >> (x_ijf.sum(v, '*', m, g) == 1) 
                            for m in trans for g in gpus for v in gpus), "routing_src")
        model.addConstrs((((dst_f[m, g] - v) == 0) >> (x_ijf.sum('*', v, m, g) == 1) 
                            for m in trans for g in gpus for v in gpus), "routing_dst")

        # if x > y, b = 1
        # model.addConstr(x >= y + eps - M * (1 - b), name="bigM_constr1")
        # model.addConstr(x <= y + M * b, name="bigM_constr2")
        eps = 0.0001
        M = 10 + eps
        b1 = model.addVars(trans, gpus, vtype=GRB.BINARY, name="b1")
        b2 = model.addVars(trans, gpus, vtype=GRB.BINARY, name="b2")
        b3 = model.addVars(trans, gpus, vtype=GRB.BINARY, name="b3")
        b4 = model.addVars(trans, gpus, vtype=GRB.BINARY, name="b4")
        b5 = model.addVars(trans, gpus, vtype=GRB.BINARY, name="b5")
        b6 = model.addVars(trans, gpus, vtype=GRB.BINARY, name="b6")
        b7 = model.addVars(trans, gpus, vtype=GRB.BINARY, name="b7")

        # if src >= v + eps, b1 = 1  OR
        # if v - eps >= src, b2 = 1
        for m in trans:
            for g in gpus:
                for v in gpus:
                    indicator_src = []
                    # if src >= v + eps: b1=1
                    model.addConstr(src_f[m, g] >= v + eps + eps - M * (1 - b1[m, g]))
                    model.addConstr(src_f[m, g] <= v + eps + M * b1[m, g])
                    # if v - eps >= src: b2=1
                    model.addConstr(v - eps >= src_f[m, g] + eps - M * (1 - b2[m, g]))
                    model.addConstr(v - eps <= src_f[m, g] + M * b2[m, g])

                    indicator_src.append(b1[m, g])
                    indicator_src.append(b2[m, g])
                    model.addConstr(b5[m, g] == gp.or_(indicator_src))
                    
                    indicator_dst = []
                    # if dst >= v + eps: b3=1
                    model.addConstr(dst_f[m, g] >= v + eps + eps - M * (1 - b3[m, g]))
                    model.addConstr(dst_f[m, g] <= v + eps + M * b3[m, g])
                    # if v - eps >= dst: b4=1
                    model.addConstr(v - eps >= dst_f[m, g] + eps - M * (1 - b4[m, g]))
                    model.addConstr(v - eps <= dst_f[m, g] + M * b4[m, g])

                    indicator_dst.append(b3[m, g])
                    indicator_dst.append(b4[m, g])
                    model.addConstr(b6[m, g] == gp.or_(indicator_dst))
                    
                    indicator = []
                    indicator.append(b5[m, g])
                    indicator.append(b6[m, g])
                    model.addConstr(b7[m, g] == gp.and_(indicator))
                    model.addGenConstrIndicator(b7[m, g], True, x_ijf.sum('*', v, m, g) \
                                                    == x_ijf.sum(v, '*', m, g))

        c_m = model.addVars(trans, vtype=GRB.INTEGER, name="c_m")
        beta_ij = model.addVars(edges, name="beta_ij")
        t_ijf = model.addVars(edges, trans, gpus, name="t_ijf")

        model.addConstrs((t_ijf[e[0], e[1], m, g] == (links_latency[e] + c_m[m] * beta_ij[e]) 
                                for e in edges for m in trans for g in gpus), "cons_tijf")

        a_mj = model.addVars(trans, gpus, vtype=GRB.BINARY, name="aggregation_control")
        h_jf = model.addVars(gpus, trans, gpus, name="h_ijf")
        L_jf = model.addVars(gpus, trans, gpus)
        temp_jf = model.addVars(gpus, trans, gpus)
        for j in gpus:
            for m in trans:
                for g in gpus:
                    for i in gpus:
                        model.addConstr(
                            temp_jf[j, m, g] >=
                            (x_ijf[i, j, m, g] * (h_jf[i, m, g] + t_ijf[i, j, m, g]))
                        )
                    model.addGenConstrIndicator(a_mj[m, j], False, 
                                            h_jf[j, m, g] == temp_jf[j, m, g])
                    
                    for i in gpus:
                        for m_ in trans:
                            for g_ in gpus:
                                model.addConstr(
                                    L_jf[j, m, g] >= 
                                    x_ijf[i, j, m_, g_] * (h_jf[i, m_, g_] + t_ijf[i, j, m_, g_])
                                )

                    model.addGenConstrIndicator(a_mj[m, j], True, 
                                                h_jf[j, m, g] == L_jf[j, m, g])

        N_mij = model.addVars(trans, edges, name="link_load")
        b8 = model.addVars(trans, edges, name="link_load_indicator")
        temp_nmij = model.addVars(trans, edges)

        for m in trans:
            for e in edges:
                indicator_load = []
                for g in gpus:
                    indicator_load.append(x_ijf[e[0], e[1], m, g])
                
                model.addConstr(b8[m, e[0], e[1]] == gp.or_(indicator_load))

                if prim == 'reduce':
                    # if a_im = 1: N_mij = I(...)
                    model.addGenConstrIndicator(
                        a_mj[m, e[0]], True, 
                        N_mij[m, e[0], e[1]] == b8[m, e[0], e[1]]
                    ) 
                    # if a_im = 0: N_mij = I(...)*(... + ...)
                    model.addConstr(temp_nmij[m, e[0], e[1]] == 
                        b8[m, e[0], e[1]] * (1 + gp.quicksum(
                            N_mij[m, k, e[0]]
                            for k in gpus
                        ))
                    )
                    model.addGenConstrIndicator(
                        a_mj[m, e[0]], False, 
                        N_mij[m, e[0], e[1]] == temp_nmij[m, e[0], e[1]]
                    )
                elif prim == 'broadcast':
                    model.addGenConstrIndicator(
                        a_mj[m, e[0]], True, 
                        N_mij[m, e[0], e[1]] == b8[m, e[0], e[1]]
                    )
                    model.addGenConstrIndicator(
                        a_mj[m, e[0]], False, 
                        N_mij[m, e[0], e[1]] == b8[m, e[0], e[1]]
                    )
                else:
                    model.addGenConstrIndicator(
                        a_mj[m, e[0]], False, 
                        N_mij[m, e[0], e[1]] == gp.quicksum(
                            x_ijf[e[0], e[1], m, k]
                            for k in gpus
                        )
                    ) 

        for e in edges:
            model.addConstr(beta_ij[e[0], e[1]] == gp.quicksum(
                N_mij[m, e[0], e[1]] * links_bandwidth[(e[0], e[1])]
                for m in trans
            ))

        T_max = model.addVar(name="T_max_flows") 
        for m in trans:
            for g in gpus: 
                for e in edges:
                    model.addConstr(T_max >= h_jf[e[1], m, g] * x_ijf[e[0], e[1], m, g])

        T_max = model.addVar(name="T_max_flows")
        T_bottlef = model.addVars(trans, gpus)
        num_chunks_m = model.addVars(trans, vtype=GRB.INTEGER)
        for m in trans:
            # for upper boud of number of chunks
            model.addConstr(num_chunks_m[m] * c_m[m] >= s_m[m])
            model.addConstr((num_chunks_m[m] - 1) * c_m[m] <= s_m[m])
            for g in gpus:         
                for e in edges:
                    model.addConstr(
                        T_bottlef[m, g] >= 
                        x_ijf[e[0], e[1], m, g] * (h_jf[e[1], m, g] - h_jf[e[0], m, g])
                    )
                    model.addConstr(
                        T_max >= h_jf[e[1], m, g] * x_ijf[e[0], e[1], m, g] 
                        + num_chunks_m[m] * T_bottlef[m, g]
                    )

        model.setObjective(T_max, GRB.MINIMIZE)
        model.update()

        return model.getAttr('x', c_m)
        