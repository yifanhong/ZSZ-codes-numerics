import numpy as np
from edge_coloring import edge_color_bipartite
import stim
from networkx import relabel_nodes
from networkx.algorithms import bipartite

def generate_synd_circuit(H, checks, stab_type, p1, p2, seed):
    m, n = H.shape
    tanner_graph = bipartite.from_biadjacency_matrix(H)
    mapping = {i: checks[i] for i in range(m)}
    mapping.update({i:i-m for i in range(m,n+m)})
    tanner_graph = relabel_nodes(tanner_graph, mapping)
    coloring = edge_color_bipartite(tanner_graph)
    if seed != 0:
        rng = np.random.default_rng(seed=seed)
        rng.shuffle(coloring, axis=0)

    c = stim.Circuit()

    if stab_type:
        c.append("H", checks)
        c.append("DEPOLARIZE1", checks, p1)

    for r in coloring:
        data_qbts = set(np.arange(H.shape[1]))
        for g in r:
            data_qbts.remove(g[0])
            targets = g[::-1] if stab_type else g
            c.append("CX", targets)
            c.append("DEPOLARIZE2", targets, p2)
        c.append("DEPOLARIZE1", data_qbts, p1)

    if stab_type:
        c.append("H", checks)
        c.append("DEPOLARIZE1", checks, p1)
    return c

# Only tracks Z syndrome measurements
def generate_full_circuit(code, rounds, p1, p2, p_spam, seed):
    mx, n = code.hx.shape
    mz = code.hz.shape[0]
    data_qubits = range(n)
    x_checks = range(n, n+mx)
    z_checks = range(n+mx, n+mx+mz)
    c = stim.Circuit()
    z_synd_circuit = generate_synd_circuit(code.hz, z_checks, 0, p1, p2, seed)
    x_synd_circuit = generate_synd_circuit(code.hx, x_checks, 1, p1, p2, seed)
    # ancilla initialization errors
    c.append("X_ERROR", z_checks, p_spam)
    c.append("X_ERROR", x_checks, p_spam)

    # syndrome extraction rounds
    c_se = stim.Circuit()
    # Z syndrome measurement
    c_se += z_synd_circuit
    c_se.append("X_ERROR", z_checks, p_spam)
    c_se.append("MR", z_checks)
    c_se.append("X_ERROR", z_checks, p_spam)
    # X syndrome measurement
    c_se += x_synd_circuit
    c_se.append("R", x_checks)
    c_se.append("X_ERROR", x_checks, p_spam)

    c += c_se * rounds

    # Final transversal measurement
    c.append("X_ERROR", data_qubits, p_spam)
    c.append("MR", data_qubits)
    return c