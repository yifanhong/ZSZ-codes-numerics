import numpy as np
import networkx as nx
from ldpc import BpOsdDecoder
from numba import njit, int8
import circuit_utils

def get_qubit_order(H):
    Q = H.T @ H
    Q.setdiag(0)
    G = nx.from_scipy_sparse_array(Q, create_using=nx.Graph())
    color_dict = nx.greedy_color(G, strategy='largest_first')
    num_colors = max(list(color_dict.values())) + 1
    qubit_coloring = []
    for i in range(num_colors):
        qubit_coloring.append([key for key, value in color_dict.items() if value == i])
    qubit_order = [x for color in qubit_coloring for x in color]
    return qubit_order


@njit(fastmath=True)
def noisy_greedy_decode(input_syndromes, qubit_order, check_inds, p, cycles):
    syndromes = input_syndromes.copy()
    n = len(check_inds)
    m = syndromes.shape[1]
    corr_syndrome = np.zeros(m, dtype=int8)
    correction = np.zeros(n, dtype=int8)
    for c in range(cycles):
        syndrome = syndromes[c] ^ corr_syndrome
        for site in qubit_order:
            red_syndrome = syndrome[check_inds[site]]
            dE = len(red_syndrome) - 2*np.sum(red_syndrome)
            rand = np.random.rand()
            condition = (dE<0 and rand>p) or (dE==0 and rand>0.5) or (dE>0 and rand<p)
            if condition:
                syndrome[check_inds[site]] = 1 - syndrome[check_inds[site]]
                corr_syndrome[check_inds[site]] = 1 - corr_syndrome[check_inds[site]]
                correction[site] = 1 - correction[site]
    return correction, corr_syndrome


def get_greedy_failures(code, pars, p1, p2, p_spam, cycles, iters, seed):
    H = code.hz
    logicals = code.lz
    m = H.shape[0]
    n = code.N

    bpd = BpOsdDecoder(H, error_rate = float(p2), max_iter = pars[0], bp_method = 'ms', osd_method = 'osd_cs', osd_order = pars[1])
    c = circuit_utils.generate_full_circuit(code, rounds=cycles, p1=p1, p2=p2, p_spam=p_spam, seed=seed)
    sampler = c.compile_sampler()
    outputs = sampler.sample(shots=iters)
    failures = 0
    for i in range(iters):
        output = outputs[i]
        final_syndrome = H @ output[-n:] % 2
        final_correction = bpd.decode(final_syndrome)
        final_state = output[-n:] ^ final_correction
        if (logicals@final_state%2).any():
            failures += 1
    return failures