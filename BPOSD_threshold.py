import numpy as np
from scipy.sparse import csr_matrix
from ldpc import BpOsdDecoder
from ldpc.bplsd_decoder import BpLsdDecoder
import circuit_utils

def get_BPOSD_failures(code, par, p1, p2, p_spam, iters, rounds, seed):
    # par = [bp_iters, osd_sweeps]
    H = code.hz.toarray()
    m, n = H.shape
    
    # Construct spacetime decoding graph
    H_dec = np.kron(np.eye(rounds+1,dtype=int), H)
    H_dec = np.concatenate((H_dec,np.zeros([m*(rounds+1),m*rounds],dtype=int)), axis=1)
    for j in range(m*rounds):
        H_dec[j,n*(rounds+1)+j] = 1
        H_dec[m+j,n*(rounds+1)+j] = 1
    H_dec = csr_matrix(H_dec)
    
    bpd = BpOsdDecoder(H_dec, error_rate = float(5*p2), max_iter = par[0], bp_method = 'ms', osd_method = 'osd_cs', osd_order = par[1])
    # lsd = BpLsdDecoder(H_dec, error_channel = list(channel_probs), max_iter = par[0], bp_method = 'ms', osd_method = 'lsd_cs', osd_order = 0, schedule = 'serial')
    
    c = circuit_utils.generate_full_circuit(code, rounds=rounds, p1=p1, p2=p2, p_spam=p_spam, seed=seed)
    sampler = c.compile_sampler()
    failures = 0
    outer_reps = iters//256    # Stim samples a minimum of 256 shots at a time
    remainder = iters % 256
    for j in range(outer_reps+1):
        num_shots = 256
        if j == outer_reps:
            num_shots = remainder
        output = sampler.sample(shots=num_shots)
        for i in range(num_shots):
            syndromes = np.zeros([rounds+1,m], dtype=int)
            syndromes[:rounds] = output[i,:-n].reshape([rounds,m])
            syndromes[-1] = H @ output[i,-n:] % 2
            syndromes[1:] = syndromes[1:] ^ syndromes[:-1]   # Difference syndrome
            bpd_output = np.reshape(bpd.decode(np.ravel(syndromes))[:n*(rounds+1)], [rounds+1,n])
            correction = bpd_output.sum(axis=0) % 2
            final_state = output[i,-n:] ^ correction
            if (code.lz@final_state%2).any():
                failures += 1
                
    return failures