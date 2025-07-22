import numpy as np
from scipy.sparse import csr_matrix
from pymatching import Matching
import circuit_utils

def get_MWPM_failures(code, p1, p2, p_spam, iters, rounds, seed):
    H = code.hz.toarray()
    m, n = H.shape
    
    # Construct spacetime decoding graph
    H_dec = np.kron(np.eye(rounds+1,dtype=int), H)
    H_dec = np.concatenate((H_dec,np.zeros([m*(rounds+1),m*rounds],dtype=int)), axis=1)
    for j in range(m*rounds):
        H_dec[j,n*(rounds+1)+j] = 1
        H_dec[m+j,n*(rounds+1)+j] = 1
    H_dec = csr_matrix(H_dec)
    
    matching = Matching(H_dec, weights=np.log((1-p2)/p2))
    
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
            mwpm_output = np.reshape(matching.decode(np.ravel(syndromes))[:n*(rounds+1)], [rounds+1,n])
            correction = mwpm_output.sum(axis=0) % 2
            final_state = output[i,-n:] ^ correction
            if (code.lz@final_state%2).any():
                failures += 1
                
    return failures