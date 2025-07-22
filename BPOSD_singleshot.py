import numpy as np
from ldpc import BpOsdDecoder
from ldpc.bplsd_decoder import BpLsdDecoder
from scipy.sparse import csr_matrix, hstack, vstack, identity
import circuit_utils


def bulk_BPOSD_decode(syndromes, H, M, p, pars, cycles):
    m, n = H.shape
    H_dec1 = hstack([H,identity(m,dtype=int,format='csr')], format='csr')
    H_dec2 = hstack([csr_matrix((M.shape[0],n),dtype=int),M], format='csr')
    H_dec = vstack([H_dec1,H_dec2], format='csr')
    bpd = BpLsdDecoder(H_dec, error_rate = float(5*p), max_iter = pars[0], bp_method = 'ms', osd_method = 'lsd_cs', osd_order = pars[1], schedule = 'serial')
    bulk_correction = np.zeros(n, dtype=int)
    corr_syndrome = np.zeros(m, dtype=int)
    for c in range(cycles):
        syndrome = syndromes[c] ^ corr_syndrome
        correction = bpd.decode(np.concatenate((syndrome,np.zeros(M.shape[0],dtype=int))))[:n]
        bulk_correction = bulk_correction ^ correction
        corr_syndrome = corr_syndrome ^ (H@correction%2)
    return bulk_correction, corr_syndrome


def get_BPOSD_failures(code, Mz, pars, noise_pars, cycles, iters, seed):
    H = code.hz
    logicals = code.lz
    m = H.shape[0]
    n = code.N
    p1, p2, p_spam = noise_pars

    # bpd = BpOsdDecoder(H, error_rate = float(p2), max_iter = pars[0], bp_method = 'ms', osd_method = 'osd_cs', osd_order = pars[1])
    bpd = BpLsdDecoder(H, error_rate = float(5*p2), max_iter = pars[0], bp_method = 'ms', osd_method = 'lsd_cs', osd_order = pars[1], schedule = 'serial')
    c = circuit_utils.generate_full_circuit(code, rounds=cycles, p1=p1, p2=p2, p_spam=p_spam, seed=seed)
    sampler = c.compile_sampler()
    failures = 0
    outer_reps = iters//256    # Stim samples a minimum of 256 shots at a time
    remainder = iters % 256
    for j in range(outer_reps+1):
        num_shots = 256
        if j == outer_reps:
            num_shots = remainder
        outputs = sampler.sample(shots=num_shots)
        for i in range(num_shots):
            output = outputs[i]
            syndromes = np.zeros([cycles+1,m], dtype=int)
            syndromes[:cycles] = output[:-n].reshape([cycles,m])
            syndromes[-1] = H @ output[-n:] % 2
            bulk_correction, corr_syndrome = bulk_BPOSD_decode(syndromes, H, Mz, p2, pars, cycles)
            final_correction = bpd.decode(corr_syndrome ^ syndromes[-1])
            final_state = output[-n:] ^ bulk_correction ^ final_correction
            if (logicals@final_state%2).any():
                failures += 1
    return failures