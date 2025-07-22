import numpy as np
from scipy.sparse import csr_matrix
from ldpc import BpOsdDecoder
from ldpc.bplsd_decoder import BpLsdDecoder

def get_min_logical_weight(code, p, pars, iters, Ptype):
    if Ptype == 0:
        H = code.hx
        logicals = code.lx
    else:
        H = code.hz
        logicals = code.lz
    n = code.N
    bp_iters = pars[0]
    osd_order = pars[1]
    # Weigh syndrome error probabilities based on weight

    bposd = BpOsdDecoder(H, error_rate=p, max_iter = bp_iters, bp_method = 'minimum_sum', osd_method = 'osd_cs', osd_order = osd_order)
    bplsd = BpLsdDecoder(H, error_rate=p, max_iter = bp_iters, bp_method = 'minimum_sum', osd_method = 'lsd_cs', bits_per_step=5, osd_order = 0)
    min_weight = n
    errors = (np.random.rand(iters,n)<p).astype(int)
    for i in range(iters):
        state = errors[i]
        syndrome = H @ state % 2
        correction = bposd.decode(syndrome)
        final_state = state ^ correction
        if (logicals@final_state%2).any():
            weight = np.sum(final_state)
            if weight > 0 and weight < min_weight:
                min_weight = weight
    return min_weight