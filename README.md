# ZSZ-codes-numerics
Python code for the numerical simulations in arXiv:2507.xxxx. The main workhorse files are the Jupyter notebooks ending in ".ipynb": "BPOSD simulation.ipynb" is for d rounds of syndrome extraction, and "greedy simulation.ipynb" is for passive decoding with local majority vote.

Python version: 3.10

Packages used: [LDPC](https://github.com/quantumgizmos/ldpc), [bposd](https://github.com/quantumgizmos/bp_osd), [NetworkX](https://networkx.org/), [Stim](https://github.com/quantumlib/Stim), [PyMatching](https://github.com/oscarhiggott/PyMatching), [Numba](https://numba.pydata.org/), [multiprocess](https://github.com/uqfoundation/multiprocess) (for multicore parallelization)

"circuit_utils.py" is adapted from [this repository](https://github.com/noahberthusen/adaptive_qec), and "edge_coloring.py" is adapted from [this repository](https://github.com/qldpc/exp_ldpc).
