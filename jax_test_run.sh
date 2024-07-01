#!/bin/bash

echo "JAX output (IC): "
~/CPUJAX/bin/python -c 'import sys; sys.path.append("./src/main/Python-JAX/"); from AM_in_JAX import *; main(d=100, n=10000, thinrate=100, burnin=0, write_files=True, trace_file="./Figures/jax_trace_basetest_IC.png", mix=False, sample_file="./data/jax_sample_basetest_IC", use_64=False)'

echo "JAX output (MD): "
~/CPUJAX/bin/python -c 'import sys; sys.path.append("./src/main/Python-JAX/"); from AM_in_JAX import *; main(d=100, n=10000, thinrate=100, burnin=0, write_files=True, trace_file="./Figures/jax_trace_basetest_MD.png", mix=True, sample_file="./data/jax_sample_basetest_MD", use_64=False)'


