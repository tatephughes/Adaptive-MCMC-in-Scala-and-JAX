#!/bin/bash

#echo "Scala output (IC): "
#sbt "runMain simple_run_IC"

#echo "Scala output (MD): "
#sbt "runMain simple_run_MD"

#echo "R output (IC): "
#Rscript -e 'source("./src/main/R/AM_in_R.R"); sample <- main(d=100, n=10000, thinrate=100, burnin=0, write_files = TRUE, trace_file="./Figures/r_trace_basetest_IC.png", mix = FALSE, sample_file="./data/r_sample_basetest_IC")'

#echo "R output (MD): "
#Rscript -e 'source("./src/main/R/AM_in_R.R"); sample <- main(d=100, n=10000, thinrate=100, burnin=0, write_files = TRUE, trace_file="./Figures/r_trace_basetest_MD.png", mix = TRUE, sample_file="./data/r_sample_basetest_MD")'

echo "JAX output (IC): "
~/CPUJAX/bin/python -c 'import sys; sys.path.append("./src/main/Python-JAX/"); from AM_in_JAX import *; main(d=100, n=10000, thinrate=100, burnin=0, write_files=True, trace_file="./Figures/jax_trace_basetest_IC.png", mix=False, sample_file="./data/jax_sample_basetest_IC", use_64=False)'

echo "JAX output (MD): "
~/CPUJAX/bin/python -c 'import sys; sys.path.append("./src/main/Python-JAX/"); from AM_in_JAX import *; main(d=100, n=10000, thinrate=100, burnin=0, write_files=True, trace_file="./Figures/jax_trace_basetest_MD.png", mix=True, sample_file="./data/jax_sample_basetest_MD", use_64=False)'


