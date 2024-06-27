#!/bin/bash

echo "Scala output (IC): "
sbt "runMain simple_run_IC"

echo "Scala output (MD): "
sbt "runMain simple_run_MD"

# Run R function
echo "R output (IC): "
Rscript -e 'source("./src/main/R/AM_in_R.R"); sample <- main(d=100, n=10000, thinrate=100, burnin=0, write_files = TRUE, trace_file="./Figures/r_trace_basetest_IC.png", mix = FALSE, csv_file="./data/r_sample_basetest_IC.csv")'

# Run R function
echo "R output (MD): "
Rscript -e 'source("./src/main/R/AM_in_R.R"); sample <- main(d=100, n=10000, thinrate=100, burnin=0, write_files = TRUE, trace_file="./Figures/r_trace_basetest_MD.png", csv_file="./data/r_sample_basetest_MD.csv")'

echo "JAX output (IC): "
~/CPUJAX/bin/python -c 'import sys; sys.path.append("./src/main/Python-JAX/"); from AM_in_JAX import *; main(d=100, n=10000, thinrate=100, burnin=0, write_files=True, trace_file="./Figures/jax_trace_basetest_IC.png", mix=False, csv_file="./data/jax_sample_basetest_IC.csv")'

echo "JAX output (MD): "
~/CPUJAX/bin/python -c 'import sys; sys.path.append("./src/main/Python-JAX/"); from AM_in_JAX import *; main(d=100, n=10000, thinrate=100, burnin=0, write_files=True, trace_file="./Figures/jax_trace_basetest_MD.png", mix=True, csv_file="./data/jax_sample_basetest_MD.csv")'


