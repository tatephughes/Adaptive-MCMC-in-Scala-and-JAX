#!/bin/bash

echo "Scala output: "
sbt "runMain simple_run"

# Run R function
echo "R output: "
Rscript -e 'source("./src/main/R/AM_in_R.R"); sample <- main(d=100, n=1000, thinrate=1000, burnin=0, write_files = TRUE, trace_file="./Figures/r_trace_basetest.png", csv_file="./data/r_sample.csv")'

echo "JAX output: "
~/CPUJAX/bin/python -c 'import sys; sys.path.append("./src/main/Python-JAX/"); from AM_in_JAX import *; main(d=100, n=1000, thinrate=1000, burnin=0, write_files=True, trace_file="./Figures/jax_trace_basetest.png", csv_file="./data/jax_sample_IC.csv")'

echo "JAX output (mixing): "
~/CPUJAX/bin/python -c 'import sys; sys.path.append("./src/main/Python-JAX/"); from AM_in_JAX import *; main(d=100, n=1000, thinrate=1000, burnin=0, write_files=True, trace_file="./Figures/jax_trace_basetest.png", mix=True, csv_file="./data/jax_sample_MD.csv")'


