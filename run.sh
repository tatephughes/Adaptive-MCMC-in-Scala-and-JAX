#!/bin/bash

echo "Scala Outputs (IC)"
for ((i = 1; i < 6; i++)); do
    sbt "runMain simple_run_IC ./Figures/scala_trace_basetest_IC_$i.png ./data/scala_sample_basetest_IC_$i" 
done

echo "Scala Outputs (MD)"
for ((i = 1; i < 6; i++)); do
    sbt "runMain simple_run_MD ./Figures/scala_trace_basetest_MD_$i.png ./data/scala_sample_basetest_MD_$i" 
done


echo "R Outputs (IC)"
for ((i = 1; i < 6; i++)); do
    Rscript -e 'source("./src/main/R/AM_in_R.R"); sample <- main(d=100, n=10000, thinrate=100, burnin=0, write_files = TRUE, trace_file="./Figures/r_trace_basetest_IC_$i.png", mix = FALSE, sample_file="./data/r_sample_basetest_IC_$i")'
done

echo "R Outputs (MD)"
for ((i = 1; i < 6; i++)); do
    Rscript -e 'source("./src/main/R/AM_in_R.R"); sample <- main(d=100, n=10000, thinrate=100, burnin=0, write_files = TRUE, trace_file="./Figures/r_trace_basetest_MD_$i.png", mix = TRUE, sample_file="./data/r_sample_basetest_MD_$i")'
done


echo "JAX Outputs (IC)"
for ((i = 1; i < 6; i++)); do
    ~/CPUJAX/bin/python -c 'import sys; sys.path.append("./src/main/Python-JAX/"); from AM_in_JAX import *; main(d=100, n=10000, thinrate=100, burnin=0, write_files=True, trace_file="./Figures/jax_trace_basetest_IC_$i.png", mix=False, sample_file="./data/jax_sample_basetest_IC_$i", use_64=False)'
done

echo "JAX Outputs (MD)"
for ((i = 1; i < 6; i++)); do
    ~/CPUJAX/bin/python -c 'import sys; sys.path.append("./src/main/Python-JAX/"); from AM_in_JAX import *; main(d=100, n=10000, thinrate=100, burnin=0, write_files=True, trace_file="./Figures/jax_trace_basetest_MD_$i.png", mix=True, sample_file="./data/jax_sample_basetest_MD_$i", use_64=False)'
done
