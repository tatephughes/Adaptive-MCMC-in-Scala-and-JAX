#!/bin/bash

# Run R function
echo "R output: "
Rscript -e 'source("./src/main/R/AM_in_R.R"); compute_time_graph(read_sigma(50), csv_file="./data/R_compute_times_laptop_1_d50.csv")'

echo "JAX output (32-bit): "
~/CPUJAX/bin/python -c 'import sys; sys.path.append("./src/main/Python-JAX/"); from AM_in_JAX import *; compute_time_graph(read_sigma(d=50), csv_file="./data/JAX_32_compute_times_laptop_1_d50.csv", is_64_bit=False)'

echo "JAX output (64-bit): "
~/CPUJAX/bin/python -c 'import sys; sys.path.append("./src/main/Python-JAX/"); from AM_in_JAX import *; compute_time_graph(read_sigma(d=50), csv_file="./data/JAX_64_compute_times_laptop_1_d50.csv", is_64_bit=True)'

echo "Scala output: "
sbt "runMain complexity_run"
