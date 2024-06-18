#!/bin/bash

# Run R function
echo "R output: "
Rscript -e 'source("./src/main/R/AM_in_R.R"); compute_time_graph(read_sigma(100), csv_file="./data/R_compute_times_laptop_1.csv")'

echo "JAX output (32-bit): "
~/CPUJAX/bin/python -c 'import sys; sys.path.append("./src/main/Python-JAX/"); from AM_in_JAX import *; compute_time_graph(read_sigma(d=100), csv_file="./data/JAX_32_compute_times_laptop_1.csv", is_64_bit=False)'

echo "JAX output (64-bit): "
~/CPUJAX/bin/python -c 'import sys; sys.path.append("./src/main/Python-JAX/"); from AM_in_JAX import *; compute_time_graph(read_sigma(d=100), csv_file="./data/JAX_64_compute_times_laptop_1.csv", is_64_bit=True)'

echo "Scala output: "
sbt run
