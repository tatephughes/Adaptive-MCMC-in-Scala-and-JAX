#!/bin/bash

# Run R function
echo "R output (MD): "
Rscript -e 'source("./src/main/R/AM_in_R.R"); compute_time_graph(read_sigma(100), csv_file="./data/R_compute_times_laptop_1_MD.csv")'

echo "R output (ID): "
Rscript -e 'source("./src/main/R/AM_in_R.R"); compute_time_graph(read_sigma(100), csv_file="./data/R_compute_times_laptop_1_IC.csv")'

echo "JAX output (IC, 32-bit): "
~/CPUJAX/bin/python -c 'import sys; sys.path.append("./src/main/Python-JAX/"); from AM_in_JAX import *; compute_time_graph(read_sigma(d=100), mix=False, csv_file="./data/JAX_32_compute_times_laptop_1_IC.csv", is_64_bit=False)'

echo "JAX output (IC, 64-bit): "
~/CPUJAX/bin/python -c 'import sys; sys.path.append("./src/main/Python-JAX/"); from AM_in_JAX import *; compute_time_graph(read_sigma(d=100), mix=True, csv_file="./data/JAX_64_compute_times_laptop_1_IC.csv", is_64_bit=True)'

echo "JAX output (MD, 32-bit): "
~/CPUJAX/bin/python -c 'import sys; sys.path.append("./src/main/Python-JAX/"); from AM_in_JAX import *; compute_time_graph(read_sigma(d=100), mix=False, csv_file="./data/JAX_32_compute_times_laptop_1_MD.csv", is_64_bit=False)'

echo "JAX output (MD, 64-bit): "
~/CPUJAX/bin/python -c 'import sys; sys.path.append("./src/main/Python-JAX/"); from AM_in_JAX import *; compute_time_graph(read_sigma(d=100), mix=True, csv_file="./data/JAX_64_compute_times_laptop_1_MD.csv", is_64_bit=True)'

echo "Scala output (IC): "
#sbt "runMain complexity_run_IC"

echo "Scala output (MD): "
sbt "runMain complexity_run_MD"

