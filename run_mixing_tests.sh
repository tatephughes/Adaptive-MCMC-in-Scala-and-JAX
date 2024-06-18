#!/bin/bash

echo "Mixing-style JAX output: "
~/CPUJAX/bin/python -c 'import sys; sys.path.append("./src/main/Python-JAX/"); from AM_in_JAX import *; mixing_test(read_sigma, mix=True,csvfile = "./data/so_factor_mixing.csv")'

echo "Non Mixing-style JAX output: "
~/CPUJAX/bin/python -c 'import sys; sys.path.append("./src/main/Python-JAX/"); from AM_in_JAX import *; mixing_test(read_sigma, mix=False,csvfile = "./data/so_factor_mixing.csv")'
