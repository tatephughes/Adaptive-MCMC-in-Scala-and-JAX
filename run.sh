#!/bin/bash

echo "Scala output: "
sbt "runMain simple_run"

# Run R function
echo "R output: "
Rscript -e 'source("./src/main/R/AM_in_R.R"); sample <- main()'

echo "JAX output: "
~/CPUJAX/bin/python -c 'import sys; sys.path.append("./src/main/Python-JAX/"); from AM_in_JAX import *; main()'


