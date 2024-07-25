#!/bin/bash

read -p "Enter the file label (e.g. pc, laptop, test, hamilton): " label

#echo "Original C code output:"
#cd ./src/main/adaptchol/
#gcc adaptchol.c -o adaptchol -lm
#./adaptchol
#
#cd ../../../

echo "Extracting the generated target varience..."
Rscript -e "source('./src/main/adaptchol/adaptb'); write.table(sig, file = './data/target_variance.csv', sep = ',', col.names = FALSE, row.names = FALSE)"

echo "Scala Outputs (IC)"
for ((i = 1; i < 6; i++)); do
    sample_file="./data/scala_sample_IC_${label}_${i}"
    var_label="${label}_${i}" 
    sbt "runMain simple_run_IC ${sample_file} ${var_label} 100 ./data/target_variance.csv" 
done

echo "Scala Outputs (MD)"
for ((i = 1; i < 6; i++)); do
    sample_file="./data/scala_sample_MD_${label}_${i}"
    var_label="${label}_${i}" 
    sbt "runMain simple_run_MD ${sample_file} ${var_label} 100 ./data/target_variance.csv" 
done


echo "R Outputs (IC)"
for ((i = 1; i < 6; i++)); do
    sample_file="'./data/r_sample_IC_${label}_${i}'"
    var_label="'${label}_${i}'" 
    Rscript -e "source('./src/main/R/AM_in_R.R'); sample <- main(sigma=read_sigma(100, './data/target_variance.csv'), n=10000, thinrate=100, burnin=0, write_files = TRUE, mix = FALSE, sample_file=${sample_file}, var_labels=${var_label})"
done

echo "R Outputs (MD)"
for ((i = 1; i < 6; i++)); do
    sample_file="'./data/r_sample_MD_${label}_${i}'"
    var_label="'${label}_${i}'" 
    Rscript -e "source('./src/main/R/AM_in_R.R'); sample <- main(sigma=read_sigma(100, './data/target_variance.csv'), n=10000, thinrate=100, burnin=0, write_files = TRUE, mix = TRUE, sample_file=${sample_file}, var_labels=${var_label})"
done


echo "JAX Outputs (IC)"
for ((i = 1; i < 6; i++)); do
    sample_file="'./data/jax_sample_IC_${label}_${i}'"
    var_label="'${label}_${i}'"
    ~/CPUJAX/bin/python -c "import sys; sys.path.append('./src/main/Python-JAX/'); from AM_in_JAX import *; main(sigma=read_sigma(100, './data/target_variance.csv'), n=10000, thinrate=100, burnin=0, write_files=True, mix=False, sample_file=${sample_file}, use_64=False, var_labels=${var_label}, seed=${i})"
done

echo "JAX Outputs (MD)"
for ((i = 1; i < 6; i++)); do
    sample_file="'./data/jax_sample_MD_${label}_${i}'"
    var_label="'${label}_${i}'" 
    ~/CPUJAX/bin/python -c "import sys; sys.path.append('./src/main/Python-JAX/'); from AM_in_JAX import *; main(sigma=read_sigma(100, './data/target_variance.csv'), n=10000, thinrate=100, burnin=0, write_files=True, mix=True, sample_file = ${sample_file}, use_64=False, var_labels=${var_label})"
done
