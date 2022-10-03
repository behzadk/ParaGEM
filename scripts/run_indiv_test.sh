#!/bin/bash

which python

cd ..

python setup.py build install


HYDRA_FULL_ERROR=1 python -u run_tests.py  --config-name=template_indiv_NSGAII user=hpc \
model_names.0=E_coli_IAI1 \
run_idx=0 \
data="M3" \
experiment_name=test_indiv \
algorithm.generation_idx=0 \
algorithm.hotstart_particles_regex=null \
algorithm.max_generations=1 \
algorithm.population_size=50 \
filter.min_growth.0='0.01' \
filter.max_growth.0='50.0' \
n_processes=10 \
algorithm.n_particles_batch=10 \


