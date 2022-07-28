
which python

python setup.py build install

CONFIG_NAME='M2_multi_NSGAII.yaml'
EXPERIMENT_NAME="m2_multi_2"
GENERTATION_IDX=0
HOTSTART_PARTICLES_REGEX=null
MAX_GENERATIONS=1
POPULATION_SIZE=5
DATA="M2_multi"


HYDRA_FULL_ERROR=1  python -u run_bkcomms.py --config-name="$CONFIG_NAME" user=local \
run_idx=0 \
data="$DATA" \
experiment_name="$EXPERIMENT_NAME" \
algorithm.generation_idx="$GENERTATION_IDX" \
algorithm.hotstart_particles_regex="$HOTSTART_PARTICLES_REGEX" \
algorithm.max_generations="$MAX_GENERATIONS" \
algorithm.population_size="$POPULATION_SIZE" \
algorithm.crossover_type=specieswise \
algorithm.mutate_type=resample_prior \
n_processes=5 \
algorithm.n_particles_batch=5 \
parallel=True
