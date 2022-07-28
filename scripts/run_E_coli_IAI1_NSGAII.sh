



python setup.py build install

CONFIG_NAME='template_indiv_NSGAII.yaml'
EXPERIMENT_NAME="indiv_growth_test"
MODEL_NAME=E_coli_IAI1
GENERTATION_IDX=0
HOTSTART_PARTICLES_REGEX=null
MAX_GENERATIONS=1
POPULATION_SIZE=5
DATA="M2_M3"
FILTER_MINGROWTH="0.01"
# FILTER_MINGROWTH="-0.0001"
FILTER_MAXGROWTH="35.0"


species_arr=(E_coli_IAI1 C_perfringens_S107 L_plantarum C_ramosum B_longum_subsp_longum L_paracasei S_salivarius)
species_arr=(E_coli_IAI1)

for species in "${species_arr[@]}"
do
    python -u run_bkcomms.py --config-name="$CONFIG_NAME" user=local \
    model_names.0="$species" \
    run_idx=0 \
    data="$DATA" \
    experiment_name="$EXPERIMENT_NAME" \
    algorithm.generation_idx="$GENERTATION_IDX" \
    algorithm.hotstart_particles_regex="$HOTSTART_PARTICLES_REGEX" \
    algorithm.max_generations="$MAX_GENERATIONS" \
    algorithm.population_size="$POPULATION_SIZE" \
    sampler=default_log_uniform_constr_biomass \
    n_processes=10 \
    algorithm.n_particles_batch=10 \
    parallel=True
done

# arr_variable=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)

arr_variable=(1)

for i in "${arr_variable[@]}"
do  
    x=$(($i - 1))
    y=$(($i + 1))

    GENERTATION_IDX=$i
    HOTSTART_PARTICLES_REGEX="\${user.wd}/output/$EXPERIMENT_NAME/\${model_names.0}/generation_$x/run_*/"
    MAX_GENERATIONS=$y

    python -u run_bkcomms.py --config-name="$CONFIG_NAME" user=local \
    model_names.0="$MODEL_NAME" \
    run_idx=0 \
    data="$DATA" \
    experiment_name="$EXPERIMENT_NAME" \
    algorithm.generation_idx="$GENERTATION_IDX" \
    algorithm.hotstart_particles_regex="$HOTSTART_PARTICLES_REGEX" \
    algorithm.max_generations="$MAX_GENERATIONS" \
    algorithm.population_size="$POPULATION_SIZE" \
    filter.min_growth.0="$FILTER_MINGROWTH" \
    filter.max_growth.0="$FILTER_MAXGROWTH" \
    n_processes=5 \
    algorithm.n_particles_batch=5 \
    algorithm.population_size="$POPULATION_SIZE"
done

