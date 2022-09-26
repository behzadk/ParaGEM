## Parameterising a monoculture with NSGAII
Here we are going to run the simulations to fit individual GSMMs to experimental growth curves. Make sure you have setup your `user.yaml` config file as described in the previous section.

**Overriding the default config file**

Here's an example of oveerriding the default config file. 

```zsh
python -u run_paragem.py --config-name='template_indiv_NSGAII' \
user='local' \ 
model_names.0='E_coli_IAI1'
algorithm='NSGAII'
algorithm.population_size=100
algorithm.max_generations=10
algorithm.n_particles_batch=10
n_processes=10
```

Let's talk through what this is doing and build it back up

**Setting the configuration file**

First of all we are setting the configuration file using:
```zsh
--config-name='template_indiv_NSGAII'
```
The default configuration files can be found in `/configs/`. `template_indiv_NSGAII` refers to one of the default configuration file. We use this config file to run parameterisation of monouclture communities, using monoculture experimental growth data.

**Setting the user**

As discussed in the previous section, the `user` configuration file is essential for simulating with comets. It defines the working directory and location of gurobi and comets libraries.

We also want to override the defaut `user=local`, so we can run using settings of our local machine.

**Setting models to be simulated**

Open the config file 
```zsh
/configs/template_indiv_NSGAII.yaml
``` 
You will see that the model names have not been set. You can also see that the `model_paths` depend upon the `model_names` being set. `${model_names.0}` will use interpolation to get the required value.

```yaml
# Names of models being simulated
model_names:
  - null

# Paths to each model in the community
model_paths:
- ${user.wd}/models/melanie_data_models/${model_names.0}.xml
```
Let's set the model name to an E. coli model:
```zsh
`model_names.0='E_coli_IAI1'
```
**Setting algorithm hyperparameters**
We are overriding these default values, setting the population size (`algorithm.population_size`), number of generations to run (`algorithm.max_generations`) and the number of particles in each batch of simulations (`algorithm.n_particles_batch`).

The following overrides relate to parameters of the algorithm:
```
algorithm.population_size=100
algorithm.max_generations=10
algorithm.n_particles_batch=10
```
The default values for NSGAII can be found here:
```
/configs/algorithm/NSGAII.yaml
```


**Bash script example**

Here, we iterate through a series of models and run the same parameterisation experiment for each of them. 

```bash
# Set parameters
CONFIG_NAME='template_indiv_NSGAII.yaml'
EXPERIMENT_NAME="indiv_growth"
GENERTATION_IDX=0
HOTSTART_PARTICLES_REGEX=null
MAX_GENERATIONS=5
POPULATION_SIZE=5
DATA="M2_M3"

# Models to parameterise
models_arr=(E_coli_IAI1 C_perfringens_S107 L_plantarum C_ramosum B_longum_subsp_longum L_paracasei S_salivarius)

# Iterate models 
for model_name in "${models_arr[@]}"
do
    # Run with defined parameters
    HYDRA_FULL_ERROR=1 python -u run_paragem.py --config-name="$CONFIG_NAME" user=local \
    model_names.0="$model_name" \
    run_idx=0 \
    data="$DATA" \
    experiment_name="$EXPERIMENT_NAME" \
    algorithm.generation_idx="$GENERTATION_IDX" \
    algorithm.hotstart_particles_regex="$HOTSTART_PARTICLES_REGEX" \
    algorithm.max_generations="$MAX_GENERATIONS" \
    algorithm.population_size="$POPULATION_SIZE" \
    sampler=default_log_uniform_constr_biomass \
    n_processes=5 \
    algorithm.n_particles_batch=5 \
    parallel=True
done
```