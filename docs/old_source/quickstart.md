## Sample and simulate example
In this tutorial we run one of the example configs

The following command will use the default config file, which you can find at `/ParaGem/configs/Ecoli_sample_simulate.yaml`.

It samples parameters from a prior distribution, and performs a dFBA simulation. It performs no parameterisation!

```zsh
python -u run_paragem.py --config-name='Ecoli_sample_simulate' \
user='local' \
model_names.0='E_coli_IAI1'\
algorithm.population_size=5 \
algorithm.n_particles_batch=5 \
n_processes=5
```


We override the population size using `algorithm.population_size`, meaning we sample 
from the prior 5 times and perform 5 simulations

`n_processes` defines the number of simulations to be run in parallel. We recommend you set to less than the number of cores if running locally.

`algorithm.n_particles_batch` defines the number of samples we perform simultaneously. This is normally limited by memory availability.

