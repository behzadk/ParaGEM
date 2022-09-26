# Running individual growth fitting
Here we are going to run the simulations to fit individual GSMMs to experimental growth curves.

Make sure you have setup your `user.yaml` config file as described in the previous section.

## Overriding the default config file
The config file for fitting individual growth curves can be found here. We use this config as a template for when you want to fit an individual model, to a monoculture growth curve. Open the file in your text editor
```zsh
/configs/template_indiv_NSGAII.yaml
```

### Setting model names
We can see that there is no value for `model_names`:
```yaml
model_names:
  - null
```
Let's set that with an override by setting the first item in the yaml list to an E_.coli model, `model_names.0='E_coli_IAI1'`.

You can see that by setting `model_names.0`, `model_paths` will use interpolation to point to the model file with the same name.
```yaml
model_paths:
- ${user.wd}/models/melanie_data_models/${model_names.0}.xml
```
Interpolates to:
```yaml
model_paths:
- ${user.wd}/models/melanie_data_models/E_coli_IAI1.xml
```
### Setting user
We also want to override the defaut `user=local`, so we can run using settings of our local maching.

### Putting it together
Putting this together looks something like this:
```zsh
python -u run_bkcomms.py --config-name='template_indiv_NSGAII' \
user='local' \ 
model_names.0='E_coli_IAI1'
```

We can add further overrides to customise your run. For example, if we want to run NSGAII with a population size of 100, running 5 processes in parallel, for 10 generations

```zsh
python -u run_bkcomms.py --config-name='template_indiv_NSGAII' \
user='local' \ 
model_names.0='E_coli_IAI1'
algorithm='NSGAII'
algorithm.population_size=100
algorithm.max_generations=10
algorithm.n_particles_batch=10
n_processes=10
```

