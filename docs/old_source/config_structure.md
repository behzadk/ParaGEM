## Config Structure

Below shows the config folder structure. Have a look inside each folder to see the available options. Each .yaml file refers to a differnt config that can be used.

```
configs
│
└───algorithm
└───community
└───data
└───distance
└───filter
└───sampler
└───user 
└───visualisation
```

`configs/algorithm`: Defines the parameterisation algorithm that will be run. Settings in here will include things like number of particles to simulate, batch sizes and algorithm hyper parameters

`configs/community`: Defines the community to be simulated/parameterised. Here the community members, their model paths and 

`configs/distance` The distance function we use to compare the experimental data and the simulation data

`configs/filter` Filter step sometimes used by an algorithm. Normally used to filter out particles before they have been simualted. For example using a min/max growth rate filter can be useful to avoid wasting compute time on undesrirable solutions.

`configs/sampler` The prior distributions to sample from

`configs/user` Settings for the user working directory, comets and gurobi directories.

`configs/visualisation` Visualisaton pipeline

