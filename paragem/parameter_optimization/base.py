from paragem.utils import logger_wraps
import numpy as np
import copy
from loguru import logger
import gc
import psutil
import os
import paragem
from typing import List
import cobra
import glob
from paragem import utils

class ParameterOptimization:
    def init_particles(self, n_particles, assign_model=True, set_media=True):
        """ Initialise new particles
        
        Initialise new particles with an optional assignment of population models
        and setting of default media

        Args:
            n_particles : number of particles to initialise
            assign_model : bool, whether to assign models to each population
            set_media : bool, whether to set default media
        """
        particles = np.zeros(shape=n_particles, dtype=object)

        for i in range(n_particles):
            # Make independent copy of base community
            comm = copy.deepcopy(self.base_community)
            
            # Assign population models
            if assign_model:
                for idx, pop in enumerate(comm.populations):
                    pop.model = self.models[i][idx]

            comm.sample_parameters_from_prior()

            particles[i] = comm
            particles[i].distance = []
            particles[i].sol = {}
            particles[i].t = []

            for media in self.sim_media_names:
                particles[i].sol[media] = []

            if set_media:
                particles[i].set_media_conditions(self.sim_media_names[0])

        return particles

    def crossover_parameterwise(self, batch_particles: List[paragem.Community], population):
        """ Crossover parameterwise for genetic algorithms

        Performs elementwise crossover between population members. Crossover includes
        initial conditions and any other available parameters

        Args:
            batch_particles : particles to which new parameters will be assigned
            population : population of particles to perform crossover with
        """

        for p_batch_idx in range(len(batch_particles)):
            # Randomly choose two particles from the population
            male_vec = np.random.choice(population).generate_parameter_vector()
            female_vec = np.random.choice(population).generate_parameter_vector()

            # Generate child parameter vector by uniform crossover
            child_vec = np.zeros(len(female_vec))
            for idx in range(len(female_vec)):
                child_vec[idx] = np.random.choice(
                    [female_vec[idx][0], male_vec[idx][0]]
                )

            batch_particles[p_batch_idx].load_parameter_vector(child_vec)

        return batch_particles

    def crossover_species_wise(self, batch_particles, population):
        """ Crossover specieswise for genetic algorithms

        Given communities that have multiple populations, crossover occurs within the species
        specific groups.

        e.g 
            Parents:
                [Ecoli_male, L_lactis_male] 
                [Ecoli_female, L_lactis_female] 
            
            Offspring:
                [Ecoli_male, L_lactis_female] 
            
        Args:
            batch_particles : particles to which new parameters will be assigned
            population : population of particles to perform crossover with
        """

        for p_batch_idx in range(len(batch_particles)):
            # Randomly choose two particles from the population
            male_part = np.random.choice(population)
            female_part = np.random.choice(population)

            # For each model in the community, randomly select parameters from male or female
            for model_idx, particle_model_name in enumerate(
                batch_particles[p_batch_idx].model_names
            ):
                # Randomly choose male or female particle and update parameters for that model
                if np.random.randint(2) == 0:
                    batch_particles[p_batch_idx].k_vals[model_idx] = male_part.k_vals[
                        model_idx
                    ].copy()

                    batch_particles[p_batch_idx].max_exchange_mat[
                        model_idx
                    ] = male_part.max_exchange_mat[model_idx].copy()

                    batch_particles[p_batch_idx].biomass_constraints[
                        model_idx
                    ] = male_part.biomass_constraints[model_idx].copy()

                    batch_particles[p_batch_idx].toxin_mat[
                        model_idx
                    ] = male_part.toxin_mat[model_idx].copy()

                else:
                    batch_particles[p_batch_idx].k_vals[model_idx] = female_part.k_vals[
                        model_idx
                    ].copy()
                    batch_particles[p_batch_idx].max_exchange_mat[
                        model_idx
                    ] = female_part.max_exchange_mat[model_idx].copy()

                    batch_particles[p_batch_idx].biomass_constraints[
                        model_idx
                    ] = female_part.biomass_constraints[model_idx].copy()

                    batch_particles[p_batch_idx].toxin_mat[
                        model_idx
                    ] = female_part.toxin_mat[model_idx].copy()

        return batch_particles

    def mutate_parameterwise(self, batch_particles, mutation_probability):
        """ Mutate parameterwise for genetic algorithms

        For each element in the parameter vector, there is a probability that the element
        is resampled from the prior.
            
        Args:
            batch_particles : particles to which new parameters will be assigned
            mutation_probability: probability of a parameter being mutated
        """

        for particle in batch_particles:
            particle_param_vec = particle.generate_parameter_vector()

            # Generate a 'mutation particle'
            mut_particle = self.init_particles(
                1,
            )[0]
            mut_params_vec = mut_particle.generate_parameter_vector()

            for idx in range(len(particle_param_vec)):
                particle_param_vec[idx] = np.random.choice(
                    [particle_param_vec[idx][0], mut_params_vec[idx][0]],
                    p=[1 - mutation_probability, mutation_probability],
                ).copy()
            particle.load_parameter_vector(particle_param_vec)

    def mutate_resample_from_prior(self, batch_particles, mutation_probability):
        """ Mutate by resampling whole particle from prior

        For particle, there is a probability that the particle should be replaced
        by resampling from the prior.
            
        Args:
            batch_particles : particles to which new parameters will be assigned
            mutation_probability: probability of a particle being resampled from prior
        """

        for particle in batch_particles:
            particle_param_vec = particle.generate_parameter_vector()

            # Generate a 'mutation particle'
            mut_particle = self.init_particles(
                1,
            )[0]
            mut_params_vec = mut_particle.generate_parameter_vector()

            if np.random.uniform() < self.mutation_probability:
                particle.load_parameter_vector(mut_params_vec)

    def mutate_community_from_prior(self, batch_particles, mutation_probability):
        """ Mutate by resampling population from prior

        For each population in a community, given a muatation probability, mutate by resampling
        the population from the prior.
            
        Args:
            batch_particles : particles to which new parameters will be assigned
            mutation_probability: probability of a population being resampled from prior
        """

        for particle in batch_particles:
            # Generate a 'mutation particle'
            mut_particle = self.init_particles(
                1,
            )[0]

            if np.random.uniform() < self.mutation_probability:
                # For each model in the community, randomly select parameters from male or female
                for model_idx, particle_model_name in enumerate(particle.model_names):
                    # Either mutate the metabolism or the toxins (not both at the same time)
                    if np.random.randint(2) == 0:
                        particle.k_vals[model_idx] = mut_particle.k_vals[
                            model_idx
                        ].copy()

                        particle.max_exchange_mat[
                            model_idx
                        ] = mut_particle.max_exchange_mat[model_idx].copy()

                    else:
                        particle.toxin_mat[model_idx] = mut_particle.toxin_mat[
                            model_idx
                        ].copy()

    @logger_wraps()
    def gen_initial_population(self, n_processes, parallel):
        """ Generates a population of particles

        Generates a population of particles by sampling from the prior,
        simulating and calculating their distances from the objective.
            
        Args:
            n_processes : number of processes to use for multiprocessing
            parallel: bool, should simulations be run in parallel
        """
        logger.info("Generating initial population")

        # Generate initial population
        particles = []
        batch_idx = 0
        while len(particles) < self.population_size:
            logger.info(f"Initial accepted particles: {len(particles)}")
            candidate_particles = []
            while len(candidate_particles) < self.n_particles_batch:
                new_particles = self.init_particles(self.n_particles_batch)

                for media_name in self.sim_media_names:
                    [p.set_media_conditions(media_name) for p in new_particles]

                    if not isinstance(self.filter, type(None)):
                        new_particles = self.filter.filter_particles(new_particles)

                candidate_particles.extend(new_particles)

            logger.info(f"Simulating candidates {len(candidate_particles)}")
            for media_idx, media_name in enumerate(self.sim_media_names):
                self.simulator.simulate_particles(
                    candidate_particles,
                    sol_key=media_name,
                    n_processes=n_processes,
                    parallel=parallel,
                )
                self.calculate_and_set_particle_distances(
                    candidate_particles, media_name
                )

            logger.info("Finished simulating")

            logger.info("Deleting models")
            self.delete_particle_fba_models(candidate_particles)

            if len(candidate_particles) > 0:
                particles.extend(candidate_particles)

            logger.info(psutil.Process(os.getpid()).memory_info().rss / 1024**2)

            batch_idx += 1

        particles = particles[: self.population_size]

        return particles

    def generate_models_list(self, n_models) -> List[cobra.core.model.Model]:
        """ Generates list of models

        Makes deep copies of the algorithms base model. This is
        necessary to simualate in parallel.

        Args:
            n_models: number of models to generate.

        Returns:
            models: list of cobrapy models
        """

        models = []
        logger.info("Copying base comm")

        for _ in range(n_models):
            proc_models = []
            for pop in self.base_community.populations:
                proc_models.append(copy.deepcopy(pop.model))
            models.append(proc_models)

        logger.info("Deleting basecomm models")
        for pop in self.base_community.populations:
            del pop.model

        return models

    def delete_particle_fba_models(self, particles):
        """ Deletes attributes of particles

        Deletes attributes of particles that are not needed after 
        they have been sampled and simulated

        Args:
            particles: particles to be cleaned

        Returns:
            
        """
        
        for part in particles:
            try:
                del part.initial_population_prior
                del part.max_exchange_prior
                del part.k_val_prior
                del part.toxin_interaction_prior

            except:
                pass

            for p in part.populations:
                del p.model

        gc.collect()

    def get_particle_distances(self, particles) -> np.ndarray:
        """Gets distances of particles

        Returns array of distances in shape [particles, distances]

        Args:
            particles: particles to get distances from

        Returns:
            distances: array of distances
        """
        n_distances = len(particles[0].distance)
        n_particles = len(particles)
        distances = np.zeros(shape=[n_particles, n_distances])

        for idx, p in enumerate(particles):
            distances[idx] = p.distance

        return distances

    def hotstart_particles(self, hostart_parameter_dir_regex):
        """Load a population of particles to the algorithm.

        Will skip directories that do not the necessary files

        Args:
            hostart_parameter_dir_regex: glob pattern to find hotstart directories
        """
        # Load all populations
        hotstart_particles = []
        hotstart_directories = glob.glob(hostart_parameter_dir_regex)

        for hotstart_dir in hotstart_directories:

            try:
                # Make parameter paths
                init_populations_arr = np.load(
                    f"{hotstart_dir}/particle_init_populations.npy"
                )
                k_values_arr = np.load(
                    f"{hotstart_dir}/particle_k_values.npy",
                )
                max_exchange_arr = np.load(
                    f"{hotstart_dir}/particle_max_exchange.npy",
                )
                toxin_arr = np.load(f"{hotstart_dir}/particle_toxin.npy")

                distance_arr = np.load(f"{hotstart_dir}/particle_distance_vectors.npy")

                biomass_rate_constraints_arr = np.load(
                    f"{hotstart_dir}/particle_biomass_rate_constr_vectors.npy"
                )

                # biomass_fluxes = np.load(f"{hotstart_dir}/biomass_flux.npy")

            except FileNotFoundError:
                continue

            # Print array shapes
            logger.info(f"init_populations_arr shape: {init_populations_arr.shape}")
            logger.info(f"k_values_arr shape: {k_values_arr.shape}")
            logger.info(f"max_exchange_arr shape: {max_exchange_arr.shape}")
            logger.info(f"toxin_arr shape: {toxin_arr.shape}")
            logger.info(f"distance_arr shape: {distance_arr.shape}")
            logger.info(
                f"biomass_rate_constraints_arr shape: {biomass_rate_constraints_arr.shape}"
            )

            naked_particles = self.init_particles(
                n_particles=len(max_exchange_arr), assign_model=False, set_media=False
            )

            print("Naked Particles: ", len(naked_particles))

            for idx, p in enumerate(naked_particles):
                p.set_initial_populations(init_populations_arr[idx])
                p.set_k_value_matrix(k_values_arr[idx])
                p.set_max_exchange_mat(max_exchange_arr[idx])
                p.set_toxin_mat(toxin_arr[idx])
                p.set_media_conditions(self.sim_media_names[0], set_media=False)
                p.set_biomass_rate_constraints(biomass_rate_constraints_arr[idx])

                # p.biomass_flux = biomass_fluxes[idx]
                p.distance = distance_arr[idx]

            hotstart_particles.extend(naked_particles)
            print("hotstart_particles: ", len(hotstart_particles))

        hotstart_particles = utils.get_unique_particles(hotstart_particles)
        print("hotstart_particles: ", len(hotstart_particles))

        self.population = hotstart_particles

    @logger_wraps()
    def calculate_and_set_particle_distances(self, particles, sol_distance_key):
        """Calculates and sets distances for each particle

        Executes the algorithm distance function to calculate the distance of each particle. 
        Distacnce vectors are concatenated into a single distance vector.

        Args:
            particles: particles to calculate distances for
            sol_distance_key: Key to match solution column and experimental data column

        """

        # Calculate distances
        for p in particles:
            for distance_func in self.distance_object:
                p.distance.extend(distance_func.calculate_distance(p, sol_distance_key))
