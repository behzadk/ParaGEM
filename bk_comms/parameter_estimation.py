from bk_comms import utils

import copy
import pickle
import numpy as np

import multiprocess as mp
import time
from loguru import logger
from pathlib import Path
import gc
import psutil
import os

from sympy.core.cache import *
import time
import sys
import pygmo

import glob

from bk_comms.utils import logger_wraps

class ParameterEstimation:
    def init_particles(
        self,
        n_particles, assign_model=True, set_media=True
    ):
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


    def crossover_parameterwise(self, n_particles, population):
        batch_particles = self.init_particles(
            n_particles,
        )

        for p_batch_idx in range(n_particles):
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

    def selection_tournament(self, population, n_particles, tournament_size):
        selected_parents = []

        # One winner per tournament
        for _ in range(n_particles):
            contestants = list(
                np.random.choice(population, size=tournament_size, replace=False)
            )
            contestants.sort(key=lambda x: sum(x.distance))
            selected_parents.append(contestants[0])

        return selected_parents

    def crossover_species_wise(self, n_particles, population):
        batch_particles = self.init_particles(
            n_particles,
        )

        for p_batch_idx in range(n_particles):
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

                    batch_particles[p_batch_idx].toxin_mat[model_idx] = male_part.toxin_mat[model_idx].copy()

                else:
                    batch_particles[p_batch_idx].k_vals[model_idx] = female_part.k_vals[
                        model_idx
                    ].copy()
                    batch_particles[p_batch_idx].max_exchange_mat[
                        model_idx
                    ] = female_part.max_exchange_mat[model_idx].copy()

                    batch_particles[p_batch_idx].toxin_mat[model_idx] = female_part.toxin_mat[model_idx].copy()
                    
        return batch_particles

    def mutate_parameterwise(self, batch_particles):
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
                    p=[1 - self.mutation_probability, self.mutation_probability],
                ).copy()
            particle.load_parameter_vector(particle_param_vec)

    def mutate_resample_from_prior(self, batch_particles):
        for particle in batch_particles:
            particle_param_vec = particle.generate_parameter_vector()

            # Generate a 'mutation particle'
            mut_particle = self.init_particles(
                1,
            )[0]
            mut_params_vec = mut_particle.generate_parameter_vector()

            if np.random.uniform() < self.mutation_probability:
                particle.load_parameter_vector(mut_params_vec)

    def mutate_community_from_prior(self, batch_particles):

        
        for particle in batch_particles:
            # Generate a 'mutation particle'
            mut_particle = self.init_particles(
                1,
            )[0]

            if np.random.uniform() < self.mutation_probability:
                # For each model in the community, randomly select parameters from male or female
                for model_idx, particle_model_name in enumerate(
                    particle.model_names
                ):
                    # Either mutate the metabolism or the toxins (not both at the same time)
                    if np.random.randint(2) == 0:
                        particle.k_vals[model_idx] = mut_particle.k_vals[
                            model_idx
                        ].copy()
                        
                        particle.max_exchange_mat[
                            model_idx
                        ] = mut_particle.max_exchange_mat[model_idx].copy()

                    else:
                        particle.toxin_mat[model_idx] = mut_particle.toxin_mat[model_idx].copy()                        


    @logger_wraps()
    def gen_initial_population(self, n_processes, parallel):
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
                    candidate_particles, sol_key=media_name, n_processes=n_processes, parallel=parallel
                )
                self.calculate_and_set_particle_distances(candidate_particles, media_name)

            logger.info(f"Finished simulating")

            logger.info(f"Deleting models")
            self.delete_particle_fba_models(candidate_particles)

            if len(candidate_particles) > 0:
                particles.extend(candidate_particles)

            logger.info(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)

            batch_idx += 1

        particles = particles[: self.population_size]

        return particles

    def save_particles(self, particles, output_dir):
    
        init_populations_arr = np.array([particle.init_population_values for particle in particles])
        k_values_arr = np.array([particle.k_vals for particle in particles])
        max_exchange_arr = np.array([particle.max_exchange_mat for particle in particles])
        toxin_arr = np.array([particle.toxin_mat for particle in particles])

        distance_vectors = np.array([particle.distance for particle in particles])

        # Write arrays to output_dir
        np.save(f"{output_dir}/particle_init_populations.npy", init_populations_arr)
        np.save(f"{output_dir}/particle_k_values.npy", k_values_arr)
        np.save(f"{output_dir}/particle_max_exchange.npy", max_exchange_arr)
        np.save(f"{output_dir}/particle_toxin.npy", toxin_arr)
        np.save(f"{output_dir}/solution_keys.npy", particles[0].solution_keys)

        if hasattr(particles[0],'distance'):
            np.save(f"{output_dir}/particle_distance_vectors.npy", distance_vectors)

        if hasattr(particles[0],'sol'):
            for media in self.sim_media_names:
                sol_arr = np.array([particle.sol[media] for particle in particles])
                np.save(f"{output_dir}/particle_sol_{media}.npy", sol_arr)

            t_vectors = np.array([particle.t for particle in particles])

            np.save(f"{output_dir}/particle_t_vectors.npy", t_vectors)


    def generate_models_list(self, n_models):
        # Generate a list of models that will be assigned
        # to new particles. Avoids repeatedly copying models
        models = []
        logger.info(f"Copying base comm")

        for _ in range(n_models):
            proc_models = []
            for pop in self.base_community.populations:
                proc_models.append(copy.deepcopy(pop.model))
            models.append(proc_models)

        logger.info(f"Deleting basecomm models")
        for pop in self.base_community.populations:
            del pop.model

        return models

    def delete_particle_fba_models(self, particles):
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

    def media_transfer_particles(particles, media_set_base_particle):
        media_set_particles = []
        for p in particles:
            media_set_particle = copy.deepcopy(media_set_base_particle)

            # Transfer data from particle particle to media set base particle
            media_set_particle.distance = p.distance
            media_set_particle.sol = p.sol

            # Transfer parameters from particle to media set particle
            
            
            media_set_particle.set_init_y()

    def get_particle_distances(self, particles):
        n_distances = len(particles[0].distance)
        n_particles = len(particles)
        distances = np.zeros(shape=[n_particles, n_distances])

        for idx, p in enumerate(particles):
            distances[idx] = p.distance

        return distances

    def hotstart(self, hotstart_directory_regex):
        # Load all populations
        population_paths = glob.glob(hotstart_directory_regex)

        hotstart_particles = []
        for f in population_paths:
            particles = utils.load_pickle(f)

            for p in particles:
                try:
                    # del p.sol
                    del p.flux_log
                    del p.experiment
                
                except:
                    pass

            hotstart_particles.extend(particles)
            gc.collect()
            logger.info(
                f"Hotstart memory usage: {psutil.Process(os.getpid()).memory_info().rss / 1024**2}"
            )

        # Set new generation
        hotstart_particles = utils.get_unique_particles(hotstart_particles)
        logger.info(
            f"Number of hotstart particles: {len(hotstart_particles)}"
        )

        self.population = hotstart_particles

    def hotstart_particles(self, hostart_parameter_dir_regex):
        # Load all populations
        hotstart_particles = []
        hotstart_directories = glob.glob(hostart_parameter_dir_regex)

        for hotstart_dir in hotstart_directories:
            
            # Make parameter paths
            init_populations_arr = np.load(f"{hotstart_dir}/particle_init_populations.npy")
            k_values_arr = np.load(f"{hotstart_dir}/particle_k_values.npy", )
            max_exchange_arr = np.load(f"{hotstart_dir}/particle_max_exchange.npy", )
            toxin_arr = np.load(f"{hotstart_dir}/particle_toxin.npy")

            distance_arr = np.load(f"{hotstart_dir}/particle_distance_vectors.npy")

            # Print array shapes
            logger.info(f"init_populations_arr shape: {init_populations_arr.shape}")
            logger.info(f"k_values_arr shape: {k_values_arr.shape}")
            logger.info(f"max_exchange_arr shape: {max_exchange_arr.shape}")
            logger.info(f"toxin_arr shape: {toxin_arr.shape}")
            logger.info(f"distance_arr shape: {distance_arr.shape}")
            
            naked_particles = self.init_particles(n_particles=len(max_exchange_arr), assign_model=False, set_media=False)

            for idx, p in enumerate(naked_particles):
                p.set_initial_populations(init_populations_arr[idx])
                p.set_k_value_matrix(k_values_arr[idx])
                p.set_max_exchange_mat(max_exchange_arr[idx])
                p.set_toxin_mat(toxin_arr[idx])
                p.set_media_conditions(self.sim_media_names[0], set_media=False)


                p.distance = distance_arr[idx]

            hotstart_particles.extend(naked_particles)


        self.population = hotstart_particles

    @logger_wraps()
    def calculate_and_set_particle_distances(self, particles, sol_distance_key):
        # Calculate distances
        for p in particles:
            p.distance.extend(self.distance_object.calculate_distance(p, sol_distance_key))


class NSGAII(ParameterEstimation):
    def __init__(
        self,
        experiment_name,
        distance_object,
        base_community,
        output_dir,
        simulator,
        sim_media_names,
        crossover_type='parameterwise',
        mutate_type='parameterwise',
        n_particles_batch=8,
        particle_filter=None,
        mutation_probability=0.0,
        generation_idx=0,
        hotstart_particles_regex=None,
        population_size=32,
        max_generations=10,
    ):
        self.experiment_name = experiment_name
        self.base_community = base_community

        # self.media_set_base_communities = {}
        # for media_name in self.sim_media_names:
        #     comm = copy.deepcopy(base_community)
        #     comm.set_media_conditions(media_name)
        #     self.media_set_base_communities[media_name] = comm

        self.n_particles_batch = n_particles_batch

        self.sim_media_names = sim_media_names

        self.output_dir = output_dir
        self.population_size = population_size
        self.distance_object = distance_object
        self.mutation_probability = mutation_probability
        self.simulator = simulator

        self.gen_idx = int(generation_idx)
        self.final_generation = False

        if crossover_type == 'parameterwise':
            self.crossover = self.crossover_parameterwise
        
        elif crossover_type == 'specieswise':
            self.crossover = self.crossover_species_wise
        
        else:
            raise ValueError(f"Unknown crossover type: {crossover_type}")

        if mutate_type == "parameterwise":
            self.mutate = self.mutate_parameterwise
        
        elif mutate_type == "resample_prior":
            # self.mutate = self.mutate_resample_from_prior
            self.mutate = self.mutate_community_from_prior
        
        else:
            raise ValueError(f"Unknown mutation type: {mutate_type}")

        self.filter = particle_filter
        self.models = self.generate_models_list(n_models=self.n_particles_batch)

        self.max_generations = max_generations


        # if not isinstance(hotstart_particles_regex, type(None)):
        #     self.hotstart_particles(hotstart_particles_regex)


    @logger_wraps()
    def run(self, n_processes=1, parallel=False):
        logger.info(f"Running NSGAII")

        # Generate first generation
        if self.gen_idx == 0:
            self.population = self.gen_initial_population(n_processes, parallel)

            self.save_particles(
                self.population,
                output_dir=f"{self.output_dir}",
            )
            self.gen_idx += 1

        while self.gen_idx < self.max_generations:
            logger.info(f"Running generation {self.gen_idx}")
            logger.info(f"Selecting parents")

            new_offspring = []
            new_parents = []
            batch_idx = 0
            while len(new_offspring) < self.population_size:

                population_average_distance = np.mean(
                [max(p.distance) for p in self.population]
                )
                population_mediain_distance = np.median(
                    [max(p.distance) for p in self.population]
                )
                population_min_distance = np.min([sum(p.distance) for p in self.population])

                logger.info(
                    f"Pop mean distance: {population_average_distance}, pop median distance: {population_mediain_distance}, pop min distance: {population_min_distance}"
                )

                # Select parents population by non-dominated sorting and crowd distance
                parent_particles = self.non_dominated_sort_parent_selection(
                    self.population, self.n_particles_batch
                )


                logger.info(f"Performing crossover to produce offspring")
                # Perform crossover and mutation to generate offspring of size N
                offspring_particles = self.crossover(
                    self.n_particles_batch, parent_particles
                )

                logger.info(f"Mutating offspring")
                self.mutate(offspring_particles)
                offspring_particles = list(offspring_particles)


                logger.info(f"Simulating offspring")

                logger.info(f"Simulating offspring batch: {batch_idx}")
                
                for media_idx, media_name in enumerate(self.sim_media_names):
                    [p.set_media_conditions(media_name) for p in offspring_particles]

                    if not isinstance(self.filter, type(None)):
                        offspring_particles = self.filter.filter_particles(offspring_particles)

                    # Simulate particle for media name
                    self.simulator.simulate_particles(
                        offspring_particles, sol_key=media_name, n_processes=n_processes, parallel=parallel
                    )

                    # Calculate distance for media_name
                    self.calculate_and_set_particle_distances(offspring_particles, media_name)
                
                offspring_particles = [p for p in offspring_particles if not np.isnan(sum(p.distance))]

                self.delete_particle_fba_models(offspring_particles)

                new_offspring.extend(offspring_particles)
                new_parents.extend(parent_particles)

                batch_idx += 1

            new_offspring = new_offspring[: self.population_size]

            # Combine population of parents and offspring (2N)
            self.population = new_offspring + new_parents


            self.save_particles(
                self.population,
                output_dir=f"{self.output_dir}",
            )
            new_offspring = []
            new_parents = []
            self.gen_idx += 1

        logger.info(f"Finished")

    def non_dominated_sort_parent_selection(self, population, n_particles):
        # Use binary tournament to generate population of parents:
        # Randomly select two solutions, keep the better one with
        # respect to non-domination rank. Solutions in same front
        # are compared by crowding distance.

        population_particle_indexes = list(range(len(population)))

        population_distances = self.get_particle_distances(population)
        ndf, dl, dc, ndr = pygmo.fast_non_dominated_sorting(population_distances)
        crowd_distances = pygmo.crowding_distance(population_distances)

        find_front = lambda idx: [
            front_idx for front_idx, front in enumerate(ndf) if idx in front
        ]

        parent_indexes = []
        for x in range(n_particles):
            # Randomly sample two particles from poplation
            candidate_0_idx, candidate_1_idx = np.random.choice(
                population_particle_indexes, size=2, replace=False
            )

            print(candidate_0_idx, candidate_1_idx)

            # Identify which front they belong to
            candidate_0_front = find_front(candidate_0_idx)
            candidate_1_front = find_front(candidate_1_idx)

            # If belonging to same front, choose by largest crowd distance
            if candidate_0_front == candidate_1_front:
                candidate_0_crowd_dist = crowd_distances[candidate_0_idx]
                candidate_1_crowd_dist = crowd_distances[candidate_1_idx]

                if candidate_0_crowd_dist > candidate_1_crowd_dist:
                    parent_indexes.append(candidate_0_idx)

                else:
                    parent_indexes.append(candidate_1_idx)

            elif candidate_0_front < candidate_1_front:
                parent_indexes.append(candidate_0_idx)

            else:
                parent_indexes.append(candidate_1_idx)

        parent_particles = [population[idx] for idx in parent_indexes]

        return parent_particles


class SimpleSimulate(ParameterEstimation):
    def __init__(
        self,
        experiment_name,
        base_community,
        output_dir,
        simulator,
        n_particles_batch,
        hotstart_particles_regex,
        max_simulations=32,
        particle_filter=None,
    ):

        self.experiment_name = experiment_name
        self.base_community = base_community

        self.output_dir = output_dir
        self.simulator = simulator
        self.n_particles_batch = n_particles_batch
        self.filter = particle_filter
        self.max_simulations = max_simulations

        self.models = self.generate_models_list(n_models=self.n_particles_batch)


        # Load hotstart particles
        if not isinstance(hotstart_particles_regex, type(None)):
            self.hotstart(hotstart_particles_regex)
            sum_distances = [max(p.distance) for p in self.population]
            self.population = utils.get_unique_particles(self.population)

            self.population = sorted(
                self.population, key=lambda x: sum_distances[self.population.index(x)]
            )


            self.population = self.population
            for p in self.population:
                pass

        else:
            self.hotstart_particles = None

        for d in base_community.dynamic_compounds:
            if d not in self.population[0].dynamic_compounds:
                print(d)
        
        self.save_particles(
                self.population,
                output_dir=f"{self.output_dir}",
            )

    def initialize_fresh_particle(self):
        pass

    def run(self, n_processes=1, parallel=False):
        particles_simulated = 0
        batch_idx = 0

        particle_idx = 0

        while particles_simulated < self.max_simulations or particle_idx > len(self.population):
        
            batch_particles = []

            for i in range(self.n_particles_batch):
                # Make independent copy of base community
                comm = copy.deepcopy(self.base_community)
                # Assign population models
                for idx, pop in enumerate(comm.populations):
                    pop.model = self.models[i][idx]
                batch_particles.append(comm)

            # batch_particles = self.init_particles(self.n_particles_batch)

            for batch_p_idx in range(self.n_particles_batch):
                batch_particles[batch_p_idx].set_k_value_matrix(self.population[particle_idx].k_vals)
                batch_particles[batch_p_idx].set_max_exchange_mat(self.population[particle_idx].max_exchange_mat)


                batch_particles[batch_p_idx].set_toxin_mat(self.population[particle_idx].toxin_mat)
                batch_particles[batch_p_idx].set_initial_populations(self.population[particle_idx].init_population_values)


                batch_particles[batch_p_idx].set_init_y()
                
                particle_idx += 1

            self.simulator.simulate_particles(
                batch_particles, n_processes=n_processes, parallel=parallel
            )

            print("Saving particles")

            self.delete_particle_fba_models(batch_particles)
            self.save_particles(
                batch_particles,
                output_dir=f"{self.output_dir}",
            )

            particles_simulated += len(batch_particles)
            batch_idx += 1
