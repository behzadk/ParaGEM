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


class ParameterEstimation:
    def init_particles(
        self,
        n_particles,
    ):
        particles = np.zeros(shape=n_particles, dtype=object)

        for i in range(n_particles):
            # Make independent copy of base community
            comm = copy.deepcopy(self.base_community)
            # Assign population models
            for idx, pop in enumerate(comm.populations):
                pop.model = self.models[i][idx]
            
            comm.sample_parameters_from_prior()
            comm.set_init_y()
            particles[i] = comm

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
                    # Randomly choose male or female particle and update parameters for that model
                    if np.random.randint(2) == 0:
                        particle.k_vals[model_idx] = mut_particle.k_vals[
                            model_idx
                        ].copy()
                        
                        particle.max_exchange_mat[
                            model_idx
                        ] = mut_particle.max_exchange_mat[model_idx].copy()

                        particle.toxin_mat[model_idx] = mut_particle.toxin_mat[model_idx].copy()                        



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

                if not isinstance(self.filter, type(None)):
                    new_particles = self.filter.filter_particles(new_particles)

                candidate_particles.extend(new_particles)

            logger.info(f"Simulating candidates {len(candidate_particles)}")

            self.simulator.simulate_particles(
                candidate_particles, n_processes=n_processes, parallel=parallel
            )
            logger.info(f"Finished simulating")

            logger.info(f"Deleting models")
            self.delete_particle_fba_models(candidate_particles)

            if len(candidate_particles) > 0:
                particles.extend(candidate_particles)

            logger.info(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)

            batch_idx += 1

        particles = particles[: self.population_size]

        return particles

    def save_particles(self, particles, output_path):
        logger.info(f"Saving particles {output_path}")

        with open(f"{output_path}", "wb") as handle:
            pickle.dump(particles, handle)

    def save_checkpoint(self, output_dir):
        time_stamp = time.strftime("%Y-%m-%d_%H%M%S")
        output_path = f"{output_dir}{self.experiment_name}_checkpoint_{time_stamp}.pkl"

        logger.info(f"Saving checkpoint {output_path}")

        with open(output_path, "wb") as f:
            pickle.dump(self, f)

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

    def calculate_and_set_particle_distances(self, particles):
        # Calculate distances
        for p in particles:
            p.distance = self.distance_object.calculate_distance(p)


class NSGAII(ParameterEstimation):
    def __init__(
        self,
        experiment_name,
        distance_object,
        base_community,
        output_dir,
        simulator,
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
        self.base_community.set_init_y()
        
        self.n_particles_batch = n_particles_batch

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

        if not isinstance(hotstart_particles_regex, type(None)):
            self.hotstart(hotstart_particles_regex)



    def run(self, n_processes=1, parallel=False):
        logger.info(f"Running NSGAII")

        # Generate first generation
        if self.gen_idx == 0:
            self.population = self.gen_initial_population(n_processes, parallel)
            self.calculate_and_set_particle_distances(self.population)

            self.save_particles(
                self.population,
                output_path=f"{self.output_dir}particles_gen_{self.gen_idx}.pkl",
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
                self.simulator.simulate_particles(
                    offspring_particles, n_processes=n_processes, parallel=parallel
                )

                self.calculate_and_set_particle_distances(offspring_particles)

                self.delete_particle_fba_models(offspring_particles)

                new_offspring.extend(offspring_particles)
                new_parents.extend(parent_particles)

                batch_idx += 1

            new_offspring = new_offspring[: self.population_size]

            # Combine population of parents and offspring (2N)
            self.population = new_offspring + new_parents


            self.save_particles(
                self.population,
                output_path=f"{self.output_dir}particles_gen_{self.gen_idx}.pkl",
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

        print(parent_indexes)
        parent_particles = [population[idx] for idx in parent_indexes]

        return parent_particles


class GeneticAlgorithm(ParameterEstimation):
    def __init__(
        self,
        experiment_name,
        distance_object,
        base_community,
        output_dir,
        simulator,
        n_particles_batch,
        hotstart_particles_regex=None,
        population_size=32,
        mutation_probability=0.1,
        epsilon_alpha=0.2,
        generation_idx=0,
        max_generations=1,
        tournament_size=2,
        particle_filter=None,
    ):
        logger.info(f"Initialising GA")

        self.experiment_name = experiment_name
        self.base_community = base_community
        self.n_particles_batch = n_particles_batch

        self.output_dir = output_dir
        self.population_size = population_size
        self.distance_object = distance_object
        self.mutation_probability = mutation_probability
        self.epsilon_alpha = epsilon_alpha
        self.simulator = simulator

        self.gen_idx = int(generation_idx)
        self.final_generation = False

        self.filter = particle_filter
        self.max_generations = max_generations

        self.tournament_size = tournament_size

        # Generate a list of models that will be assigned
        # to new particles. Avoids repeatedly copying models
        self.models = self.generate_models_list(n_models=self.n_particles_batch)

        print(hotstart_particles_regex)
        if not isinstance(hotstart_particles_regex, type(None)):
            self.hotstart(hotstart_particles_regex)
            self.population = utils.get_unique_particles(self.population)

            logger.info(f"Hotstart complete, population size: {len(self.population)}")

    def selection(self, particles):
        accepted_particles = []

        for p in particles:
            accepted_flag, distance = self.distance_object.assess_particle(p)

            with np.printoptions(precision=4, suppress=True):
                print(f"{accepted_flag}\t {distance}")

            if accepted_flag:
                accepted_particles.append(p)

            p.accepted_flag = accepted_flag
            p.distance = distance

        return accepted_particles

    def update_epsilon(self, particles):
        particle_distances = []
        # Collate distances
        for p in particles:
            particle_distances.append(p.distance)

        particle_distances = np.array(particle_distances)
        new_epsilon = []
        # For each distance, update epsilon
        for dist_idx in range(particle_distances.shape[1]):
            dists = particle_distances[:, dist_idx]
            dists.sort()

            # Trim distances to largest epsilon out of smallest x
            dists = dists[: int(self.population_size * self.epsilon_alpha)]
            new_epsilon = max(self.distance_object.final_epsilon[dist_idx], dists[-1])

            self.distance_object.epsilon[dist_idx] = new_epsilon

    def run(self, n_processes=1, parallel=False):
        logger.info("Running genetic algorithm")

        if self.gen_idx == 0:
            # Generate initial population
            self.population = self.gen_initial_population(n_processes, parallel)
            self.calculate_and_set_particle_distances(self.population)

            output_path = f"{self.output_dir}particles_{self.experiment_name}_gen_{self.gen_idx}.pkl"

            self.save_particles(self.population, output_path)
            self.gen_idx += 1

        # Core genetic algorithm loop
        while not self.final_generation and self.gen_idx < self.max_generations:
            batch_idx = 0
            accepted_particles = []


            logger.info(f"Performing crossover...")
            new_offspring = []
            new_parents = []

            # Update epsilon
            # self.update_epsilon(self.population)

            if self.distance_object.final_epsilon == self.distance_object.epsilon:
                self.final_generation = True

            population_average_distance = np.mean(
                [max(p.distance) for p in self.population]
            )
            population_mediain_distance = np.median(
                [max(p.distance) for p in self.population]
            )
            population_min_distance = np.min([max(p.distance) for p in self.population])

            logger.info(
                f"Pop mean distance: {population_average_distance}, pop median distance: {population_mediain_distance}, pop min distance: {population_min_distance}"
            )
            while len(new_offspring) < self.population_size:
                logger.info(
                    f"Gen: {self.gen_idx}, batch: {batch_idx}, accepted: {len(new_offspring)}, mem usage (mb): {utils.get_mem_usage()}"
                )

                offspring_particles = []
                while len(offspring_particles) < self.n_particles_batch:
                    parents = self.selection_tournament(
                        self.population, 2, tournament_size=self.tournament_size
                    )

                    # Generate new batch by crossover
                    candidate_particles = self.crossover_parameterwise(1, parents)

                    # candidate_particles = self.crossover_species_wise(1, parents)


                    # Mutate batch
                    self.mutate_parameterwise(candidate_particles)
                    # self.mutate_resample_from_prior(candidate_particles)

                    for p in candidate_particles:
                        p.set_init_y()

                    if self.filter is not None:
                        candidate_particles = self.filter.filter_particles(
                            candidate_particles
                        )

                    if len(candidate_particles) > 0:
                        offspring_particles.extend(candidate_particles)
                        new_parents.extend(parents)

                offspring_particles = offspring_particles[: self.n_particles_batch]

                logger.info(f"Simulating particles...")
                # Simulate particles
                self.simulator.simulate_particles(
                    offspring_particles, n_processes=n_processes, parallel=parallel
                )
                self.calculate_and_set_particle_distances(offspring_particles)

                self.delete_particle_fba_models(offspring_particles)

                # offspring_particles = self.selection(offspring_particles)
                new_offspring.extend(offspring_particles)

                batch_idx += 1

            new_offspring = new_offspring[: self.population_size]
            self.population = new_offspring + new_parents

            output_path = f"{self.output_dir}particles_{self.experiment_name}_gen_{self.gen_idx}.pkl"

            self.save_particles(
                self.population,
                output_path=f"{self.output_dir}particles_gen_{self.gen_idx}.pkl",
            )
            self.gen_idx += 1


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
                print(sum(p.distance))
                pass

        else:
            self.hotstart_particles = None

        for d in base_community.dynamic_compounds:
            if d not in self.population[0].dynamic_compounds:
                print(d)
        
        self.save_particles(
                self.population,
                f"{self.output_dir}hotstart_particles_{self.experiment_name}.pkl",
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
                f"{self.output_dir}particles_{self.experiment_name}_batch_{batch_idx}.pkl",
            )

            particles_simulated += len(batch_particles)
            batch_idx += 1
