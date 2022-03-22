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
            particles[i] = comm

        return particles

    def crossover(self, n_particles, population):
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

    def crossover_species_wise(self, n_particles, population):
        batch_particles = self.init_particles(
            n_particles,
        )

        for p_batch_idx in range(n_particles):
            # Randomly choose two particles from the population
            male_part = np.random.choice(population)
            female_part = np.random.choice(population)

            # For each model in the community, randomly select parameters from male or female 
            for model_idx, particle_model_name in enumerate(batch_particles[p_batch_idx].model_names):
                # Randomly choose male or female particle and update parameters for that model
                if np.random.randint == 0:
                    batch_particles[p_batch_idx].k_vals[model_idx] = male_part.k_vals[model_idx].copy()
                    batch_particles[p_batch_idx].max_exchange_mat[model_idx] = male_part.max_exchange_mat[model_idx].copy()

                else:
                    batch_particles[p_batch_idx].k_vals[model_idx] = female_part.k_vals[model_idx].copy()
                    batch_particles[p_batch_idx].max_exchange_mat[model_idx] = female_part.max_exchange_mat[model_idx].copy()

            # Randomly sample toxin interactions from either male or female
            if np.random.randint == 0:
                batch_particles[p_batch_idx].set_toxin_mat(male_part.toxin_mat.copy())

            else:
                batch_particles[p_batch_idx].set_toxin_mat(female_part.toxin_mat.copy())

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

            if np.random.uniform() > self.mutation_probability:
                particle.load_parameter_vector(mut_params_vec)



    def gen_initial_population(self, n_processes, parallel):
        logger.info("Generating initial population")

        # Generate initial population
        accepted_particles = []
        batch_idx = 0
        while len(accepted_particles) < self.population_size:
            logger.info(f"Initial accepted particles: {len(accepted_particles)}")

            particles = []
            while len(particles) <= self.n_particles_batch:
                candidate_particles = self.init_particles(self.n_particles_batch)
                if not isinstance(self.filter, type(None)):
                    candidate_particles = self.filter.filter_particles(
                        candidate_particles
                    )

                particles.extend(candidate_particles)

            particles = particles[: self.n_particles_batch]

            if len(particles) == 0:
                continue

            self.simulator.simulate_particles(
                particles,
                n_processes=n_processes,
                parallel=parallel,
            )

            logger.info(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)

            accepted_particles.extend(particles)
            self.delete_particle_fba_models(particles)

            batch_idx += 1

        return accepted_particles

    def save_particles(self, particles, output_path):
        particles_out = []
        for idx, p in enumerate(particles):
            p_copy = copy.deepcopy(p)

            particles_out.append(p_copy)

        logger.info(f"Saving particles {output_path}")

        with open(f"{output_path}", "wb") as handle:
            pickle.dump(particles_out, handle)

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

    @classmethod
    def load_checkpoint(cls, checkpoint_path):
        logger.info(f"Loading checkpoint {checkpoint_path}")

        with open(checkpoint_path, "rb") as f:
            return pickle.load(f)


class NSGAII(ParameterEstimation):
    def __init__(
        self,
        experiment_name,
        distance_object,
        base_community,
        max_uptake_sampler,
        k_val_sampler,
        output_dir,
        simulator,
        particle_filter=None,
        mutation_probability=0.0,
        generation_idx=0,
        hotstart_particles_regex=None,
        toxin_interaction_sampler=None,
        init_population_sampler=None,
        population_size=32,
        max_generations=10,
    ):
        self.experiment_name = experiment_name
        self.base_community = base_community
        self.n_particles_batch = population_size
        self.max_uptake_sampler = max_uptake_sampler
        self.k_val_sampler = k_val_sampler
        self.init_population_sampler = init_population_sampler
        self.toxin_interaction_sampler = toxin_interaction_sampler

        self.output_dir = output_dir
        self.population_size = population_size
        self.distance_object = distance_object
        self.mutation_probability = mutation_probability
        self.simulator = simulator

        self.gen_idx = int(generation_idx)
        self.final_generation = False

        self.filter = particle_filter
        self.models = self.generate_models_list(n_models=self.population_size)

        self.max_generations = max_generations

        if not isinstance(hotstart_particles_regex, type(None)):
            self.hotstart(hotstart_particles_regex)

    def run(self, n_processes=1, parallel=False):
        # Generate first generation
        if self.gen_idx == 0:
            self.population = self.gen_initial_population(n_processes, parallel)
            self.calculate_and_set_particle_distances(self.population)

            self.save_particles(
                self.population,
                output_path=f"{self.output_dir}particles_gen_{self.gen_idx}.pkl",
            )

        while self.gen_idx < self.max_generations:
            self.gen_idx += 1
            logger.info(f"Running generation {self.gen_idx}")

            logger.info(f"Selecting parents")
            # Select parents population by non-dominated sorting and crowd distance
            parent_particles = self.non_dominated_sort_parent_selection(
                self.population, self.population_size
            )

            logger.info(f"Performing crossover to produce offspring")
            # Perform crossover and mutation to generate offspring of size N
            offspring_particles = self.crossover_species_wise(self.population_size, parent_particles)

            logger.info(f"Mutating offspring")
            self.mutate_resample_from_prior(offspring_particles)
            offspring_particles = list(offspring_particles)

            logger.info(f"Simulating offspring")
            self.simulator.simulate_particles(
                offspring_particles, n_processes=n_processes, parallel=parallel
            )
            self.delete_particle_fba_models(offspring_particles)

            # Combine population of parents and offspring (2N)
            self.population = offspring_particles + parent_particles

            # Calculate distances
            self.calculate_and_set_particle_distances(self.population)

            self.save_particles(
                self.population,
                output_path=f"{self.output_dir}particles_gen_{self.gen_idx}.pkl",
            )

        logger.info(f"Finished")

    def calculate_and_set_particle_distances(self, particles):
        # Calculate distances
        for p in particles:
            p.distance = self.distance_object.calculate_distance(p)

    def hotstart(self, hotstart_directory_regex):
        # Load all populations
        population_paths = glob.glob(hotstart_directory_regex)

        hotstart_particles = []
        for f in population_paths:
            particles = utils.load_pickle(f)
            hotstart_particles.extend(particles)

        # Set new generation
        self.population = hotstart_particles

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
        max_uptake_sampler,
        k_val_sampler,
        output_dir,
        simulator,
        n_particles_batch,
        toxin_interaction_sampler=None,
        init_population_sampler=None,
        population_size=32,
        mutation_probability=0.1,
        epsilon_alpha=0.2,
        filter=None,
    ):
        logger.info(f"Initialising GA")

        self.experiment_name = experiment_name
        self.base_community = base_community
        self.n_particles_batch = n_particles_batch
        self.max_uptake_sampler = max_uptake_sampler
        self.k_val_sampler = k_val_sampler
        self.init_population_sampler = init_population_sampler
        self.toxin_interaction_sampler = toxin_interaction_sampler

        self.output_dir = output_dir
        self.population_size = population_size
        self.distance_object = distance_object
        self.mutation_probability = mutation_probability
        self.epsilon_alpha = epsilon_alpha
        self.simulator = simulator

        self.gen_idx = 0
        self.final_generation = False

        self.filter = filter

        # Generate a list of models that will be assigned
        # to new particles. Avoids repeatedly copying models
        self.models = []
        logger.info(f"Copying base comm")

        for _ in range(self.n_particles_batch):
            proc_models = []
            for pop in self.base_community.populations:
                proc_models.append(copy.deepcopy(pop.model))
            self.models.append(proc_models)

        logger.info(f"Deleting basecomm models")
        for pop in self.base_community.populations:
            del pop.model

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

        particle_distances = np.vstack(particle_distances)

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

            self.update_epsilon(self.population)

            output_path = f"{self.output_dir}particles_{self.experiment_name}_gen_{self.gen_idx}.pkl"
            self.save_particles(self.population, output_path)
            self.gen_idx += 1

        # Core genetic algorithm loop
        while not self.final_generation:
            batch_idx = 0
            accepted_particles = []

            if self.distance_object.final_epsilon == self.distance_object.epsilon:
                self.final_generation = True

            while len(accepted_particles) <= self.population_size:
                logger.info(
                    f"Gen: {self.gen_idx}, batch: {batch_idx}, epsilon: {self.distance_object.epsilon}, accepted: {len(accepted_particles)}, mem usage (mb): {utils.get_mem_usage()}"
                )

                logger.info(f"Performing crossover...")
                batch_particles = []

                while len(batch_particles) <= self.n_particles_batch:
                    # Generate new batch by crossover
                    candidate_particles = self.crossover(
                        self.n_particles_batch, self.population
                    )

                    # Mutate batch
                    self.mutate_parameterwise(candidate_particles)

                    for p in candidate_particles:
                        p.set_init_y()

                    if self.filter is not None:
                        candidate_particles = self.filter.filter_particles(
                            candidate_particles
                        )

                    batch_particles.extend(candidate_particles)

                batch_particles = batch_particles[: self.n_particles_batch]

                logger.info(f"Simulating particles...")
                # Simulate
                self.simulator.simulate_particles(
                    batch_particles, n_processes=n_processes, parallel=parallel
                )

                clear_cache()

                logger.info(f"Selecting particles...")
                # Select particles
                batch_accepted = self.selection(batch_particles)

                self.delete_particle_fba_models(batch_particles)
                accepted_particles.extend(batch_accepted)

                batch_idx += 1

            accepted_particles = accepted_particles[: self.population_size]

            output_path = f"{self.output_dir}particles_{self.experiment_name}_gen_{self.gen_idx}.pkl"
            self.save_particles(accepted_particles, output_path)

            # Set new population
            self.population = accepted_particles
            # Update epsilon
            self.update_epsilon(self.population)
            self.gen_idx += 1

        output_path = f"{self.output_dir}particles_{self.experiment_name}_final_gen_{self.gen_idx}.pkl"
        self.save_particles(accepted_particles, output_path)
