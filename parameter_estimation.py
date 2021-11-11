import copy
import pickle
import numpy as np

# import multiprocessing as mp
import multiprocess as mp
from pathos.multiprocessing import ProcessPool
from matplotlib.backends.backend_pdf import PdfPages
import time
from loguru import logger
from pathlib import Path
import gc
import psutil
import os
import utils
from sympy.core.cache import *
from guppy import hpy
import dask
import functools
from dask.distributed import Client
# pool = ProcessPool(nodes=8)

def clear_lrus():
    gc.collect()
    wrappers = [a for a in gc.get_objects() if
        isinstance(a, functools._lru_cache_wrapper)]

    for wrapper in wrappers:
        wrapper.cache_clear()

def simulate_particles(particles, dask_client=None, parallel=True):
    if parallel:
        print("running parallel")
        pool = mp.get_context("spawn").Pool(8, maxtasksperchild=1)
        pool.imap(sim_community, particles)
        futures_mp_sol = pool.imap(sim_community, particles)
        for particle in particles:
            try:
                sol, t = futures_mp_sol.next(timeout=45.0)
                particle.sol = sol
                particle.t = t
            except mp.context.TimeoutError:
                print("TIMEOUT ERROR")
                print("time out closing")
                pool.close()
                print("Assigining solutions for unfinished particles")
                particle.sol = None
                particle.t = None
                for p in particles:
                    if not hasattr(p, 'sol'):
                        particle.sol = None
                        particle.t = None

    else:
        p_idx = 0
        for p in particles:
            print(f"Simulating particle idx: {p_idx}")
            output = sim_community(p)
            p.sol = output[0]
            p.t = output[1]
            p_idx += 1


def filter(particles):
    filtered_particles = []

    for p in particles:
        p.sim_step(p.init_y)

        df = p.populations[0].model.optimize().to_frame()
        df["name"] = df.index
        df.reset_index(drop=True, inplace=True)

        ser_flux = df.loc[df["name"] == "EX_ser__L_e"]["fluxes"].values[0]
        # ala_flux = df.loc[df["name"] == "EX_ala__L_e"]["fluxes"].values[0]
        # glu_flux = df.loc[df["name"] == "EX_glu__L_e"]["fluxes"].values[0]
        # gly_flux = df.loc[df["name"] == "EX_gly_e"]["fluxes"].values[0]
        biomass_flux = df.loc[df["name"] == "BIOMASS_SC5_notrace"]["fluxes"].values[0]

        # num_efflux = sum([1 for f in [ser_flux, ala_flux, glu_flux, gly_flux] if f > 0])

        if biomass_flux > 0:
            # if ser_flux > 0:
            filtered_particles.append(p)

    return filtered_particles


def sim_community(community):
    sol, t = community.simulate_community("vode")
    print("returning sim")
    return [sol, t]

   


class ParameterEstimation:
    def init_particles(self, n_particles):
        particles = np.zeros(shape=n_particles, dtype=object)

        for i in range(n_particles):
            # Make independent copy of base community
            comm = copy.deepcopy(self.base_community)

            # Assign population models
            for idx, pop in enumerate(comm.populations):
                pop.model = self.models[i][idx]

            array_size = [len(comm.populations), len(comm.dynamic_compounds)]
            # Sample new max uptake matrix
            max_exchange_mat = self.max_uptake_sampler.sample(size=array_size)
            comm.set_max_exchange_mat(max_exchange_mat)

            #  Sample new K value matrix
            k_val_mat = self.k_val_sampler.sample(size=array_size)
            comm.set_k_value_matrix(k_val_mat)

            particles[i] = comm

        return particles

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

    def delete_particle_fba_models(self, particles):
        for part in particles:
            for p in part.populations:
                del p.model
            # del part.sol
            # del part.t

        gc.collect()

    @classmethod
    def load_checkpoint(cls, checkpoint_path):
        logger.info(f"Loading checkpoint {checkpoint_path}")

        with open(checkpoint_path, "rb") as f:
            return pickle.load(f)


class SpeedTest(ParameterEstimation):
    def __init__(self, particles_path, base_community):

        with open(particles_path, "rb") as f:
            particles = pickle.load(f)

        particles = particles
        self.particles = particles
        self.base_community = base_community

        # Generate a list of models that will be assigned
        # to new particles. Avoids repeatedly copying models
        self.models = []
        for _ in range(len(self.particles)):
            proc_models = []
            for pop in self.base_community.populations:
                proc_models.append(pop.model)
            self.models.append(proc_models)

        for pop in self.base_community.populations:
            del pop.model

        for particle_idx, comm in enumerate(self.particles):
            # Assign population models
            for pop_idx, pop in enumerate(comm.populations):
                pop.model = self.models[particle_idx][pop_idx]

    def speed_test(self, dask_client):
        start_time = time.time()
        simulate_particles(
            self.particles,
            dask_client=dask_client,
            parallel=True,
        )
        end_time = time.time()
        # dask_client.restart()

        for p in self.particles:
            print(p.sol.shape, p.t.shape)

        print("logging")
        logger.info(f"Dask parallel: {end_time - start_time}")

        start_time = time.time()
        simulate_particles(self.particles, parallel=False)
        end_time = time.time()

        logger.info(f"Serial: {end_time - start_time}")


class GeneticAlgorithm(ParameterEstimation):
    def __init__(
        self,
        experiment_name,
        distance_object,
        base_community,
        max_uptake_sampler,
        k_val_sampler,
        output_dir,
        n_particles_batch=32,
        population_size=32,
        mutation_probability=0.1,
        epsilon_alpha=0.2,
        parallel=True,
    ):
        self.experiment_name = experiment_name
        self.base_community = base_community
        self.n_processes = n_particles_batch
        self.n_particles_batch = n_particles_batch
        self.max_uptake_sampler = max_uptake_sampler
        self.k_val_sampler = k_val_sampler
        self.output_dir = output_dir
        self.population_size = population_size
        self.distance_object = distance_object
        self.mutation_probability = mutation_probability
        self.epsilon_alpha = epsilon_alpha

        self.gen_idx = 0
        self.final_generation = False

        # Generate a list of models that will be assigned
        # to new particles. Avoids repeatedly copying models
        self.models = []
        for _ in range(self.n_particles_batch):
            proc_models = []
            for pop in self.base_community.populations:
                proc_models.append(copy.deepcopy(pop.model))
            self.models.append(proc_models)

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

    def crossover(self, n_particles, population):
        batch_particles = self.init_particles(n_particles)

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

    def mutate_particles(self, batch_particles):
        for particle in batch_particles:
            particle_param_vec = particle.generate_parameter_vector()

            # Generate a 'mutation particle'
            mut_particle = self.init_particles(1)[0]
            mut_params_vec = mut_particle.generate_parameter_vector()

            for idx in range(len(particle_param_vec)):
                particle_param_vec[idx] = np.random.choice(
                    [particle_param_vec[idx][0], mut_params_vec[idx][0]],
                    p=[1 - self.mutation_probability, self.mutation_probability],
                )

            particle.load_parameter_vector(particle_param_vec)

    def gen_initial_population(self, dask_client, parallel):
        logger.info("Generating initial population")

        # Generate initial population
        accepted_particles = []
        batch_idx = 0
        while len(accepted_particles) < self.population_size:
            logger.info(f"Initial accepted particles: {len(accepted_particles)}")

            particles = self.init_particles(self.n_particles_batch)
            particles = filter(particles)
            if len(particles) == 0:
                continue

            simulate_particles(
                particles,
                parallel=parallel,
                dask_client=dask_client,
            )

            from pympler import muppy
            all_objects = muppy.get_objects()

            from pympler import summary
            sum1 = summary.summarize(all_objects)
            summary.print_(sum1)   
            logger.info(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
            logger.info(summary.print_(sum1))

            accepted_particles.extend(self.selection(particles))
            self.delete_particle_fba_models(particles)

            batch_idx += 1

        return accepted_particles

    def update_epsilon(self, particles):
        particle_distances = []
        # Collate distances
        for p in particles:
            particle_distances.append(p.distance)

        particle_distances = np.vstack(particle_distances)

        print(particle_distances.shape)

        new_epsilon = []
        # For each distance, update epsilon
        for dist_idx in range(particle_distances.shape[1]):
            dists = particle_distances[:, dist_idx]
            dists.sort()

            # Trim distances to largest epsilon out of smallest x
            dists = dists[: int(self.population_size * self.epsilon_alpha)]
            new_epsilon = max(self.distance_object.final_epsion[dist_idx], dists[-1])

            self.distance_object.epsilon[dist_idx] = new_epsilon

    def run(self, dask_client=None, parallel=False):
        logger.info("Running genetic algorithm")

        if self.gen_idx == 0:
            # Generate initial population
            self.population = self.gen_initial_population(dask_client, parallel)
            self.gen_idx += 1

            self.save_checkpoint(self.output_dir)

        # Core genetic algorithm loop
        while not self.final_generation:
            # dask_client.shutdown()
            # dask_client = Client(processes=True, 
            # n_workers=6, threads_per_worker=1, silence_logs=False,
            # timeout="3600s")

            batch_idx = 0
            accepted_particles = []

            if self.distance_object.final_epsion == self.distance_object.epsilon:
                self.final_generation = True

            while len(accepted_particles) < self.population_size:
                logger.info(
                    f"Gen: {self.gen_idx}, batch: {batch_idx}, epsilon: {self.distance_object.epsilon}, accepted: {len(accepted_particles)}, mem usage (mb): {utils.get_mem_usage()}"
                )

                logger.info(f"Performing crossover...")

                # batch_particles = self.population[0:self.n_particles_batch]

                # Generate new batch by crossover
                batch_particles = self.crossover(
                    self.n_particles_batch, self.population
                )

                logger.info(f"Mutating particles...")

                # Mutate batch
                self.mutate_particles(batch_particles)

                logger.info(f"Simulating particles...")
                output_path = f"{self.output_dir}particles_{self.experiment_name}_latest_batch.pkl"
                self.save_particles(batch_particles, output_path)

                # Simulate
                simulate_particles(
                    batch_particles,
                    parallel=parallel,
                    dask_client=dask_client,
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
            self.save_checkpoint(self.output_dir)
            # Update epsilon
            self.update_epsilon(self.population)
            self.gen_idx += 1

        output_path = f"{self.output_dir}particles_{self.experiment_name}_final_gen_{self.gen_idx}.pkl"
        self.save_particles(accepted_particles, output_path)
        self.save_checkpoint(self.output_dir)


class RejectionAlgorithm(ParameterEstimation):
    def __init__(
        self,
        distance_object,
        base_community,
        max_uptake_sampler,
        k_val_sampler,
        n_particles_batch=32,
        population_size=32,
    ):
        self.base_community = base_community
        self.n_processes = n_particles_batch
        self.n_particles_batch = n_particles_batch
        self.max_uptake_sampler = max_uptake_sampler
        self.k_val_sampler = k_val_sampler
        self.output_dir = "./output/exp_test/"
        self.population_size = population_size
        self.distance_object = distance_object

    def assess_particles(self, particles):
        acceptance_status = []
        for p in particles:
            accepted_flag, distance = self.distance_object.assess_particle(p)

            with np.printoptions(precision=4, suppress=True):
                print(f"{accepted_flag}\t {distance}")

            p.distance = distance
            p.accepted_flag = accepted_flag

            acceptance_status.append(accepted_flag)

        return acceptance_status

        # np.save(f"{self.output_dir}{output_name}.npy",  out_dict)

    def run(self, output_name):
        accepted_particles = []

        batch_idx = 0
        while len(accepted_particles) < self.population_size:
            print(f"Name: {output_name}")
            print(f"Batch {batch_idx}, Accepted particles: {len(accepted_particles)}")

            print(f"\t Initialising batch {batch_idx}")
            particles = self.init_particles(self.n_particles_batch)
            particles = filter(particles)

            print(f"\t Simulating batch {batch_idx}")
            simulate_particles(particles, parallel=True)

            acceptance_status = self.assess_particles(particles)

            for idx, p in enumerate(particles):
                if acceptance_status[idx]:
                    accepted_particles.append(p)

            batch_idx += 1

        if len(accepted_particles) > 0:
            # Trim population to exact max pop size
            accepted_particles = accepted_particles[: self.population_size]

            with PdfPages(f"{self.output_dir}{output_name}_{batch_idx}.pdf") as pdf:
                for acc_idx, p in enumerate(accepted_particles):
                    self.distance_object.plot_community(
                        p, pdf=pdf, prefix=output_name, comm_idx=acc_idx, batch_idx=0
                    )

            # Save accepted particles
            self.save_particles(accepted_particles, output_name)
