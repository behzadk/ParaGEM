import multiprocessing
import copy
import sampling
import pickle
import numpy as np
import multiprocessing as mp
from matplotlib.backends.backend_pdf import PdfPages
import time
from loguru import logger
import sys

def simulate_particles(particles, n_processes=4, parallel=True):
    if parallel:
        p = mp.Pool(n_processes)
        mp_solutions = p.map(sim_community, particles)

        for idx in range(len(particles)):
            particles[idx].sol = mp_solutions[idx][0]
            particles[idx].t = mp_solutions[idx][1]

    else:
        for p in particles:
            output = sim_community(p)
            p.sol = output[0]
            p.t = output[1]


def filter(particles):
    filtered_particles = []

    for p in particles:
        p.sim_step(p.init_y)

        df = p.populations[0].model.optimize().to_frame()
        df["name"] = df.index
        df.reset_index(drop=True, inplace=True)

        ser_flux = df.loc[df["name"] == "EX_ser__L_e"]["fluxes"].values[0]
        biomass_flux = df.loc[df["name"] == "BIOMASS_SC5_notrace"]["fluxes"].values[0]

        if ser_flux > 0 and biomass_flux > 0:
            filtered_particles.append(p)

    return filtered_particles


def sim_community(community):
    sol, t = community.simulate_community("vode")
    return [sol, t]


class ParameterEstimation:
    def init_particles(self, n_particles):
        particles = np.zeros(shape=n_particles, dtype=object)

        for i in range(n_particles):
            # Make independent copy of base community
            comm = copy.deepcopy(self.base_community)

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

            for pop in p_copy.populations:
                del pop.model

            particles_out.append(p_copy)

        logger.info(f'Saving checkpoint {output_path}')

        with open(f"{output_path}", "wb") as handle:
            pickle.dump(particles_out, handle)        

    def save_checkpoint(self, output_dir):
        time_stamp = time.strftime("%Y-%m-%d_%H%M%S")
        output_path = f'{output_dir}/{self.experiment_name}_checkpoint_{time_stamp}.pkl'

        logger.info(f'Saving checkpoint {output_path}')

        with open(output_path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load_checkpoint(cls, checkpoint_path):
        logger.info(f'Loading checkpoint {checkpoint_path}')

        with open(checkpoint_path, 'rb') as f:
            return pickle.load(f)

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
    ):
        self.experiment_name = experiment_name
        self.base_community = base_community
        self.n_processes = n_particles_batch
        self.n_particles_batch = n_particles_batch
        self.max_uptake_sampler = max_uptake_sampler
        self.k_val_sampler = k_val_sampler
        self.output_dir = "./output/exp_test/"
        self.population_size = population_size
        self.distance_object = distance_object
        self.mutation_probability = mutation_probability
        self.epsilon_alpha = epsilon_alpha

        self.gen_idx = 0
        self.final_generation = False


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
        batch_particles = np.zeros(n_particles, dtype=object)

        for p_batch_idx in range(n_particles):
            # Randomly choose two particles from the population
            male_vec = np.random.choice(population).generate_parameter_vector()
            female_vec = np.random.choice(population).generate_parameter_vector()

            # Generate child parameter vector by uniform crossover
            child_vec = np.zeros(len(female_vec))
            for idx in range(len(female_vec)):
                child_vec[idx] = np.random.choice([female_vec[idx][0], male_vec[idx][0]])

            # Generate child paticle
            child_particle = copy.deepcopy(self.base_community)
            child_particle.load_parameter_vector(child_vec)

            batch_particles[p_batch_idx] = child_particle

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

    def gen_initial_population(self):
        logger.info('Generating initial population')

        # Generate initial population
        accepted_particles = []
        batch_idx = 0
        while len(accepted_particles) < self.population_size:
            particles = self.init_particles(self.n_particles_batch)
            particles = filter(particles)

            simulate_particles(particles, n_processes=self.n_processes, parallel=True)

            accepted_particles.extend(self.selection(particles))

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

    
    def run(self):
        logger.info('Running genetic algorithm')

        if self.gen_idx == 0:
            # Generate initial population
            self.population = self.gen_initial_population()
            self.gen_idx += 1

            self.save_checkpoint(self.output_dir)
            exit()

        # Core genetic algorithm loop
        while not self.final_generation:
            batch_idx = 0
            accepted_particles = []

            if self.distance_object.final_epsion == self.distance_object.epsilon:
                final_generation = True

            while len(accepted_particles) < self.population_size:
                print(
                    "Gen: ",
                    self.gen_idx,
                    "batch: ",
                    batch_idx,
                    "epsilon: ",
                    self.distance_object.epsilon,
                    "accepted: ",
                    len(accepted_particles),
                )
                # Generate new batch by crossover
                batch_particles = self.crossover(self.n_particles_batch, self.population)

                # Mutate batch
                self.mutate_particles(batch_particles)


                print("Simulating... ")
                # Simulate
                simulate_particles(
                    batch_particles, n_processes=self.n_processes, parallel=True
                )

                print("Selecting... ")

                # Select particles
                batch_accepted = self.selection(batch_particles)
                accepted_particles.extend(batch_accepted)

                batch_idx += 1

            accepted_particles = accepted_particles[: self.population_size]

            output_path = f'{self.output_dir}/particles_{self.experiment_name}_gen_{self.gen_idx}.pkl'
            self.save_particles(accepted_particles, output_path)

            # Set new population
            self.population = accepted_particles

            # Update epsilon
            self.update_epsilon(self.population)
            self.gen_idx += 1

        output_path = f'{self.output_dir}/particles_{self.experiment_name}_final_gen_{gen_idx}.pkl'
        self.save_particles(particles, output_path)
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
            simulate_particles(particles, n_processes=self.n_processes, parallel=True)

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
