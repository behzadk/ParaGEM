import multiprocessing
import copy
import sampling
import pickle
import numpy as np
import multiprocessing as mp
from matplotlib.backends.backend_pdf import PdfPages

def sim_community(community):
    sol, t = community.simulate_community("vode")
    return [sol, t]


class RejectionAlgorithm:
    def __init__(self, distance_object, 
    base_community, 
    max_uptake_sampler, 
    k_val_sampler,
    n_particles_batch=32,
    max_population_size=32):
        self.base_community = base_community

        self.n_processes = n_particles_batch
        self.n_particles_batch = n_particles_batch
        self.max_uptake_sampler = max_uptake_sampler

        self.k_val_sampler = k_val_sampler

        self.output_dir = "./output/exp_test/"

        self.max_population_size = max_population_size

        self.distance_object = distance_object

    def init_particles(self):
        particles = np.zeros(shape=self.n_particles_batch, dtype=object)

        for i in range(self.n_particles_batch):
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

    def simulate_particles(self, particles, parallel=True):
        if parallel:
            num_processes = self.n_processes
            p = mp.Pool(num_processes)
            mp_solutions = p.map(sim_community, particles)

            for idx in range(len(particles)):
                particles[idx].sol = mp_solutions[idx][0]
                particles[idx].t = mp_solutions[idx][1]

        else:
            for p in particles:
                output = sim_community(p)
                p.sol = output[0]
                p.t = output[1]

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

    def save_particles(self, particles, output_name):
        out_dict = {}
        for idx, p in enumerate(particles):
            out_dict[idx] = {}
            out_dict[idx]['k_vals'] = p.k_vals
            out_dict[idx]['max_exchange_mat'] = p.max_exchange_mat
            out_dict[idx]['distance'] = p.distance
            out_dict[idx]['accepted_flag'] = p.accepted_flag

        np.save(f"{self.output_dir}{output_name}.npy",  out_dict)    

    def run(self, output_name):
        accepted_particles = []

        batch_idx = 0
        while len(accepted_particles) < self.max_population_size:
            print(f"Name: {output_name}")
            print(f"Batch {batch_idx}, Accepted particles: {len(accepted_particles)}")

            print(f"\t Initialising batch {batch_idx}")
            particles = self.init_particles()

            print(f"\t Simulating batch {batch_idx}")
            self.simulate_particles(particles, parallel=True)

            acceptance_status = self.assess_particles(particles)
            
            for idx, p in enumerate(particles):
                if acceptance_status[idx]:
                    accepted_particles.append(p)
            
            batch_idx += 1

            if batch_idx > 250:
                break
        
        if len(accepted_particles) > 0:
            # Trim population to exact max pop size
            accepted_particles = accepted_particles[: self.max_population_size]

            with PdfPages(f'{self.output_dir}{output_name}_{batch_idx}.pdf') as pdf:
                for acc_idx, p in enumerate(accepted_particles):
                    self.distance_object.plot_community(p, pdf=pdf, prefix=output_name, comm_idx=acc_idx, batch_idx=0)

            # Save accepted particles
            self.save_particles(accepted_particles, output_name)
            
