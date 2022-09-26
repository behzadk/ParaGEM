import numpy as np

from loguru import logger

from paragem.utils import logger_wraps
from paragem.utils import save_particles
import cobra
from paragem.parameter_optimization.base import ParameterOptimization

class SampleSimulate(ParameterOptimization):
    """
    A simple algorithm without parameterisation. Simply samples from the given prior distributions and simulates them.
    All particles are accepted. 

    Can be useful for debugging, testing and sanity checking
    """
    def __init__(
        self,
        experiment_name: str,
        distance_object,
        base_community: cobra.core.model.Model,
        experiment_dir: str,
        simulator,
        sim_media_names,
        n_particles_batch=8,
        generation_idx=0,
        run_idx=0,
        population_size=32,
        particle_filter=None,
    ):
        """
        Args:
            experiment_name : experiment_name
            distance_object : distance_object
            base_community : community to be parameterised
            experiment_dir : experiment directory
            simulator : Simulator object (used to simulate community)
            sim_media_names : list of media names to simulate the community in
            n_particles_batch : number of particles to simulate per batch
            generation_idx : generation index
            run_idx : run index
            population_size : population size
            particle_filter : function to filter particles
        """

        self.experiment_name = experiment_name
        self.base_community = base_community
        self.distance_object = distance_object
        self.filter = particle_filter

        self.n_particles_batch = n_particles_batch

        self.sim_media_names = sim_media_names
        self.experiment_dir = experiment_dir
        self.population_size = population_size
        self.simulator = simulator

        self.gen_idx = int(generation_idx)
        self.run_idx = run_idx

        self.output_dir = (
            f"{self.experiment_dir}/generation_{self.gen_idx}/run_{self.run_idx}/"
        )

        # Generate model instances for each element in batch
        self.models = self.generate_models_list(n_models=self.n_particles_batch)

        self.population = []

    @logger_wraps()
    def run(self, n_processes=1, parallel=False):
        """
        Run NSGAII algorithm

        Args:
            n_processes : number of processes to use when running in parallel
            parallel : bool to run simulations in parallel
        """

        logger.info("Running SampleSimulate")

        while len(self.population) < self.population_size:
            logger.info(f"Accepted particles: {len(self.population)}")

            new_particles = self.init_particles(self.n_particles_batch)

            for media_idx, media_name in enumerate(self.sim_media_names):
                [p.set_media_conditions(media_name) for p in new_particles]

                if not isinstance(self.filter, type(None)):
                    new_particles = self.filter.filter_particles(
                        new_particles
                    )

                # Simulate particle for media name
                self.simulator.simulate_particles(
                    new_particles,
                    sol_key=media_name,
                    n_processes=n_processes,
                    parallel=parallel,
                )

                # Calculate distance for media_name
                self.calculate_and_set_particle_distances(
                    new_particles, media_name
                )

            new_particles = [
                p for p in new_particles if not np.isnan(sum(p.distance))
            ]

            self.delete_particle_fba_models(new_particles)
            self.population.extend(new_particles)

        save_particles(
            self.population,
            self.sim_media_names,
            output_dir=f"{self.output_dir}",
        )
        self.gen_idx += 1

        logger.info("Finished")
