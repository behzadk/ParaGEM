import numpy as np

from loguru import logger
from pathlib import Path

import pygmo


from bk_comms.utils import logger_wraps
from bk_comms.utils import save_particles
import cobra
from bk_comms.parameter_optimization.base import ParameterOptimization


class RejectionAlgorithm(ParameterOptimization):
    """Simple rejection algorithm"""

    def __init__(
        self,
        experiment_name: str,
        distance_object,
        base_community: cobra.core.model.Model,
        experiment_dir: str,
        simulator,
        sim_media_names,
        n_particles_batch=8,
        particle_filter=None,
        run_idx=0,
        hotstart_particles_regex=None,
        population_size=32,
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
            particle_filter : function to filter particles
            run_idx : run index
            hotstart_particles_regex : pathname glob expression e.g `experiment_dir/generation_*/run_*`. If None, hotstart is not conducted.
            population_size : population size
        """
        self.experiment_name = experiment_name
        self.base_community = base_community

        self.n_particles_batch = n_particles_batch

        self.sim_media_names = sim_media_names
        self.experiment_dir = experiment_dir
        self.population_size = population_size
        self.distance_object = distance_object
        self.simulator = simulator
        self.run_idx = run_idx

        self.output_dir = f"{self.experiment_dir}/generation_0/run_{self.run_idx}/"

        self.filter = particle_filter

        # Generate model instances for each element in batch
        self.models = self.generate_models_list(n_models=self.n_particles_batch)

        if not isinstance(hotstart_particles_regex, type(None)):
            self.hotstart_particles(hotstart_particles_regex)

    @logger_wraps()
    def run(self, n_processes=1, parallel=False):
        """Run rejection algorithm

        Args:
            n_processes : number of processes to use when running in parallel
            parallel : bool to run simulations in parallel
        """
        logger.info("Running Rejection algorithm")

        accepted_particles = []

        while len(accepted_particles) < self.population_size:
            candidate_particles = self.init_particles(self.n_particles_batch)

            # For each media, simulate and calculate distances
            for media_idx, media_name in enumerate(self.sim_media_names):
                [p.set_media_conditions(media_name) for p in candidate_particles]

                if not isinstance(self.filter, type(None)):
                    candidate_particles = self.filter.filter_particles(
                        candidate_particles
                    )

                # Simulate particle for media name
                self.simulator.simulate_particles(
                    candidate_particles,
                    sol_key=media_name,
                    n_processes=n_processes,
                    parallel=parallel,
                )

                # Calculate distance for media_name
                self.calculate_and_set_particle_distances(
                    candidate_particles, media_name
                )

            candidate_particles = [
                p for p in candidate_particles if not np.isnan(sum(p.distance))
            ]

            self.delete_particle_fba_models(candidate_particles)

            # DO COMPARISON TO DISTANCE THRESHOLD