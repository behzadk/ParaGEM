import numpy as np

from loguru import logger
from pathlib import Path

import pygmo


from paragem.utils import logger_wraps
from paragem.utils import save_particles
import cobra
from paragem.parameter_optimization.base import ParameterOptimization

class NSGAII(ParameterOptimization):
    """
    Non-dominated sorting genetic algorithm II
    """
    def __init__(
        self,
        experiment_name: str,
        distance_object,
        base_community: cobra.core.model.Model,
        experiment_dir: str,
        simulator,
        sim_media_names,
        crossover_type="parameterwise",
        mutate_type="parameterwise",
        n_particles_batch=8,
        particle_filter=None,
        mutation_probability=0.0,
        generation_idx=0,
        run_idx=0,
        hotstart_particles_regex=None,
        population_size=32,
        max_generations=10,
    ):
        """
        Args:
            experiment_name : experiment_name
            distance_object : distance_object
            base_community : community to be parameterised
            experiment_dir : experiment directory
            simulator : Simulator object (used to simulate community)
            sim_media_names : list of media names to simulate the community in
            crossover_type : string defining type of crossover function to use
            mutate_type : string defining mutation function to use
            n_particles_batch : number of particles to simulate per batch
            particle_filter : function to filter particles
            mutation_probability : probability of mutation occuring
            generation_idx : generation index
            run_idx : run index
            hotstart_particles_regex : pathname glob expression e.g `experiment_dir/generation_*/run_*`. If None, hotstart is not conducted.
            population_size : population size
            max_generations : max number of generations to run for
        """

        self.experiment_name = experiment_name
        self.base_community = base_community

        self.n_particles_batch = n_particles_batch

        self.sim_media_names = sim_media_names
        self.experiment_dir = experiment_dir
        self.population_size = population_size
        self.distance_object = distance_object
        self.mutation_probability = mutation_probability
        self.simulator = simulator

        self.gen_idx = int(generation_idx)
        self.final_generation = False
        self.run_idx = run_idx

        self.output_dir = (
            f"{self.experiment_dir}/generation_{self.gen_idx}/run_{self.run_idx}/"
        )

        if crossover_type == "parameterwise":
            self.crossover = self.crossover_parameterwise

        elif crossover_type == "specieswise":
            self.crossover = self.crossover_species_wise

        else:
            raise ValueError(f"Unknown crossover type: {crossover_type}")

        if mutate_type == "parameterwise":
            self.mutate = self.mutate_parameterwise

        elif mutate_type == "resample_prior":
            self.mutate = self.mutate_resample_from_prior
            # self.mutate = self.mutate_community_from_prior

        else:
            raise ValueError(f"Unknown mutation type: {mutate_type}")

        self.filter = particle_filter

        # Generate model instances for each element in batch
        self.models = self.generate_models_list(n_models=self.n_particles_batch)

        self.max_generations = max_generations

        if not isinstance(hotstart_particles_regex, type(None)):
            self.hotstart_particles(hotstart_particles_regex)

    @logger_wraps()
    def run(self, n_processes=1, parallel=False):
        """
        Run NSGAII algorithm

        Args:
            n_processes : number of processes to use when running in parallel
            parallel : bool to run simulations in parallel
        """

        logger.info("Running NSGAII")

        # Generate first generation
        if self.gen_idx == 0:
            self.output_dir = (
                f"{self.experiment_dir}/generation_{self.gen_idx}/run_{self.run_idx}/"
            )
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)

            self.population = self.gen_initial_population(n_processes, parallel)

            save_particles(
                self.population,
                self.sim_media_names,
                output_dir=f"{self.output_dir}",
            )
            self.gen_idx += 1

        while self.gen_idx < self.max_generations:
            self.output_dir = (
                f"{self.experiment_dir}/generation_{self.gen_idx}/run_{self.run_idx}/"
            )
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)

            logger.info(f"Running generation {self.gen_idx}")
            logger.info("Selecting parents")

            new_offspring = []
            new_parents = []
            batch_idx = 0
            while len(new_offspring) <= self.population_size:

                population_average_distance = np.mean(
                    [max(p.distance) for p in self.population]
                )
                population_mediain_distance = np.median(
                    [max(p.distance) for p in self.population]
                )
                population_min_distance = np.min(
                    [sum(p.distance) for p in self.population]
                )

                logger.info(
                    f"Pop mean distance: {population_average_distance}, pop median distance: {population_mediain_distance}, pop min distance: {population_min_distance}"
                )


                # Select parents population by non-dominated sorting and crowd distance
                parent_particles = self.non_dominated_sort_parent_selection(
                    self.population, self.n_particles_batch
                )

                logger.info("Performing crossover to produce offspring")
                # Perform crossover and mutation to generate offspring of size N
                offspring_particles = self.crossover(
                    self.init_particles(self.n_particles_batch), parent_particles
                )

                logger.info("Mutating offspring")
                self.mutate(offspring_particles, self.mutation_probability)
                offspring_particles = list(offspring_particles)

                logger.info("Simulating offspring")

                logger.info(f"Simulating offspring batch: {batch_idx}")

                for media_idx, media_name in enumerate(self.sim_media_names):
                    [p.set_media_conditions(media_name) for p in offspring_particles]

                    if not isinstance(self.filter, type(None)):
                        offspring_particles = self.filter.filter_particles(
                            offspring_particles
                        )

                    # Simulate particle for media name
                    self.simulator.simulate_particles(
                        offspring_particles,
                        sol_key=media_name,
                        n_processes=n_processes,
                        parallel=parallel,
                    )

                    # Calculate distance for media_name
                    self.calculate_and_set_particle_distances(
                        offspring_particles, media_name
                    )

                offspring_particles = [
                    p for p in offspring_particles if not np.isnan(sum(p.distance))
                ]

                self.delete_particle_fba_models(offspring_particles)

                new_offspring.extend(offspring_particles)
                new_parents.extend(parent_particles)

                batch_idx += 1

            new_offspring = new_offspring[: self.population_size]

            # Combine population of parents and offspring (2N)
            self.population = new_offspring + new_parents

            save_particles(
                self.population,
                self.sim_media_names,
                output_dir=f"{self.output_dir}",
            )
            new_offspring = []
            new_parents = []
            self.gen_idx += 1

        logger.info("Finished")

    def non_dominated_sort_parent_selection(self, population, n_particles):
        """
        Use binary tournament to generate population of parents:
        Randomly select two solutions, keep the better one with
        respect to non-domination rank. Solutions in same front
        are compared by crowding distance.
    
        Args:
            population : a population of particles
            n_particles : number of new particles to generate
        """

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
