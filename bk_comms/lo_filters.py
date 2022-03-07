from bk_comms import utils
import numpy as np
import copy
from loguru import logger
import gc


class Filter:
    def filter_particles(self, particles):
        return self.filter(particles)

    def filter_growth_rates(self, particles):
        filtered_particles = []

        for p in particles:
            p.sim_step(p.init_y)
            keep = True

            for pop_idx, _ in enumerate(p.populations):
                population = p.populations[pop_idx]

                for name_idx, name in enumerate(self.population_names):
                    if name == population.name:
                        growth_key = self.growth_keys[name_idx]
                        df = population.model.optimize().to_frame()
                        df["name"] = df.index

                        df.reset_index(drop=True, inplace=True)
                        biomass_flux = df.loc[df["name"] == growth_key][
                            "fluxes"
                        ].values[0]
                        if (
                            biomass_flux >= self.min_growth[name_idx]
                            and biomass_flux <= self.max_growth[name_idx]
                        ):
                            print(name, biomass_flux)
                        else:
                            keep = False
                            break

                if not keep:
                    break

            if keep:
                print("keep")
                print("")
                filtered_particles.append(p)

        return filtered_particles


class ViableGrowthFilter(Filter):
    def __init__(self, population_names, growth_keys, min_growth, max_growth):
        self.population_names = population_names
        self.growth_keys = growth_keys

        self.min_growth = min_growth
        self.max_growth = max_growth

    def filter(self, particles):
        return self.filter_growth_rates(particles)


class ViableGrowthCombineParticles(Filter):
    """
    Updates the parameter of a particle from a randomly selected particle
    of another
    """

    def __init__(
        self,
        input_experiment_dirs,
        epsilon,
        population_names,
        growth_keys,
        min_growth,
        max_growth,
    ):
        self.input_experiment_dirs = input_experiment_dirs
        self.population_names = population_names
        self.growth_keys = growth_keys
        self.min_growth = min_growth
        self.max_growth = max_growth

        repeat_prefix = "run_"
        run_dirs = [
            utils.get_experiment_repeat_directories(
                exp_dir=x, repeat_prefix=repeat_prefix
            )
            for x in self.input_experiment_dirs
        ]

        # self.input_particles = [utils.load_all_particles(x) for x in run_dirs]
        self.input_particles = []

        for idx, x in enumerate(run_dirs):
            logger.info(f"Loading {x},  mem usage (mb): {utils.get_mem_usage()}")
            gc.collect()

            filtered_particles = utils.load_all_particles(x, epsilon[idx])

            self.input_particles.append(filtered_particles)

        # Clean up unwanted data
        for idx, _ in enumerate(self.input_particles):
            for p in self.input_particles[idx]:
                del p.sol
                del p.t
                del p.media_df
                p.set_init_y()

        self.filter = self.filter

    def filter(self, particles):
        self.update_particle_parameters(particles)
        particles = self.filter_growth_rates(particles)

        return particles

    def update_particle_parameters(self, particles):
        for p in particles:
            for idx, _ in enumerate(self.input_particles):
                input_particle = np.random.choice(self.input_particles[idx])
                p.update_parameters_from_particle(
                    input_particle, model_names=input_particle.model_names
                )
