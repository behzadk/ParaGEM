import numpy as np
from bk_comms import utils
from loguru import logger
import gc
from glob import glob


class SampleDistribution:
    def sample(self, size):
        return self.distribution(size)


class MultiDistribution(SampleDistribution):
    def __init__(self, dist_1, dist_2, prob_dist_1):
        self.dist_1 = dist_1
        self.dist_2 = dist_2

        self.prob_dist_1 = prob_dist_1

        self.distribution = self.random_multi_dist

    def random_multi_dist(self, size):
        x = np.zeros(size)

        for i in range(size[0]):
            for j in range(size[1]):
                if np.random.uniform() < self.prob_dist_1:
                    x[i][j] = self.dist_1.sample(1)[0]

                else:
                    x[i][j] = self.dist_2.sample(1)[0]

        return x


class SampleSkewNormal(SampleDistribution):
    def __init__(self, loc, scale, alpha, clip_above_zero=False, clip_below_zero=False):
        self.alpha = alpha
        self.loc = loc
        self.scale = scale
        self.clip_above_zero = clip_above_zero
        self.clip_below_zero = clip_below_zero

        self.distribution = self.random_skew_normal

    def random_skew_normal(self, size):
        sigma = self.alpha / np.sqrt(1.0 + self.alpha**2)
        u0 = np.random.standard_normal(size)
        v = np.random.standard_normal(size)
        u1 = (sigma * u0 + np.sqrt(1.0 - sigma**2) * v) * self.scale
        u1[u0 < 0] *= -1
        u1 = u1 + self.loc

        if self.clip_below_zero:
            u1 = u1.clip(0)

        if self.clip_above_zero:
            u1 = np.clip(u1, a_min=None, a_max=0)

        return u1


class SampleUniform(SampleDistribution):
    def __init__(self, min_val, max_val, distribution_type="uniform"):
        self.min_val = min_val
        self.max_val = max_val
        self.distribution_type = distribution_type

        dist_options = ["uniform", "log_uniform"]
        assert (
            self.distribution_type in dist_options
        ), f"Error distribution, {self.distribution_type}, not in {dist_options}"

        self.distribution = self.sample_random_matrix

    def sample_random_matrix(self, size):

        if self.distribution_type == "uniform":
            mat = np.random.uniform(self.min_val, self.max_val, size=size)

        elif self.distribution_type == "log_uniform":
            is_negative = False

            if self.min_val < 0 and self.max_val < 0:
                is_negative = True

            elif self.min_val < 0 or self.max_val < 0:
                raise ValueError

            mat = np.exp(
                np.random.uniform(
                    np.log(abs(self.min_val)), np.log(abs((self.max_val))), size=size
                )
            )

            if is_negative:
                mat = mat * -1

        else:
            raise ValueError("incorrect distribution definition")

        return mat


class SampleUniformConstant(SampleDistribution):
    def __init__(self, min_val, max_val, distribution_type="uniform"):
        self.min_val = min_val
        self.max_val = max_val
        self.distribution_type = distribution_type

        dist_options = ["uniform", "log_uniform"]
        assert (
            self.distribution_type in dist_options
        ), f"Error distribution, {self.distribution_type}, not in {dist_options}"

        self.distribution = self.sample_constant_matrix

    def sample_constant_matrix(self, size):
        if self.distribution_type == "uniform":
            value = np.random.uniform(self.min_val, self.max_val)
            mat = np.ones(shape=size)
            mat *= value

        elif self.distribution_type == "log_uniform":
            is_negative = False

            if self.min_val < 0 and self.max_val < 0:
                is_negative = True

            elif self.min_val < 0 or self.max_val < 0:
                raise ValueError

            value = np.exp(
                np.random.uniform(
                    np.log(abs(self.min_val)), np.log(abs((self.max_val)))
                )
            )

            mat = np.ones(shape=size)
            mat *= value

            if is_negative:
                mat = mat * -1

        else:
            raise ValueError("incorrect distribution definition")

        return mat


class SampleCombinationParticles:
    def __init__(
        self,
        input_particles_parameter_vector_regex,
        population_names,
    ):
        self.input_particles_parameter_vector_regex = (
            input_particles_parameter_vector_regex
        )
        self.population_names = population_names

        particles_input_info = {}

        for reg_ex, pop_name in zip(
            self.input_particles_parameter_vector_regex, self.population_names
        ):
            particles_input_info[pop_name] = reg_ex

        self.particle_counts = {}
        for p in self.population_names:
            self.particle_counts[p] = 0

        self.params_dict = self.load_params_dict(
            particles_input_info, self.population_names
        )

    def load_particles(self, particles_input_info, population_names, epsilon):

        particles = {}
        for species_idx, species_particle_paths in enumerate(particle_paths):
            logger.info(
                f"Loading {species_particle_paths},  mem usage (mb): {utils.get_mem_usage()}"
            )

            filtered_particles = []

            for p_path in species_particle_paths:
                particle_population = utils.load_pickle(p_path)
                logger.info(
                    f"Loaded {species_particle_paths},  particles: {len(particle_population)}, mem usage (mb): {utils.get_mem_usage()}"
                )
                particle_population = utils.filter_particles_by_distance(
                    particle_population, epsilon[species_idx]
                )
                logger.info(
                    f"After filtering,  particles: {len(particle_population)}, mem usage (mb): {utils.get_mem_usage()}"
                )

                filtered_particles.extend(particle_population)

            # Clean up unwanted data
            for p in filtered_particles:
                del p.sol
                del p.t
                del p.media_df
                p.set_init_y()

            gc.collect()

            particles[population_names[species_idx]] = filtered_particles
            self.particle_counts[population_names[species_idx]] = len(
                filtered_particles
            )

            logger.info(
                f"{population_names[species_idx]},  particles loaded: {len(particles[population_names[species_idx]])}, mem usage (mb): {utils.get_mem_usage()}"
            )

        return particles

    def load_params_dict(self, particles_input_info, population_names):
        particles_dict = {}

        for pop_name in population_names:
            particles_dict[pop_name] = {}
            particles_dict[pop_name]["k_vals"] = []
            particles_dict[pop_name]["max_exchange_mat"] = []
            particles_dict[pop_name]["initial_population"] = []
            particles_dict[pop_name]["biomass_rate_constraints"] = []

            particles_dict[pop_name]["toxin_mat"] = []

        for pop_name in population_names:
            params_dirs = glob(particles_input_info[pop_name])
            for d in params_dirs:
                # Make parameter paths
                init_populations_arr = np.load(f"{d}/particle_init_populations.npy")
                k_values_arr = np.load(
                    f"{d}/particle_k_values.npy",
                )
                max_exchange_arr = np.load(
                    f"{d}/particle_max_exchange.npy",
                )
                toxin_arr = np.load(f"{d}/particle_toxin.npy")
                biomass_rate_constr = np.load(
                    f"{d}/particle_biomass_rate_constr_vectors.npy"
                )

                for idx, _ in enumerate(init_populations_arr):
                    particles_dict[pop_name]["k_vals"].append(k_values_arr[idx])
                    particles_dict[pop_name]["max_exchange_mat"].append(
                        max_exchange_arr[idx]
                    )
                    particles_dict[pop_name]["initial_population"].append(
                        init_populations_arr[idx]
                    )
                    particles_dict[pop_name]["toxin_mat"].append(toxin_arr[idx])
                    particles_dict[pop_name]["biomass_rate_constraints"].append(
                        biomass_rate_constr[idx]
                    )

            self.particle_counts[pop_name] = len(
                particles_dict[pop_name]["max_exchange_mat"]
            )
            print(self.particle_counts)

        return particles_dict

    def generate_random_index_combination(self):
        index_combination = {}
        for p in self.population_names:
            index_combination[p] = np.random.randint(0, self.particle_counts[p])

        return index_combination

    def sample(
        self,
        model_names,
        data_field,
        index_combination=None,
    ):
        legal_data_fields = [
            "k_vals",
            "max_exchange_mat",
            "initial_population",
            "biomass_rate_constraints",
        ]
        assert (
            data_field in legal_data_fields
        ), f"{data_field} not in legal datafields: {legal_data_fields}"
        output_sample = {}

        if index_combination is None:
            index_combination = self.generate_random_index_combination()

        sampled_data = []

        for name in model_names:
            idx = index_combination[name]
            output_data = self.params_dict[name][data_field][idx].reshape(-1)

            sampled_data.append(output_data)

        return np.array(sampled_data)
