import numpy as np
from bk_comms import utils
from loguru import logger
import gc


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
        sigma = self.alpha / np.sqrt(1.0 + self.alpha ** 2)
        u0 = np.random.standard_normal(size)
        v = np.random.standard_normal(size)
        u1 = (sigma * u0 + np.sqrt(1.0 - sigma ** 2) * v) * self.scale
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


class SampleCombinationParticles:
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

        particles_dict = self.load_particles(run_dirs, self.population_names, epsilon)
        self.params_dict = self.generate_parameter_dict(particles_dict)

        self.particle_counts = {}
        for p in self.population_names:
            self.particle_counts[p] = len(self.params_dict[p]["k_vals"])

    def load_particles(self, run_dirs, population_names, epsilon):

        particles = {}

        for idx, x in enumerate(run_dirs):
            logger.info(f"Loading {x},  mem usage (mb): {utils.get_mem_usage()}")

            filtered_particles = utils.load_all_particles(x, epsilon[idx])

            # Clean up unwanted data
            for p in filtered_particles:
                del p.sol
                del p.t
                del p.media_df
                p.set_init_y()

            gc.collect()

            particles[population_names[idx]] = filtered_particles

        return particles

    def generate_parameter_dict(self, particles_dict):
        # Unpack particles into parameters dictionary. Each key refers
        # to a population name
        params = {}

        for key in particles_dict:
            params[key] = {}
            params[key]["k_vals"] = []
            params[key]["max_exchange_mat"] = []
            params[key]["initial_population"] = []

            for p in particles_dict[key]:
                params[key]["k_vals"].append(p.k_vals)
                params[key]["max_exchange_mat"].append(p.max_exchange_mat)
                params[key]["initial_population"].append(p.init_population_values)

        return params

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
        legal_data_fields = ["k_vals", "max_exchange_mat", "initial_population"]
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
