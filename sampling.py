import numpy as np


class MultiDistribution:
    def __init__(self, dist_1, dist_2, prob_dist_1):
        self.dist_1 = dist_1
        self.dist_2 = dist_2

        self.prob_dist_1 = prob_dist_1

    def random_multi_dist(self, size):
        x = np.zeros(size)

        for i in range(size[0]):
            for j in range(size[1]):
                if np.random.uniform() < self.prob_dist_1:
                    x[i][j] = self.dist_1.sample(1)[0]

                else:
                    x[i][j] = self.dist_2.sample(1)[0]

        return x

    def sample(self, size):
        return self.random_multi_dist(size)


class SampleSkewNormal:
    def __init__(self, loc, scale, alpha, clip_above_zero=False, clip_below_zero=False):
        self.alpha = alpha
        self.loc = loc
        self.scale = scale
        self.clip_above_zero = clip_above_zero
        self.clip_below_zero = clip_below_zero

    def random_skew_normal(self, size):
        sigma = self.alpha / np.sqrt(1.0 + self.alpha**2) 
        u0 = np.random.standard_normal(size)
        v = np.random.standard_normal(size)
        u1 = (sigma*u0 + np.sqrt(1.0 - sigma**2)*v) * self.scale
        u1[u0 < 0] *= -1
        u1 = u1 + self.loc
        
        if self.clip_below_zero:
            u1 = u1.clip(0)
        
        if self.clip_above_zero:
            u1 = np.clip(u1, a_min=None, a_max=0)


        return u1

    def sample(self, size):
        return self.random_skew_normal(size)

class SampleUniform:
    def __init__(self, min_val, max_val, distribution='uniform'):
        self.min_val = min_val
        self.max_val = max_val
        self.distribution = distribution

        dist_options = ["uniform", "log_uniform"]
        assert (
            self.distribution in dist_options
        ), f"Error distribution, {self.distribution}, not in {dist_options}"


    def sample_random_matrix(self, size):

        if self.distribution == "uniform":
            mat = np.random.uniform(self.min_val, self.max_val, size=size)

        elif self.distribution == "log_uniform":
            mat = np.exp(
                np.random.uniform(np.log(self.min_val), np.log(self.max_val), size=size)
            )

        else:
            raise ("ValueError")

        return mat

    def sample(self, size):
        return self.sample_random_matrix(size)
