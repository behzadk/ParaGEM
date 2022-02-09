import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger
from typing import List


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def get_solution_index(comm, sol_element_key):
    idx = comm.solution_keys.index(sol_element_key)
    return idx


class DistanceGrowthRateMinMax:
    def __init__(
        self,
        min_growth: List[str],
        max_growth: List[str],
        pop_keys: List[str],
        mode: str,
    ):
        self.min_growth = min_growth
        self.max_growth = max_growth
        self.mode = mode

        available_modes = ["max", "mean", "median"]

        if self.mode not in available_modes:
            raise ValueError(f"{self.mode} not in {available_modes}")

    def calculate_distance(self, community):
        n_distances = len(self.pop_keys)
        distances = np.zeros(n_distances)

        if community.sol is None or community.t is None:
            return [np.inf for d in range(n_distances)]

        for idx, p_key in enumerate(self.pop_keys):
            if self.mode == "max":
                distances[idx] = np.max(community.sol[:, idx])

            elif self.mode == "mean":
                distances[idx] = np.mean(community.sol[:, idx])

            elif self.mode == "median":
                distances[idx] = np.median(community.sol[:, idx])

    def assess_particle(self, community):
        distance = self.calculate_distance(community)

        for idx, d in enumerate(distance):
            if d > self.min_growth and d < self.max_growth:
                continue

            else:
                return False, distance

        return True, distance


class DistanceTimeseriesEuclidianDistance:
    def __init__(
        self,
        exp_data_path: str,
        exp_t_key,
        exp_sol_keys,
        epsilon=1.0,
        final_epsilon=1.0,
    ):
        self.exp_data = pd.read_csv(exp_data_path)
        self.exp_t_key = exp_t_key
        self.exp_sol_keys = exp_sol_keys

        self.epsilon = epsilon
        self.final_epsilon = final_epsilon

    def calculate_distance(self, community):
        n_distances = len(self.exp_sol_keys)
        distances = np.zeros(n_distances)

        if community.sol is None or community.t is None:
            return [np.inf for d in range(n_distances)]

        sim_data = community.sol
        sim_t = community.t

        for distance_idx, key_pair in enumerate(self.exp_sol_keys):
            for exp_data_idx, t in enumerate(self.exp_data[self.exp_t_key].values):
                sim_t_idx = find_nearest(sim_t, t)

                sol_idx = get_solution_index(community, key_pair[1])

                sim_val = sim_data[:, sol_idx][sim_t_idx]
                exp_val = self.exp_data.loc[self.exp_data[self.exp_t_key] == t][
                    key_pair[0]
                ].values[0]

                if np.isnan(exp_val):
                    continue

                distances[distance_idx] += abs(exp_val - sim_val)

            # Temp fix
            print(
                "end to end diff: ",
                abs(sim_data[:, sol_idx][0] - sim_data[:, sol_idx][-1]),
            )
            # if abs(sim_data[:, sol_idx][0] - sim_data[:, sol_idx][-1]) < 1e-10:
            #     distances[distance_idx] = 1000

        return distances

    def assess_particle(self, community):
        distance = self.calculate_distance(community)

        for idx, d in enumerate(distance):
            if d < self.epsilon[idx]:
                continue

            else:
                return False, distance

        return True, distance

    def plot_community(self, community, pdf, comm_idx, batch_idx):
        for idx, key_pair in enumerate(self.exp_sol_keys):
            f, ax = plt.subplots()
            sol_idx = get_solution_index(community, key_pair[1])
            ax.plot(
                community.t,
                community.sol[:, sol_idx],
                label=f"sim_{sol_idx}",
                color="orange",
            )
            ax.set_title(
                f"key:{key_pair[1]}, eps: {self.epsilon[idx]:.3}, dist: {community.distance[idx]:.4}"
            )

            for idx, t in enumerate(self.exp_data[self.exp_t_key].values):

                exp_val = self.exp_data.loc[self.exp_data[self.exp_t_key] == t][
                    key_pair[0]
                ].values[0]

                if np.isnan(exp_val):
                    continue
                else:
                    ax.scatter(t, exp_val, label=key_pair[0], color="black")

            # Temp fix
            print(
                "end to end diff: ",
                abs(sim_data[:, sol_idx][0] - sim_data[:, sol_idx][-1]),
            )
            if abs(sim_data[:, sol_idx][0] - sim_data[:, sol_idx][-1]) < 1e-10:
                distances[distance_idx] = 1000

            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())
            pdf.savefig()
            plt.close()


class DistanceFoldChangeError:
    def __init__(
        self,
        exp_data_path: str,
        exp_t_key,
        exp_sol_keys,
        epsilon=1.0,
        final_epsilon=1.0,
    ):

        self.exp_data = pd.read_csv(exp_data_path)
        self.exp_t_key = exp_t_key
        self.exp_sol_keys = exp_sol_keys

        # Here, epsilon behaves as a percentage error tolerance
        self.epsilon = epsilon
        self.final_epsilon = final_epsilon

    def calculate_distance(self, community):
        n_distances = len(self.exp_sol_keys)
        distances = np.zeros(n_distances)

        if community.sol is None or community.t is None:
            return [np.inf for d in range(n_distances)]

        sim_data = community.sol
        sim_t = community.t

        for distance_idx, key_pair in enumerate(self.exp_sol_keys):
            for exp_data_idx, t in enumerate(self.exp_data[self.exp_t_key].values):
                sim_t_idx = find_nearest(sim_t, t)

                sol_idx = get_solution_index(community, key_pair[1])

                # Calculate fold change compared with starting value
                init_sim_val = sim_data[:, sol_idx][0]
                this_sim_val = sim_data[:, sol_idx][sim_t_idx]


                exp_val = self.exp_data.loc[self.exp_data[self.exp_t_key] == t][
                    key_pair[0]
                ].values[0]

                if np.isnan(exp_val):
                    continue

                fold_change_sim_val = (this_sim_val - init_sim_val) / init_sim_val

                if exp_val == 0.0:
                    print("here")
                    fold_change_error = ((fold_change_sim_val + 1) - (exp_val + 1)) / (exp_val + 1)

                else:
                    fold_change_error = (fold_change_sim_val - exp_val) / exp_val
                
                distances[distance_idx] += abs(fold_change_error)

                print(
                    f"FoldChangeError: {init_sim_val}, {this_sim_val}, {exp_val}, {fold_change_sim_val}, {distances[distance_idx]}"
                )

        return distances

    def assess_particle(self, community):
        distance = self.calculate_distance(community)

        for idx, d in enumerate(distance):
            if d < self.epsilon[idx]:
                continue

            else:
                return False, distance

        return True, distance
