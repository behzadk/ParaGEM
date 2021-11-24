import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def get_solution_index(comm, sol_element_key):
    idx = comm.solution_keys.index(sol_element_key)
    return idx


class DistanceTimeseriesEuclidianDistance:
    def __init__(self, exp_data_path: str, exp_t_key, exp_sol_keys, epsilon=1.0, final_epsion=1.0):
        self.exp_data = pd.read_csv(exp_data_path)
        self.exp_t_key = exp_t_key
        self.exp_sol_keys = exp_sol_keys

        self.epsilon = epsilon
        self.final_epsion = final_epsion

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
            ax.plot(community.t, community.sol[:, sol_idx], label=f'sim_{sol_idx}', color='orange')
            ax.set_title(f"key:{key_pair[1]}, eps: {self.epsilon[idx]:.3}, dist: {community.distance[idx]:.4}")

            for idx, t in enumerate(self.exp_data[self.exp_t_key].values):
                
                exp_val = self.exp_data.loc[self.exp_data[self.exp_t_key] == t][
                    key_pair[0]
                ].values[0]

                if np.isnan(exp_val):
                    continue
                else:
                    ax.scatter(t, exp_val, label=key_pair[0], color='black')
            
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())
            pdf.savefig()
            plt.close()
