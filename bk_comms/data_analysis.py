import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from pathlib import Path
import pandas as pd
import os

from plotly.subplots import make_subplots

from omegaconf import OmegaConf
import plotly.express as px
import plotly.graph_objects as go

colours = ["#003f5c", "#58508d", "#bc5090", "#ff6361", "#ffa600"]


class DataAnalysis:
    def __init__(
        self,
        experiment_dir="./output/mel_indiv_growth/exp_B_longum_subsp_longum_ga_fit_batch_tests",
        data_path="./experimental_data/mel_target_data/mel_indiv_target_data.csv",
    ):
        self.experiment_dir = experiment_dir
        self.run_directories = glob(experiment_dir + "/run_*/")
        self.config = self.read_omega_config()
        self.particles = self.load_all_particles(self.run_directories)

        self.output_dir = self.experiment_dir + "/data_analysis/"

        self.target_data = pd.read_csv(data_path)

        self.make_output_dir()

    def read_omega_config(self):
        # We are assuing all included run directories
        # have the same config (dangerous)
        conf = OmegaConf.load(self.run_directories[0] + "/cfg.yaml")

        return conf

    def make_output_dir(self):
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

    def load_pickle(self, pickle_path):
        with open(pickle_path, "rb") as f:
            data = pickle.load(f)
            return data

    def filter_particles_by_distance(self, epsilon):
        filtered_particles = []
        for p in self.particles:
            flag = True
            for idx, eps in enumerate(epsilon):
                if p.distance[idx] > eps:
                    flag = False

            if flag:
                filtered_particles.append(p)

        return filtered_particles

    def load_all_particles(self, repeat_dirs):
        all_particles = []
        for d in repeat_dirs:
            particle_paths = glob(f"{d}particles_*.pkl")
            for p_path in particle_paths:
                all_particles.extend(self.load_pickle(p_path))

        return all_particles

    def get_solution_index(self, comm, sol_element_key):
        idx = comm.solution_keys.index(sol_element_key)
        return idx

    def plot_population_timeseries(self, plot_n_particles=10):
        n_sol_keys = len(self.config["exp_sol_keys"])
        fig = make_subplots(rows=n_sol_keys, cols=1)

        for idx, key_pair in enumerate(self.config["exp_sol_keys"]):
            # Plot particle simulations
            for p in self.particles[:plot_n_particles]:

                sim_sol = p.sol[:, self.get_solution_index(p, key_pair[1])]

                fig.add_trace(
                    go.Line(
                        x=p.t,
                        y=sim_sol,
                        name="Simulation",
                        legendgroup="Simulation",
                        opacity=0.2,
                        marker={"color": colours[0]},
                    ),
                    row=idx + 1,
                    col=1,
                )

        names = set()
        fig.for_each_trace(
            lambda trace: trace.update(showlegend=False)
            if (trace.name in names)
            else names.add(trace.name)
        )

        fig.update_xaxes(title="Time")
        fig.update_yaxes(title="Population")
        fig.update_layout(template="simple_white")

        fig.write_image(f"{self.output_dir}/solution_timeseries.png")

    def plot_fold_change_timeseries(self, plot_n_particles=10):
        n_sol_keys = len(self.config["exp_sol_keys"])
        fig = make_subplots(rows=n_sol_keys, cols=1)

        for idx, key_pair in enumerate(self.config["exp_sol_keys"]):
            # Plot particle simulations
            for p in self.particles[:plot_n_particles]:

                sim_sol = p.sol[:, self.get_solution_index(p, key_pair[1])]

                # Convert to fold change
                init_sim = sim_sol[0]
                calc_fc = lambda x: (x - init_sim) / init_sim
                sim_sol = [calc_fc(n) for n in sim_sol]

                fig.add_trace(
                    go.Line(
                        x=p.t,
                        y=sim_sol,
                        name="Simulation",
                        legendgroup="Simulation",
                        opacity=0.2,
                        marker={"color": colours[0]},
                    ),
                    row=idx + 1,
                    col=1,
                )

            # Plot experimental data
            exp_data = self.target_data[key_pair[0]].values

            fig.add_trace(
                go.Scatter(
                    x=self.target_data.time,
                    y=exp_data,
                    name="Experiment",
                    legendgroup="Experiment",
                    marker={"color": colours[-1]},
                ),
                row=idx + 1,
                col=1,
            )

        names = set()
        fig.for_each_trace(
            lambda trace: trace.update(showlegend=False)
            if (trace.name in names)
            else names.add(trace.name)
        )

        fig.update_xaxes(title="Time")
        fig.update_yaxes(title="Population fold change")
        fig.update_layout(template="simple_white")

        fig.write_image(f"{self.output_dir}/fold_change_timeseries.png")

    def plot_distance_distributions(self):
        # Make dataframe
        n_distances = len(self.particles[0].distance)
        distance_columns = [f"dist_{x}" for x in range(n_distances)]

        df = pd.DataFrame(
            [p.distance for p in self.particles], columns=distance_columns
        )

        fig = make_subplots(rows=1, cols=n_distances)

        for idx, d in enumerate(distance_columns):

            fig.add_trace(
                go.Histogram(
                    x=df[d],
                    marker={"color": colours[0]},
                ),
                row=1,
                col=idx + 1,
            )

        fig.update_xaxes(title="Distance")
        fig.update_layout(template="simple_white")

        fig.write_image(f"{self.output_dir}/distance_distribution.png")


if __name__ == "__main__":

    data_directories = glob("./output/mel_indiv_growth/**/")
    data_directories = [
        "./output/mel_indiv_growth/exp_C_perfringens_S107_ga_fit_batch_tests/"
    ]
    for x in data_directories:
        d = DataAnalysis(experiment_dir=x)
        d.particles = d.filter_particles_by_distance([0.30])

        if len(d.particles) == 0:
            print(x)
            continue

        d.plot_distance_distributions()
        d.plot_population_timeseries(plot_n_particles=100)
        d.plot_fold_change_timeseries(plot_n_particles=100)
