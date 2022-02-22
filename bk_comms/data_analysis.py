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
from sklearn import manifold


colours = ["#003f5c", "#58508d", "#bc5090", "#ff6361", "#ffa600", "#ffa800",]


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

        return conf['cfg']

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

    def get_total_biomass(self, particle, sim_data, sim_t_idx):
        species_initial_abundance = 0

        for pop in particle.populations:
            sol_idx = self.get_solution_index(particle, pop.name)

            sim_val = sim_data[:, sol_idx][sim_t_idx]
            species_initial_abundance += sim_val

        return species_initial_abundance

    def generate_particle_parameter_df(self, particles):
        init_particle = particles[0]
        
        init_species_col = [f'init_species_{x}' for x in init_particle.model_names]
        init_conc_cols = [f'init_met_{x}' for x in init_particle.dynamic_compounds]

        k_val_cols = []
        lb_constraint_cols = []
        
        for m in init_particle.model_names:
            # Make column headings
            k_val_cols += [f'K_{x}_{m}' for x in init_particle.dynamic_compounds]
            lb_constraint_cols += [f'lb_constr_{x}_{m}' for x in init_particle.dynamic_compounds] 
        column_headings = []
        column_headings += init_conc_cols + init_species_col + k_val_cols + lb_constraint_cols

        parameters = np.zeros([len(particles), len(column_headings)])
        for idx, p in enumerate(particles):
            parameters[idx] = p.generate_parameter_vector().reshape(-1)

        df = pd.DataFrame(data=parameters, columns=column_headings)

        return df

    def plot_tsne(self, include_init_metabolites=True, include_lb_constraints=True, include_k_values=True, include_init_species=True):
        params_df = self.generate_particle_parameter_df(self.particles)

        if not include_init_species:
            params_df = params_df.loc[:, ~params_df.columns.str.contains('init_species_')]

        if not include_init_metabolites:
            params_df = params_df.loc[:, ~params_df.columns.str.contains('init_met_')]

        if not include_lb_constraints:
            params_df = params_df.loc[:, ~params_df.columns.str.contains('lb_constr_')]

        if not include_k_values:
            params_df = params_df.loc[:, ~params_df.columns.str.contains('K_')]

        tsne = manifold.TSNE(n_components=2, n_jobs=5).fit_transform(params_df)
        
        
        fig = px.scatter(x=tsne[:, 0], y=tsne[:, 1])
        fig.update_layout(template="simple_white", width=800, height=800)
        fig.write_image(f"{self.output_dir}/parameter_tsne.png")




    def plot_population_timeseries(self, plot_n_particles=10):
        n_sol_keys = len(self.config["exp_sol_keys"])
        fig = make_subplots(rows=n_sol_keys, cols=1, shared_xaxes=True, shared_yaxes='all')

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
        # fig.update_yaxes(title="Population", type='log')
        fig.update_layout(template="simple_white")

        fig.write_image(f"{self.output_dir}/solution_timeseries.png")

    def plot_population_abundance_timeseries(self, plot_n_particles=10):
        n_sol_keys = len(self.config["exp_sol_keys"])
        fig = make_subplots(rows=n_sol_keys, cols=1, shared_xaxes=True, shared_yaxes='all')

        for idx, key_pair in enumerate(self.config["exp_sol_keys"]):
            # Plot particle simulations
            for p in self.particles[:plot_n_particles]:
                total_biomasses = np.array([self.get_total_biomass(p, p.sol, sim_t_idx) for sim_t_idx, t in enumerate(p.t)])
                sim_sol = p.sol[:, self.get_solution_index(p, key_pair[1])]

                abundance_sol = sim_sol / total_biomasses

                fig.add_trace(
                    go.Line(
                        x=p.t,
                        y=sim_sol,
                        name=key_pair[1],
                        opacity=0.2,
                        marker={"color": colours[idx]},
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
        # fig.update_yaxes(title="Abundance", type='log')
        fig.update_layout(template="simple_white", width=800, height=800)

        fig.write_image(f"{self.output_dir}/species_abundance_timeseries.png")

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

        fig = make_subplots(rows=n_distances, cols=1)

        for idx, d in enumerate(distance_columns):

            fig.add_trace(
                go.Histogram(
                    x=df[d],
                    marker={"color": colours[0]},
                ),
                row=idx + 1,
                col=1,
            )

        fig.update_xaxes(title="Distance")
        fig.update_layout(template="simple_white")

        fig.write_image(f"{self.output_dir}/distance_distribution.png")

    def plot_abundance_bar(self):
        n_sol_keys = len(self.config["exp_sol_keys"])

        for p_idx, p in enumerate(self.particles):
            # Make df containing experimental and simulation abundances

            data_dict = {'dataset_label': [], 'species_label': [], 'abundance': []}
            for idx, key_pair in enumerate(self.config["exp_sol_keys"]):
                exp_data = self.target_data[key_pair[0]].values

                data_dict['dataset_label'].append('experiment')
                data_dict['species_label'].append(key_pair[1])
                data_dict['abundance'].append(exp_data[0])


                total_biomasses = np.array([self.get_total_biomass(p, p.sol, sim_t_idx) for sim_t_idx, t in enumerate(p.t)])
                sim_sol = p.sol[:, self.get_solution_index(p, key_pair[1])]

                abundance_sol = sim_sol / total_biomasses

                data_dict['dataset_label'].append('simulation')
                data_dict['species_label'].append(key_pair[1])
                data_dict['abundance'].append(abundance_sol[-1])

            df = pd.DataFrame.from_dict(data_dict)

            # Make figure
            fig = px.bar(df, x="dataset_label", y="abundance", color="species_label")
            fig.update_layout(template="simple_white")
            # fig.update_yaxes(title="Abundance", type='log')
            fig.write_image(f"{self.output_dir}/abundance_bar_particle_{p_idx}.png")



if __name__ == "__main__":

    data_directories = glob("./output/mel_indiv_growth/**/")
    # data_directories = [
    #     "./output/mel_indiv_growth/exp_S_salivarius_ga_fit_batch_tests/"
    # ]

    # data_directories = [
    #     "/Users/bezk/Documents/CAM/research_code/yeast_LAB_coculture/output/mel_mixes_growth/mel_multi_mix2_m2_growers/"
    # ]

    mix_target_data_path = "/Users/bezk/Documents/CAM/research_code/yeast_LAB_coculture/experimental_data/mel_target_data/target_data_pH71_Mix2_Med2.csv"
    mel_indiv_target_data_path = "/Users/bezk/Documents/CAM/research_code/yeast_LAB_coculture/experimental_data/mel_target_data/mel_indiv_target_data.csv"

    target_data_path = mel_indiv_target_data_path

    for x in data_directories:
        if not ('E_coli') in (x):
            continue
        print(x)
        d = DataAnalysis(experiment_dir=x, data_path=target_data_path)
        # d.particles = d.filter_particles_by_distance([1.0, 1.0, 1.0, 1.5, 1.5])

        if len(d.filter_particles_by_distance([5.0])) == 0:
            print(x)
            d.particles = d.filter_particles_by_distance([5.0])
            
            
        else:
            d.particles = d.filter_particles_by_distance([5.0])

        print(x, len(d.particles))

        # d.plot_tsne(include_init_metabolites=False, include_lb_constraints=True, include_k_values=False, include_init_species=False)
        # d.plot_abundance_bar()
        d.plot_distance_distributions()
        d.plot_population_timeseries(plot_n_particles=100)
        d.plot_fold_change_timeseries(plot_n_particles=100)
        d.plot_population_abundance_timeseries(plot_n_particles=100)
