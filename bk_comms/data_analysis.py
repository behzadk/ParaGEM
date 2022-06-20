import numpy as np
import pickle
import matplotlib.pyplot as plt
from glob import glob
from pathlib import Path
import pandas as pd
import os

from plotly.subplots import make_subplots
import datapane as dp

from omegaconf import OmegaConf
import plotly.express as px
import plotly.graph_objects as go
from sklearn import manifold

import pygmo

colours = [
    "#003f5c",
    "#58508d",
    "#bc5090",
    "#ff6361",
    "#ffa600",
    "#ffa800",
]


class DataAnalysis:
    def __init__(
        self,
        experiment_dir,
        particle_pickle_regex,
        data_path="./experimental_data/mel_target_data/mel_indiv_target_data.csv",
    ):
        self.experiment_dir = experiment_dir
        self.run_directories = glob(particle_pickle_regex)
        self.config = self.read_omega_config()
        self.particles = self.load_all_particles(self.run_directories)

        self.output_dir = self.experiment_dir + "/data_analysis/"

        self.target_data = pd.read_csv(data_path)

        self.make_output_dir()

    def read_omega_config(self):
        # We are assuing all included run directories
        # have the same config (dangerous)
        conf = OmegaConf.load(self.experiment_dir + "generation_11/run_1/cfg.yaml")

        return conf["cfg"]

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

    def load_all_particles(self, particle_paths):
        all_particles = []
        for p_path in particle_paths:
            particle = self.load_pickle(p_path)

            try:
                del particle.max_exchange_prior
                del particle.k_val_prior
                del particle.toxin_interaction_prior

            except:
                pass
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

        init_species_col = [f"init_species_{x}" for x in init_particle.model_names]
        init_conc_cols = [f"init_met_{x}" for x in init_particle.dynamic_compounds]

        k_val_cols = []
        lb_constraint_cols = []
        toxin_cols = []

        for m in init_particle.model_names:
            # Make column headings
            k_val_cols += [f"K_{x}_{m}" for x in init_particle.dynamic_compounds]
            lb_constraint_cols += [
                f"lb_constr_{x}_{m}" for x in init_particle.dynamic_compounds
            ]

        # Make toxin interaction headings
        for donor_model in init_particle.model_names:
            for recipient_model in init_particle.model_names:
                toxin_cols += [f"toxin_{donor_model}_{recipient_model}"]

        column_headings = []
        column_headings += (
            init_conc_cols
            + init_species_col
            + k_val_cols
            + lb_constraint_cols
            + toxin_cols
        )

        parameters = np.zeros([len(particles), len(column_headings)])
        for idx, p in enumerate(particles):
            parameters[idx] = p.generate_parameter_vector().reshape(-1)

        df = pd.DataFrame(data=parameters, columns=column_headings)

        return df

    def get_particle_distances(self, particles):
        n_distances = len(particles[0].distance)
        n_particles = len(particles)
        distances = np.zeros(shape=[n_particles, n_distances])

        for idx, p in enumerate(particles):
            distances[idx] = p.distance

        return distances

    def plot_tsne(
        self,
        include_init_metabolites=True,
        include_lb_constraints=True,
        include_k_values=True,
        include_init_species=True,
        include_toxin_values=True,
        colour_value_vector=None,
    ):
        params_df = self.generate_particle_parameter_df(self.particles)

        if not include_init_species:
            params_df = params_df.loc[
                :, ~params_df.columns.str.contains("init_species_")
            ]

        if not include_init_metabolites:
            params_df = params_df.loc[:, ~params_df.columns.str.contains("init_met_")]

        if not include_lb_constraints:
            params_df = params_df.loc[:, ~params_df.columns.str.contains("lb_constr_")]

        if not include_k_values:
            params_df = params_df.loc[:, ~params_df.columns.str.contains("K_")]

        if not include_toxin_values:
            params_df = params_df.loc[:, ~params_df.columns.str.contains("toxin_")]

        tsne = manifold.TSNE(n_components=2, n_jobs=5).fit_transform(params_df)

        # fig = px.scatter(x=tsne[:, 0], y=tsne[:, 1],
        #     marker=dict(
        #     size=16,
        #     color=colour_value_vector, #set color equal to a variable
        #     colorscale='Viridis', # one of plotly colorscales
        #     showscale=True
        #     )
        # )

        fig = go.Figure(
            data=go.Scatter(
                x=tsne[:, 0],
                y=tsne[:, 1],
                mode="markers",
                marker=dict(
                    color=colour_value_vector,  # set color equal to a variable
                    colorscale="Viridis",  # one of plotly colorscales
                    showscale=True,
                ),
            )
        )

        fig.update_layout(template="simple_white", width=800, height=800)
        fig.write_image(f"{self.output_dir}/parameter_tsne.png")

    def plot_paerto_front(self):
        distances = self.get_particle_distances(self.particles)
        ndf, dl, dc, ndr = pygmo.fast_non_dominated_sorting(distances)

        front_1_distances = distances[ndf[0]]

        for idx_i, name_i in enumerate(self.particles[0].model_names):
            for idx_j, name_j in enumerate(self.particles[0].model_names):
                if idx_i == idx_j:
                    continue

                fig = px.scatter(
                    x=front_1_distances[:, idx_i], y=front_1_distances[:, idx_j]
                )
                fig.update_xaxes(title_text=name_i)

                fig.update_yaxes(title_text=name_j)

                fig.write_image(f"{self.output_dir}/pareto_front_{idx_i}_{idx_j}.png")

    def plot_population_timeseries(self, plot_n_particles=10):
        n_sol_keys = len(self.config["exp_sol_keys"])
        fig = make_subplots(
            rows=n_sol_keys, cols=1, shared_xaxes=True, shared_yaxes="all"
        )

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
        fig = make_subplots(
            rows=n_sol_keys, cols=1, shared_xaxes=True, shared_yaxes="all"
        )

        for idx, key_pair in enumerate(self.config["exp_sol_keys"]):
            # Plot particle simulations
            for p in self.particles[:plot_n_particles]:
                total_biomasses = np.array(
                    [
                        self.get_total_biomass(p, p.sol, sim_t_idx)
                        for sim_t_idx, t in enumerate(p.t)
                    ]
                )
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

            data_dict = {"dataset_label": [], "species_label": [], "abundance": []}
            for idx, key_pair in enumerate(self.config["exp_sol_keys"]):
                exp_data = self.target_data[key_pair[0]].values

                data_dict["dataset_label"].append("experiment")
                data_dict["species_label"].append(key_pair[1])
                data_dict["abundance"].append(exp_data[0])

                total_biomasses = np.array(
                    [
                        self.get_total_biomass(p, p.sol, sim_t_idx)
                        for sim_t_idx, t in enumerate(p.t)
                    ]
                )
                sim_sol = p.sol[:, self.get_solution_index(p, key_pair[1])]

                abundance_sol = sim_sol / total_biomasses

                data_dict["dataset_label"].append("simulation")
                data_dict["species_label"].append(key_pair[1])
                data_dict["abundance"].append(abundance_sol[-1])

            df = pd.DataFrame.from_dict(data_dict)
            title_str = ""
            for idx, d in enumerate(p.distance):
                title_str += p.model_names[idx][:3] + " " + str(round(d, 3))
                title_str += "  "
            # Make figure
            fig = px.bar(df, x="dataset_label", y="abundance", color="species_label")
            fig.update_layout(template="simple_white", title=title_str)

            # fig.update_yaxes(title="Abundance", type='log')
            fig.write_image(f"{self.output_dir}/abundance_bar_particle_{p_idx}.png")

    def plot_toxin_interaction_matricies(self):
        for p_idx, p in enumerate(self.particles):
            print(p.toxin_mat)
            fig = make_subplots(rows=1, cols=1, shared_xaxes=True, shared_yaxes="all")

            population_names = [pop.name for pop in p.populations]

            heatmap = go.Heatmap(
                x=list(range(len(population_names))),
                y=list(range(len(population_names))),
                z=p.toxin_mat,
                colorscale="inferno",
            )

            heatmap["x"] = population_names
            heatmap["y"] = population_names

            fig.add_trace(heatmap)
            fig.update_layout(template="simple_white", title="Toxin interaction mat")
            fig.write_image(f"{self.output_dir}/toxin_mat_{p_idx}.png")

    def plot_metabolite_exchanges(self):
        figures = []
        for p_idx, p in enumerate(self.particles):
            fig = make_subplots(rows=1, cols=1, shared_xaxes=True, shared_yaxes="all")

            population_names = [pop.name for pop in p.populations]

            exchange_mat = p.max_exchange_mat
            print(sum(exchange_mat))
            # Apply mask

            for pop_idx, pop in enumerate(p.populations):
                exchange_mat[pop_idx] = (
                    exchange_mat[pop_idx] * pop.dynamic_compound_mask
                )

            # Binarise
            exchange_mat = (exchange_mat > 0) + (exchange_mat < 0) * -1

            keep_cmpd_indexes = []
            for cmpd_idx, _ in enumerate(p.dynamic_compounds):
                if any(exchange_mat[:, cmpd_idx] == 1) and any(
                    exchange_mat[:, cmpd_idx] == -1
                ):
                    keep_cmpd_indexes.append(cmpd_idx)

            exchange_mat = exchange_mat[:, keep_cmpd_indexes]
            dynamic_compounds = [p.dynamic_compounds[idx] for idx in keep_cmpd_indexes]

            heatmap = go.Heatmap(
                y=list(range(len(population_names))),
                x=list(range(len(dynamic_compounds))),
                z=exchange_mat,
                colorscale="picnic",
            )

            heatmap["y"] = population_names
            heatmap["x"] = dynamic_compounds

            fig.add_trace(heatmap)
            fig.update_layout(template="simple_white", title="exchanges")
            fig.write_image(f"{self.output_dir}/exchange_mat_{p_idx}.png")

            figures.append(fig)
        return figures


def check_particle_equality(patricle_0, particle_1):
    if not np.array_equal(patricle_0.toxin_mat, particle_1.toxin_mat):
        return False

    if not np.array_equal(patricle_0.max_exchange_mat, particle_1.max_exchange_mat):
        return False

    if not np.array_equal(patricle_0.k_vals, particle_1.k_vals):
        return False

    return True


def get_unique_particles(particles_list):
    unique_particles = []
    for p in particles_list:
        is_unique = True
        for uniq_p in unique_particles:
            if check_particle_equality(p, uniq_p):
                is_unique = False
                break

        if is_unique:
            unique_particles.append(p)

    return unique_particles


def vis_multi_datapane():
    data_directories = [
        "/Users/bezk/Documents/CAM/research_code/yeast_LAB_coculture/output/mel_mixes_growth/mel_multi_mix2_m2_growers/"
    ]

    data_dir = "/Users/bezk/Documents/CAM/research_code/yeast_LAB_coculture/output/mel_mixes_growth/mel_multi_mix2_m2_growers/"

    mix_target_data_path = "/Users/bezk/Documents/CAM/research_code/yeast_LAB_coculture/experimental_data/mel_target_data/target_data_pH7_Mix2_Med2.csv"

    target_data_path = mix_target_data_path

    particle_path_regex = f"{data_dir}/generation_11/run_1/*.pkl"
    d = DataAnalysis(
        experiment_dir=data_dir,
        particle_pickle_regex=particle_path_regex,
        data_path=target_data_path,
    )

    distances = d.get_particle_distances(d.particles)
    sum_distances = [sum(d) for d in distances]

    sorted_particles = sorted(
        d.particles, key=lambda x: sum_distances[d.particles.index(x)]
    )
    d.particles = sorted_particles[:10]
    # d.plot_toxin_interaction_matricies()

    exchange_figures = d.plot_metabolite_exchanges()
    toxin_mat_figures = d.plot_toxin_interaction_matricies()

    labels = [f"Solution {str(x)}" for x in range(len(exchange_figures))]
    blocks = [dp.Plot(f, label=labels[idx]) for idx, f in enumerate(exchange_figures)]

    print(labels)
    report = dp.Report(dp.Select(blocks=blocks, type=dp.SelectType.DROPDOWN))

    report.preview(
        open=True, formatting=dp.ReportFormatting(width=dp.ReportWidth.MEDIUM)
    )


def vis_multi():
    data_directories = [
        "/Users/bezk/Documents/CAM/research_code/yeast_LAB_coculture/output/mel_mixes_growth/mel_multi_mix2_m2_growers_test/"
    ]

    data_directories = [
        "/Users/bezk/Documents/CAM/research_code/yeast_LAB_coculture/output/mel_mixes_growth/mel_multi_mix2_m2_growers/"
    ]

    mix_target_data_path = "/Users/bezk/Documents/CAM/research_code/yeast_LAB_coculture/experimental_data/mel_target_data/target_data_pH7_Mix2_Med2.csv"

    target_data_path = mix_target_data_path

    for x in data_directories:
        particle_path_regex = f"{x}/generation_*/run_*/*.pkl"
        d = DataAnalysis(
            experiment_dir=x,
            particle_pickle_regex=particle_path_regex,
            data_path=target_data_path,
        )

        distances = d.get_particle_distances(d.particles)
        sum_distances = [sum(d) for d in distances]

        sorted_particles = sorted(
            d.particles, key=lambda x: sum_distances[d.particles.index(x)]
        )

        keep_n_particles = 1000
        d.particles = sorted_particles[:keep_n_particles]
        d.particles = get_unique_particles(d.particles)

        sum_distances = sum_distances[:keep_n_particles]

        distances = d.get_particle_distances(d.particles)
        sum_distances = [sum(d) for d in distances]

        sorted_particles = sorted(
            d.particles, key=lambda x: sum_distances[d.particles.index(x)]
        )

        keep_n_particles = 25
        d.particles = sorted_particles[:keep_n_particles]
        d.particles = get_unique_particles(d.particles)

        sum_distances = sum_distances[:keep_n_particles]

        # d.plot_toxin_interaction_matricies()

        # ndf, dl, dc, ndr = pygmo.fast_non_dominated_sorting(distances)

        # d.particles = [ d.particles[idx]  for idx in ndf[0]]

        # for p in d.particles:
        #     print("tox_sum", p.toxin_mat.sum())
        #     print(sum(p.distance))

        d.plot_tsne(
            include_init_metabolites=False,
            include_lb_constraints=True,
            include_k_values=True,
            include_init_species=False,
            include_toxin_values=True,
            colour_value_vector=sum_distances,
        )

        d.plot_abundance_bar()
        d.plot_metabolite_exchanges()
        d.plot_toxin_interaction_matricies()
        d.plot_distance_distributions()
        d.plot_population_timeseries(plot_n_particles=100)
        d.plot_fold_change_timeseries(plot_n_particles=100)
        d.plot_population_abundance_timeseries(plot_n_particles=100)


def vis_indiv():
    data_directories = glob("./output/mel_indiv_growth/**/")
    mel_indiv_target_data_path = "/Users/bezk/Documents/CAM/research_code/yeast_LAB_coculture/experimental_data/mel_target_data/mel_indiv_target_data.csv"

    target_data_path = mel_indiv_target_data_path

    for x in data_directories:
        d = DataAnalysis(experiment_dir=x, data_path=target_data_path)
        # d.particles = d.filter_particles_by_distance([1.0, 1.0, 1.0, 1.5, 1.5])

        epsilon = 0.24
        while len(d.filter_particles_by_distance([epsilon])) < 100:
            # print(f'epsilon: {epsilon}', f'particles: {len(d.filter_particles_by_distance([epsilon]))}')
            epsilon += 0.01

        d.particles = d.filter_particles_by_distance([epsilon])

        print(x, f"epsilon: {epsilon}", f"particles: {len(d.particles)}")

        d.plot_distance_distributions()
        d.plot_population_timeseries(plot_n_particles=100)
        d.plot_fold_change_timeseries(plot_n_particles=100)
        d.plot_population_abundance_timeseries(plot_n_particles=100)


if __name__ == "__main__":
    vis_multi_datapane()
    # exit()
    # vis_multi()
