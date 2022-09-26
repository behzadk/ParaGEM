from glob import glob
import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
import webcolors
from paragem import utils
from glob import glob
import pickle
from plotly.subplots import make_subplots
import numpy as np
import plotly
import plotly.express as px
import plotly.graph_objects as go
import datapane as dp
import pandas as pd
from paragem import distances
from paragem.data_analysis.visualisation_utils import load_particles

import os
import re

colours = [
    "#003f5c",
    "#58508d",
    "#bc5090",
    "#ff6361",
    "#ffa600",
    "#ffa800",
]


def get_target_data_columns(target_species_name, target_media_name, all_target_columns):
    target_data_columns = []
    for c in all_target_columns:
        if target_species_name not in c:
            continue

        species, media, repeat_idx = split_target_column(c)

        if species == target_species_name and media == target_media_name:
            target_data_columns.append(c)

    return target_data_columns


def split_target_column(target_col):
    target_col = target_col.split("_")
    media = target_col[-3]
    media_index = target_col.index(media)

    species = "_".join(target_col[0:media_index])
    repeat_idx = target_col[media_index + 1]

    return species, media, repeat_idx


def load_solution(directory, media_name):
    sol_path = f"{directory}/particle_sol_{media_name}.npy"
    sol_vec = np.load(sol_path, allow_pickle=True)

    return sol_vec

def load_toxin_mat(directory):
    toxin_mat_path = f"{directory}/particle_toxin.npy"

    return np.load(toxin_mat_path)


def load_distance_vector(directory):
    distance_vector_path = f"{directory}/particle_distance_vectors.npy"
    distance_vector = np.load(distance_vector_path)
    return distance_vector


def load_solution_keys(directory):
    solution_keys_path = f"{directory}/solution_keys.npy"
    solution_keys = np.load(solution_keys_path)
    return solution_keys


def load_time_vector(directory):
    time_vector_path = f"{directory}/particle_t_vectors.npy"
    time_vector = np.load(time_vector_path, allow_pickle=True)
    return time_vector


def figure_all_particle_fold_change_timeseries(
    particles_df,
    species_name,
    target_data_path,
    exp_sol_keys,
    sim_media_name,
    n_particles_vis,
    output_dir,
):
    """
    Plot time series of particles with experimental data overlayed

    Args:
        particles_df: dataframe containing information of all particles in the experiment
        species name: name of species to be plotted
        target_data_path: path to experimental data that should be overlayed
        exp_sol_keys: keys indicating column in experiment data linked to solution in model
        sim_media_name: A key to the solution dictioanry in the case of multiple simulations
        n_particles_vis: number of particle solutions to visualise
    """

    particles_df = particles_df.head(n_particles_vis)
    target_data_df = pd.read_csv(target_data_path)

    fig = go.Figure()

    all_distances = []
    sim_sols = []
    all_t_vecs = []

    # Iterate each data directory and unpack particles
    for d in particles_df.data_dir.unique():
        # Subset for this data directory
        df = particles_df[particles_df.data_dir == d]
        print(sim_media_name)
        # Load solutions
        sol_vec = load_solution(d, sim_media_name)
        distance_vec = load_distance_vector(d)
        solution_keys = list(load_solution_keys(d))
        t_vec = load_time_vector(d)

        for idx, row in df.iterrows():
            p_idx = row.particle_index
            sum_dist = row.sum_distance
            all_distances.append(sum_dist)

            
            sim_sol = sol_vec[p_idx][
                :, solution_keys.index(exp_sol_keys[sim_media_name][0][1])
            ]
            t = t_vec[p_idx]

            sim_sols.append(sim_sol)
            all_t_vecs.append(t)

    # Sort particles by distance
    sol_indexes = list(range(len(all_distances)))
    sorted_indexes = [
        i for _, i in sorted(zip(all_distances, sol_indexes), key=lambda pair: pair[0])
    ]

    all_distances = [all_distances[i] for i in sorted_indexes]
    sim_sols = [sim_sols[i] for i in sorted_indexes]
    all_t_vecs = [all_t_vecs[i] for i in sorted_indexes]

    for idx, _ in enumerate(sorted_indexes):
        sim_sol = sim_sols[idx]
        t = all_t_vecs[idx]

        # Convert to fold change
        init_sim = sim_sol[0]
        calc_fc = lambda x: (x - init_sim) / init_sim
        sim_sol = [calc_fc(n) for n in sim_sol]

        fig.add_trace(
            go.Line(
                x=np.around(t, decimals=2),
                y=sim_sol,
                name=f"Simulation {sim_media_name}",
                opacity=0.1,
                marker={"color": colours[1]},
                hovertext=str(round(sum_dist, 3)),
            ),
        )

    exp_columns = get_target_data_columns(
        species_name, sim_media_name, target_data_df.columns
    )

    for c in exp_columns:
        species, media, repeat_idx = split_target_column(c)

        exp_data = target_data_df[c].values

        fig.add_trace(
            go.Scatter(
                x=target_data_df.time,
                y=exp_data,
                name=f"Experimental data rep: {repeat_idx}",
                legendgroup="Experiment",
                marker={"color": colours[-1]},
            ),
        )

    # Create and add slider
    steps = []
    for i in range(len(fig.data)):
        step_updates = []
        for x in range(len(fig.data)):
            if "Experimental data" in fig.data[x].name:
                step_updates.append(True)

            elif x <= i:
                step_updates.append(True)

            else:
                step_updates.append(False)

        if "Experimental data" in fig.data[i].name:
            distance = 0.0
        else:
            distance = all_distances[i]

        step = dict(
            method="update",
            label=f"Rank {i}",
            args=[
                {"visible": step_updates},
                {
                    "title": "Fold change error threshold: "
                    + str(round(distance, 3))
                    + f"\t Num particles: {sum(step_updates)}"
                },
                {"name": f"Rank {i}"},
            ],  # layout attribute
        )
        # step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    fig.data[0].visible = True

    sliders = [
        dict(
            active=10,
            currentvalue={"prefix": "Particle Rank: "},
            pad={"t": 50},
            steps=steps,
        )
    ]

    names = set()
    fig.for_each_trace(
        lambda trace: trace.update(showlegend=False)
        if (trace.name in names)
        else names.add(trace.name)
    )

    # fig.update_xaxes(title="Time")
    fig.update_xaxes(title="Time")
    fig.update_yaxes(title="Fold change")
    fig.update_layout(template="simple_white", width=1000, height=500, sliders=sliders)

    fig.write_html(f"{output_dir}/fold_change_timeseries_{sim_media_name}.html")

    return fig


def figure_plot_distances(particles_df, output_dir):
    generations = particles_df.generation.unique()
    generations.sort()
    fig = go.Figure()

    for gen in generations:
        sub_df = particles_df.loc[particles_df.generation == gen]

        fig.add_trace(
            go.Box(y=sub_df.sum_distance, name=f"{gen}", marker_color="lightseagreen")
        )

    fig.update_layout(template="simple_white", width=1000, height=1000)
    fig.update_xaxes(title="Generation")
    fig.update_yaxes(title="Sum distances")
    fig.update_yaxes(type="log")

    fig.write_html(f"{output_dir}/generation_distances.html")

    return fig


def split_experiment_dir(exp_dir):
    exp_dir = exp_dir.split("/")
    run_num = int(re.findall(r"\d+", exp_dir[-2])[0])
    generation_num = int(re.findall(r"\d+", exp_dir[-3])[0])

    return generation_num, run_num


def load_particles_dataframe(sim_media_names, particle_directories):
    data_df = {
        "particle_index": [],
        "data_dir": [],
        "generation": [],
        "run_idx": [],
        "sum_distance": [],
    }

    # Get distance vector paths
    for d in particle_directories:
        solution_keys = load_solution_keys(d)

        distance_vec = load_distance_vector(d)

        solution_vectors = []
        for m in sim_media_names:
            sol_vec = load_solution(d, m)
            solution_vectors.append(sol_vec)

        for p_idx, _ in enumerate(distance_vec):
            keep = True

            for sol_vec in solution_vectors:
                if len(sol_vec[p_idx]) == 0:
                    keep = False

            if len(distance_vec[p_idx]) == 0:
                keep = False

            if keep:
                generation, run_idx = split_experiment_dir(d)

                data_df["generation"].append(generation)
                data_df["run_idx"].append(run_idx)

                data_df["particle_index"].append(p_idx)
                data_df["data_dir"].append(d)
                data_df["sum_distance"].append(sum(distance_vec[p_idx]))

    particle_df = pd.DataFrame(data_df)

    return particle_df


def load_particle_solutions(particle_df, solution_keys):
    particle_solutions = []
    for d in particle_df.data_dir.unique():
        particle_solutions.append(load_particles(d))


def filter_unfinished_experiments(experiment_folders):
    finished_experiments = []

    for d in experiment_folders:
        if len(glob(f"{d}/*.npy")) > 0:
            finished_experiments.append(d)

    return finished_experiments


def prepare_particles_df(cfg):
    output_dir = f"{cfg.experiment_dir}/"
    experiment_folders = glob(f"{output_dir}/generation_*/run_**/")
    experiment_folders = filter_unfinished_experiments(experiment_folders)
    print(experiment_folders)

    particles_df = load_particles_dataframe(cfg.sim_media_names, experiment_folders)
    particles_df.sort_values(by="sum_distance", inplace=True)
    particles_df.reset_index(inplace=True)
    particles_df.to_csv(f"{output_dir}/particles_df.csv")

    return particles_df

def get_total_biomass(sol, pop_names, solution_keys, sim_data, sim_t_idx):
    species_initial_abundance = 0

    for pop in pop_names:
        sol_idx = solution_keys.index(pop)


        sim_val = sim_data[:, sol_idx][sim_t_idx]
        species_initial_abundance += sim_val

    return species_initial_abundance


def figure_particle_abundance_timeseries(
    particles_df,
    target_data_path,
    exp_sol_keys,
    sim_media_name,
    n_particles_vis,
    output_dir,
):
    particles_df = particles_df.head(n_particles_vis)
    target_data_df = pd.read_csv(target_data_path)

    abundance_timeseries_colours = px.colors.qualitative.Dark24


    for part_idx, row in particles_df.iterrows():

        d = row["data_dir"]
        p_idx = row.particle_index

        # Load particle data
        sol_vec = load_solution(d, sim_media_name)[p_idx]
        distance_vec = load_distance_vector(d)[p_idx]
        solution_keys = list(load_solution_keys(d))
        t_vec = load_time_vector(d)[p_idx]
        target_data_df = pd.read_csv(target_data_path)


        pop_names = [x[1] for x in exp_sol_keys]

        # Plot particle simulations
        total_biomasses = np.array(
            [
                get_total_biomass(sol_vec, pop_names, solution_keys, sol_vec, sim_t_idx)
                for sim_t_idx, t in enumerate(t_vec)
            ]
        )

        fig = make_subplots(
            rows=len(pop_names), cols=1, shared_xaxes=True, shared_yaxes="all"
        )

        for idx, sol_key_pair in enumerate(exp_sol_keys):
            model_name = sol_key_pair[1]

            sim_sol = sol_vec[:, solution_keys.index(model_name)]

            abundance_sol = sim_sol / total_biomasses

            # Plot experimental data
            exp_data = target_data_df[model_name].values

            fig.add_trace(
                go.Scatter(
                    x=target_data_df.time,
                    y=exp_data,
                    name="Experiment",
                    legendgroup="Experiment",
                    marker={"color": abundance_timeseries_colours[-1]},
                ),
                row=idx + 1,
                col=1,
            )

            fig.add_trace(
                go.Line(
                    x=t_vec,
                    y=abundance_sol,
                    name=model_name,
                    opacity=1.0,
                    marker={"color": abundance_timeseries_colours[idx]},
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

        # fig.update_xaxes(title="Time")
        fig.update_xaxes(title="Time", row=len(pop_names) + 1, col=1)
        # fig.update_yaxes(title="Abundance", type='log')
        fig.update_layout(template="simple_white", width=800, height=600)

        fig.write_html(f"{output_dir}/particle_rank_{part_idx}_media_{sim_media_name}_ts_abundance.html")
        fig.write_json(f"{output_dir}/particle_rank_{part_idx}_media_{sim_media_name}_ts_abundance.json")

    return fig

def figure_particle_endpoint_abundance(particles_df,
    target_data_path,
    exp_sol_keys,
    sim_media_name,
    n_particles_vis,
    output_dir,
):
    particles_df = particles_df.head(n_particles_vis)
    target_data_df = pd.read_csv(target_data_path)


    for idx, row in particles_df.iterrows():
        d = row["data_dir"]
        p_idx = row.particle_index

        # Load particle data
        sol_vec = load_solution(d, sim_media_name)[p_idx]
        distance_vec = load_distance_vector(d)[p_idx]
        solution_keys = list(load_solution_keys(d))
        t_vec = load_time_vector(d)[p_idx]
        target_data_df = pd.read_csv(target_data_path)

        data_dict = {"dataset_label": [], "species_label": [], "abundance": []}

        model_names = [x[1] for x in exp_sol_keys]
        
        for model_name in model_names:
            exp_data = target_data_df[model_name].values

            data_dict["dataset_label"].append("experiment")
            data_dict["species_label"].append(model_name)
            data_dict["abundance"].append(exp_data[0])

            total_biomasses = np.array(
                [
                    get_total_biomass(sol_vec, model_names, solution_keys, sol_vec, sim_t_idx)
                    for sim_t_idx, t in enumerate(t_vec)
                ]
            )

            sim_sol = sol_vec[:, solution_keys.index(model_name)]

            abundance_sol = sim_sol / total_biomasses

            data_dict["dataset_label"].append("simulation")
            data_dict["species_label"].append(model_name)
            data_dict["abundance"].append(abundance_sol[-1])

        df = pd.DataFrame.from_dict(data_dict)

        # Make figure
        fig = px.bar(df, x="dataset_label", y="abundance", color="species_label")
        fig.update_layout(template="simple_white", title=f"Sum distances = {sum(distance_vec)}", 
        width=600, height=500, autosize=False)

        fig.write_html(f"{output_dir}/particle_rank_{idx}_media_{sim_media_name}_endpoint_abundance.html")
        fig.write_json(f"{output_dir}/particle_rank_{idx}_media_{sim_media_name}_endpoint_abundance.json")

    return fig

def figure_particle_toxin_interactions(particles_df, n_particles_vis, model_names, output_dir):
    particles_df = particles_df.head(n_particles_vis)
    model_names = list(model_names)

    for particle_rank, row in particles_df.iterrows():
        d = row["data_dir"]
        p_idx = row.particle_index

        toxin_mat = load_toxin_mat(d)[p_idx]

        # Bsinarize
        toxin_mat = (toxin_mat > 0) + (toxin_mat < 0) * -1

        # Setup colours
        producer_colour = webcolors.hex_to_rgb(px.colors.qualitative.Dark24[0])
        consumer_colour = webcolors.hex_to_rgb(px.colors.qualitative.Dark24[1])
        neutral_colour = webcolors.hex_to_rgb(px.colors.qualitative.Dark24[5])

        # Make colours matrix and information matrix
        img_colours = np.zeros(shape=toxin_mat.shape, dtype=object)
        info_mat = np.zeros(shape=toxin_mat.shape, dtype=object)

        for x_idx in range(toxin_mat.shape[0]):
            for y_idx in range(toxin_mat.shape[1]):
                if toxin_mat[x_idx][y_idx] > 0:
                    img_colours[x_idx][y_idx] = producer_colour
                elif toxin_mat[x_idx][y_idx] < 0:
                    img_colours[x_idx][y_idx] = consumer_colour

                else:
                    img_colours[x_idx][y_idx] = neutral_colour

                info_mat[
                    x_idx, y_idx
                ] = f"Value: {toxin_mat[x_idx, y_idx]}\t Donor: {model_names[y_idx]} \t Recipient: {model_names[x_idx]}"

        # Setup fig
        fig = go.Figure()

        fig.add_trace(
            go.Image(
                z=img_colours,
                ids=info_mat,
                hovertext=info_mat,
            ),
        )

        fig.update_layout(
            width=600,
            height=600,
            showlegend=True,
            title="Toxin Interactions",
            xaxis_title="Toxin Sensitivity",
            yaxis_title="Toxin Producer",
        )

        fig.update_xaxes(
            dict(
                tickmode="array",
                tickvals=[x for x in range(len(model_names))],
                ticktext=model_names,
            )
        )

        fig.update_yaxes(
            dict(
                tickmode="array",
                tickvals=[y for y in range(len(model_names))],
                ticktext=model_names,
            )
        )

        fig.write_html(f"{output_dir}/particle_rank_{particle_rank}_toxin_interactions.html")
        fig.write_json(f"{output_dir}/particle_rank_{particle_rank}_toxin_interactions.json")


def figure_particle_biomass_timeseries(particles_df, model_names, media_name):

    for particle_rank, row in particles_df.iterrows():
        d = row["data_dir"]
        p_idx = row.particle_index

        solution_keys = load_solution_keys(d)
        solution = load_solution(d, media_name)

        for idx, model_name in enumerate(model_names):
            sim_sol = solution[:, solution_keys.index(model_name)]
        pass

def figure_community_biomass(particles_df, n_particles_vis, model_names, media_name, output_dir):
    particles_df = particles_df.head(n_particles_vis)
    model_names = list(model_names)

    for particle_rank, row in particles_df.iterrows():
        d = row["data_dir"]
        p_idx = row.particle_index

        solution_keys = list(load_solution_keys(d))
        solution = load_solution(d, media_name)[p_idx]
        t = np.around(load_time_vector(d)[p_idx], decimals=2)

        community_biomass = np.zeros(solution[:, solution_keys.index(model_names[0])].shape)
        for idx, model_name in enumerate(model_names):
            community_biomass += solution[:, solution_keys.index(model_name)]

        fig = make_subplots(
            rows=1, cols=1, shared_xaxes=True, shared_yaxes="all"
        )

        fig.add_trace(
            go.Line(
                x=t,
                y=community_biomass,
                name="Community biomass",
                opacity=1.0,
                marker={"color": colours[0]},
            ),
            row=1,
            col=1,
        )

        # fig.update_xaxes(title="Time")
        fig.update_xaxes(title="Time", row=1, col=1)
        fig.update_yaxes(title="Biomass", type='log')
        fig.update_layout(template="simple_white", width=800, height=600)
        
        fig.write_html(f"{output_dir}/particle_rank_{particle_rank}_media_{media_name}_community_biomass.html")
        fig.write_json(f"{output_dir}/particle_rank_{particle_rank}_media_{media_name}_community_biomass.json")


def figure_datapane(particles_df, sim_media_names, n_particles_vis, output_dir):
    particle_blocks = []
    for particle_rank in range(n_particles_vis):
        abundance_figures = []
        community_biomass_figures = []

        for media_name in sim_media_names:
            particle_endpoint_path = f"{output_dir}/particle_rank_{particle_rank}_media_{media_name}_endpoint_abundance.json"
            particle_community_biomass_path = f"{output_dir}/particle_rank_{particle_rank}_media_{media_name}_community_biomass.json"

            fig_endpoint = dp.Plot(plotly.io.read_json(particle_endpoint_path), caption=f"Media {media_name}")
            fig_community_biomass = dp.Plot(plotly.io.read_json(particle_community_biomass_path), caption=f"Media {media_name}")

            abundance_figures.append(fig_endpoint)
            community_biomass_figures.append(fig_community_biomass)


        toxin_fig_path = f"{output_dir}/particle_rank_{particle_rank}_toxin_interactions.json"
        toxin_interaction_figure = dp.Plot(plotly.io.read_json(toxin_fig_path), caption=f"Binarised toxin interactions")

        abundance_group = dp.Group(
            blocks=[x for x in abundance_figures],
            label='Endpoint Abundance',
            # responsive=False,
            columns=(len(sim_media_names))
        )

        community_biomass_group = dp.Group(
            blocks=[x for x in community_biomass_figures],
            label='Community biomass',
            # responsive=False,
            columns=(len(sim_media_names))
        )

        toxin_group = dp.Group(
            blocks=[toxin_interaction_figure],
            label='Toxin Interactions',
            # responsive=False,
        )

        particle_block = dp.Group(
            dp.Select(blocks=[abundance_group, community_biomass_group, toxin_group]),
            label=f"Particle_rank_{particle_rank}",
        )

        particle_blocks.append(particle_block)

    report = dp.Report(dp.Select(blocks=particle_blocks, type=dp.SelectType.DROPDOWN))
    report.save(f"{output_dir}/report.html", open=False, formatting=dp.ReportFormatting(width=dp.ReportWidth.MEDIUM))


def pipeline(cfg):
    wd = cfg.user.wd
    experiment_dir = f"{wd}/output/{cfg.experiment_name}/"

    output_dir = f"{experiment_dir}/{cfg.model_names[0]}/"
    experiment_folders = glob(
        f"{experiment_dir}/{cfg.model_names[0]}/generation_*/run_**/"
    )
    experiment_folders = filter_unfinished_experiments(experiment_folders)

    exp_sol_keys = cfg.data.exp_sol_keys
    target_data_df = pd.read_csv(cfg.data.exp_data_path)

    particles_df = load_particles_dataframe(cfg.sim_media_names, experiment_folders)
    particles_df.sort_values(by="sum_distance", inplace=True)
    particles_df.reset_index(inplace=True)
    particles_df.to_csv(f"{output_dir}/particles_df.csv")

    fig = figure_plot_distances(particles_df, output_dir)
    fig.write_html(f"{output_dir}/generation_distances.html")

    particles_df = particles_df.head(100)
    fig = figure_all_particle_fold_change_timeseries(
        particles_df,
        cfg.model_names[0],
        target_data_df,
        exp_sol_keys,
        cfg.sim_media_names[0],
    )

    # Save fig
    fig.write_html(f"{output_dir}/fold_change_timeseries_{cfg.sim_media_names[0]}.html")

    fig = figure_all_particle_fold_change_timeseries(
        particles_df,
        cfg.model_names[0],
        target_data_df,
        exp_sol_keys,
        cfg.sim_media_names[1],
        output_dir,
    )

    # Save fig
    fig.write_html(f"{output_dir}/fold_change_timeseries_{cfg.sim_media_names[1]}.html")

    distance_vector_path = f"{output_dir}/particle_distance_vector.npy"
    init_populations_vector_path = f"{output_dir}/particle_init_populations.npy.npy"
    k_val_vector_path = f"{output_dir}/particle_k_val.npy"
    max_exchange_vector_path = f"{output_dir}/particle_max_exchange.npy.npy"


if __name__ == "__main__":
    cfg = {}
    pipeline()
