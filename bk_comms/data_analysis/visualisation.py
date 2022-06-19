from glob import glob
import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
import webcolors
from bk_comms import utils
from glob import glob
import pickle
from plotly.subplots import make_subplots
import numpy as np
import plotly
import plotly.express as px
import plotly.graph_objects as go
import datapane as dp
import pandas as pd
from bk_comms import distances
from bk_comms.data_analysis.visualisation_utils import load_particles

import os

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
    particles_df, species_name, target_data, exp_sol_keys, sim_media_name
):

    
    fig = go.Figure()

    all_distances = []

    # Iterate each data directory
    for d in particles_df.data_dir.unique():
        # Subset for this data directory
        df = particles_df[particles_df.data_dir == d]
        
        # Load solutions
        sol_vec = load_solution(d, sim_media_name)
        distance_vec = load_distance_vector(d)
        solution_keys = list(load_solution_keys(d))
        t_vec = load_time_vector(d)

        for idx, row in df.iterrows():
            p_idx = row.particle_index
            sum_dist = row.sum_distance

            sim_sol = sol_vec[p_idx][:, solution_keys.index(exp_sol_keys[sim_media_name][0][1])]
            t = t_vec[p_idx]

            # Convert to fold change
            init_sim = sim_sol[0]
            calc_fc = lambda x: (x - init_sim) / init_sim
            sim_sol = [calc_fc(n) for n in sim_sol]

            all_distances.append(sum_dist)
            fig.add_trace(
                go.Line(
                    x=t,
                    y=sim_sol,
                    name=f"Simulation {sim_media_name}",
                    opacity=0.1,
                    marker={"color": colours[1]},
                    hovertext=str(round(sum_dist, 3)),
                ),
            ) 


    exp_columns = get_target_data_columns(species_name, sim_media_name, target_data.columns)
    
    for c in exp_columns:
        species, media, repeat_idx = split_target_column(c)

        exp_data = target_data[c].values

        fig.add_trace(
            go.Scatter(
                x=target_data.time,
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

    return fig


def load_particles_dataframe(sim_media_names, particle_directories):
    particle_indexes = []
    particle_path = []
    particle_sum_distance = []

    data_df = {'particle_index': [], 'data_dir': [], 'sum_distance': []}
    
    # Get distance vector paths
    for d in particle_directories:
        distance_vec = load_distance_vector(d)
        solution_keys = load_solution_keys(d)

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
                data_df['particle_index'].append(p_idx)
                data_df['data_dir'].append(d)
                data_df['sum_distance'].append(sum(distance_vec[p_idx]))
            

    particle_df = pd.DataFrame(data_df)

    return particle_df

def load_particle_solutions(particle_df, solution_keys):
    particle_solutions = []
    for d in particle_df.data_dir.unique():
        particle_solutions.append(load_particles(d))


def pipeline(cfg):
    wd = cfg.user.wd
    output_dir = cfg.output_dir
    experiment_dir = f"{wd}/output/{cfg.experiment_name}/"
    experiment_folders = glob(f"{experiment_dir}/{cfg.model_names[0]}/generation_*/run_**/")

    exp_sol_keys = cfg.data.exp_sol_keys
    target_data_df = pd.read_csv(cfg.data.exp_data_path)

    particles_df = load_particles_dataframe(cfg.sim_media_names, experiment_folders)
    print(particles_df)
    particles_df.sort_values(by='sum_distance', inplace=True)
    
    fig = figure_all_particle_fold_change_timeseries(
    particles_df, cfg.model_names[0], target_data_df, exp_sol_keys, cfg.sim_media_names[0])
    # Save fig
    fig.write_html(f"{output_dir}/fold_change_timeseries_{cfg.sim_media_names[0]}.html")

    fig = figure_all_particle_fold_change_timeseries(
    particles_df, cfg.model_names[0], target_data_df, exp_sol_keys, cfg.sim_media_names[1])

    # Save fig
    fig.write_html(f"{output_dir}/fold_change_timeseries_{cfg.sim_media_names[1]}.html")

    distance_vector_path = f"{output_dir}/particle_distance_vector.npy"
    init_populations_vector_path = f"{output_dir}/particle_init_populations.npy.npy"
    k_val_vector_path = f"{output_dir}/particle_k_val.npy"
    max_exchange_vector_path = f"{output_dir}/particle_max_exchange.npy.npy"



if __name__ == "__main__":
    
    cfg = {}
    pipeline()