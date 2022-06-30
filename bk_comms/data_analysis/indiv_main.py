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
from pathlib import Path
from bk_comms.data_analysisvisualisation_utils import generate_particle_parameter_df
import scipy

colours = [
    "#003f5c",
    "#58508d",
    "#bc5090",
    "#ff6361",
    "#ffa600",
    "#ffa800",
]


def split_target_column(target_col):
    target_col = target_col.split("_")
    media = target_col[-3]
    media_index = target_col.index(media)

    species = "_".join(target_col[0:media_index])
    repeat_idx = target_col[media_index + 1]

    return species, media, repeat_idx


def get_target_data_columns(target_species_name, target_media_name, all_target_columns):
    target_data_columns = []
    for c in all_target_columns:
        if target_species_name not in c:
            continue

        species, media, repeat_idx = split_target_column(c)

        if species == target_species_name and media == target_media_name:
            target_data_columns.append(c)

    return target_data_columns


def load_particle(particle_regex, epsilon=None):
    particle_files = glob(particle_regex)
    particles = []

    for pickle_path in particle_files:
        with open(pickle_path, "rb") as f:
            if isinstance(epsilon, type(None)):
                try:
                    particles.extend(pickle.load(f))
                except:
                    continue

            else:
                try:
                    batch_particles = pickle.load(f)
                except:
                    continue
                for p in batch_particles:
                    if p.distance[0] <= epsilon:
                        particles.append(p)
    return particles


def load_fitting_figures(fit_dir, species_name, media_name):
    fig_paths = glob(f"{fit_dir}/{species_name}_{media_name}_*_fitted.json")
    print(fig_paths)
    figs = []
    for fig_path in fig_paths:
        print(fig_path)
        fig = plotly.io.read_json(fig_path)
        figs.append(fig)

    return figs


def figure_all_particle_fold_change_timeseries(
    particle_list, target_data, target_data_columns
):
    fig = go.Figure()

    for particle in particle_list:
        for idx, model_name in enumerate(particle.model_names):
            sim_sol = particle.sol[:, particle.solution_keys.index(model_name)]

            # Convert to fold change
            init_sim = sim_sol[0]
            calc_fc = lambda x: (x - init_sim) / init_sim
            sim_sol = [calc_fc(n) for n in sim_sol]

            fig.add_trace(
                go.Line(
                    x=particle.t,
                    y=sim_sol,
                    name=f"Simulation {model_name}",
                    opacity=0.1,
                    marker={"color": colours[1]},
                    hovertext=str(round(list(particle.distance)[0], 3)),
                ),
            )

    # Plot experimental data
    for c in target_data_columns:
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
        distances = []
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
            distance = particle_list[i].distance[0]

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


def figure_all_particle_timeseries(particle_list):
    fig = go.Figure()

    for particle in particle_list:
        for idx, model_name in enumerate(particle.model_names):
            sim_sol = particle.sol[:, particle.solution_keys.index(model_name)]

            # Convert to fold change
            init_sim = sim_sol[0]

            fig.add_trace(
                go.Line(
                    x=particle.t,
                    y=sim_sol,
                    name=f"Simulation {model_name}",
                    opacity=0.1,
                    marker={"color": colours[1]},
                ),
            )

    # Create and add slider
    steps = []
    for i in range(len(fig.data)):
        step_updates = []
        for x in range(len(fig.data)):
            if x <= i:
                step_updates.append(True)

            else:
                step_updates.append(False)

        step = dict(
            method="update",
            label=f"Rank {i}",
            args=[
                {"visible": step_updates},
                {
                    "title": "Fold change error threshold: "
                    + str(round(particle_list[i].distance[0], 3))
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
    fig.update_yaxes(title="g/L")
    fig.update_layout(template="simple_white", width=1000, height=500, sliders=sliders)

    # fig.show()
    return fig

def figure_plot_experimental_max_OD(individual_growth_df, species_names, media_name):
    sub_df = individual_growth_df.loc[individual_growth_df["medium"] == media_name]

    sub_df = sub_df.loc[sub_df["species"].isin(species_names)]

    sub_df = sub_df.groupby('species').agg({'OD': 'max'})
    print(sub_df)
    exit()

def sort_particles_by_min_distance(particles):
    particles.sort(key=lambda x: max(x.distance))


def recalculate_particle_distances(
    particles, target_data_path, target_data_key="E_coli_IAI1_M2_ODfc"
):
    exp_sol_keys = [
        [target_data_key, particles[0].model_names[idx]]
        for idx in range(len(particles[0].model_names))
    ]
    dist = distances.DistanceFoldChangeError(
        target_data_path, "time", exp_sol_keys, epsilon=1.0, final_epsilon=1.0
    )

    for p in particles:
        d = dist.calculate_distance(p)
        p.distance = d


def filter_particles(particles, epsilon):
    filtered_particles = []

    for p in particles:
        if p.distance[0] < epsilon:
            filtered_particles.append(p)

    return filtered_particles


def recalculate_particle_distances_test(
    particles, target_data_path, target_data_key="E_coli_IAI1_M2_ODfc"
):

    exp_sol_keys = [
        [target_data_key, particles[0].model_names[idx]]
        for idx in range(len(particles[0].model_names))
    ]

    dist = distances.DistanceFoldChangeError(
        target_data_path, "time", exp_sol_keys, epsilon=1.0, final_epsilon=1.0
    )

    for p in particles:
        print(p.distance)
        print(sum(p.distance))
        d = dist.calculate_distance(p)
        print(d)
        exit()

def calculate_posterior_volume(particles, output_path):
    df = generate_particle_parameter_df(particles)
    
    param_analysis_data = {'param': [], 'range': [], 'max': [], 
    'min': [], 'norm_test_p': [], 'norm_test_stat': []}

    for c in df.columns:
        data = df[c].values
        norm_test = scipy.stats.normaltest(data)
        param_analysis_data['param'].append(c)
        param_analysis_data['max'].append(max(data))
        param_analysis_data['min'].append(min(data))

        param_analysis_data['norm_test_stat'].append(norm_test[0])
        param_analysis_data['norm_test_p'].append(norm_test[1])

        param_analysis_data['range'].append(max(data) - min(data))

    param_analysis_df = pd.DataFrame(param_analysis_data)
    param_analysis_df.sort_values('range', inplace=True)
    param_analysis_df.to_csv(output_path)
    exit()


def write_manuscript_figure(figure, output_path):
    figure = go.Figure(figure)
    figure.layout.sliders = None
    figure.update_layout(showlegend=False)
    # figure.update_xaxes(slider=False)

    figure.write_image(output_path)

def datapane_main():
    wd = "/Users/bezk/Documents/CAM/research_code/yeast_LAB_coculture/"
    data_directories = glob(f"{wd}/output/mel_indiv_growth_8_filtered/**/")
    target_data_path = "/Users/bezk/Documents/CAM/research_code/yeast_LAB_coculture/experimental_data/mel_target_data/mel_indiv_target_data.csv"
    target_data_path = "/Users/bezk/Documents/CAM/research_code/community_study_mel_krp/target_data/individual_growth/mel_indiv_target_data.csv"

    fitted_curves_dir = "/Users/bezk/Documents/CAM/research_code/community_study_mel_krp/target_data/individual_growth/fitted_curves/"
    individual_growth_df = pd.read_csv('/Users/bezk/Documents/CAM/research_code/community_study_mel_krp/target_data/individual_growth/formatted_individual_growth.csv')


    output_dir = f"{wd}/output/mel_indiv_growth_8_filtered/"



    target_df = pd.read_csv(target_data_path, index_col=0)

    figure_blocks = []

    epsilon_dict = {
        "B_longum_subsp_longum": 13.00,
        "C_perfringens_S107": 37.644,
        "E_coli_IAI1": 9.354,
        "L_paracasei": 9.60,
        "L_plantarum": 15.0,
        "S_salivarius": 25.0,
        "C_ramosum": 13.7,
    }


    epsilon_dict = {
        "B_longum_subsp_longum": 1000,
        "C_perfringens_S107": 1000,
        "E_coli_IAI1": 1000,
        "L_paracasei": 1000,
        "L_plantarum": 1000,
        "S_salivarius": 1000,
        "C_ramosum": 16.85
    }

    # figure_plot_experimental_max_OD(individual_growth_df,epsilon_dict.keys(), 'M2')
    Path(f'{output_dir}/figures/').mkdir(parents=True, exist_ok=True)

    for d in data_directories:
        print(d)
        # Get top directory name
        dir_name = d.split("/")[-2]



        # if dir_name != "S_salivarius":
        #     continue

        # Load all particles
        particle_regex = f"{d}/run_*/particles_*.pkl"
        particle_regex = f"{d}/generation_*/run_*/particles_*.pkl"

        particles = load_particles(particle_regex)
        # recalculate_particle_distances(
        #     particles, target_data_path, target_data_key=f"{dir_name}_M2_ODfc"
        # )
        particles = [p for p in particles if sum(p.distance) < epsilon_dict[dir_name]]
        particles = [p for p in particles if hasattr(p, 'sol')]

        print(len(particles))

        calculate_posterior_volume(particles, output_path=f"{output_dir}/figures/{dir_name}_posterior_volume.csv")
        exit()

        if len(particles) == 0:
            continue

        particles = utils.get_unique_particles(particles)
        sort_particles_by_min_distance(particles)

        particles = particles

        particles = particles
        # skipped_particles = []
        # for p_idx, p in enumerate(particles):
        #     if p_idx % 10 == 0.0:
        #         skipped_particles.append(p)

        # particles = skipped_particles

        print(len(particles))

        exp_columns = get_target_data_columns(f"{dir_name}", "M2", target_df.columns)

        fig_fold_change_timeseries = figure_all_particle_fold_change_timeseries(
            particles, target_df, exp_columns
        )


        fig_timeseries = figure_all_particle_timeseries(particles)

        fitting_figs = load_fitting_figures(
            fit_dir=fitted_curves_dir, species_name=dir_name, media_name="M2"
        )

        fig_list = [fig_fold_change_timeseries] + fitting_figs

        print(len(fig_list))
        species_block = dp.Group(dp.Select(blocks=fig_list), label=f"{dir_name}")


        write_manuscript_figure(fig_fold_change_timeseries, f"{output_dir}/figures/{dir_name}_fold_change_timeseries.pdf")
        write_manuscript_figure(fig_timeseries, f"{output_dir}/figures/{dir_name}_biomass_timeseries.pdf")

        for idx, fig_rep in enumerate(fitting_figs):
            write_manuscript_figure(fig_rep, f"{output_dir}/figures/{dir_name}_fitting_rep_{idx}.pdf")


        # species_block = dp.Group(
        #     blocks=[fig_fold_change_timeseries],
        #     label=f"{dir_name}"
        # )

        figure_blocks.append(species_block)

        # report = dp.Report(species_block)
        # report.preview(
        #     open=True, formatting=dp.ReportFormatting(width=dp.ReportWidth.MEDIUM)
        # )

    report = dp.Report(dp.Select(blocks=figure_blocks, type=dp.SelectType.DROPDOWN))

    report.save(output_dir + "report.html", open=True, formatting=dp.ReportFormatting(width=dp.ReportWidth.MEDIUM))

if __name__ == "__main__":
    datapane_main()
