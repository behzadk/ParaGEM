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

from visualisation_utils import load_particles

colours = [
    "#003f5c",
    "#58508d",
    "#bc5090",
    "#ff6361",
    "#ffa600",
    "#ffa800",
]

colours = px.colors.qualitative.Dark24


def figure_particle_metabolite_exchange(particle, binarize=True):
    population_names = [pop.name for pop in particle.populations]

    exchange_mat = particle.max_exchange_mat

    for pop_idx, pop in enumerate(particle.populations):
        exchange_mat[pop_idx] = exchange_mat[pop_idx] * pop.dynamic_compound_mask

    # Binarize
    exchange_mat = (exchange_mat > 0) + (exchange_mat < 0) * -1

    # Get exchanges involved in crossfeeding
    keep_cmpd_indexes = []
    for cmpd_idx, _ in enumerate(particle.dynamic_compounds):
        if any(exchange_mat[:, cmpd_idx] == 1) and any(exchange_mat[:, cmpd_idx] == -1):
            keep_cmpd_indexes.append(cmpd_idx)

    # Keep only metabolites involved in crossfeed
    exchange_mat = exchange_mat[:, keep_cmpd_indexes]
    dynamic_compounds = [particle.dynamic_compounds[idx] for idx in keep_cmpd_indexes]

    # Setup colours
    producer_colour = webcolors.hex_to_rgb(px.colors.qualitative.Dark24[0])
    consumer_colour = webcolors.hex_to_rgb(px.colors.qualitative.Dark24[1])
    neutral_colour = webcolors.hex_to_rgb(px.colors.qualitative.Dark24[5])

    # Make colours matrix and information matrix
    img_colours = np.zeros(shape=exchange_mat.shape, dtype=object)
    info_mat = np.zeros(shape=exchange_mat.shape, dtype=object)

    for x_idx in range(exchange_mat.shape[0]):
        for y_idx in range(exchange_mat.shape[1]):
            if exchange_mat[x_idx][y_idx] > 0:
                img_colours[x_idx][y_idx] = producer_colour
            elif exchange_mat[x_idx][y_idx] < 0:
                img_colours[x_idx][y_idx] = consumer_colour

            else:
                img_colours[x_idx][y_idx] = neutral_colour

            info_mat[
                x_idx, y_idx
            ] = f"Value: {exchange_mat[x_idx, y_idx]}\t Compound: {dynamic_compounds[y_idx]} \t Species: {population_names[x_idx]}"

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
        width=600, height=600, showlegend=True, title="Metabolic Interactions"
    )

    fig.update_xaxes(
        dict(
            tickmode="array",
            tickvals=[x for x in range(len(dynamic_compounds))],
            ticktext=dynamic_compounds,
        )
    )

    fig.update_yaxes(
        dict(
            tickmode="array",
            tickvals=[y for y in range(len(population_names))],
            ticktext=population_names,
        )
    )

    return fig


def get_total_biomass(particle, sim_data, sim_t_idx):
    species_initial_abundance = 0

    for pop in particle.populations:
        sol_idx = particle.solution_keys.index(pop.name)

        sim_val = sim_data[:, sol_idx][sim_t_idx]
        species_initial_abundance += sim_val

    return species_initial_abundance


def figure_particle_toxin_interactions(particle):
    population_names = [pop.name for pop in particle.populations]

    toxin_mat = particle.toxin_mat

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
            ] = f"Value: {toxin_mat[x_idx, y_idx]}\t Donor: {population_names[y_idx]} \t Recipient: {population_names[x_idx]}"

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
            tickvals=[x for x in range(len(population_names))],
            ticktext=population_names,
        )
    )

    fig.update_yaxes(
        dict(
            tickmode="array",
            tickvals=[y for y in range(len(population_names))],
            ticktext=population_names,
        )
    )

    return fig


def figure_particle_endpoint_abundance(particle, target_data):
    data_dict = {"dataset_label": [], "species_label": [], "abundance": []}
    for model_name in particle.model_names:
        exp_data = target_data[model_name].values

        data_dict["dataset_label"].append("experiment")
        data_dict["species_label"].append(model_name)
        data_dict["abundance"].append(exp_data[0])

        total_biomasses = np.array(
            [
                get_total_biomass(particle, particle.sol, sim_t_idx)
                for sim_t_idx, t in enumerate(particle.t)
            ]
        )

        sim_sol = particle.sol[:, particle.solution_keys.index(model_name)]

        abundance_sol = sim_sol / total_biomasses

        data_dict["dataset_label"].append("simulation")
        data_dict["species_label"].append(model_name)
        data_dict["abundance"].append(abundance_sol[-1])

    df = pd.DataFrame.from_dict(data_dict)

    # Make figure
    fig = px.bar(df, x="dataset_label", y="abundance", color="species_label")
    fig.update_layout(template="simple_white", width=600, height=500, autosize=False)

    return fig


def figure_particle_abundance_timeseries(particle, target_data):
    fig = make_subplots(
        rows=len(particle.model_names), cols=1, shared_xaxes=True, shared_yaxes="all"
    )

    for idx, model_name in enumerate(particle.model_names):
        print(model_name)
        # Plot particle simulations
        total_biomasses = np.array(
            [
                get_total_biomass(particle, particle.sol, sim_t_idx)
                for sim_t_idx, t in enumerate(particle.t)
            ]
        )
        sim_sol = particle.sol[:, particle.solution_keys.index(model_name)]

        abundance_sol = sim_sol / total_biomasses

        # Plot experimental data
        exp_data = target_data[model_name].values

        fig.add_trace(
            go.Scatter(
                x=target_data.time,
                y=exp_data,
                name="Experiment",
                legendgroup="Experiment",
                marker={"color": colours[-1]},
            ),
            row=idx + 1,
            col=1,
        )

        fig.add_trace(
            go.Line(
                x=particle.t,
                y=abundance_sol,
                name=model_name,
                opacity=1.0,
                marker={"color": colours[idx]},
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
    fig.update_xaxes(title="Time", row=len(particle.model_names) + 1, col=1)
    # fig.update_yaxes(title="Abundance", type='log')
    fig.update_layout(template="simple_white", width=800, height=600)

    return fig


def figure_particle_biomass_timeseries(particle, target_data):
    fig = make_subplots(
        rows=len(particle.model_names), cols=1, shared_xaxes=True, shared_yaxes="all"
    )

    for idx, model_name in enumerate(particle.model_names):
        print(model_name)

        sim_sol = particle.sol[:, particle.solution_keys.index(model_name)]

        abundance_sol = sim_sol

        # # Plot experimental data
        # exp_data = target_data[model_name].values

        # fig.add_trace(
        #     go.Scatter(
        #         x=target_data.time,
        #         y=exp_data,
        #         name="Experiment",
        #         legendgroup="Experiment",
        #         marker={"color": colours[-1]},
        #     ),
        #     row=idx + 1,
        #     col=1,
        # )

        fig.add_trace(
            go.Line(
                x=particle.t,
                y=abundance_sol,
                name=model_name,
                opacity=1.0,
                marker={"color": colours[idx]},
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
    fig.update_xaxes(title="Time", row=len(particle.model_names) + 1, col=1)
    fig.update_yaxes(title="Abundance", type='log')
    fig.update_layout(template="simple_white", width=800, height=600)

    return fig


def figure_flux_map_continuous(particle):
    print(particle.dynamic_compounds)

    exchange_reactions = [x.replace("M_", "EX_") for x in particle.dynamic_compounds]


    all_species_fluxes = []
    for species in particle.flux_log:
        species_data = [species]
        for ex_r in exchange_reactions:
            if ex_r in particle.flux_log[species].columns:
                
                species_data.append(particle.flux_log[species][ex_r].mean())
            
            else:
                species_data.append(0)
        all_species_fluxes.append(species_data)
    
    df = pd.DataFrame(all_species_fluxes, columns=["species"] + exchange_reactions)
    
    # Remove columns that are all zero
    drop_cols = []
    for col in df.columns[1:]:
        if (df[col] == 0).all():
            drop_cols.append(col)
        
        elif (df[col] >= 0).all() or (df[col] <= 0).all():
            drop_cols.append(col)
    
    df = df.drop(drop_cols, axis=1)

    fig = go.Figure(data=go.Heatmap(
        z=df[df.columns[~df.columns.isin(['species'])]].values,
        y=df.species,
        x=df.columns,
        colorscale='RdBu',
    ))

    return fig

def figure_flux_map_discrete(particle):
    exchange_reactions = [x.replace("M_", "EX_") for x in particle.dynamic_compounds]
    all_species_fluxes = []

    if not hasattr(particle, "flux_log") or particle.flux_log is None:
        print("No flux log found")
        # exit()
        return go.Figure()


    for species in particle.flux_log:
        species_data = [species]
        for ex_r in exchange_reactions:
            if ex_r in particle.flux_log[species].columns:
                
                species_data.append(particle.flux_log[species][ex_r].mean())
            
            else:
                species_data.append(0)
        all_species_fluxes.append(species_data)
    
    df = pd.DataFrame(all_species_fluxes, columns=["species"] + exchange_reactions)
    
    # Remove columns that are all zero
    drop_cols = []
    for col in df.columns[1:]:
        if (df[col] == 0).all():
            drop_cols.append(col)
        
        elif (df[col] >= 0).all() or (df[col] <= 0).all():
            drop_cols.append(col)
    
    df = df.drop(drop_cols, axis=1)

    producer_colour = webcolors.hex_to_rgb(px.colors.qualitative.Dark24[0])
    consumer_colour = webcolors.hex_to_rgb(px.colors.qualitative.Dark24[1])
    neutral_colour = webcolors.hex_to_rgb(px.colors.qualitative.Dark24[5])

    z = df[df.columns[~df.columns.isin(['species'])]].values
    img_colours = np.zeros(shape=z.shape, dtype=object)
    info_mat = np.zeros(shape=z.shape, dtype=object)

    print(z.shape)
    for x_idx in range(z.shape[0]):
        for y_idx in range(z.shape[1]):
            if z[x_idx][y_idx] == 0.0:
                img_colours[x_idx][y_idx] = neutral_colour
            
            elif z[x_idx][y_idx] > 0.0:
                img_colours[x_idx][y_idx] = consumer_colour
            
            elif z[x_idx][y_idx] < 0.0:
                img_colours[x_idx][y_idx] = producer_colour
    
            info_mat[
                x_idx, y_idx
            ] = f"Value: {z[x_idx, y_idx]}\t Species: {df.species.values[x_idx]} \t Compound: {df[df.columns[~df.columns.isin(['species'])]].columns[y_idx]}"

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
        # width=600,
        # height=600,
        showlegend=True,
        title="Discretised mean average flux",
        yaxis_title="Species",
        xaxis_title="Compound",
    )

    fig.update_yaxes(
        dict(
            tickmode="array",
            tickvals=[y for y in range(len(df.species.values))],
            ticktext=df.species.values,
        )
    )

    fig.update_xaxes(
        dict(
            tickmode="array",
            tickvals=[x for x in range(len(df[df.columns[~df.columns.isin(['species'])]].columns))],
            ticktext=df[df.columns[~df.columns.isin(['species'])]].columns,
        )
    )
    
    return fig

def figure_community_biomass(particle):
    fig = make_subplots(
        rows=1, cols=1, shared_xaxes=True, shared_yaxes="all"
    )

    community_biomass = np.zeros(particle.sol[:, particle.solution_keys.index(particle.model_names[0])].shape)
    for idx, model_name in enumerate(particle.model_names):
        community_biomass += particle.sol[:, particle.solution_keys.index(model_name)]

    fig.add_trace(
        go.Line(
            x=particle.t,
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

    return fig


def recalculate_particle_distances(particles, target_data_path):
    exp_sol_keys = [
        [particles[0].model_names[idx], particles[0].model_names[idx]]
        for idx in range(len(particles[0].model_names))
    ]
    dist = distances.DistanceAbundanceError(
        target_data_path, "time", exp_sol_keys, epsilon=1.0, final_epsilon=1.0
    )

    for p in particles:
        d = dist.calculate_distance(p)
        p.distance = d


def main():
    wd = "/Users/bezk/Documents/CAM/research_code/yeast_LAB_coculture/"
    mix_target_data_path = f"{wd}/experimental_data/mel_target_data/target_data_pH7_Mix2_Med2.csv"
    target_data = pd.read_csv(mix_target_data_path)
    experiment_dir = f"{wd}/output/mel_mixes_growth_test_2/mel_multi_mix2_m2_growers/"

    particle_regex = f"{experiment_dir}/generation_*/run_*/*.pkl"
    output_dir = f"{experiment_dir}/"

    particles = load_particles(particle_regex)
    particles = [p for p in particles if hasattr(p, "sol")]

    particles = utils.get_unique_particles(particles)
    # recalculate_particle_distances(particles, mix_target_data_path)

    sum_distances = [sum(p.distance) for p in particles]

    sorted_particles = sorted(
        particles, key=lambda x: sum_distances[particles.index(x)]
    )


    particles = sorted_particles[:25]

    for p in sorted_particles:
        print(max(p.distance))

    particle_blocks = []
    for p_idx, p in enumerate(particles):
        endpoint_abundance_plot = figure_particle_endpoint_abundance(p, target_data)
        timeseries_plot = figure_particle_abundance_timeseries(p, target_data)
        biomass_plot = figure_particle_biomass_timeseries(p, target_data)

        toxin_exchange_fig = figure_particle_toxin_interactions(p)
        met_exchange_fig = figure_particle_metabolite_exchange(p)

        flux_mat_fig = figure_flux_map_discrete(p)

        community_biomass_fig = figure_community_biomass(p)

        abundance_block = dp.Group(
            dp.Plot(endpoint_abundance_plot, responsive=False),
            label="Endpoint abundance",
        )

        timeseries_bloc = dp.Group(
            blocks=[dp.Plot(timeseries_plot, responsive=True), dp.Plot(biomass_plot, responsive=True), dp.Plot(community_biomass_fig, responsive=True)],
            label="Timeseries",
        )

        interactions_block = dp.Group(
            dp.Plot(toxin_exchange_fig, responsive=False),
            dp.Plot(met_exchange_fig, responsive=False),
            dp.Plot(flux_mat_fig, responsive=False),
            label="Interactions",
        )

        particle_block = dp.Group(
            dp.Select(blocks=[abundance_block, timeseries_bloc, interactions_block]),
            label=f"Solution_{p_idx}",
        )
        particle_blocks.append(particle_block)

        # if p_idx == 5:
        #     break

    report = dp.Report(dp.Select(blocks=particle_blocks, type=dp.SelectType.DROPDOWN))
    # report.preview(
    #     open=True, formatting=dp.ReportFormatting(width=dp.ReportWidth.MEDIUM)
    # )

    report.save(output_dir + "report.html", open=False, formatting=dp.ReportFormatting(width=dp.ReportWidth.MEDIUM))


if __name__ == "__main__":
    main()
