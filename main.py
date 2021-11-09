import sys
from loguru import logger
from community import Community
import distances
from parameter_estimation import RejectionAlgorithm
from parameter_estimation import GeneticAlgorithm
from parameter_estimation import SpeedTest

import pandas as pd
import sampling
import numpy as np
from community import sim_community
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

import argparse
import psutil
import os
import dask
from dask.distributed import Client

def rejection_sampling():
    model_paths = [
        # "./models/L_lactis/L_lactis_fbc.xml",
        "./models/S_cerevisiae/iMM904.xml",
    ]

    model_names = ["iMM904"]

    smetana_analysis_path = "./carveme_output/lactis_cerevisiae_detailed.tsv"
    media_path = "./media_db_CDM35.tsv"

    comm = Community(
        model_names,
        model_paths,
        smetana_analysis_path,
        media_path,
        "CDM35",
        use_parsimonius_fba=False,
    )

    print(len(comm.reaction_keys))
    print(len(comm.dynamic_compounds))
    print(comm.dynamic_compounds)
    print(comm.reaction_keys)

    exp_data = pd.read_csv("./data/Figure1B_fake_data.csv")

    exp_sol_keys = [
        ["yeast_dcw", "iMM904"],
        ["yeast_ser_mm", "M_ser__L_e"],
        ["yeast_ala_mm", "M_ala__L_e"],
        ["yeast_glyc_mm", "M_glyc_e"],
    ]

    epslilon = [3.0]
    for _ in range(1, len(exp_sol_keys)):
        epslilon.append(1000.0)

    epslilon[0] = 1.5
    epslilon[1] = 0.019

    distance = distances.DistanceTimeseriesEuclidianDistance(
        exp_data, exp_t_key="time", exp_sol_keys=exp_sol_keys, epsilon=epslilon
    )

    alpha_vals = [-5.0]
    for alpha in alpha_vals:
        dist_1 = sampling.SampleSkewNormal(
            loc=-2.0, scale=0.1, alpha=0.0, clip_zero=False
        )
        dist_2 = sampling.SampleUniform(
            min_val=1e-3, max_val=1e-1, distribution="log_uniform"
        )
        max_uptake_sampler = multi_dist = sampling.MultiDistribution(
            dist_1, dist_2, prob_dist_1=0.95
        )

        k_val_sampler = sampling.SampleUniform(
            min_val=1e-5, max_val=2.0, distribution="log_uniform"
        )

        rej = RejectionAlgorithm(
            distance_object=distance,
            base_community=comm,
            max_uptake_sampler=max_uptake_sampler,
            k_val_sampler=k_val_sampler,
            n_particles_batch=6,
            max_population_size=2,
        )

        rej.run(output_name=f"test_alpha_{alpha}")


def genetic_algorithm(experiment_name, output_dir):
    # Make output directories
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(f"./{output_dir}{experiment_name}.log", level="DEBUG")

    model_paths = [
        # "./models/L_lactis/L_lactis_fbc.xml",
        "./models/S_cerevisiae/iMM904.xml",
    ]

    model_names = ["iMM904"]

    smetana_analysis_path = "./carveme_output/lactis_cerevisiae_detailed.tsv"
    media_path = "./media_db_CDM35.tsv"

    comm = Community(
        model_names,
        model_paths,
        smetana_analysis_path,
        media_path,
        "CDM35",
        use_parsimonius_fba=False,
    )

    exp_data = pd.read_csv("./data/Figure1B_fake_data.csv")
    exp_sol_keys = [
        ["yeast_dcw", "iMM904"],
        ["yeast_ser_mm", "M_ser__L_e"],
        ["yeast_ala_mm", "M_ala__L_e"],
    ]

    epslilon = [3.0, 1000, 1000]
    final_epsion = [0.5, 0.008, 0.008]

    distance = distances.DistanceTimeseriesEuclidianDistance(
        exp_data,
        exp_t_key="time",
        exp_sol_keys=exp_sol_keys,
        epsilon=epslilon,
        final_epsion=final_epsion,
    )

    dist_1 = sampling.SampleSkewNormal(loc=-2.0, scale=0.1, alpha=0.0, clip_zero=False)
    dist_2 = sampling.SampleUniform(
        min_val=1e-3, max_val=1e-1, distribution="log_uniform"
    )
    max_uptake_sampler = multi_dist = sampling.MultiDistribution(
        dist_1, dist_2, prob_dist_1=0.95
    )

    k_val_sampler = sampling.SampleUniform(
        min_val=1e-5, max_val=2.0, distribution="log_uniform"
    )

    ga = GeneticAlgorithm(
        experiment_name=experiment_name,
        distance_object=distance,
        base_community=comm,
        max_uptake_sampler=max_uptake_sampler,
        k_val_sampler=k_val_sampler,
        output_dir=output_dir,
        n_particles_batch=6,
        population_size=5,
        mutation_probability=0.1,
        epsilon_alpha=0.3,
        parallel=True
    )

    # checkpoint_path = './output/exp_yeast_ga_fit/run_0/yeast_ga_0_checkpoint_2021-11-08_171838.pkl'
    # ga = ga.load_checkpoint(checkpoint_path)
    # logger.info(f"Checkpoint loaded. mem usage: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2}")

    dask_client = Client(processes=True, 
    threads_per_worker=1, 
    n_workers=6, 
    timeout="3600s")

    ga.run(dask_client, parallel=True)

    dask_client.shutdown()


def example_simulation():
    model_paths = [
        # "./models/L_lactis/L_lactis_fbc.xml",
        "./models/S_cerevisiae/iMM904.xml",
    ]

    model_names = ["iMM904"]

    smetana_analysis_path = "./carveme_output/lactis_cerevisiae_detailed.tsv"
    media_path = "./media_db_CDM35.tsv"

    comm = Community(
        model_names,
        model_paths,
        smetana_analysis_path,
        media_path,
        "CDM35",
        use_parsimonius_fba=False,
    )

    # Sample
    array_size = [len(comm.populations), len(comm.dynamic_compounds)]

    max_uptake_sampler = sampling.SampleSkewNormal(
        loc=-1.0, scale=0.75, alpha=0.0, clip_zero=False
    )
    k_val_sampler = sampling.SampleUniform(
        min_val=1e-5, max_val=2.0, distribution="log_uniform"
    )

    # init_y = np.concatenate(
    #         (comm.init_population_values, comm.init_compound_values), axis=None
    #     )
    dist_1 = sampling.SampleSkewNormal(loc=-1.0, scale=0.1, alpha=0.0, clip_zero=False)
    dist_2 = sampling.SampleUniform(
        min_val=1e-5, max_val=1e-3, distribution="log_uniform"
    )
    multi_dist = sampling.MultiDistribution(dist_1, dist_2, prob_dist_1=0.95)

    for x in range(100):
        max_uptake_mat = multi_dist.sample(size=array_size)
        comm.set_max_exchange_mat(max_uptake_mat)

        #  Sample new K value matrix
        k_val_mat = k_val_sampler.sample(size=array_size)
        comm.set_k_value_matrix(k_val_mat)

        comm.sim_step(comm.init_y)

        df = comm.populations[0].model.optimize().to_frame()
        df["name"] = df.index
        df.reset_index(drop=True, inplace=True)

        # ser_flux = df[df['name'] == 'EX_ser__L_e'].values
        ser_flux = df.loc[df["name"] == "EX_ser__L_e"]["fluxes"].values[0]
        # biomass_flux = df[df['name'] == 'BIOMASS_SC5_notrace'].values
        biomass_flux = df.loc[df["name"] == "BIOMASS_SC5_notrace"]["fluxes"].values[0]

        if ser_flux > 0 and biomass_flux > 0:
            print(ser_flux, biomass_flux)
            # sol, t = sim_community(comm)
            # plt.plot(t, sol[:, 0])
            # plt.show()
            # plt.close()

    # exp_data = pd.read_csv("./data/Figure1B_fake_data.csv")

def speed_test():
    output_dir = "./output/exp_yeast_ga_fit/"
    experiment_name = 'speed_test'
    logger.remove()
    logger.add(f"./{output_dir}{experiment_name}.log", level="DEBUG")
    
    model_paths = [
        # "./models/L_lactis/L_lactis_fbc.xml",
        "./models/S_cerevisiae/iMM904.xml",
    ]

    model_names = ["iMM904"]
    smetana_analysis_path = "./carveme_output/lactis_cerevisiae_detailed.tsv"
    media_path = "./media_db_CDM35.tsv"

    comm = Community(
        model_names,
        model_paths,
        smetana_analysis_path,
        media_path,
        "CDM35",
        use_parsimonius_fba=False,
    )

    particles_path = './output/exp_test/particles_yeast_ga_0_final_gen_8.pkl'
    s = SpeedTest(particles_path, comm, n_processes=6)
    s.speed_test()

    
if __name__ == "__main__":
    # example_simulation()
    # rejection_sampling()
    # speed_test()
    # exit()

    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-r','--run_idx', help='Description for foo argument', required=True)
    args = vars(parser.parse_args())



    run_idx = args['run_idx']
    output_dir = f"./output/exp_yeast_ga_fit/run_{run_idx}/"

    genetic_algorithm(f"yeast_ga_{run_idx}", output_dir)
