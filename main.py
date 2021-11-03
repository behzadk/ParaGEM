from community import Community
import distances
from parameter_estimation import RejectionAlgorithm
import pandas as pd
import sampling
import numpy as np


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

    epslilon[1] = 0.019

    distance = distances.DistanceTimeseriesEuclidianDistance(
        exp_data, exp_t_key="time", exp_sol_keys=exp_sol_keys, epsilon=epslilon
    )

    alpha_vals = [-10.0, -9.0, -8.0, -7.0, -5.0]
    for alpha in alpha_vals:
        max_uptake_sampler = sampling.SampleSkewNormal(loc=0.0, scale=1.0, alpha=alpha, clip_zero=False)
        k_val_sampler = sampling.SampleUniform(min_val=1e-5, max_val=2.0, distribution='log_uniform')

        rej = RejectionAlgorithm(
            distance_object=distance,
            base_community=comm,
            max_uptake_sampler=max_uptake_sampler,
            k_val_sampler=k_val_sampler,
            n_particles_batch=6,
            max_population_size=1,
        )

        rej.run(output_name=f"test_alpha_{alpha}")

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

    max_uptake_sampler = sampling.SampleSkewNormal(loc=-1.0, scale=0.75, alpha=0.0, clip_zero=False)
    k_val_sampler = sampling.SampleUniform(min_val=1e-5, max_val=2.0, distribution='log_uniform')

    init_y = np.concatenate(
            (comm.init_population_values, comm.init_compound_values), axis=None
        )

    for x in range(100):
        max_uptake_mat = max_uptake_sampler.sample(size=array_size)
        comm.set_max_exchange_mat(max_uptake_mat)

        #  Sample new K value matrix
        k_val_mat = k_val_sampler.sample(size=array_size)
        comm.set_k_value_matrix(k_val_mat)
        
        comm.sim_step(init_y)

        df = comm.populations[0].model.optimize().to_frame()
        df['name'] = df.index
        df.reset_index(drop=True, inplace=True)
        # print(df)
        # exit()

        # ser_flux = df[df['name'] == 'EX_ser__L_e'].values
        ser_flux = df.loc[df['name'] == 'EX_ser__L_e']['fluxes'].values[0]
        # biomass_flux = df[df['name'] == 'BIOMASS_SC5_notrace'].values
        biomass_flux = df.loc[df['name'] == 'BIOMASS_SC5_notrace']['fluxes'].values[0]

        # if ser_flux > 0:
        print(ser_flux, biomass_flux)

        
    # exp_data = pd.read_csv("./data/Figure1B_fake_data.csv")


if __name__ == "__main__":
    # example_simulation()
    rejection_sampling()
