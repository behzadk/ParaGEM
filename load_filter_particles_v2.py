from unittest.mock import patch
import pandas as pd
import numpy as np
import bk_comms
from bk_comms.community import Community
import copy
import pickle
from pathlib import Path

def update_particle(particle, dir, particle_index):
    init_populations_arr = np.load(f"{dir}/particle_init_populations.npy")
    k_values_arr = np.load(f"{dir}/particle_k_values.npy", )
    max_exchange_arr = np.load(f"{dir}/particle_max_exchange.npy", )

    particle.set_initial_populations(init_populations_arr[particle_index])
    particle.set_k_value_matrix(k_values_arr[particle_index])
    particle.set_max_exchange_mat(max_exchange_arr[particle_index])


def cleanup_data(particle):
    try:
        del particle.model
        del particle.initial_population_prior
        del particle.max_exchange_prior
        del particle.k_val_prior
        del particle.toxin_interaction_prior

    except:
        pass

def main():
    wd = "/rds/user/bk445/hpc-work/yeast_LAB_coculture/"
    models_dir = f'{wd}/models/melanie_data_models'
    input_dir = f"{wd}/output/indiv_growth_4/"
    output_dir = f"{wd}/output/indiv_growth_4_filtered/"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    smetana_analysis_path = f"{wd}/smetana_analysis/melanie_data_detailed.tsv"
    media_path = f"{wd}/media/melanie_data_media.tsv"

    population_epsilons = {
        "B_longum_subsp_longum": 4.0,
        "C_perfringens_S107": 14.0,
        "E_coli_IAI1": 9.172,
        "L_paracasei": 10.36,
        "L_plantarum": 14.57,
        "S_salivarius": 18.1,
        "C_ramosum": 7.30,
    }


    for pop_name in population_epsilons:
        kept_particles = []
        pop_out_dir = f"{output_dir}/{pop_name}/generation_0/run_0/"
        Path(pop_out_dir).mkdir(parents=True, exist_ok=True)

        base_comm = Community(model_names=[pop_name], 
        model_paths=[f'{models_dir}/{pop_name}.xml'],
        smetana_analysis_path=smetana_analysis_path,
        media_path=media_path, 
        objective_reaction_keys=['Growth'],
        enable_toxin_interactions=False)

        particles_df = pd.read_csv(f'{input_dir}/{pop_name}/particles_df.csv')
        particles_df = particles_df.loc[particles_df['sum_distance'] < population_epsilons[pop_name]]

        for idx, row in particles_df.iterrows():
            new_particle = copy.deepcopy(base_comm)
            d = row['data_dir']
            p_idx = row['particle_index']

            update_particle(new_particle, d, p_idx)

            cleanup_data(new_particle)
            kept_particles.append(new_particle)


        kept_particles = bk_comms.utils.get_unique_particles(kept_particles)

        output_path = f"{pop_out_dir}/particles_filtered_0.pkl"
        
        print(pop_name, len(kept_particles))

        with open(f"{output_path}", "wb") as handle:
            pickle.dump(kept_particles, handle)





if __name__ == "__main__":
    main()
