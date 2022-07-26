from bk_comms import utils
import pickle
from pathlib import Path
import sys
import numpy as np
import gc
from glob import glob


def load_filter_save_particles(run_dirs, population_names, epsilon, output_wd):
    print("Loading and filtering particles...")
    for idx, x in enumerate(run_dirs):
        sys.stdout.flush()

        filtered_particles = utils.load_all_particles(x, [10000])


        print("Before distance filter: ", len(filtered_particles))
        filtered_particles = [p for p in filtered_particles if hasattr(p, 'sol')]

        min_dist = 1000 
        for p in filtered_particles:
            if sum(p.distance) < min_dist:
                min_dist = sum(p.distance)
        
        print("")
        print(population_names[idx])
        print("Min population distance: ", min_dist)
        print("")

        filtered_particles = [p for p in filtered_particles if sum(p.distance) < epsilon[idx]]

        print("Filtered particles:", len(filtered_particles))
        sys.stdout.flush()

        filtered_particles = utils.get_unique_particles(filtered_particles)

        for p in filtered_particles:
            cleanup_data(p)

        print(population_names[idx], len(filtered_particles), epsilon[idx])
        output_dir = f"{output_wd}/{population_names[idx]}/generation_0/run_0/"
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        chunked_filtered_particles = np.array_split(filtered_particles, 1)

        for chunk_idx, chunk in enumerate(chunked_filtered_particles):
            output_path = f"{output_dir}/particles_filtered_{chunk_idx}.pkl"

            with open(f"{output_path}", "wb") as handle:
                pickle.dump(chunk, handle)
        
        del chunked_filtered_particles
        del filtered_particles
        gc.collect()



def cleanup_data(particle):
    try:
        del particle.initial_population_prior
        del particle.max_exchange_prior
        del particle.k_val_prior
        del particle.toxin_interaction_prior

    except:
        pass

def main_multi():
    input_dir = "./output/mel_mixes_growth_10/mel_multi_mix2_m3_growers/"
    output_dir = "./output/mel_mixes_growth_10_filtered/mel_multi_mix2_m3_growers/generation_0/run_0/"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    particle_paths = glob(f"{input_dir}/generation_*/run_*/*.pkl")

    epsilon = 0.41

    particle_count = 0
    for idx, part_path in enumerate(particle_paths):
        particles = utils.load_pickle(part_path)
        filtered_particles = [p for p in particles if hasattr(p, 'sol')]
        filtered_particles = [p for p in filtered_particles if sum(p.distance) < epsilon]
        output_path = f"{output_dir}/particles_filtered_{idx}.pkl"
        
        particle_count += len(filtered_particles)
        if len(filtered_particles) == 0:
            continue

        with open(f"{output_path}", "wb") as handle:
            pickle.dump(filtered_particles, handle)

        print(particle_count)


def main():
    input_dir = "./output/mel_indiv_growth_10/"
    output_wd = "./output/mel_indiv_growth_10_filtered/"

    population_names = ['B_longum_subsp_longum',
    'C_perfringens_S107',
    'E_coli_IAI1',
    'L_paracasei',
    'L_plantarum',
    'S_salivarius',
    "C_ramosum"
    ]

    species_folders = [
        "B_longum_subsp_longum",
        "C_perfringens_S107",
        "E_coli_IAI1",
        "L_paracasei",
        "L_plantarum",
        "S_salivarius",
        "C_ramosum"
    ]


    epsilon = [
        [2.0],
        [13.5],
        [5.55],
        [7.5],
        [19.0],
        [5.0],
        [5.5]]

    input_experiment_dirs = []

    for s in species_folders:
        input_experiment_dirs.append(input_dir + s + "/")

    repeat_prefix = "run_"

    run_dirs = []

    for x in input_experiment_dirs:
        run_dirs.append(
            utils.get_experiment_repeat_directories(
                exp_dir=x, repeat_prefix=repeat_prefix
            )
        )

    load_filter_save_particles(run_dirs, population_names, epsilon, output_wd)



if __name__ == "__main__":\
    main_multi()
    # main()
