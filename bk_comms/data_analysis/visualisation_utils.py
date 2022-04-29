import pickle
import pandas as pd
from glob import glob
import numpy as np

def generate_particle_parameter_df(particles):
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
        + init_species_col[0:-1]
        + k_val_cols
        + lb_constraint_cols
        + toxin_cols
    )

    

    parameters = np.zeros([len(particles), len(column_headings)])
    for idx, p in enumerate(particles):
        # parameters[idx] = p.generate_parameter_vector().reshape(-1)
        param_vec, initial_concs_vec, k_val_vec, max_exchange_vec, toxin_mat = p.generate_parameter_vector()


    df = pd.DataFrame(data=parameters, columns=column_headings)
    print(df)
    return df


def load_particles(particle_regex):
    particle_files = glob(particle_regex)
    particles = []

    for pickle_path in particle_files:
        with open(pickle_path, "rb") as f:
            particles.extend(pickle.load(f))

    return particles
