import numpy as np
import pandas as pd
from typing import List
import glob

from smetana.legacy import Community
from reframed import Environment
from smetana.interface import load_communities
from pathlib import Path
import psutil
import os
import pickle
import cobra
from loguru import logger
import functools


def load_model(model_path, model_name):
    """
    Loads models from model paths
    """

    model = cobra.io.read_sbml_model(model_path, name=model_name)
    model.solver = "cplex"

    return model


def check_particle_equality(patricle_0, particle_1):
    if not np.array_equal(patricle_0.toxin_mat, particle_1.toxin_mat):
        return False

    if not np.array_equal(patricle_0.max_exchange_mat, particle_1.max_exchange_mat):
        return False

    if not np.array_equal(patricle_0.k_vals, particle_1.k_vals):
        return False

    return True


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def get_experiment_repeat_directories(exp_dir, repeat_prefix):
    exp_path = Path(exp_dir)
    sub_dirs = list(exp_path.glob("**"))
    repeat_dirs = []

    for d in sub_dirs:
        if repeat_prefix in d.stem:
            repeat_dirs.append(d.absolute())

    return repeat_dirs


def load_pickle(pickle_path):
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
        return data


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


def load_all_particles(repeat_dirs, distance_filter_epsilon):
    all_particles = []
    for d in repeat_dirs:
        particle_paths = d.glob("particles_*.pkl")
        for p_path in particle_paths:
            p = load_pickle(p_path)
            filtered_p = filter_particles_by_distance(p, distance_filter_epsilon)
            all_particles.extend(filtered_p)

    return all_particles


def filter_particles_by_distance(particles, epsilon):
    filtered_particles = []
    for p in particles:
        for idx, eps in enumerate(epsilon):
            if p.distance[idx] < eps:
                filtered_particles.append(p)

    return filtered_particles


def get_solution_index(comm, sol_element_key):
    idx = comm.solution_keys.index(sol_element_key)
    return idx


def get_mem_usage():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2


def get_community_complete_environment(
    model_paths: List[str], max_uptake=10.0, flavor="cobra"
):
    """
    Generates a complete environment for a given list of model paths.
    """
    # Load smetana community
    model_cache, comm_dict, other_models = load_communities(
        model_paths, None, None, flavor=flavor
    )

    comm_models = [
        model_cache.get_model(org_id, reset_id=True) for org_id in comm_dict["all"]
    ]
    community = Community("all", comm_models, copy_models=False)

    # Use reframed to generate complete environment
    compl_env = Environment.complete(community.merged, max_uptake=max_uptake)

    # Convert to dataframe
    compl_env = pd.DataFrame.from_dict(
        compl_env, columns=["lwr", "upr"], orient="index"
    )
    compl_env.index.name = "reaction"
    compl_env.reset_index(inplace=True)

    # Reformat reaction names
    get_compound = lambda x: x.replace("_pool", "").replace("R_EX_", "")

    compl_env["compound"] = compl_env["reaction"].apply(get_compound)

    return compl_env


def get_crossfeeding_compounds(smetana_df: pd.DataFrame):
    """
    Generates list of metabolites involved in crossfeeding using the output of a SMETANA
    run
    """
    return list(smetana_df.loc[smetana_df["smetana"] > 0.0].compound)


def get_competition_compounds(smetana_df: pd.DataFrame, all_compounds):
    """
    Generates list of metabolites involved in crossfeeding using the output of a SMETANA
    run
    """
    competition_compounds = []

    for m in all_compounds:
        sub_df = smetana_df.loc[smetana_df["compound"] == m]

        if sub_df["mus"].sum() > 1.0:
            competition_compounds.append(m)

    return competition_compounds


def logger_wraps(*, entry=True, exit=True, level="DEBUG"):
    def wrapper(func):
        name = func.__name__

        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            logger_ = logger.opt(depth=1)
            if entry:
                logger_.log(
                    level, "Entering '{}' (args={}, kwargs={})", name, args, kwargs
                )
            result = func(*args, **kwargs)
            if exit:
                logger_.log(level, "Exiting '{}' (result={})", name, result)
            return result

        return wrapped

    return wrapper
