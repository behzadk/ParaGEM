from omegaconf import DictConfig, open_dict

from hydra import compose, initialize
from omegaconf import OmegaConf
import psutil
import os
import hydra
from hydra.utils import instantiate
from pathlib import Path
from hydra import initialize, compose
import functools

from loguru import logger

import bk_comms

from bk_comms.sampling import SampleCombinationParticles
from bk_comms.data_analysis.visualisation import pipeline as vis_pipeline

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


@hydra.main(version_base=None, config_path="./configs")
def test_main(cfg: DictConfig):
    """
    Test the main function
    """
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, f"{cfg.output_dir}/cfg.yaml")

    simulator = test_instantiate_simulator(cfg)

    alg = test_instantiate_algorithm(cfg)


    particles = test_gen_initial_population(alg, cfg)

    test_calculate_particle_distances(alg, particles, 'M3')
    test_save_particles(alg, particles)
    test_load_particles(alg)
    test_combination_particle_sampler(alg)


    vis_pipeline(cfg)



@logger_wraps()
def test_particle_simulation(simulator, media_name, particles, n_processes=2, parallel=True):
    """
    Test the particle simulation
    """
    simulator.simulate_particles(
        sol_key=media_name, n_processes=n_processes, parallel=parallel
    )


@logger_wraps()
def test_instantiate_simulator(cfg):
    """
    Test the instantiation of the simulator
    """
    simulator = instantiate(cfg.simulator)
    return simulator


@logger_wraps()
def test_instantiate_algorithm(cfg: DictConfig):
    """
    Test the instantiation of algorithm function
    """
    alg = instantiate(cfg.algorithm)
    return alg


@logger_wraps()
def test_gen_initial_population(alg, cfg: DictConfig):
    n_processes = cfg.n_processes
    parallel = cfg.parallel

    particles = alg.gen_initial_population(n_processes=n_processes, parallel=parallel)
    return particles


@logger_wraps()
def test_calculate_particle_distances(alg, particles, distance_key):
    alg.calculate_and_set_particle_distances(particles, distance_key)


@logger_wraps()
def test_save_particles(alg, particles):
    """
    Test the load and save of particles
    """
    alg.save_particles(
        particles,
        output_dir=f"{alg.output_dir}",
    )


@logger_wraps()
def test_load_particles(alg):
    """
    Test the load and save of particles
    """
    print(f"{alg.output_dir}/particles_parameter_vectors.npy")
    alg.hotstart_particles(
        hostart_parameter_dir_regex=f"{alg.output_dir}/",
    )


@logger_wraps()
def test_combination_particle_sampler(alg):
    """
    test combination particle sampler

    Depends on previous step success
    """
    input_particles_regex = [alg.output_dir, alg.output_dir]
    population_names = ["E_coli_A", "E_coli_B"]

    sampler = SampleCombinationParticles(
        input_particles_parameter_vector_regex=input_particles_regex,
        population_names=population_names,
    )

    sampler.sample(population_names, data_field="k_vals")


if __name__ == "__main__":
    test_main()
