from omegaconf import DictConfig, open_dict

from hydra import compose, initialize
from omegaconf import OmegaConf

import hydra
from hydra.utils import instantiate
from pathlib import Path

from loguru import logger
from bk_comms.parameter_estimation import GeneticAlgorithm

import warnings

warnings.filterwarnings("ignore")


@hydra.main(config_path="../configs", config_name="base")
def run_algorithm(cfg: DictConfig):
    if cfg["load_path"] is not None:
        cfg["cfg"] = OmegaConf.load(cfg.load_path)

    Path(cfg.cfg.output_dir).mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, f"{cfg.cfg.output_dir}/cfg.yaml")

    cfg = cfg["cfg"]
    
    logger.remove()
    logger.add(f"{cfg.output_dir}info_log.log", level="DEBUG")

    print(cfg['filter'])

    alg = instantiate(cfg.algorithm)

    # logger.info("Instantiating algorithm")

    # alg = GeneticAlgorithm(
    #     n_particles_batch=n_particles_batch,
    #     population_size=population_size,
    #     mutation_probability=mutation_probability,
    #     epsilon_alpha=epsilon_alpha,
    #     simulator=simulator,
    #     base_community=base_community,
    #     experiment_name=experiment_name,
    #     distance_object=distance_object,
    #     max_uptake_sampler=max_uptake_sampler,
    #     k_val_sampler=k_val_sampler,
    #     init_population_sampler=init_population_sampler,
    #     output_dir=output_dir,
    #     filter=filter
    #     )
    # logger.info("Run")

    # Write config once folder structuer has ben made
    alg.run(n_processes=cfg.n_processes, parallel=cfg.parallel)
