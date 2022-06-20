from omegaconf import DictConfig, open_dict

from hydra import compose, initialize
from omegaconf import OmegaConf

import hydra
from hydra.utils import instantiate
from pathlib import Path

from loguru import logger

import warnings
from bk_comms.data_analysis.visualisation import pipeline as vis_pipeline

# warnings.filterwarnings("ignore")


@hydra.main(version_base=None, config_path="./configs")
def run_algorithm(cfg: DictConfig):

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, f"{cfg.output_dir}/cfg.yaml")

    logger.remove()
    logger.add(f"{cfg.output_dir}info_log.log", level="DEBUG")

    alg = instantiate(cfg.algorithm)

    if not isinstance(cfg.algorithm.hotstart_particles_regex, type(None)):
        alg.hotstart_particles(cfg.algorithm.hotstart_particles_regex)

    # Write config once folder structuer has ben made
    alg.run(n_processes=cfg.n_processes, parallel=cfg.parallel)

    vis_pipeline(cfg)


if __name__ == "__main__":
    run_algorithm()
