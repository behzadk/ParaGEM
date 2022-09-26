from omegaconf import DictConfig

from omegaconf import OmegaConf

import hydra
from hydra.utils import instantiate
from pathlib import Path

from loguru import logger
from paragem.data_analysis.visualisation import prepare_particles_df


@hydra.main(version_base=None, config_path="./configs")
def run_algorithm(cfg: DictConfig):

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, f"{cfg.experiment_dir}/cfg.yaml")

    logger.remove()
    logger.add(f"{cfg.experiment_dir}info_log.log", level="DEBUG")

    alg = instantiate(cfg.algorithm)
    alg.run(n_processes=cfg.n_processes, parallel=cfg.parallel)

    particles_df = prepare_particles_df(cfg)
    print(particles_df.head(5))

    visualisation_pipeline = cfg.visualisation

    for step in visualisation_pipeline:
        print(step)
        step = instantiate(step, particles_df=particles_df)

    # vis_pipeline(cfg)


if __name__ == "__main__":
    run_algorithm()
