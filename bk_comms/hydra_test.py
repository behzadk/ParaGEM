from omegaconf import DictConfig, MissingMandatoryValue, open_dict

from hydra import compose, initialize
from omegaconf import OmegaConf

import hydra
from hydra.utils import instantiate
from pathlib import Path

from loguru import logger


@hydra.main(config_path="../configs", config_name="base")
def run_algorithm(cfg: DictConfig):

    if cfg['load_path'] is not None:
        cfg['cfg'] = OmegaConf.load(cfg.load_path)
    

    cfg = cfg['cfg']
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(f"{cfg.output_dir}info_log.log", level="DEBUG")


    OmegaConf.save(cfg, f'{cfg.output_dir}/cfg.yaml')

    alg = instantiate(cfg.algorithm)
    
    # Write config once folder structuer has ben made
    alg.run(n_processes=cfg.n_processes, parallel=cfg.parallel)