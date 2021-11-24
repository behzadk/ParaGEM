from omegaconf import DictConfig, MissingMandatoryValue, open_dict

from hydra import compose, initialize
from omegaconf import OmegaConf

import hydra
from hydra.utils import instantiate

@hydra.main(config_path="../configs", config_name="config")
def my_app(cfg: DictConfig):
    # # context initialization
    # with initialize(config_path="configs", job_name="test_app"):
    #     cfg = compose(config_name="config")

    # print(instantiate(cfg.sampler['max_uptake_dist']))
    # dist_1 = sampling.SampleSkewNormal(loc=-2.0, scale=0.25, alpha=0.0, clip_above_zero=True)
    # print(cfg)
    print(cfg)
    # from pathlib import Path

    # fpath = Path(cfg['community']['media_path']).absolute()
    # alg = instantiate(cfg.algorithm)
    # cfg.algorithm.max_uptake_sampler = dist_1
    alg = instantiate(cfg.algorithm)
    # comm = instantiate(cfg.community)
    
    # alg.base_community = comm
    exit()

    # print(comm)
    # print(type(comm))

if __name__ == "__main__":
    my_app()