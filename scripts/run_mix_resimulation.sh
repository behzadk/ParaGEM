#!/bin/bash

which python

cd ..

python setup.py build install
# python bk_comms cfg=mel_multi_mix2_m2_growers cfg/algorithm=default_simulate cfg.simulator.t_end=432 cfg.algorithm.max_simulations=25 cfg.algorithm.n_particles_batch=5 cfg.algorithm.hotstart_particles_regex="\${cfg.wd}/output/mel_mixes_growth_2/mel_multi_mix2_m2_growers/generation_*/run_*/*.pkl" +cfg.simulator.flux_log_rate=1.0 cfg.output_dir="\${cfg.wd}/output/mel_mixes_growth_2/resim_mel_multi_mix2_m2_growers/generation_0/"

python -u bk_comms cfg=mel_multi_mix2_m2_growers cfg/algorithm=default_simulate cfg.simulator.t_end=95.9 cfg.community.media_name=M2 cfg.algorithm.max_simulations=50 cfg.algorithm.n_particles_batch=3 +cfg.simulator.flux_log_rate=1.0 cfg.algorithm.hotstart_particles_regex="\${cfg.wd}/output/mel_mixes_growth_9_filtered/mel_multi_mix2_m3_growers/generation_*/run_*/*.pkl" cfg.output_dir="\${cfg.wd}/output/mel_mixes_growth_9_filtered/resim_2_mel_multi_mix2_m2_growers_M2/generation_0/run_0/"

# python -u bk_comms cfg=mel_multi_mix2_m2_growers cfg/algorithm=default_simulate cfg.simulator.t_end=239.9 cfg.community.media_name=M2 cfg.algorithm.max_simulations=25 cfg.algorithm.n_particles_batch=5 cfg.algorithm.hotstart_particles_regex="\${cfg.wd}/output/mel_mixes_growth_9_filtered/mel_multi_mix2_m3_growers/generation_*/run_*/*.pkl" +cfg.simulator.flux_log_rate=1.0 cfg.output_dir="\${cfg.wd}/output/mel_mixes_growth_9_filtered/resim_2_mel_multi_mix2_m2_growers_M2/generation_0/run_0/"

# python -u bk_comms cfg=mel_multi_mix2_m2_growers cfg/algorithm=default_simulate cfg.community.media_name=M3 cfg.algorithm.max_simulations=5 cfg.algorithm.n_particles_batch=5 cfg.algorithm.hotstart_particles_regex="\${cfg.wd}/output/mel_mixes_growth_4/mel_multi_mix2_m2_growers/generation_*/run_*/*.pkl" +cfg.simulator.flux_log_rate=1.0 cfg.output_dir="\${cfg.wd}/output/mel_mixes_growth_4/resim_2_mel_multi_mix2_m2_growers_M3/generation_0/run_0/"

