import multiprocess as mp
import copy
import cometspy
import numpy as np
import time

from loguru import logger
from bk_comms import utils
import os

import shutil


class CometsTimeSeriesSimulation:
    def __init__(
        self,
        t_end,
        dt,
        gurobi_home_dir,
        comets_home_dir,
        batch_dilution=False,
        dilution_factor=0.0,
        dilution_time=0.0,
        media_log_rate=5,
        flux_log_rate=0.0,
    ):
        self.dt = dt
        self.max_cycles = int(np.ceil(t_end / dt))

        self.batch_dilution = batch_dilution
        self.dilution_factor = dilution_factor
        self.dilution_time = dilution_time



        self.comets_home_dir = comets_home_dir
        self.gurobi_home_dir = gurobi_home_dir

        self.media_log_rate = int(media_log_rate)
        self.flux_log_rate = int(flux_log_rate)
        

        os.environ["GUROBI_HOME"] = gurobi_home_dir
        os.environ["GUROBI_COMETS_HOME"] = gurobi_home_dir
        os.environ["COMETS_HOME"] = comets_home_dir

    def convert_models(self, community):
        """Converts cobrapy models of community to comets model inplace"""
        for idx, population in enumerate(community.populations):

            if isinstance(population.model, cometspy.model):
                continue

            else:
                population.model = cometspy.model(population.model)
                population.model.id = population.name
                population.model.open_exchanges()

    def set_lb_constraint(self, layout, community):
        """
        Updates the model exchange bounds according to the community
        max uptake matrix
        """
        for idx, model in enumerate(layout.models):
            lower_bound_constraints = community.max_exchange_mat[idx]

            for cmpd_idx, cmpd in enumerate(community.dynamic_compounds):
                cmpd_str = community.dynamic_compounds[cmpd_idx]
                cmpd_str = cmpd_str.replace("M_", "EX_")

                # Change michaelis menten vmax
                # model.change_vmax(cmpd_str, lower_bound_constraints[cmpd_idx])

                # Update model lower bound
                model.change_bounds(cmpd_str, lower_bound_constraints[cmpd_idx], 1000)


    def set_k_values(self, layout, community):
        for idx, model in enumerate(layout.models):
            k_values = community.k_vals[idx]

            for cmpd_idx, cmpd in enumerate(community.dynamic_compounds):
                cmpd_str = community.dynamic_compounds[cmpd_idx]
                cmpd_str = cmpd_str.replace("M_", "EX_")

                # Change michaelis menten vmax
                # model.change_vmax(cmpd_str, lower_bound_constraints[cmpd_idx])

                # Update model lower bound
                model.change_km(cmpd_str, k_values[cmpd_idx])

    def set_layout_metabolite_concentrations(self, layout, community):
        # Set toxin compound initial concentrations to zero
        # will be overriden if defined in media file
        for idx, toxin_name in enumerate(community.toxin_names):
            toxin_name = toxin_name + "_e"
            layout.set_specific_metabolite(toxin_name, 1e-30)

        # Set dynamic compound initial concentrations
        for idx, cmpd in enumerate(community.dynamic_compounds):
            metabolite_str = cmpd.replace("M_", "")
            layout.set_specific_metabolite(
                metabolite_str, community.init_compound_values[idx]
            )

        #  Fill non dynamic compound concentrations
        non_dynamic_media_df = community.curr_media_df.loc[
            community.media_df["dynamic"] == 0
        ]
        for idx, row in non_dynamic_media_df.iterrows():
            cmpd_str = row.compound
            cmpd_str = cmpd_str.replace("M_", "")
            cmpd_str = cmpd_str + "_e"
            layout.set_specific_metabolite(cmpd_str, row.mmol_per_L, static=True)



    def set_model_initial_pop(self, layout, community):
        for idx, pop in enumerate(community.populations):
            model_init_pop = community.init_population_values[idx]
            pop.model.initial_pop = [0, 0, model_init_pop]

    def set_toxin_interactions(self, layout, community):
        for idx_i, donor_pop in enumerate(community.populations):
            for idx_j, recipient_pop in enumerate(community.populations):
                if community.toxin_mat[idx_i][idx_j] != 0:

                    # Find the index of the toxin in the recipient
                    # population metabolite list
                    toxin_e_index = (
                        list(recipient_pop.model.get_exchange_metabolites()).index(
                            f"toxin_{donor_pop.name}_e"
                        )
                        + 1
                    )

                    recipient_pop.model.add_signal(
                        rxn_num="death",
                        exch_ind=toxin_e_index,
                        bound="met_unchanged",
                        function="linear",
                        parms=[community.toxin_mat[idx_i][idx_j], 0.0],
                    )

    def load_layout_models(self, layout, community):
        for pop in community.populations:
            layout.add_model(pop.model)

    def process_experiment(self, comm, experiment):
        media_df = experiment.media
        media_df["t"] = media_df["cycle"] * self.dt
        met_df = experiment.get_metabolite_time_series(upper_threshold=None)
        met_df["t"] = met_df["cycle"] * self.dt
        biomass_df = experiment.total_biomass
        biomass_df["t"] = biomass_df["cycle"] * self.dt

        t = biomass_df["t"].values
        sol = np.zeros([len(t), len(comm.solution_keys)])

        for idx, s in enumerate(comm.solution_keys):
            if s in comm.model_names:
                for t_val in t:
                    t_idx = utils.find_nearest(biomass_df["t"].values, t_val)
                    sol[:, idx][t_idx] = biomass_df[s].values[t_idx]

            else:
                s = s.replace("M_", "")
                try:
                    s_df = met_df[[s, "t"]]
                    for t_val in t:
                        t_idx = utils.find_nearest(s_df["t"].values, t_val)
                        sol[:, idx][t_idx] = s_df[s].values[t_idx]
                    

                except KeyError:
                    for t_val in t:
                        t_idx = utils.find_nearest(s_df["t"].values, t_val)
                        sol[:, idx][t_idx] = np.nan

        return sol, t        

    def simulate(self, community, idx=0):
        self.convert_models(community)

        layout = cometspy.layout()
        self.set_model_initial_pop(layout, community)
        self.load_layout_models(layout, community)
        self.set_lb_constraint(layout, community)
        self.set_k_values(layout, community)
        self.set_toxin_interactions(layout, community)

        self.set_layout_metabolite_concentrations(layout, community)

        layout.media.reset_index(inplace=True)

        sim_params = cometspy.params()
        sim_params.set_param("defaultVmax", 18.5)
        sim_params.set_param("defaultKm", 0.000015)
        sim_params.set_param("maxCycles", self.max_cycles)
        sim_params.set_param("timeStep", self.dt)
        sim_params.set_param("spaceWidth", 1)
        sim_params.set_param("maxSpaceBiomass", 10)
        sim_params.set_param("minSpaceBiomass", 1e-11)
        sim_params.set_param("writeMediaLog", True)
        sim_params.set_param("MediaLogRate", self.media_log_rate)


        # Optional parameters
        if self.batch_dilution:
            sim_params.set_param("batchDilution", True)
            sim_params.set_param("dilFactor", self.dilution_factor)
            sim_params.set_param("dilTime", self.dilution_time)

        if self.flux_log_rate == 0.0:
            sim_params.set_param("writeFluxLog", False)

        else:
            sim_params.set_param("writeFluxLog", True)
            sim_params.set_param("FluxLogRate", self.flux_log_rate)


        tmp_dir = f"./tmp_{os.getpid()}/"
        if not os.path.exists(tmp_dir):
            os.mkdir(tmp_dir)


        experiment = cometspy.comets(layout, sim_params, relative_dir=tmp_dir)
        self.experiment = experiment

        experiment.set_classpath("bin", f"{self.comets_home_dir}/bin/comets.jar")

        # experiment.set_classpath(
        #     "gurobi", f"{self.gurobi_home_dir}/lib/gurobi.jar"
        # )

        experiment.set_classpath(
            "jdistlib", f"{self.comets_home_dir}/lib/jdistlib/jdistlib-0.4.5-bin.jar"
        )

        experiment.run(delete_files=False)

        sol, t = self.process_experiment(community, experiment)
        
        if self.flux_log_rate == 0.0:
            experiment.fluxes_by_species = None
        


        print(tmp_dir)
        shutil.rmtree(tmp_dir)

        return sol, t, experiment

    def simulate_particles(
        self, particles, sol_key, n_processes=1, sim_timeout=1000.0, parallel=True
    ):
        
        if parallel:
            print("running parallel")

            def wrapper(args):
                idx, args = args
                sol, t, experiment= self.simulate(args)
                return (idx, sol, t, experiment)

            pool = mp.get_context("spawn").Pool(n_processes, maxtasksperchild=1)
            futures_mp_sol = pool.imap_unordered(wrapper, enumerate(particles))

            for particle in particles:
                try:
                    idx, sol, t, experiment = futures_mp_sol.next(timeout=sim_timeout)
                    particles[idx].sol[sol_key] = sol
                    particles[idx].t = t

                    
                except mp.context.TimeoutError:
                    print("TIMEOUT ERROR")
                    break

                try:
                    particles[idx].flux_log = experiment.fluxes_by_species

                except:
                    pass

            print("Terminating pool")
            pool.terminate()

            for idx, p in enumerate(particles):
                if not hasattr(p, "sol"):
                    print(f"Particle {idx} has no sol")
                    p.sol = None
                    p.t = None

        else:
            p_idx = 0
            for p in particles:
                start = time.time()
                print(f"Simulating particle idx: {p_idx}")
                sol, t, experiment = self.simulate(p)
                p.sol = sol
                p.t = t
                p.flux_log = experiment.fluxes_by_species
                p_idx += 1
                end = time.time()
                print("Sim time: ", end - start)


class GrowthRateConditionedMedia:
    def __init__(self, conditioner_particles_dir, condition_media_time_sample):
        self.conditioner_particles_dir = conditioner_particles_dir
        self.condition_media_time_sample = condition_media_time_sample

        repeat_prefix = "run_"
        run_dirs = [
            utils.get_experiment_repeat_directories(
                exp_dir=x, repeat_prefix=repeat_prefix
            )
            for x in self.conditioner_particles_dir
        ]

        self.conditoner_particles = [utils.load_all_particles(x) for x in run_dirs]

        for idx, _ in enumerate(self.input_particles):
            self.input_particles[idx] = utils.filter_particles_by_distance(
                self.input_particles[idx], epsilon=epsilon[idx]
            )

    def simulate(self, particle):
        n_populations = len(particle.populations)
        n_conditioned_medias = len(self.conditoner_particles)

        particle_growth_rates = np.zeros(shape=[n_conditioned_medias, n_populations])

        for idx, condit_part in enumerate(self.conditoner_particles):
            t_idx = utils.find_nearest(condit_part.t, self.condition_media_time_sample)
            conditioned_sol = condit_part.sol[t_idx]

            growth_rates, flux_matrix = particle.sim_step(conditioned_sol)
            particle_growth_rates[idx] = growth_rates

        return particle_growth_rates

    def simulate_particles(self, particles):
        for p in particles:
            p.sol = self.simulate(p)


class GrowthRate:
    def __init__(self, t_end):
        self.t_end = t_end

    def simulate(self, particle):
        n_populations = len(particle.populations)
        particle_growth_rates = np.zeros(shape=[1, n_populations])

        particle.set_init_y()
        
        growth_rates, flux_matrix = particle.sim_step(particle.init_y)
        particle_growth_rates[0] = growth_rates

        return particle_growth_rates

    def simulate_particles(self, particles, n_processes=None, parallel=None):
        for p in particles:
            p.sol = self.simulate(p)
            p.t = [self.t_end]
