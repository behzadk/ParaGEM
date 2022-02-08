import multiprocess as mp
import copy
import cometspy
import numpy as np
import time
from scipy.integrate import ode
from scipy.integrate import odeint

from loguru import logger
from bk_comms import utils
import os

import shutil


class TimeSeriesSimulation:
    def __init__(self, t_0, t_end, steps, method="vode"):
        self.t_0 = t_0
        self.t_end = t_end
        self.steps = steps
        self.method = method

    def simulate(self, particle):
        y0 = copy.deepcopy(particle.init_y)
        t_0 = self.t_0
        t_end = self.t_end
        steps = self.steps
        method = self.method

        if method == "odeint":
            t = np.linspace(t_0, t_end, steps)
            sol = odeint(
                particle.diff_eqs,
                y0,
                t,
                args=(),
                Dfun=particle.calculate_jacobian,
                col_deriv=True,
                mxstep=10000,
            )

        elif method == "vode":
            sol = []
            t = []
            steps = 100
            t_points = list(np.linspace(t_0, t_end, steps))

            # Approximately set atol, rtol
            atol_list = []
            rtol_list = []
            for y in y0:
                if y > 50.0:
                    atol_list.append(1e-3)
                    rtol_list.append(1e-3)
                else:
                    atol_list.append(1e-6)
                    rtol_list.append(1e-3)

            start = time.time()
            solver = ode(particle.diff_eqs_vode, jac=None).set_integrator(
                "vode", method="bdf", atol=1e-4, rtol=1e-3, max_step=0.01
            )
            # print("solver initiated")

            solver.set_initial_value(y0, t=t_0)

            while solver.successful() and solver.t < t_end:
                step_out = solver.integrate(t_end, step=True)

                if solver.t >= t_points[0]:
                    mx = np.ma.masked_array(step_out, mask=step_out == 0)
                    sol.append(step_out)
                    t.append(solver.t)
                    t_points.pop(0)

            sol = np.array(sol)
            t = np.array(t)
            end = time.time()
            logger.info(f"Simulation time: {end - start}")

        return sol, t

    def simulate_particles(
        self, particles, n_processes=1, sim_timeout=360.0, parallel=True
    ):
        if parallel:
            print("running parallel")

            def wrapper(args):
                idx, args = args
                sol, t = self.simulate(args)
                return (idx, sol, t)

            pool = mp.get_context("spawn").Pool(n_processes, maxtasksperchild=1)
            futures_mp_sol = pool.imap_unordered(wrapper, enumerate(particles))

            for particle in particles:
                try:
                    idx, sol, t = futures_mp_sol.next(timeout=sim_timeout)
                    particles[idx].sol = sol
                    particles[idx].t = t

                except mp.context.TimeoutError:
                    print("TIMEOUT ERROR")
                    break

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
                sol, t = self.simulate(args)
                p.sol = sol
                p.t = t
                p_idx += 1
                end = time.time()
                print("Sim time: ", end - start)


class CometsTimeSeriesSimulation:
    def __init__(self, t_end, dt, gurobi_home_dir, comets_home_dir, batch_dilution=False, dilution_factor=0.0, dilution_time=0.0):
        self.dt = dt
        self.max_cycles = int(np.ceil(t_end / dt))

        self.batch_dilution = batch_dilution
        self.dilution_factor = dilution_factor
        self.dilution_time = dilution_time

        os.environ["GUROBI_HOME"] = gurobi_home_dir
        os.environ["GUROBI_COMETS_HOME"] = gurobi_home_dir
        os.environ["COMETS_HOME"] = comets_home_dir

    def convert_models(self, community):
        """Conversts cobrapy models of community to comets model inplace"""
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

    def set_layout_metabolite_concentrations(self, layout, community):
        # Set dynamic compound initial concentrations
        for idx, cmpd in enumerate(community.dynamic_compounds):
            metabolite_str = cmpd.replace("M_", "")
            layout.set_specific_metabolite(
                metabolite_str, community.init_compound_values[idx]
            )

        #  Fill non dynamic compound concentrations
        non_dynamic_media_df = community.media_df.loc[
            community.media_df["dynamic"] == 0
        ]
        for idx, row in non_dynamic_media_df.iterrows():
            cmpd_str = row.compound
            cmpd_str = cmpd_str + "_e"

            layout.set_specific_metabolite(cmpd_str, row.mmol_per_L)

            # Make static
            layout.set_specific_static(cmpd_str, row.mmol_per_L)

    def set_model_initial_pop(self, layout, community):
        for idx, pop in enumerate(community.populations):
            model_init_pop = community.init_population_values[idx]
            pop.model.initial_pop = [0, 0, model_init_pop]

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
                s_df = met_df[[s, "t"]]
                for t_val in t:
                    t_idx = utils.find_nearest(s_df["t"].values, t_val)
                    sol[:, idx][t_idx] = s_df[s].values[t_idx]

        return sol, t

    def simulate(self, community, idx=0):
        self.convert_models(community)

        layout = cometspy.layout()
        self.set_model_initial_pop(layout, community)
        self.load_layout_models(layout, community)
        self.set_layout_metabolite_concentrations(layout, community)

        self.set_lb_constraint(layout, community)

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

        # Optional parameters
        if self.batch_dilution:
            sim_params.set_param('batchDilution', True)
            sim_params.set_param('dilFactor', self.dilution_factor)
            sim_params.set_param('dilTime', self.dilution_time)

        tmp_dir = f"./tmp_{os.getpid()}/"
        if not os.path.exists(tmp_dir):
            os.mkdir(tmp_dir)

        experiment = cometspy.comets(layout, sim_params, relative_dir=tmp_dir)
        self.experiment = experiment

        comets_lib = "/Users/bezk/Documents/CAM/research_code/comets/lib"

        experiment.set_classpath(
            "bin", "/Users/bezk/Documents/CAM/research_code/comets/bin/comets.jar"
        )
        experiment.set_classpath(
            "gurobi", "/Library/gurobi950/macos_universal2/lib/gurobi.jar"
        )
        experiment.set_classpath(
            "jdistlib", f"{comets_lib}/jdistlib/jdistlib-0.4.5-bin.jar"
        )

        experiment.run()

        sol, t = self.process_experiment(community, experiment)

        shutil.rmtree(tmp_dir)

        return sol, t

    def simulate_particles(
        self, particles, n_processes=1, sim_timeout=360.0, parallel=True
    ):
        if parallel:
            print("running parallel")

            def wrapper(args):
                idx, args = args
                sol, t = self.simulate(args)
                return (idx, sol, t)

            pool = mp.get_context("spawn").Pool(n_processes, maxtasksperchild=1)
            futures_mp_sol = pool.imap_unordered(wrapper, enumerate(particles))

            for particle in particles:
                try:
                    idx, sol, t = futures_mp_sol.next(timeout=sim_timeout)
                    particles[idx].sol = sol
                    particles[idx].t = t

                except mp.context.TimeoutError:
                    print("TIMEOUT ERROR")
                    break

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
                sol, t = self.simulate(p)
                p.sol = sol
                p.t = t
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
            p.sol = self.simulate(particle)
