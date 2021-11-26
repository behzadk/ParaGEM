import numpy as np
import scipy as sp
from reframed import Environment
from typing import List
import pandas as pd

import time

from scipy.integrate import ode

from bk_comms import utils
import cobra
from scipy.integrate import odeint

from loguru import logger

import matplotlib.pyplot as plt
import copy

import warnings

# import multiprocessing as mp
import multiprocess as mp


class Population:
    def __init__(
        self,
        name,
        model_path,
        dynamic_compounds,
        reaction_keys,
        media_df,
        use_parsimonius_fba=False,
    ):
        self.name = name
        self.model = self.load_model(model_path)
        self.media_df = media_df

        self.dynamic_compounds = dynamic_compounds
        self.reaction_keys = reaction_keys

        self.dynamic_compound_mask = self.set_dynamic_compound_mask(dynamic_compounds)
        self.set_media()

        self.use_parsimonius_fba = use_parsimonius_fba

    def load_model(self, model_path):
        """
        Loads models from model paths
        """

        model = cobra.io.read_sbml_model(model_path, name=str(np.random.randint(10)))
        model.solver = "glpk"

        return model

    def set_media(self):
        dynamic_compounds = [x.replace("M_", "EX_") for x in self.dynamic_compounds]

        defined_mets = ["EX_" + x + "_e" for x in self.media_df["compound"].values]
        
        for met in list(self.model.medium):
            if met in defined_mets:
                print("Defined in media file and model media", met)
                key = met.replace("EX_", "").replace("_e", "")

                medium = self.model.medium
                medium[met] = self.media_df.loc[self.media_df["compound"] == key][
                    "mmol_per_L"
                ].values[0]

                self.model.medium = medium

            elif met in dynamic_compounds:
                # Compounds that are not in the media but are involved
                # in interactions
                medium = self.model.medium
                medium[met] = 0.0
                self.model.medium = medium

            else:
                medium = self.model.medium
                medium[met] = 0.0
                self.model.medium = medium

    def set_dynamic_compound_mask(self, community_dynamic_compounds):
        dynamic_compound_mask = []
        for metabolite in community_dynamic_compounds:
            key = metabolite.replace("M_", "EX_")

            if self.model.reactions.has_id(key):
                dynamic_compound_mask.append(1)

            else:
                dynamic_compound_mask.append(0)

        return dynamic_compound_mask

    def set_max_growth_rate(self, max_growth_rate=2.0):
        self.model.reactions.get_by_id("Growth").upper_bound = max_growth_rate

    # @staticmethod
    # @numba.jit(parallel=False)
    # def calculate_monod_uptake(max_uptake, cmpd_conc, k):
    #     return max_uptake * (cmpd_conc / (k + cmpd_conc))

    def update_reaction_constraints(self, lower_constraints):
        for idx, _ in enumerate(lower_constraints):
            if self.dynamic_compound_mask[idx]:
                self.model.reactions.get_by_id(
                    self.reaction_keys[idx]
                ).lower_bound = lower_constraints[idx]

    def optimize(self):
        if self.use_parsimonius_fba:
            self.opt_sol = cobra.flux_analysis.pfba(self.model)
        
        
        else:
            self.opt_sol = self.model.optimize()
        

    def get_dynamic_compound_fluxes(self):
        compound_fluxes = np.zeros(len(self.dynamic_compounds))

        for compound_idx, reaction in enumerate(self.dynamic_compounds):
            if self.dynamic_compound_mask[compound_idx]:
                key = reaction.replace("M_", "EX_")
                compound_fluxes[compound_idx] = self.opt_sol.get_primal_by_id(key)

        return compound_fluxes

    def get_growth_rate(self):
        return self.opt_sol.objective_value
        # return self.opt_sol.get_primal_by_id('Growth')


class Community:
    def __init__(
        self,
        model_names,
        model_paths: List[str],
        smetana_analysis_path: str,
        media_path: str,
        media_name: str,
        use_parsimonius_fba: bool,
        initial_populations: List[float]
    ):
        self.model_names = model_names
        self.model_paths = model_paths
        self.smetana_analysis_path = smetana_analysis_path
        self.media_path = media_path
        self.use_parsimonius_fba = use_parsimonius_fba

        self.media_df = pd.read_csv(self.media_path, delimiter="\t")
        self.media_df = self.media_df.loc[
            self.media_df["medium"] == media_name
        ].reset_index(drop=True)


        self.dynamic_compounds = self.get_dynamic_compounds(
            model_paths, smetana_analysis_path, self.media_df, flavor="fbc2"
        )

        self.reaction_keys = [x.replace("M_", "EX_") for x in self.dynamic_compounds]

        self.populations = self.load_populations(
            self.model_names,
            self.model_paths,
            self.dynamic_compounds,
            self.reaction_keys,
            self.media_df,
            self.use_parsimonius_fba,
        )

        self.init_compound_values = self.load_initial_compound_values(
            self.dynamic_compounds
        )

        self.init_population_values = initial_populations

        k_vals_mat = self.generate_default_k_values(
            self.init_compound_values, media_name
        )
        self.set_k_value_matrix(k_vals_mat)

        self.set_max_exchange_mat(
            np.ones(shape=[len(self.populations), len(self.dynamic_compounds)]) * -1
        )

        self.population_indexes = self.set_population_indexes()
        self.compound_indexes = self.set_compound_indexes()
        self.solution_keys = self.set_solution_key_order()

        self.set_init_y()

    def generate_initial_population_densities(self):
        return np.array([0.01 * 0.56 for x in self.populations])

    def generate_parameter_vector(self):
        # Initial conditions
        # exchange_constraints
        # k values

        initial_concs_vec = self.init_y.reshape(1, -1)
        k_val_vec = self.k_vals.reshape(1, -1)
        max_exchange_vec = self.max_exchange_mat.reshape(1, -1)
        param_vec = np.concatenate(
            [initial_concs_vec, k_val_vec, max_exchange_vec], axis=1
        ).reshape(-1, 1)

        return param_vec

    def update_parameters_from_particle(self, particle, model_names):
        # Update self arrays according to the specific model of the particle
        for particle_model_name in model_names:
            if particle_model_name in self.model_names:
                particle_model_idx = particle.model_names.index(particle_model_name)
                this_model_idx = self.model_names.index(particle_model_name)

                self.k_vals[this_model_idx] = particle.k_vals[particle_model_idx]
                self.max_exchange_mat[this_model_idx] = particle.max_exchange_mat[
                    particle_model_idx
                ]

    def load_parameter_vector(self, parameter_vec):
        n_variables = len(self.init_y)
        n_k_vals = self.k_vals.shape[0] * self.k_vals.shape[1]
        n_max_exchange_vals = (
            self.max_exchange_mat.shape[0] * self.max_exchange_mat.shape[1]
        )

        self.init_y = parameter_vec[0:n_variables]
        self.k_vals = parameter_vec[n_variables : n_variables + n_k_vals].reshape(
            self.k_vals.shape
        )
        self.max_exchange_mat = parameter_vec[
            n_variables + n_k_vals : n_variables + n_k_vals + n_max_exchange_vals
        ].reshape(self.max_exchange_mat.shape)

    def load_initial_compound_values(self, dynamic_compounds):
        init_compound_values = np.zeros(len(dynamic_compounds))

        for compound_idx, dynm_cmpd in enumerate(dynamic_compounds):
            found_match = False
            for idx, row in self.media_df.iterrows():
                media_cmpd = "M_" + row["compound"] + "_e"
                if dynm_cmpd == media_cmpd:
                    found_match = True

                    init_compound_values[compound_idx] = row["mmol_per_L"]
                    break

            if not found_match:
                init_compound_values[compound_idx] = 0.00

        return init_compound_values

    def generate_default_k_values(self, dynm_compound_concs, media_name):
        """
        Heuristic choices for picking default k values
        produces a matrix of species x compounds
        """

        k_vals = np.zeros(shape=[len(self.populations), len(dynm_compound_concs)])

        generic_k_val = 0.1

        for species_idx, _ in enumerate(self.populations):
            for compound_idx, _ in enumerate(dynm_compound_concs):
                k = generic_k_val

                if self.populations[species_idx].dynamic_compound_mask:
                    if dynm_compound_concs[compound_idx] > 0.0:
                        k = dynm_compound_concs[compound_idx] * 0.01

                k_vals[species_idx][compound_idx] = k

        return k_vals

    def set_k_value_matrix(self, k_val_mat):
        assert k_val_mat.shape == (
            len(self.populations),
            len(self.dynamic_compounds),
        ), f"Error: shape of kval mat, {k_val_mat.shape} does not match expected  {(len(self.populations), len(self.dynamic_compounds))}"

        self.k_vals = k_val_mat

    def set_max_exchange_mat(self, max_exchange_mat):
        assert max_exchange_mat.shape == (
            len(self.populations),
            len(self.dynamic_compounds),
        ), f"Error: shape of max_exchange_mat, {max_exchange_mat.shape} does not match expected  {(len(self.populations), len(self.dynamic_compounds))}"
        self.max_exchange_mat = max_exchange_mat

    def set_solution_key_order(self):
        solution_keys = []

        for p in self.populations:
            solution_keys.append(p.name)

        for m in self.dynamic_compounds:
            solution_keys.append(m)

        return solution_keys

    def load_populations(
        self,
        model_names: List[str],
        model_paths: List[str],
        dynamic_compounds,
        reaction_keys,
        media_df,
        use_parsimonius_fba,
    ):
        """
        Load population objects from model paths
        """
        populations = []
        for idx, n in enumerate(model_names):
            populations.append(
                Population(
                    n,
                    model_paths[idx],
                    dynamic_compounds,
                    reaction_keys,
                    media_df,
                    use_parsimonius_fba,
                )
            )

        return populations

    def get_dynamic_compounds(
        self,
        model_paths: List[str],
        smetana_analysis_path: str,
        media_df: pd.DataFrame,
        flavor: str,
    ):
        """
        1. Generate complete environment reactions.
        2. Read SMETANA output full output
        3. Identify metabolites involved in crossfeeding
        4. Identify metabolites involved in competition
        """

        dynamic_media_df = media_df.loc[media_df["dynamic"] == True]
        unconstrained_compounds = media_df.loc[media_df["dynamic"] == False]
        unconstrained_compounds = [
            "M_" + m + "_e" for m in unconstrained_compounds["compound"].values
        ]

        # Generate complete environment metabolites dict
        compl_env = utils.get_community_complete_environment(
            self.model_paths, max_uptake=1.0, flavor=flavor
        )
        all_compounds = compl_env.compound.values

        # Get crossfeeding and competition metabolites
        smetana_df = pd.read_csv(smetana_analysis_path, delimiter="\t")
        cross_feeding_compounds = utils.get_crossfeeding_compounds(smetana_df)
        competition_compounds = utils.get_competition_compounds(
            smetana_df, all_compounds
        )

        media_compounds = ["M_" + m + "_e" for m in dynamic_media_df["compound"].values]

        # Aggregate dynamic compounds
        dynamic_compounds = (
            cross_feeding_compounds + competition_compounds + media_compounds
        )
        dynamic_compounds = [
            met for met in dynamic_compounds if met not in unconstrained_compounds
        ]

        # Remove duplicates
        dynamic_compounds = list(dict.fromkeys(dynamic_compounds))

        print(f"Total env compounds: {len(all_compounds)}")
        print(f"Crossfeeding compounds: {len(cross_feeding_compounds)}")
        print(f"Competition compounds: {len(competition_compounds)}")
        print(f"Media compounds: {len(media_compounds)}")
        print(f"Dynamic compounds: {len(dynamic_compounds)}")

        return dynamic_compounds

    def set_population_indexes(self):
        num_populations = len(self.populations)
        return range(num_populations)

    def set_compound_indexes(self):
        num_dynamic_cmpds = len(self.dynamic_compounds)
        num_populations = len(self.populations)

        return range(num_populations, num_populations + num_dynamic_cmpds)

    def set_init_y(self):
        self.init_y = np.concatenate(
            (self.init_population_values, self.init_compound_values), axis=None
        )

    def simulate_community(self, method="odeint"):
        y0 = copy.deepcopy(self.init_y)
        t_0 = 0.0
        t_end = 24.0
        steps = 100000

        if method == "odeint":
            t = np.linspace(t_0, t_end, steps)
            sol = odeint(self.diff_eqs, y0, t, args=())

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
            solver = ode(self.diff_eqs_vode, jac=None).set_integrator(
                "vode", method="bdf", atol=1e-4, rtol=1e-3, max_step=0.01
            )
            # print("solver initiated")

            solver.set_initial_value(y0, t=t_0)

            while solver.successful() and solver.t < t_end:
                # print(solver.t)
                step_out = solver.integrate(t_end, step=True)
                
                if solver.t >= t_points[0]:
                    mx = np.ma.masked_array(step_out, mask=step_out==0)
                    print(solver.t, mx.min())
                    sol.append(step_out)
                    t.append(solver.t)
                    t_points.pop(0)

            sol = np.array(sol)
            t = np.array(t)
            end = time.time()
            logger.info(f'Simulation time: {end - start}')

        return sol, t

    def print_sol(self, sol):
        count = 1
        for idx in self.population_indexes:
            print(f"N_{idx}: {sol[:, idx][-1]}")

        for idx, sol_idx in enumerate(self.compound_indexes):
            print(f"{self.dynamic_compounds[idx]}: {sol[:, sol_idx][-1]}")

    # @staticmethod
    # @numba.jit(parallel=False)
    def calculate_exchange_reaction_lb_constraints(
        self, compound_concs, k_mat, max_exchange_mat
    ):
        # Get matrix where exchange is uptake (negative numbers)
        uptake_mat = np.clip(max_exchange_mat, a_min=None, a_max=0)
        uptake_mat = uptake_mat * (compound_concs / (k_mat + compound_concs))

        # Get matrix where exchange is secretion (positive numbers)
        secretion_mat = np.clip(max_exchange_mat, a_min=0, a_max=None)

        return uptake_mat + secretion_mat

    def sim_step(self, y):
        compound_concs = y[self.compound_indexes]

        flux_matrix = np.zeros(
            shape=[len(self.populations), len(self.dynamic_compounds)]
        )
        growth_rates = np.zeros(len(self.populations))

        lower_constraints = self.calculate_exchange_reaction_lb_constraints(
            compound_concs, self.k_vals, self.max_exchange_mat
        )        


        self.lower_constraints = lower_constraints

        for idx, pop in enumerate(self.populations):
            pop.update_reaction_constraints(lower_constraints[idx])
            with warnings.catch_warnings():
                warnings.filterwarnings("error")
                try:
                    pop.optimize()
                    flux_matrix[idx] = pop.get_dynamic_compound_fluxes()
                    growth_rates[idx] = pop.get_growth_rate()
                    

                except UserWarning:
                    flux_matrix[idx] = np.zeros(shape=len(self.dynamic_compounds))
                    growth_rates[idx] = 0.0
        

        return growth_rates, flux_matrix

    def diff_eqs(self, y, t):
        print(t)
        y = y.clip(0)
        # y[y < 1e-25] = 0
        populations = y[self.population_indexes]
        compounds = y[self.compound_indexes]

        growth_rates, flux_matrix = self.sim_step(y)

        output = np.zeros(len(y))

        output[self.population_indexes] = growth_rates * populations
        output[self.compound_indexes] = np.dot(populations, flux_matrix)

        return output

    def diff_eqs_vode(self, t, y):
        y = y.clip(0.0)
        # y[y < 1e-25] = 0
        populations = y[self.population_indexes]
        compounds = y[self.compound_indexes]

        growth_rates, flux_matrix = self.sim_step(y)

        output = np.zeros(len(y))

        output[self.population_indexes] = growth_rates * populations
        output[self.compound_indexes] = np.dot(populations, flux_matrix)

        # # DEBUG
        # for y_idx, x in enumerate(y):
        #     if x == 0.0 or x < 0.0:
        #         print(y_idx, output[y_idx])
        #         if output[y_idx] < 0.0:
        #             print(y_idx)
        #             exit()
        # # DEBUG


        return output

