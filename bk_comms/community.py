import numpy as np
from reframed import Environment
from typing import List
import pandas as pd

import time

from scipy.integrate import ode

from bk_comms import utils
from bk_comms.sampling import SampleDistribution, SampleCombinationParticles
import cobra
from cobra import Model, Reaction, Metabolite

from scipy.integrate import odeint

from loguru import logger

import copy
import gc
import warnings

import multiprocess as mp


class Population:
    def __init__(
        self,
        name,
        model,
        dynamic_compounds,
        reaction_keys,
    ):
        self.name = name
        logger.info(f"Loading model {name}")
        self.model = model

        self.dynamic_compounds = dynamic_compounds
        self.reaction_keys = reaction_keys

        logger.info(
            f"Model  {name}, making dynamic compound mask, mem usage (mb): {utils.get_mem_usage()}"
        )
        self.dynamic_compound_mask = self.set_dynamic_compound_mask(dynamic_compounds)

        logger.info(
            f"Model  {name}, setting media, mem usage (mb): {utils.get_mem_usage()}"
        )

    def load_model(self, model_path):
        """
        Loads models from model paths
        """

        model = cobra.io.read_sbml_model(model_path, name=str(np.random.randint(10)))
        model.solver = "gurobi"

        return model

    def set_media(self, media_df):

        dynamic_compounds = [x.replace("M_", "EX_") for x in self.dynamic_compounds]

        defined_mets = ["EX_" + x + "_e" for x in media_df["compound"].values]

        new_medium = self.model.medium

        for met in list(self.model.medium):
            # M_glyb_e
            if met in defined_mets:
                key = met.replace("EX_", "").replace("_e", "")

                new_medium[met] = media_df.loc[media_df["compound"] == key][
                    "mmol_per_L"
                ].values[0]

            elif met in dynamic_compounds:
                # Compounds that are not in the media but are involved
                # in interactions
                new_medium[met] = 1e-30

            else:
                new_medium[met] = 1e-30

        self.model.medium = new_medium

    def set_dynamic_compound_mask(self, community_dynamic_compounds):
        dynamic_compound_mask = []
        for metabolite in community_dynamic_compounds:
            key = metabolite.replace("M_", "EX_")

            if self.model.reactions.has_id(key):
                dynamic_compound_mask.append(1)

            else:
                dynamic_compound_mask.append(0)

        return dynamic_compound_mask

    def update_reaction_constraints(self, lower_constraints):
        for idx, _ in enumerate(lower_constraints):
            if self.dynamic_compound_mask[idx]:
                self.model.reactions.get_by_id(
                    self.reaction_keys[idx]
                ).lower_bound = lower_constraints[idx]

    def optimize(self):
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


class Community:
    def __init__(
        self,
        model_names: List[str],
        model_paths: List[str],
        smetana_analysis_path: str,
        media_path: str,
        objective_reaction_keys: List[str],
        enable_toxin_interactions: bool,
        initial_population_prior: SampleDistribution = None,
        max_exchange_prior: SampleDistribution = None,
        k_val_prior: SampleDistribution = None,
        toxin_interaction_prior: SampleDistribution = None,
    ):
        self.model_names = model_names
        self.model_paths = model_paths

        self.initial_population_prior = initial_population_prior
        self.max_exchange_prior = max_exchange_prior
        self.k_val_prior = k_val_prior
        self.toxin_interaction_prior = toxin_interaction_prior

        # Load models
        models = [
            utils.load_model(model_path, model_name)
            for model_path, model_name in zip(self.model_paths, self.model_names)
        ]

        if enable_toxin_interactions:
            self.toxin_names = self.add_toxin_production_reactions(
                models, model_names, objective_reaction_keys
            )

        else:
            self.toxin_names = []

        self.smetana_analysis_path = smetana_analysis_path
        self.media_path = media_path

        self.media_df = pd.read_csv(self.media_path, delimiter="\t")

        self.dynamic_compounds = self.get_dynamic_compounds(
            model_paths, smetana_analysis_path, self.media_df, flavor="fbc2"
        )

        self.reaction_keys = [x.replace("M_", "EX_") for x in self.dynamic_compounds]

        # Initialise populations
        self.populations = self.load_populations(
            self.model_names,
            models,
            self.dynamic_compounds,
            self.reaction_keys,
            self.media_df,
        )

        logger.info(f"Loading compound values")

        # self.init_compound_values = self.load_initial_compound_values(
        #     self.dynamic_compounds
        # )

        logger.info(f"Generating k values, mem usage (mb): {utils.get_mem_usage()}")

        logger.info(f"Setting k val mat, mem usage (mb): {utils.get_mem_usage()}")
        self.set_k_value_matrix(
            np.ones(shape=[len(self.populations), len(self.dynamic_compounds)])
        )

        logger.info(
            f"Setting max exchange mat, mem usage (mb): {utils.get_mem_usage()}"
        )
        self.set_max_exchange_mat(
            np.ones(shape=[len(self.populations), len(self.dynamic_compounds)]) * -1
        )

        self.set_toxin_mat(
            np.zeros(shape=[len(self.populations), len(self.populations)])
        )

        logger.info(f"Setting pop indexesmem usage (mb): {utils.get_mem_usage()}")
        self.population_indexes = self.set_population_indexes()

        logger.info(
            f"Setting compound indexes, mem usage (mb): {utils.get_mem_usage()}"
        )
        self.compound_indexes = self.set_compound_indexes()

        logger.info(
            f"Setting solution key order,  mem usage (mb): {utils.get_mem_usage()}"
        )
        self.solution_keys = self.set_solution_key_order()

        logger.info(
            f"Community initialisation finished,  mem usage (mb): {utils.get_mem_usage()}"
        )

        gc.collect()

    def sample_parameters_from_prior(self):
        """
        Samples and sets community parameters and initial populations from
        prior distribution samplers
        """
        n_populations = len(self.populations)
        n_dynamic_compounds = len(self.dynamic_compounds)

        # If distributions exist, sample from them and set parameters.
        # Special case used if prior is from existing particles,

        if self.initial_population_prior is not None:
            if isinstance(self.initial_population_prior, SampleCombinationParticles):
                populations_vec = self.initial_population_prior.sample(
                    self.model_names, data_field="initial_population"
                )

            else:
                populations_vec = self.initial_population_prior.sample(
                    size=[n_populations]
                )

            populations_vec.reshape(1, -1)
            self.set_initial_populations(populations_vec)

        if self.max_exchange_prior is not None:
            if isinstance(self.max_exchange_prior, SampleCombinationParticles):
                max_exchange_mat = self.max_exchange_prior.sample(
                    self.model_names, data_field="max_exchange_mat"
                )

            else:
                max_exchange_mat = self.max_exchange_prior.sample(
                    size=[n_populations, n_dynamic_compounds]
                )
            self.set_max_exchange_mat(max_exchange_mat)

        if self.k_val_prior is not None:
            if isinstance(self.k_val_prior, SampleCombinationParticles):
                k_val_mat = self.k_val_prior.sample(
                    self.model_names, data_field="k_vals"
                )

            else:
                k_val_mat = self.k_val_prior.sample(
                    size=[n_populations, n_dynamic_compounds]
                )
            self.set_k_value_matrix(k_val_mat)

        if self.toxin_interaction_prior is not None:
            toxin_mat = self.toxin_interaction_prior.sample(
                size=[n_populations, n_populations]
            )
            self.set_toxin_mat(toxin_mat)

    def add_toxin_production_reactions(
        self, models, model_names, objective_keys, toxin_production_rate=1.0
    ):
        """
        Adds a new metabolite to each model, precursor and toxin production reactions,
        and an exchange reaction for the toxin
        """

        toxin_names = []
        toxin_metabolites = []

        for idx, _ in enumerate(models):
            model = models[idx]
            model_name = model_names[idx]
            objective_key = objective_keys[idx]

            toxin_name = f"toxin_{model_name}"

            # Initialise toxin production and precursor transport reactions
            reaction_precursor_toxin = Reaction("R_PRETOXIN")
            reaction_toxin_transport = Reaction("R_TOXINt")

            # Initialise metabolites
            toxin_c = Metabolite(f"{toxin_name}_c", compartment="c")
            pretoxin_c = Metabolite("pretoxin_c", compartment="c")
            toxin_e = Metabolite(f"{toxin_name}_e", compartment="e")

            # Add metabolites to model
            model.add_metabolites([pretoxin_c, toxin_c, toxin_e])

            # # Set toxin production stoichiometry
            reaction_precursor_toxin.add_metabolites(
                {pretoxin_c: toxin_production_rate}
            )

            reaction_toxin_transport.add_metabolites(
                {toxin_c: -1 * toxin_production_rate, toxin_e: toxin_production_rate}
            )

            # Prevent upake of toxin, allow only export
            reaction_toxin_transport.lower_bound = 0.0
            reaction_toxin_transport.upper_bound = 1000

            model.reactions.get_by_id(objective_key).add_metabolites(
                {pretoxin_c: -1 * toxin_production_rate, toxin_c: toxin_production_rate}
            )

            model.add_reactions([reaction_precursor_toxin, reaction_toxin_transport])
            model.add_boundary(
                model.metabolites.get_by_id(f"{toxin_name}_e"), type="exchange"
            )

            toxin_names.append(toxin_name)
            toxin_metabolites.append(toxin_e)

        # Add exchange reactions for all toxins for each model
        # ensuring each model is aware of other metabolite existance
        for model_idx, model in enumerate(models):
            for toxin_idx, toxin_name in enumerate(toxin_names):
                print(toxin_name, model_names[idx])
                try:
                    model.metabolites.get_by_id(f"{toxin_name}_e")

                except KeyError:
                    model.add_boundary(toxin_metabolites[toxin_idx], type="exchange")

        return toxin_names

    def generate_parameter_vector(self):
        # Initial conditions
        # k values
        # exchange_constraints

        initial_concs_vec = self.init_y.reshape(1, -1)
        k_val_vec = self.k_vals.reshape(1, -1)
        max_exchange_vec = self.max_exchange_mat.reshape(1, -1)
        toxin_mat = self.toxin_mat.reshape(1, -1)

        param_vec = np.concatenate(
            [initial_concs_vec, k_val_vec, max_exchange_vec, toxin_mat], axis=1
        ).reshape(-1, 1)

        return param_vec

    def update_parameters_from_particle(self, particle, model_names):
        # Update self arrays according to the specific model of the particle
        for particle_model_name in model_names:
            if particle_model_name in self.model_names:
                particle_model_idx = particle.model_names.index(particle_model_name)
                this_model_idx = self.model_names.index(particle_model_name)

                print(particle_model_name, self.model_names[this_model_idx])

                self.k_vals[this_model_idx] = particle.k_vals[particle_model_idx]
                self.max_exchange_mat[this_model_idx] = particle.max_exchange_mat[
                    particle_model_idx
                ]

    def load_parameter_vector(self, parameter_vec):
        """
        Loads a parameter vector consisting of initial concentrations, k values and
        max exchange parameters
        """

        n_variables = len(self.init_y)
        n_k_vals = self.k_vals.shape[0] * self.k_vals.shape[1]
        n_max_exchange_vals = (
            self.max_exchange_mat.shape[0] * self.max_exchange_mat.shape[1]
        )
        n_toxin_vals = self.toxin_mat.shape[0] * self.toxin_mat.shape[1]

        self.init_y = parameter_vec[0:n_variables]
        self.set_k_value_matrix(
            parameter_vec[n_variables : n_variables + n_k_vals].reshape(
                self.k_vals.shape
            )
        )

        self.set_max_exchange_mat(
            parameter_vec[
                n_variables + n_k_vals : n_variables + n_k_vals + n_max_exchange_vals
            ].reshape(self.max_exchange_mat.shape)
        )

        self.set_toxin_mat(
            parameter_vec[
                n_variables
                + n_k_vals
                + n_max_exchange_vals : n_variables
                + n_k_vals
                + n_max_exchange_vals
                + n_toxin_vals
            ].reshape(self.toxin_mat.shape)
        )

    def load_initial_compound_values(self, sub_media_df):
        """
        Loads vector of initial compound concentrations from a subset
        of the media df for a particular media name
        """

        init_compound_values = np.zeros(len(self.dynamic_compounds))

        for compound_idx, dynm_cmpd in enumerate(self.dynamic_compounds):
            found_match = False
            for idx, row in sub_media_df.iterrows():
                media_cmpd = "M_" + row["compound"] + "_e"
                if dynm_cmpd == media_cmpd:
                    found_match = True

                    init_compound_values[compound_idx] = row["mmol_per_L"]
                    break

            if not found_match:
                init_compound_values[compound_idx] = 1e-30

        return init_compound_values

    def set_initial_populations(self, populations_vec):
        assert len(populations_vec) == (
            len(self.populations)
        ), f"Error: length of parameter_vec, {len(populations_vec)} \
            does not match expected  {len(self.populations)}"

        self.init_population_values = populations_vec

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

    def set_toxin_mat(self, toxin_mat):
        assert toxin_mat.shape == (
            len(self.populations),
            len(self.populations),
        ), f"Error: shape of toxin_mat, {toxin_mat.shape} does not match expected  {(len(self.populations), len(self.populations))}"
        self.toxin_mat = toxin_mat

    def set_solution_key_order(self):
        solution_keys = []

        for p in self.populations:
            solution_keys.append(p.name)

        for m in self.dynamic_compounds:
            solution_keys.append(m)

        for m in self.toxin_names:
            m = f"M_{m}_e"
            solution_keys.append(m)

        return solution_keys

    def load_populations(
        self,
        model_names: List[str],
        model_paths: List[str],
        dynamic_compounds,
        reaction_keys,
        media_df,
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

        # Remove duplicates and append toxins
        # toxin_compounds = [f"M_{t}_e" for t in self.toxin_names]
        dynamic_compounds = list(dict.fromkeys(dynamic_compounds))  # + toxin_compounds
        dynamic_compounds = list(set(dynamic_compounds))

        dynamic_compounds.sort()

        print(f"Total env compounds: {len(all_compounds)}")
        print(f"Crossfeeding compounds: {len(cross_feeding_compounds)}")
        print(f"Competition compounds: {len(competition_compounds)}")
        print(f"Media compounds: {len(media_compounds)}")
        print(f"Toxin compounds: {len(self.toxin_names)}")
        print(f"Dynamic compounds: {len(dynamic_compounds)}")

        return dynamic_compounds

    def set_population_indexes(self):
        num_populations = len(self.populations)
        return list(range(num_populations))

    def set_compound_indexes(self):
        num_dynamic_cmpds = len(self.dynamic_compounds)
        num_populations = len(self.populations)

        return list(range(num_populations, num_populations + num_dynamic_cmpds))

    def set_init_y(self):
        self.init_y = np.concatenate(
            (self.init_population_values, self.init_compound_values), axis=None
        )

    def set_media_conditions(self, media_name, set_media=True):
        sub_media_df = self.media_df.loc[self.media_df["medium"] == media_name]
        self.curr_media_df = sub_media_df

        self.init_compound_values = self.load_initial_compound_values(sub_media_df)

        if set_media:
            for p in self.populations:
                p.set_media(self.curr_media_df)
        # for idx, c in enumerate(self.dynamic_compounds):
        #     print(c, self.init_compound_values[idx])

        self.set_init_y()

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

                if growth_rates[idx] < 0.0:
                    growth_rates[idx] = 0.0

        return growth_rates, flux_matrix
