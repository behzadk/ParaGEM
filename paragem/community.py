import numpy as np
from typing import List
import pandas as pd
from paragem import utils
from paragem.sampling import SampleDistribution, SampleCombinationParticles
import cobra
from loguru import logger
from paragem.utils import logger_wraps
from typing import Tuple
import gc
import warnings
from cobra import Reaction, Metabolite

class Population:
    """
    Represents a homogenous population, consisting of one model.
    """

    def __init__(
        self,
        name: str,
        model: cobra.core.model.Model,
        dynamic_compounds: List[str],
        reaction_keys: List[str],
    ):
        """
        Args:
            name : population name
            model : model
            dynamic_compounds : path to smetana output file
            media_path : path to media definition file
            reaction_keys : list of exchange reaction keys corresponding to the dynamic compounds
        """

        self.name = name
        logger.info(f"Loading model {name}")
        self.model = model

        self.dynamic_compounds = dynamic_compounds
        self.reaction_keys = reaction_keys

        self.dynamic_compound_mask = self.make_dynamic_compound_mask(dynamic_compounds)

    def set_media(self, media_df: pd.DataFrame):
        """Set model medium according to media dataframe
        Args:
            media_df : dataframe defining media
        """

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

    def make_dynamic_compound_mask(self, community_dynamic_compounds: List[str]):
        """Makes a mask for the dynamic compounds that exist in this model

        Args:
            community_dynamic_compounds : all dynamic compounds of the community
        """

        dynamic_compound_mask = []
        for metabolite in community_dynamic_compounds:
            key = metabolite.replace("M_", "EX_")

            if self.model.reactions.has_id(key):
                dynamic_compound_mask.append(1)

            else:
                dynamic_compound_mask.append(0)

        return dynamic_compound_mask

    def update_reaction_constraints(self, lower_constraints: np.ndarray):
        """Updates model reaction lowerbounds
        for the dynamic compounds that exist in this population's model
        Args:
            media_df : dataframe defining media
        """

        for idx, _ in enumerate(lower_constraints):
            if self.dynamic_compound_mask[idx]:
                self.model.reactions.get_by_id(
                    self.reaction_keys[idx]
                ).lower_bound = lower_constraints[idx]

    def optimize(self):
        """Optimize model
        """

        self.opt_sol = self.model.optimize()

    def get_dynamic_compound_fluxes(self) -> np.ndarray:
        """Gets compound fluxes of last optimize

        Creates a matrix that is compatible with the community. Compounds that do not exist in this model will be zero.

        Returns:
            compound_fluxes : matrix of compund fluxes
        """

        compound_fluxes = np.zeros(len(self.dynamic_compounds))

        for compound_idx, reaction in enumerate(self.dynamic_compounds):
            if self.dynamic_compound_mask[compound_idx]:
                key = reaction.replace("M_", "EX_")
                compound_fluxes[compound_idx] = self.opt_sol.get_primal_by_id(key)

        return compound_fluxes

    def get_growth_rate(self):
        return self.opt_sol.objective_value


class Community:
    """
    Contains all information about the community that should be simulated or parameterised. Class methods are used for
    setting parameters, media files.
    A `Community` object is treated as a unique solution.

    Initialises `Population` objects for the given models and model names. These

    Attributes:

    """

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
        biomass_rate_constraint_prior: SampleDistribution = None,
        toxin_interaction_prior: SampleDistribution = None,
    ):
        """
        Args:
            model_names : list of model names
            model_paths : list of model paths
            smetana_analysis_path : path to smetana output file
            media_path : path to media definition file
            objective_reaction_keys : list of strings defining the objective reaction for each model e.g ['Growth', 'Growth']
            enable_toxin_interactions: bool defining whether to include toxin interactioons or not,
            initial_population_prior: SampleDistribution object defining how initial populations should be sampled
            max_exchange_prior: SampleDistribution object defining how max exchange reactions should be sampled
            k_val_prior: SampleDistribution object defining how k-values should be sampled
            biomass_rate_constraint_prior: SampleDistribution object defining how biomass constraints should be sampled
            toxin_interaction_prior:  SampleDistribution object defining how toxin interactions should be sampled

        """

        self.model_names = model_names
        self.model_paths = model_paths

        self.initial_population_prior = initial_population_prior
        self.max_exchange_prior = max_exchange_prior
        self.k_val_prior = k_val_prior
        self.biomass_rate_constraint_prior = biomass_rate_constraint_prior
        self.toxin_interaction_prior = toxin_interaction_prior

        self.objective_reaction_keys = objective_reaction_keys

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

        logger.info(f"Setting toxin mat, mem usage (mb): {utils.get_mem_usage()}")
        self.set_toxin_mat(
            np.zeros(shape=[len(self.populations), len(self.populations)])
        )

        self.set_biomass_rate_constraints(np.ones(len(self.populations)) * 1000)

        self.population_indexes = self.gen_population_indexes()
        self.compound_indexes = self.gen_compound_indexes()

        logger.info(
            f"Setting solution key order,  mem usage (mb): {utils.get_mem_usage()}"
        )
        self.solution_keys = self.set_solution_key_order()

        logger.info(
            f"Community initialisation finished,  mem usage (mb): {utils.get_mem_usage()}"
        )

        gc.collect()

    def sample_parameters_from_prior(self):
        """Samples and sets community parameters from community priors

        Samples and sets initial populations, max exchange matrix, k values matrix, biomass constraints
        and toxin interaction matrix

        Args:

        Returns:

        Raises:

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

        if self.biomass_rate_constraint_prior is not None:
            if isinstance(
                self.biomass_rate_constraint_prior, SampleCombinationParticles
            ):
                biomass_constraints = self.biomass_rate_constraint_prior.sample(
                    self.model_names, data_field="biomass_rate_constraints"
                )

            else:
                biomass_constraints = self.biomass_rate_constraint_prior.sample(
                    size=[
                        n_populations,
                    ]
                )

            self.set_biomass_rate_constraints(biomass_constraints)

    @logger_wraps()
    def add_toxin_production_reactions(
        self,
        models: List[str],
        model_names: List[str],
        objective_keys: List[str],
        toxin_production_rate: float = 1.0,
    ) -> List[str]:
        """Adds metabolites and reactions for toxin interactions to each model

        Adds precursor metabolites, toxin metabolites and production reactions to each model.
        Each model produces a unique toxin. Production of the toxin is added to the biomass reaction,
        so toxin is only produced while the organism is able to grow.

        Each model has exchange reactions added for each toxin - I don't think this is necesssary and should be removed.
        Need to check that sensitivity to toxin is not dependnt upon ability to uptake (should be based just on
        environmental concentration).

        Args:
            models : List of models to add toxins to
            model_names : Names of the models, used for naming the toxins
            objective_keys : Name of the objective function. used to
            toxin_production_rate : Rate of production for toxin from organism into the environment

        Returns:
            toxin_names : List of toxin names added

        Todo: Each model has exchange reactions added for each toxin - I don't think this is necesssary and should be removed.
        Need to check that sensitivity to toxin is not dependnt upon ability to uptake (should be based just on
        environmental concentration).

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

    def generate_parameter_vector(self) -> np.ndarray:
        """Generates parameter vector by flattening and concatenating community parameters

        Concatenates parameters into a 1d vector. 1d vec produced here can be used by
        :func:`~community.Community.load_parameter_vector`.

        Returns:
            param_vec : 1d numpy array of parameters

        Raises:

        """
        initial_concs_vec = self.init_y.reshape(1, -1)
        k_val_vec = self.k_vals.reshape(1, -1)
        max_exchange_vec = self.max_exchange_mat.reshape(1, -1)
        biomass_constraints_vec = self.biomass_constraints.reshape(1, -1)
        toxin_mat = self.toxin_mat.reshape(1, -1)

        param_vec = np.concatenate(
            [
                initial_concs_vec,
                k_val_vec,
                max_exchange_vec,
                biomass_constraints_vec,
                toxin_mat,
            ],
            axis=1,
        ).reshape(-1, 1)

        return param_vec

    def load_parameter_vector(self, parameter_vec):
        """Loads a 1d parameter vector produced by :func:`~community.Community.generate_parameter_vector`

        Args:
            self : self
            param_vec : 1d numpy array of parameters

        Returns:

        Raises:

        """

        n_variables = len(self.init_y)
        n_k_vals = self.k_vals.shape[0] * self.k_vals.shape[1]
        n_max_exchange_vals = (
            self.max_exchange_mat.shape[0] * self.max_exchange_mat.shape[1]
        )
        n_toxin_vals = self.toxin_mat.shape[0] * self.toxin_mat.shape[1]
        n_biomass_constraint_vals = len(self.populations)

        self.init_y = parameter_vec[0:n_variables]
        self.set_k_value_matrix(
            parameter_vec[n_variables: n_variables + n_k_vals].reshape(
                self.k_vals.shape
            )
        )

        self.set_max_exchange_mat(
            parameter_vec[
                n_variables + n_k_vals : n_variables + n_k_vals + n_max_exchange_vals
            ].reshape(self.max_exchange_mat.shape)
        )

        self.set_biomass_rate_constraints(
            parameter_vec[
                n_variables
                + n_k_vals
                + n_max_exchange_vals : n_variables
                + n_k_vals
                + n_max_exchange_vals
                + n_biomass_constraint_vals
            ].reshape(self.biomass_constraints.shape)
        )

        self.set_toxin_mat(
            parameter_vec[
                n_variables
                + n_k_vals
                + n_max_exchange_vals
                + n_biomass_constraint_vals : n_variables
                + n_k_vals
                + n_max_exchange_vals
                + n_biomass_constraint_vals
                + n_toxin_vals
            ].reshape(self.toxin_mat.shape)
        )

    def load_initial_compound_values(self, sub_media_df: pd.DataFrame) -> np.ndarray:
        """Loads vector of initial compound concentrations from a subset of a media df, containing only one

        Args:
            sub_media_df : dataframe of media file containing only one media name

        Returns:
            init_compound_values : numpy array containing initial compound values

        Raises:

        ToDo:
            * consider renaming this function
            * Add assertion that only one media name exists

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

    def set_initial_populations(self, populations_vec: List[float]):
        """Setter for initial population

        Args:
            populations_vec : list containing intial populations
        Raises:

        """
        assert len(populations_vec) == (
            len(self.populations)
        ), f"Error: length of parameter_vec, {len(populations_vec)} \
            does not match expected  {len(self.populations)}"

        self.init_population_values = populations_vec

    def set_k_value_matrix(self, k_val_mat: np.ndarray):
        """Setter for k-values matrix

        Args:
            k_val_mat : array of k-values
        Raises:
        """

        assert k_val_mat.shape == (
            len(self.populations),
            len(self.dynamic_compounds),
        ), f"Error: shape of kval mat, {k_val_mat.shape} does not match expected  {(len(self.populations), len(self.dynamic_compounds))}"

        self.k_vals = k_val_mat

    def set_max_exchange_mat(self, max_exchange_mat: np.ndarray):
        """Setter for max exchange matrix

        Args:
            max_exchange_mat : array of max exchange values
        Raises:
        """

        assert max_exchange_mat.shape == (
            len(self.populations),
            len(self.dynamic_compounds),
        ), f"Error: shape of max_exchange_mat, {max_exchange_mat.shape} does not match expected  {(len(self.populations), len(self.dynamic_compounds))}"
        self.max_exchange_mat = max_exchange_mat

    def set_toxin_mat(self, toxin_mat: np.ndarray):
        """Setter for toxin interaction matrix

        Args:
            toxin_mat : array of toxin interaction values
        Raises:
        """

        assert toxin_mat.shape == (
            len(self.populations),
            len(self.populations),
        ), f"Error: shape of toxin_mat, {toxin_mat.shape} does not match expected  {(len(self.populations), len(self.populations))}"
        self.toxin_mat = toxin_mat

    def set_biomass_rate_constraints(self, biomass_constraints: np.ndarray):
        """Setter for biomass constraints

        Args:
            biomass_constraints : array biomass constraint values
        Raises:
        """

        assert biomass_constraints.shape[0] == (
            len(self.populations)
        ), f"Error: shape of biomass_constraints[0], {biomass_constraints.shape[0]} does not match expected  {len(self.populations)}"
        self.biomass_constraints = biomass_constraints

    def set_solution_key_order(self) -> List[str]:
        """Sets solution keys. Defining the order of species, compounds and toxins in simulation solutions

        Returns:
            solution_keys : list defining order of simulation solutions

        Raises:

        """

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
        models: List[cobra.core.model.Model],
        dynamic_compounds: List[str],
        reaction_keys: List[str],
        media_df,
    ) -> List[Population]:
        """
        Load population objects from model paths

        Args:
            model_names : list of model names
            model_paths : list of model paths
            dynamic_compounds : list of dynamic cmpounds e.g 'M_ala__L_e'
            reaction_keys : list of reaction keys corresponding to compound e.g 'EX_ala__L_e'

        Returns:
            populations : list of Population objects

        Raises:

        """
        populations = []
        for idx, n in enumerate(model_names):
            populations.append(
                Population(
                    n,
                    models[idx],
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
        flavor: str = "fbc2",
    ) -> List[str]:
        """Generates a list of dynamic compounds

        Uses SMETANA output to get compounds involved in competition and
        crossfeeding

        1. Generate complete environment reactions.
        2. Read SMETANA output full output
        3. Identify metabolites involved in crossfeeding
        4. Identify metabolites involved in competition

        Args:
            model_paths : list of model paths
            smetana_analysis_path : path to smetana analysis
            media_df : media dataframe
            flavor : flavor for carveme (use fbc2)

        Returns:
            dynamic_compounds : list of dynamic compounds

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

        dynamic_compounds = list(dict.fromkeys(dynamic_compounds))
        dynamic_compounds = list(set(dynamic_compounds))

        dynamic_compounds.sort()

        print(f"Total env compounds: {len(all_compounds)}")
        print(f"Crossfeeding compounds: {len(cross_feeding_compounds)}")
        print(f"Competition compounds: {len(competition_compounds)}")
        print(f"Media compounds: {len(media_compounds)}")
        print(f"Toxin compounds: {len(self.toxin_names)}")
        print(f"Dynamic compounds: {len(dynamic_compounds)}")

        return dynamic_compounds

    def gen_population_indexes(self):
        """
        generates population indexes

        Args:
            self : self
        Returns:
            pop_indexes : indexes for populations used in solutions
        """
        num_populations = len(self.populations)
        pop_indexes = list(range(num_populations))
        return pop_indexes

    def gen_compound_indexes(self) -> np.ndarray:
        """
        generates compound indexes

        Returns:
            pop_indexes : indexes for populations used in solutions
        """

        num_dynamic_cmpds = len(self.dynamic_compounds)
        num_populations = len(self.populations)

        compound_indexes = list(
            range(num_populations, num_populations + num_dynamic_cmpds)
        )
        return compound_indexes

    def set_init_y(self):
        """
        Sets attribute initial y, concentrations of populations and compounds

        Args:
            self : self
        """

        self.init_y = np.concatenate(
            (self.init_population_values, self.init_compound_values), axis=None
        )

    def set_media_conditions(self, media_name, set_media=True):
        """ Sets current media dataframe and applys to model

        1. Subsets media dataframe for the given media_name
        2. Sets curr_media_df attribute
        3. Sets init_compound_values attrubite from the current media
        4. Optionally sets the media for each model in the community populations

        Args:
            media_name : name of media to use
            set_media : bool to define whether media is set

        Returns:
            populations : list of Population objects

        Todo
            * This should be two functions

        """

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
    ) -> np.ndarray:
        """Calculates exchange reaction constraints based on michaelis menten dynamics

        Args:
            compound_concs : concentrations of dynamic compounds
            k_mat : matrix of k values
            max_exchange_mat : matrix of vmax values

        Returns:
            constraints_mat : numpy array of new exchange reaction constraints

        ToDo
            * This should be two functions

        """

        # Get matrix where exchange is uptake (negative numbers)
        uptake_mat = np.clip(max_exchange_mat, a_min=None, a_max=0)
        uptake_mat = uptake_mat * (compound_concs / (k_mat + compound_concs))

        # Get matrix where exchange is secretion (positive numbers)
        secretion_mat = np.clip(max_exchange_mat, a_min=0, a_max=None)

        constraints_mat = uptake_mat + secretion_mat

        return constraints_mat

    def sim_step(self, y) -> Tuple[np.ndarray, np.ndarray]:
        """Simulates a single step by calculating michaelis menten constraints,
        applying them to the population models, and optimising for the objective
        function

        Args:
            y : y


        Returns:
            growth_rates : numpy array of new exchange reaction constraints

        Todo
            * This should be two functions
        """
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

            # Set biomass rate constraint
            pop.model.reactions.get_by_id(self.objective_reaction_keys[idx]).bounds = (
                0.0,
                self.biomass_constraints[idx],
            )

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
