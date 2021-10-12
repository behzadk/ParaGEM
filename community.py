import numpy as np
import scipy as sp
import smetana
from reframed import Environment
from smetana.interface import load_media_db
from typing import List
import pandas as pd
from smetana.interface import load_communities

import utils
import cobra
import simulate_dfba
from scipy.integrate import odeint


import logging
logging.getLogger("cobra").setLevel(logging.ERROR)

import matplotlib.pyplot as plt

import numba


class Population:
    def __init__(self, model_path, dynamic_compounds, reaction_keys, media_df):
        self.model = self.load_model(model_path)
        self.media_df = media_df 

        self.dynamic_compounds = dynamic_compounds
        self.reaction_keys = reaction_keys
        
        self.dynamic_compound_mask = self.set_dynamic_compound_mask(dynamic_compounds)
        self.set_media()
        # self.set_max_growth_rate(2.0)
        
    def load_model(self, model_path):
        """
        Loads models from model paths
        """
        model = cobra.io.read_sbml_model(model_path)

        return model

    def set_media(self):
        model_medium_compounds = list(self.model.medium)
        dynamic_compounds = [x.replace('M_', 'EX_') for x in self.dynamic_compounds]

        defined_mets = ['EX_' + x + '_e' for x in self.media_df['compound'].values]

        for met in list(self.model.medium):
            if met in defined_mets:
                key = met.replace('EX_', '').replace('_e', '')
                
                medium = self.model.medium
                medium[met] = self.media_df.loc[self.media_df['compound'] == key]['mmol_per_L'].values
                self.model.medium = medium

            
            elif met in dynamic_compounds:
                medium = self.model.medium
                medium[met] = 0.0
                self.model.medium = medium

            else:
                medium = self.model.medium
                medium[met] = 0.0
                self.model.medium = medium
    
    def set_dynamic_compound_mask(self, community_dynamic_compounds):
        dynamic_compound_mask = []
        for reaction in community_dynamic_compounds:
            key = reaction.replace('M_', 'EX_')
            
            if self.model.reactions.has_id(key):
                dynamic_compound_mask.append(1)
            
            else:
                dynamic_compound_mask.append(0)

        return dynamic_compound_mask

    def set_max_growth_rate(self, max_growth_rate=2.0):
        self.model.reactions.get_by_id('Growth').upper_bound = max_growth_rate

    @staticmethod
    @numba.jit(parallel=False)
    def calculate_monod_uptake(max_uptake, cmpd_conc, k):
        return max_uptake * (cmpd_conc / (k + cmpd_conc) )

    def update_reaction_fluxes(self, new_compound_concentrations, k_values, default_max_uptake=-1):
        for idx, new_cmpd_conc in enumerate(new_compound_concentrations):
            if self.dynamic_compound_mask[idx]:
                self.model.reactions.get_by_id(self.reaction_keys[idx]).lower_bound = new_compound_concentrations[idx]

    def optimize(self):
        self.opt_sol = self.model.optimize()
      
    def get_dynamic_compound_fluxes(self):
        compound_fluxes = np.zeros(len(self.dynamic_compounds))

        for compound_idx, reaction in enumerate(self.dynamic_compounds):
            if self.dynamic_compound_mask[compound_idx]:
                key = reaction.replace('M_', 'EX_')
                compound_fluxes[compound_idx] = self.opt_sol.get_primal_by_id(key)

        return compound_fluxes

    def get_growth_rate(self):
        return self.opt_sol.objective_value
        # return self.opt_sol.get_primal_by_id('Growth')



class Community:
    def __init__(self, model_paths: List[str], smetana_analysis_path: str, media_path: str, media_name: str):
        self.model_paths = model_paths
        self.smetana_analysis_path = smetana_analysis_path
        self.media_path = media_path
        
        self.media_df = pd.read_csv(self.media_path, delimiter='\t')
        self.media_df = self.media_df.loc[self.media_df['medium'] == media_name].reset_index(drop=True)       


        self.dynamic_compounds = self.get_dynamic_compounds(model_paths, smetana_analysis_path, self.media_df)
        self.reaction_keys = [x.replace('M_', 'EX_') for x in self.dynamic_compounds]
        
        self.populations = self.load_populations(self.model_paths, self.dynamic_compounds, self.reaction_keys, self.media_df)

        self.init_compound_values = self.load_initial_compound_values(
            self.media_path, 
            self.dynamic_compounds)


        self.init_population_values = self.generate_initial_population_densities()
        
        self.k_vals = self.generate_default_k_values(self.init_compound_values, media_name)


    def generate_initial_population_densities(self):
        return np.array([0.01 for x in self.populations])

    def load_initial_compound_values(self, media_path, dynamic_compounds):
        init_compound_values = np.zeros(len(dynamic_compounds))

        for compound_idx, dynm_cmpd in enumerate(dynamic_compounds):
            found_match = False

            for idx, row in self.media_df.iterrows():
                media_cmpd = 'M_' + row['compound'] + '_e'

                if dynm_cmpd == media_cmpd:
                    found_match = True
                    init_compound_values[compound_idx] = row['mmol_per_L']
                    break
            
            if not found_match:
                print(dynm_cmpd)

                init_compound_values[compound_idx] = 0.01

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

    def load_populations(self, model_paths: List[str], dynamic_compounds, reaction_keys, media_df):
        """
        Load population objects from model paths
        """
        populations = []
        for p in model_paths:
            populations.append(Population(p, dynamic_compounds, reaction_keys, media_df))
        
        return populations

    def get_dynamic_compounds(self, model_paths: List[str], smetana_analysis_path: str, media_df: pd.DataFrame):
        """
        1. Generate complete environment reactions.
        2. Read SMETANA output full output
        3. Identify metabolites involved in crossfeeding
        4. Identify metabolites involved in competition
        """
        
        # Generate complete environment metabolites dict
        compl_env = utils.get_community_complete_environment(self.model_paths, max_uptake=10.0)
        all_compounds = compl_env.compound.values

        # Get crossfeeding and competition metabolites
        smetana_df = pd.read_csv(smetana_analysis_path, delimiter='\t')
        cross_feeding_compounds = utils.get_crossfeeding_compounds(smetana_df)
        competition_compounds = utils.get_competition_compounds(smetana_df, all_compounds)

        
        media_compounds = ['M_' + m + '_e' for m in media_df['compound'].values]

        # Aggregate dynamic compounds
        dynamic_compounds = set(cross_feeding_compounds + competition_compounds + media_compounds)

        # media_compounds = pass

        print(f'Total env compounds: {len(all_compounds)}')
        print(f'Crossfeeding compounds: {len(cross_feeding_compounds)}')
        print(f'Competition compounds: {len(competition_compounds)}')
        print(f'Media compounds: {len(media_compounds)}')
        print(f'Dynamic compounds: {len(dynamic_compounds)}')

        return list(dynamic_compounds)
    
    def simulate_community(self):
        num_populations = len(self.populations)
        num_dynamic_cmpds = len(self.dynamic_compounds)

        self.population_indexes = range(num_populations)
        self.compound_indexes = range(num_populations, num_populations + num_dynamic_cmpds)

        init_y = np.concatenate((self.init_population_values, self.init_compound_values), axis=None)

        y0 = init_y
        t_0 = 0.0
        t_end = 12.0
        steps = 1000

        t = np.linspace(t_0, t_end, steps)
        sol = odeint(self.diff_eqs, y0, t, args=())

        self.print_sol(sol)

        plt.plot(t, sol[:, 0])
        # plt.plot(t, sol[:, 1])
        plt.show()

    def print_sol(self, sol):
        count = 1
        for idx in self.population_indexes:
            print(f"N_{idx}: {sol[:, idx][-1]}")

        for idx, sol_idx in enumerate(self.compound_indexes):
            print(f"{self.dynamic_compounds[idx]}: {sol[:, sol_idx][-1]}")

    @staticmethod
    @numba.jit(parallel=False)
    def calculate_exchange_reaction_constraints(compound_concs, k_mat, default_max_uptake=-1):
        return default_max_uptake * (compound_concs / (k_mat + compound_concs))

    def sim_step(self, y):
        compound_concs = y[self.compound_indexes]

        flux_matrix = np.zeros(shape=[len(self.populations), len(self.dynamic_compounds)])
        growth_rates = np.zeros(len(self.populations))

        exchange_fluxes = self.calculate_exchange_reaction_constraints(compound_concs, self.k_vals, default_max_uptake=-1)

        for idx, pop in enumerate(self.populations):
            pop.update_reaction_fluxes(exchange_fluxes[idx], self.k_vals[idx])
            pop.optimize()

            flux_matrix[idx] = pop.get_dynamic_compound_fluxes()
            growth_rates[idx] = pop.get_growth_rate()

        return growth_rates, flux_matrix


    def diff_eqs(self, y, t):
        y = y.clip(0)
        # y[y < 1e-25] = 0
        populations = y[self.population_indexes]
        compounds = y[self.compound_indexes]

        growth_rates, flux_matrix = self.sim_step(y)
        print(t, growth_rates)

        output = np.zeros(len(y))

        output[self.population_indexes] = growth_rates * populations
        output[self.compound_indexes] = np.dot(populations, flux_matrix)

        return output



def main():
    # model_paths = ['./models/E_col/E_coli_IAI1.xml']
    # model_paths = ['./models/L_lactis/L_lactis.xml', './models/L_plantarum/L_plantarum.xml']

    model_paths = ['./models/E_col/Ecoli_K12_MG1655.xml']

    smetana_analysis_path = './carveme_output/lactis_plant_detailed.tsv'

    # smetana_analysis_path = './carveme_output/lactis_plant_ecol_detailed.tsv'


    media_path = './media_db_CDM35.tsv'

    comm = Community(model_paths, smetana_analysis_path, media_path, 'LB')
    comm.simulate_community()


if __name__ == "__main__":
    main()