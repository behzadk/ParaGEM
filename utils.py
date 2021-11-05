import numpy as np
import pandas as pd
from typing import List

from smetana.legacy import Community
from reframed import Environment
from smetana.interface import load_communities
from pathlib import Path

def get_community_complete_environment(model_paths: List[str], max_uptake=10.0, flavor='cobra'):
    """
    Generates a complete environment for a given list of model paths. 
    """
    # Load smetana community
    model_cache, comm_dict, other_models = load_communities(
        model_paths, 
        None, None, 
        flavor=flavor)
    
    comm_models = [model_cache.get_model(org_id, reset_id=True) for org_id in comm_dict['all']]
    community = Community('all', comm_models, copy_models=False)

    # Use reframed to generate complete environment
    compl_env = Environment.complete(community.merged, max_uptake=max_uptake)
    
    # Convert to dataframe
    compl_env = pd.DataFrame.from_dict(compl_env, columns=['lwr', 'upr'], 
    orient='index')
    compl_env.index.name = 'reaction'
    compl_env.reset_index(inplace=True)

    # Reformat reaction names
    get_compound = lambda x: x.replace('_pool', '').replace('R_EX_', '')

    compl_env['compound'] = compl_env['reaction'].apply(get_compound)

    return compl_env


def get_crossfeeding_compounds(smetana_df: pd.DataFrame):
    """
    Generates list of metabolites involved in crossfeeding using the output of a SMETANA
    run
    """
    return list(smetana_df.loc[smetana_df['smetana'] > 0.0].compound)


def get_competition_compounds(smetana_df: pd.DataFrame, all_compounds):
    """
    Generates list of metabolites involved in crossfeeding using the output of a SMETANA
    run
    """
    competition_compounds = []

    for m in all_compounds:
        sub_df = smetana_df.loc[smetana_df['compound'] == m]
        
        if sub_df['mus'].sum() > 1.0:
            competition_compounds.append(m)

    return competition_compounds


