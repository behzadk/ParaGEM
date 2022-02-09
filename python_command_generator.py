import pandas as pd
    

def write_execution_command(submissions_path, species_name, run_idx):
    experiment_name = species_name + '_indiv'
    print(experiment_name)
    submission_df = pd.read_csv(submissions_path)
    df = submission_df.loc[submission_df['cfg.experiment_name'] == experiment_name]

    base_cfg = 'mel_indiv_growth_base'
    python_string = f'python -u bk_comms cfg={base_cfg} cfg.run_idx={run_idx} '

    for col in df.columns:
        python_string += f' {col}={df[col].values[0]} '

    with open(f"./exec_{species_name}_{run_idx}.sh", "w") as text_file:
        text_file.write(python_string)
