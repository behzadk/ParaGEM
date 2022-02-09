from python_command_generator import write_execution_command

# Path to file with species specific configs
submission_path = './snake_make_submission_configs/submission_configs.csv'

rule all:
    output:
        temp("exec_{species_name}_{run_idx}.sh")
    run:
        # Generate execution file
        write_execution_command(submission_path, wildcards.species_name, wildcards.run_idx)
        
        # Give execution permissions
        shell('chmod +x ./exec_{wildcards.species_name}_{wildcards.run_idx}.sh \n')

rule execute_run:
    input:
        'exec_{species_name}_{run_idx}.sh'

    output:
        temp('mel_indiv_logs/{species_name}_{run_idx}.running'),

    run:
        # Signal running
        shell('touch mel_indiv_logs/{wildcards.species_name}_{wildcards.run_idx}.running')

        # Execute
        shell('./exec_{wildcards.species_name}_{wildcards.run_idx}.sh')

rule clean_up:
    input:
        'mel_indiv_logs/{species_name}_{run_idx}.running'

    output:
         'mel_indiv_logs/{species_name}_{run_idx}.finished'

    run:
        shell('echo finished')
        shell('touch mel_indiv_logs/{wildcards.species_name}_{wildcards.run_idx}.finished')

