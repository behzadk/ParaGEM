import cobra
import cobra.test
import cometspy
import os

import matplotlib.pyplot as plt
import numpy as np

def main():
    # os.environ['GUROBI_HOME'] = "/Library/gurobi912/mac64"
    os.environ['GUROBI_COMETS_HOME'] = "/Library/gurobi950/macos_universal2/"
    os.environ['COMETS_HOME'] = "/Users/bezk/Documents/CAM/research_code/comets" 
    # os.environ['JAVA_HOME'] = "/Library/Java/JavaVirtualMachines/jdk-17.0.1.jdk/Contents/Home"

    comets_home_dir =  "/Users/bezk/Documents/CAM/research_code/comets"
    gurobi_home_dir = "/Library/gurobi912/mac64"
    jdistlib_jar_path = f"{comets_home_dir}/lib/jdistlib/jdistlib-0.4.5-bin.jar"
    
    dt = 0.1
    t = (48 * 4) - 1
    max_cycles = int(np.ceil(t / dt))

    # Load a textbook example model using the COBRAPy toolbox 
    test_model = cobra.test.create_test_model('textbook')

    # Use the above model to create a COMETS model
    test_model = cometspy.model(test_model)

    # Change comets specific parameters, e.g. the initial biomass of the model
    # Notre 
    test_model.initial_pop = [0, 0, 1e-6]

    # Create a parameters object with default values 
    my_params = cometspy.params()

    # Change the value of a parameter, for example number of simulation cycles
    my_params.set_param('maxCycles', max_cycles)
    my_params.set_param('batchDilution', True)
    my_params.set_param("timeStep", dt)
    my_params.set_param('dilFactor', 0.02)            # Dilution to apply
    my_params.set_param('dilTime', 23)                # hours

    # Set some writeTotalBiomassLog parameter to True, in order to save the output
    my_params.set_param('writeTotalBiomassLog', True)

    my_layout = cometspy.layout(test_model)


    # Add 11mM glucose and remove o2
    my_layout.set_specific_metabolite('glc__D_e', 0.011)
    my_layout.set_specific_metabolite('o2_e', 0)

    # Add the rest of nutrients unlimited (ammonia, phosphate, water and protons)
    my_layout.set_specific_metabolite('nh4_e',1000);
    my_layout.set_specific_metabolite('pi_e',1000);
    my_layout.set_specific_metabolite('h2o_e',1000);
    my_layout.set_specific_metabolite('h_e',1000);

    my_layout.media

    my_simulation = cometspy.comets(my_layout, my_params)

    comets_lib = "/Users/bezk/Documents/CAM/research_code/comets/lib"

    my_simulation.set_classpath(
        "bin", "/Users/bezk/Documents/CAM/research_code/comets/bin/comets.jar"
    )

    my_simulation.set_classpath(
        "jdistlib", f"{comets_lib}/jdistlib/jdistlib-0.4.5-bin.jar"
    )

    my_simulation.run()
    # fig, ax = plt.subplots(figsize=(15, 5))

    ax = my_simulation.total_biomass.plot(x = 'cycle')
    ax.set_ylabel("Biomass (gr.)")

    plt.show()



if __name__ == '__main__':
    main()