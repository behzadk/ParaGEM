import numpy as np
import scipy as sp


if __name__ == "__main__":
    np.random.seed(seed=2)
    populations = np.random.random(size=3)
    print("populations", populations)

    fluxes = np.random.randint(-1, 2, size=[3, 5])
    print("fluxes", fluxes)
    

    changes = np.dot(populations, fluxes)
    print("changes", changes)


    flux_0 = populations[0] * fluxes[0][0] + populations[1] * fluxes[1][0] + populations[2] * fluxes[2][0]

    print(flux_0)