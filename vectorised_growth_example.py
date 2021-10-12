import numpy as np
import scipy as sp

def calculate_growth_rate(s_mat, k_vals_mat, max_growth_rates):
    growth_frac = (s_mat / (k_vals_mat + s_mat)).prod(axis=1)
    
    return growth_frac

def diff_eqs(y, t, pop_params):
    S_1, S_2, N_1, N_2 = y

    k_values = get_k_val_matrix()

    # Population
    # Compound
    # Parameters: Max growth, K

    K_vals = get_k_val_matrix()
    max_growth_rates = get_max_growth_rates()
    s_mat = make_s_mat(s_vals=[y[0], y[1]], n_populations=2)
    
    calculate_growth_rate_matrix(K_vals, s_mat)


    dN_1 = pop_params[0][0][0] * (S_1 / (S_1 + (pop_params[0][0][1]))) * (S_2 / S_2 + (pop_params[0][1][1]))
    dN_2 = pop_params[1][0][0] * (S_1 / (S_1 + (pop_params[0][0][1]))) * (S_2 / S_2 + (pop_params[0][1][1]))



def calculate_growth_rate_matrix(k_vals_mat, s_mat):
    return k_vals_mat * s_mat

def get_k_val_matrix():
    x = [
        [0.0, 0.2],
        [0.4, 0.2]
        ]

    return x

def make_s_mat(s_vals, n_populations):
    return np.tile(s_vals, n_populations).reshape(n_populations, -1)

def main():
    pass


if __name__ == "__main__":
    main()