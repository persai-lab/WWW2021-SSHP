from typing import Tuple, Dict
import numpy as np

# Initialize seed
init_seed = 0
np.random.seed(init_seed)


# The following function generates normally distributed points givem mu, sigma
def generate_normal_dist_points(mu: float, sigma: float, shape: Tuple):
    return np.random.normal(mu, sigma, shape)


# This function returns data dictionary having all the required parameters
def generate_synthetic_data() -> Dict:
    mu_a, sigma_a = 0.4, 0.1
    mu_m, sigma_m = 0, 5
    mu_gammad, sigma_gammad = 15, 3
    mu_gammao, sigma_gammao = 5, 3
    mu_gammah, sigma_gammah = 0.5, 0.1
    mu_v, sigma_v = 20, 10
    mu_b, sigma_b = 0.5, 0.3
    mu_p, sigma_p = 6, 4
    mu_c, sigma_c = 1.2, 0.1
    m, n = 10, 5

    A = generate_normal_dist_points(mu_a, sigma_a, (m, n))
    M = generate_normal_dist_points(mu_m, sigma_m, (m, n))
    Gamma_d = generate_normal_dist_points(mu_gammad, sigma_gammad, (m, n))
    Gamma_o = generate_normal_dist_points(mu_gammao, sigma_gammao, (m, n))
    Gamma_h = generate_normal_dist_points(mu_gammah, sigma_gammah, (m, n))
    v = generate_normal_dist_points(mu_v, sigma_v, (m, n))
    b = generate_normal_dist_points(mu_b, sigma_b, (m, n))
    p = generate_normal_dist_points(mu_p, sigma_p, (m, n))
    c = generate_normal_dist_points(mu_c, sigma_c, (m, n))

    data = {'A': A, 'M': M, 'Gamma_d': Gamma_d, 'Gamma_o': Gamma_o, 'Gamma_h': Gamma_h, 'v': v,
            'b': b, 'p': p, 'c': c}

    return data
