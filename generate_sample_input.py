from scipy.io import savemat
import numpy as np
from scipy.stats import bernoulli


def gen_mask(m, n, prob_masked=0.5):
    return 1 - bernoulli.rvs(p=prob_masked, size=(m, n))


x = np.array([np.array(
    [range(2, 10), range(5, 13), range(4, 12), range(4, 12), range(4, 12)])] * 10)

m = 10
n = 5
mask = gen_mask(10, 5)
gen_mask = gen_mask(10, 5)

data_dict = {'X_train': x, 'mask': mask, 'gen_mask':gen_mask, 'X':x, 'X_test':x}
savemat("data.mat", data_dict)
