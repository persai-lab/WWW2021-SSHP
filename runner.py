from inference_real import *
from scipy.io import loadmat

data = loadmat('data.mat')
X = data.get('X_train')
mask = data.get('mask')

# params initializations
M,N = X.shape
M0 = []
for i in range(M):
    for j in range(N):
        try:
            M0.append((X[i,j][0][-1]+1200)/2) # initialize M as 1200 unit time after the last interaction time
        except:
            M0.append(1200)
M0 = np.array(M0).reshape(M,N)
init = {}
init['A0'] = abs(np.random.normal(loc = 1,scale = 0.1,size = (M,N)))
init['M0'] = M0
init['b0'] = np.ones((M, N)) * 0.5
init['v0'] = np.ones((M, N)) * 10
init['c0'] = np.ones((M, N)) * 1.8
init['p0'] = np.ones((M, N)) * 8
init['Gamma_d0'] = np.ones((M,N))*10
init['Gamma_o0'] = np.ones((M,N))*10
init['Gamma_h0'] = np.ones((M,N))*10

decay = 10 #hyperparameter
est,funcVal = customized_Hawkes(data, init, mask, decay, 0.5, 25, term_cond =1)
scipy.io.savemat(f'est_decay_{decay}', mdict=est) # save estimated params
est