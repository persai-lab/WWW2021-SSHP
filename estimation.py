import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy.io import loadmat
import numpy.ma as ma


def R_function(x,beta):
    Rxnew = [0]
    for i in range(1, len(x)):
        Rprev = Rxnew[-1]
        Rcurr = (1 + Rprev) * (np.exp(-beta * (x[i] - x[i - 1])))
        Rxnew.append(Rcurr)
    return np.array(Rxnew)

def compute_intensity(x,alpha,beta,v,m,b,s,p,c,gammad, gammao, gammat):
    t = x[-1]
    if t<=m:
        mud = 1/np.sqrt(2*np.pi*v)/(m - t/s)*np.exp(-(np.log(m - t/s))**2/v)
    else:
        mud = 0
    muo = b**(t/s)
    mut = np.sin(np.pi/24*(t/s+p))+c
    R = R_function(x,beta)[-1]
    _lambda = gammad * mud + gammao * muo + gammat * mut + alpha * beta * R
    return _lambda


def sample_single_time(x,alpha,beta,v,m,b,s,p,c,gammad, gammao, gammat):
    thin = False
    while thin == False:
        I = compute_intensity(x,alpha,beta,v,m,b,s,p,c,gammad, gammao, gammat)
        next_time = x[-1]
        u = np.random.uniform(0,1)
        tau = -np.log(u)/I

        next_time = next_time + np.random.exponential(1/I)
        # next_time+=tau
        x = np.append(np.array(x), next_time)
        Is = compute_intensity(x,alpha,beta,v,m,b,s,p,c,gammad, gammao, gammat)
        c = np.random.rand()
        if c*I>Is:
            'reject'
        else:
            # print('accept')
            thin = True
        return next_time


def return_time(x,alpha,beta,v,m,b,s,p,c,gammad, gammao, gammat,N,method = ''):
    sampled_time = []
    for i in range(N):
        sample = sample_single_time(x,alpha,beta,v,m,b,s,p,c,gammad, gammao, gammat)
        sampled_time.append(sample)
    if method == 'mean':
        next_time = np.mean(sampled_time)
    elif method == 'median':
        next_time = np.median(sampled_time)
    return next_time


def sample_single_times(x,train_len,alpha,beta,v,m,b,s,p,c,gammad, gammao, gammat,n,N,add_true_next = False):

    sampled_time = []
    x_obs = np.array(x[:train_len])
    i = 1
    for i in range(n):
        nt = return_time(x_obs,alpha,beta,v,m,b,s,p,c,gammad, gammao, gammat,N,method = 'median')
        sampled_time.append(nt)
        if add_true_next == False:
            x_obs = np.append(np.array(x_obs), nt)
        else:
            x_obs = x[:train_len+i]
            i = i+1
            # print(train_len+i)
            # print(x_obs[-1])
    return np.array(sampled_time)

def run_real(decay,data,est,next_no,samp_no = 100,add_true_next = False):
    mask = data.get('mask')
    gen_mask = data.get('gen_mask')
    holdout_idx = mask-gen_mask
    train_x = data.get('X_train')
    test_x = data.get('X_test')
    x = data.get('X')
    m, n = train_x.shape
    est_A = est.get('A')
    est_M = est.get('M')
    est_Gamma_d = est.get('Gammad')
    est_Gamma_o = est.get('Gammao')
    est_Gamma_h = est.get('Gammah')
    est_v = est.get('v')
    est_b = est.get('b')
    est_p = est.get('p')
    est_c = est.get('c')
    E_test = []
    E_holdout = []
    for i in range(m)[:10]:
        for j in range(n)[:]:
            print(i,j,gen_mask[i,j],holdout_idx[i,j],)
            if gen_mask[i,j]==1:

                train_len = len(train_x[i, j][0])
                # try:
                smp_times = sample_single_times(x[i,j][0],train_len,est_A[i,j],decay,est_v[i,j],est_M[i,j],
                                  est_b[i,j],1,est_p[i,j],est_c[i,j],est_Gamma_d[i,j],
                                  est_Gamma_o[i,j], est_Gamma_h[i,j],next_no,samp_no,add_true_next)
                smp_times = smp_times[:min(next_no,len(test_x[i,j][0]))]
                err = smp_times - test_x[i,j][0][:len(smp_times)]
                err = list(err) + [np.nan]*(next_no - len(err))
                E_test.append((err))
                # print(f'test sample: {err}')
                # except:pass
            else:
                if holdout_idx[i,j] == 1:
                    smp_times = sample_single_times(x[i,j][0],1,est_A[i,j],decay,est_v[i,j],est_M[i,j],
                                      est_b[i,j],1,est_p[i,j],est_c[i,j],est_Gamma_d[i,j],
                                      est_Gamma_o[i,j], est_Gamma_h[i,j],next_no,samp_no,add_true_next)
                    smp_times = smp_times[:min(next_no,len(train_x[i,j][0])-1)]
                    err = smp_times - train_x[i,j][0][1:len(smp_times)+1]
                    err = list(err) + [np.nan]*(next_no - len(err))
                    E_holdout.append((err))

    e_test = np.where(np.isnan(E_test), ma.array(E_test, mask=np.isnan(E_test)).mean(axis=0), E_test)
    e_holdout = np.where(np.isnan(E_holdout), ma.array(E_holdout, mask=np.isnan(E_holdout)).mean(axis=0), E_holdout)

    return e_test, e_holdout


def compute_error(e):
    evl = np.mean(np.abs(e), axis=0)
    return evl


decay = 10
data_path = 'data.mat'
data = loadmat(data_path)
est = loadmat(f'canvas_traintest_decay = {decay}')


e_test, e_holdout = run_real(decay,data,est,10,50,add_true_next=False)
error_test,error_holdout = compute_error(e_test),compute_error(e_holdout)