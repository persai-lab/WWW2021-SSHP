from gradients_update import *
from trace_projection import *
from scipy.io import loadmat
import warnings
warnings.filterwarnings('ignore')
from scipy.stats import bernoulli
import scipy.io
from svt import *
def gen_mask(m, n, prob_masked=0.5):
    """
    Generate a binary mask for m by n matrix
    Note that 1 denotes observed, and 0 denotes unobserved.
    """
    return 1 - bernoulli.rvs(p=prob_masked, size=(m, n))

def customized_Hawkes(data,mask,decay,rho, itermax,term_cond = 1,train_split = 0.7):
    # x = np.array([np.array([range(2, 10), range(3, 11), range(4, 12)])] * 5)  # M x N x K, N = 3, M = 5, K = 8
    x = data.get('X_train')
    # train_x =
    true_A = data.get('A')
    true_M = data.get('M')
    true_Gamma_d = data.get('Gamma_d')
    true_Gamma_o = data.get('Gamma_o')
    true_Gamma_h = data.get('Gamma_h')
    M, N = x.shape
    true_v = data.get('v')
    true_b = data.get('b')
    true_p = data.get('p')
    true_c = data.get('c')
    # initializations
    b0 = np.ones((M,N))
    v0 = np.ones((M,N))
    c0 = np.ones((M,N))
    p0 = np.ones((M,N))

    vz,vz_old = v0, v0
    bz, bz_old = b0, b0
    pz, pz_old = p0, p0
    cz, cz_old = c0, c0


    M0 = np.ones((M,N))
    A0 = np.ones((M,N))
    Gamma_d0 = np.ones((M,N))
    Gamma_o0 = np.ones((M,N))
    Gamma_h0 = np.ones((M,N))


    Az, Az_old = A0, A0
    Mz, Mz_old = M0, M0
    Gamma_dz, Gamma_dz_old = Gamma_d0, Gamma_d0
    Gamma_oz, Gamma_oz_old = Gamma_o0, Gamma_o0
    Gamma_hz, Gamma_hz_old = Gamma_h0, Gamma_h0


    funcVal = [compute_grads(x, mask, A0, decay, v0, M0, b0, 1, p0, c0, Gamma_d0,
                             Gamma_o0, Gamma_h0, out_grad=True)[-1]]
    true_loss = [compute_grads(x, mask, true_A, decay, true_v, true_M, true_b, 1, true_p, true_c, true_Gamma_d,
                             true_Gamma_o, true_Gamma_h, out_grad=True)[-1]]
    print(f'true_loss: {true_loss}')
    t, t_old = 1,0

    iter = 0
    gamma = 100
    gamma_inc = 2
    while iter < itermax:
        print(iter)
        a = (t_old - 1) / t
        print(a)
        vs = (1 + a) * vz - a * vz_old
        bs = (1 + a) * bz - a * bz_old
        ps = (1 + a) * pz - a * pz_old
        cs = (1 + a) * cz - a * cz_old


        vs[vs <= 0] = 0.1
        bs[bs >= 1] = 0.99
        bs[bs <= 0] = 0.1
        # cs[cs <= 1] = 1.001
        # ps = np.mod(ps, np.ones((M,N))*24)
        As = (1 + a) * Az - a * Az_old
        Ms = (1 + a) * Mz - a * Mz_old
        Gamma_ds = (1 + a) * Gamma_dz - a * Gamma_dz_old
        Gamma_os = (1 + a) * Gamma_oz - a * Gamma_oz_old
        Gamma_hs = (1 + a) * Gamma_hz - a * Gamma_hz_old

        As[As <= 0] = 0.001
        As[As >= 1] = 0.99
        Ms[Ms <= 0] = 0.001
        Ms[Ms >= 100] = 100

        Gamma_ds[Gamma_ds < 0] = 0.01
        Gamma_os[Gamma_os < 0] = 0.01
        Gamma_hs[Gamma_hs < 0] = 0.01


        dalphas, dms, dvs, dbs, dps, dcs, dgammads, dgammaos, dgammahs, Fs_matrix, Fs \
            = compute_grads(x, mask, As, decay, vs, Ms, bs, 1, ps, cs, Gamma_ds,
                              Gamma_os, Gamma_hs, out_grad=True)

        while True:
            Azp, Azp_tn = trace_proj((As - dalphas / gamma / 100).astype(float), 4, 0.5,mask)
            # Azp, Azp_tn = trace_projection((As - dalphas / gamma / 100).astype(float), mask, 10)
            # Azp = svt_solve((As - dalphas / gamma / 100).astype(float), mask)
            Azp[Azp <= 0] = 0.01

            Mzp, Mzp_tn = trace_proj((Ms - dms / gamma).astype(float), 4, 0.5,mask)
            Mzp[Mzp <= 0] = 0.001
            Mzp[Mzp >= 100] = 100
            Gamma_dzp, Gamma_dzp_tn = trace_proj((Gamma_ds - dgammads / gamma / 5).astype(float), 3, 5,mask)
            Gamma_ozp, Gamma_ozp_tn = trace_proj((Gamma_os - dgammaos / gamma / 10).astype(float), 3, 5,mask)
            Gamma_hzp, Gamma_hzp_tn = trace_proj((Gamma_hs - dgammahs / gamma / 10).astype(float), 3, 1,mask)

            pzp, _ = trace_proj((ps - dps /gamma).astype(float),1,1,mask)
            bzp, _ = trace_proj((bs - dbs / gamma/10).astype(float),1,1,mask)
            czp, _ = trace_proj((cs - dcs / gamma/20).astype(float), 1, 1,mask)
            vzp, _ = trace_proj((vs - dvs / gamma).astype(float), 1,1,mask)
            bzp = np.array([np.mean(bzp,axis = 1),]*N).T
            czp= np.array([np.mean(czp,axis = 1),]*N).T
            vzp= np.array([np.mean(vzp,axis = 1),]*N).T

            vzp[vzp <= 0] = 0.1
            bzp[bzp >= 1] = 0.99
            bzp[bzp <= 0] = 0.1
            czp[czp <= 1] = 1.000001


            Fzp = compute_grads(x, mask, Azp, decay, vzp, Mzp, bzp, 1, pzp, czp, Gamma_dzp,
                             Gamma_ozp, Gamma_hzp,out_grad=False)[-1]
            delta_pzp = pzp - ps
            delta_bzp = bzp - bs
            delta_czp = czp - cs
            delta_vzp = vzp-vs
            delta_Azp = Azp - As
            delta_Mzp = Mzp - Ms
            delta_Gd = Gamma_dzp - Gamma_ds
            delta_Gh = Gamma_hzp - Gamma_hs
            delta_Go = Gamma_ozp - Gamma_os

            rsum = 0

            rsum += np.linalg.norm(delta_Mzp, 'fro') ** 2
            rsum += np.linalg.norm(delta_Azp, 'fro') ** 2
            rsum += np.linalg.norm(delta_Gd, 'fro') ** 2
            rsum += np.linalg.norm(delta_Go, 'fro') ** 2
            rsum += np.linalg.norm(delta_Gh, 'fro') ** 2
            rsum += np.linalg.norm(delta_pzp, 'fro') ** 2
            rsum += np.linalg.norm(delta_bzp, 'fro') ** 2
            rsum +=  np.linalg.norm(delta_czp, 'fro') ** 2
            rsum +=  np.linalg.norm(delta_vzp, 'fro') ** 2

            rsum = rsum/9
            Fzp_gamma = Fs
            Fzp_gamma += gamma / 2 * rsum
            Fzp_gamma += np.sum(delta_vzp * dvs)
            Fzp_gamma += np.sum(delta_bzp * dbs)
            Fzp_gamma += np.sum(delta_czp * dcs)
            Fzp_gamma += np.sum(delta_pzp * dps)

            Fzp_gamma+=sum(sum(delta_vzp * dvs))
            Fzp_gamma+=sum(sum(delta_bzp * dbs))
            Fzp_gamma+=sum(sum(delta_czp * dcs))
            Fzp_gamma+=sum(sum(delta_pzp * dps))

            Fzp_gamma += np.sum(delta_Azp * dalphas)
            Fzp_gamma += np.sum(delta_Mzp * dms)
            Fzp_gamma += np.sum(delta_Gd * dgammads)
            Fzp_gamma += np.sum(delta_Go * dgammaos)
            Fzp_gamma += np.sum(delta_Gh * dgammahs)

            Fzp_gamma += sum(sum(delta_Azp * dalphas))
            Fzp_gamma += sum(sum(delta_Mzp * dms))
            Fzp_gamma += sum(sum(delta_Gd * dgammads))
            Fzp_gamma += sum(sum(delta_Go * dgammaos))
            Fzp_gamma += sum(sum(delta_Gh * dgammahs))

            # print(f'r sum: {rsum}')
            if rsum < 0.001:
                bflag = 1
                break
            if Fzp < Fzp_gamma:
                # print('Fzp< Fzp_gamma')
                # print(Fzp,Fzp_gamma)
                break
            else:
                gamma = gamma * gamma_inc

        bz_old = bz
        cz_old = cz
        pz_old = pz
        vz_old = vz

        bz = bzp
        cz = czp
        vz = vzp
        pz = pzp
        Az_old = Az
        Mz_old = Mz
        Gamma_dz_old = Gamma_dz
        Gamma_oz_old = Gamma_oz
        Gamma_hz_old = Gamma_hz

        Az = Azp
        Mz = Mzp
        Gamma_dz = Gamma_dzp
        Gamma_oz = Gamma_ozp
        Gamma_hz = Gamma_hzp

        error_v = np.sqrt(np.sum(mask*(vzp - true_v)**2)/np.sum(mask))
        error_b = np.sqrt(np.sum(mask * (bzp - true_b) ** 2) / np.sum(mask))
        error_c = np.sqrt(np.sum(mask * (czp - true_c) ** 2) / np.sum(mask))
        error_p = np.sqrt(np.sum(mask * (pzp - true_p) ** 2) / np.sum(mask))

        error_A = np.sqrt(np.sum(mask*(Azp - true_A)**2)/np.sum(mask))
        error_M = np.sqrt(np.sum(mask*(Mzp - true_M)**2)/np.sum(mask))
        error_Gammad = np.sqrt(np.sum(mask*(Gamma_dzp - true_Gamma_d)**2)/np.sum(mask))
        error_Gammao = np.sqrt(np.sum(mask*(Gamma_ozp - true_Gamma_o)**2)/np.sum(mask))
        error_Gammah = np.sqrt(np.sum(mask*(Gamma_hzp - true_Gamma_h)**2)/np.sum(mask))



        print(f'train rmse:,'
              f'A: {error_A},\n'
              f'M:{error_M},\n'
              f'Gd:{error_Gammad},\n'
              f'Gh:{error_Gammah},\n'
              f'Go: {error_Gammao},\n'
              f' b: {error_b},\n'
              f'v: {error_v},\n'
              f'p: {error_p},\n'
              f'c: {error_c}')

        funcVal.append(Fzp)


        print(f'loss: {funcVal[-1]}')

        if term_cond == 1:
            if iter >= 3:
                if funcVal[-1]>np.max(funcVal[-3:-1]):
                    print('cond1')
                    break
        elif term_cond == 2:
            if iter >= 3:
                if abs((funcVal[-1] - funcVal[-2])/ funcVal[-2]) <= 10 ** -4 :
                    print('cond2')
                    break
        elif term_cond == 3:

            if iter >= itermax:
                print('cond3')
                break
        elif term_cond == 4:
            if funcVal[-1]<=true_loss:
                break
        iter = iter + 1
        t_old = t
        t = 0.5 * (1 + np.sqrt(1 + 4 * t ** 2))

    b = bzp
    v = vzp
    c = czp
    p = pzp
    A = Azp
    M = Mzp
    Gamma_d = Gamma_dzp
    Gamma_o = Gamma_ozp
    Gamma_h = Gamma_hzp

    train_error_A = np.sqrt(np.sum(mask * (Az - true_A) ** 2) / np.sum(mask))
    train_error_M = np.sqrt(np.sum(mask * (Mz - true_M) ** 2) / np.sum(mask))
    train_error_Gammad = np.sqrt(np.sum(mask * (Gamma_dz - true_Gamma_d) ** 2) / np.sum(mask))
    train_error_Gammao = np.sqrt(np.sum(mask * (Gamma_oz - true_Gamma_o) ** 2) / np.sum(mask))
    train_error_Gammah = np.sqrt(np.sum(mask * (Gamma_hz - true_Gamma_h) ** 2) / np.sum(mask))
    train_error_v = np.sqrt(np.sum(mask * (vz - true_v) ** 2) / np.sum(mask))
    train_error_b = np.sqrt(np.sum(mask * (bz - true_b) ** 2) / np.sum(mask))
    train_error_c = np.sqrt(np.sum(mask * (cz - true_c) ** 2) / np.sum(mask))
    train_error_p = np.sqrt(np.sum(mask * (pz - true_p) ** 2) / np.sum(mask))

    test_error_A = np.sqrt(np.sum((1-mask) * (Az - true_A) ** 2) / np.sum((1-mask)))
    test_error_M = np.sqrt(np.sum((1-mask) * (Mz - true_M) ** 2) / np.sum((1-mask)))
    test_error_Gammad = np.sqrt(np.sum((1-mask) * (Gamma_dz - true_Gamma_d) ** 2) / np.sum((1-mask)))
    test_error_Gammao = np.sqrt(np.sum((1-mask) * (Gamma_oz - true_Gamma_o) ** 2) / np.sum((1-mask)))
    test_error_Gammah = np.sqrt(np.sum((1-mask) * (Gamma_hz - true_Gamma_h) ** 2) / np.sum((1-mask)))
    test_error_v = np.sqrt(np.sum((1-mask) * (vz - true_v) ** 2) / np.sum(1-mask))
    test_error_b = np.sqrt(np.sum((1-mask) * (bz - true_b) ** 2) / np.sum((1-mask)))
    test_error_c = np.sqrt(np.sum((1-mask) * (cz - true_c) ** 2) / np.sum((1-mask)))
    test_error_p = np.sqrt(np.sum((1-mask) * (pz - true_p) ** 2) / np.sum((1-mask)))

    train_test_error = {'train_error':[train_error_A,train_error_M,train_error_Gammad,
                                       train_error_Gammah,train_error_Gammao,train_error_v,
                                       train_error_b,train_error_p,train_error_c],\
                        'test_error': [test_error_A,test_error_M,test_error_Gammad,
                                       test_error_Gammah,test_error_Gammao,
                                       test_error_v,test_error_b,test_error_p,test_error_c]}

    est = {'A':A, 'M': M,'Gammad':Gamma_d,"Gammao":Gamma_o,"Gammah":Gamma_h,
           'b':b,'v':v,'p':p,'c':c,
           'train_error':[train_error_A,train_error_M,train_error_Gammad,
                                       train_error_Gammah,train_error_Gammao,train_error_v,
                                       train_error_b,train_error_p,train_error_c],\
            'test_error': [test_error_A,test_error_M,test_error_Gammad,
                                       test_error_Gammah,test_error_Gammao,
                                       test_error_v,test_error_b,test_error_p,test_error_c]}
    return est,funcVal



