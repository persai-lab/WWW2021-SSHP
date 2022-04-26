from gradients_update import *
from trace_projection import *
from scipy.io import loadmat
import warnings
warnings.filterwarnings('ignore')
from scipy.stats import bernoulli
import scipy.io
def gen_mask(m, n, prob_masked=0.5):
    return 1 - bernoulli.rvs(p=prob_masked, size=(m, n))

def customized_Hawkes(data,init,mask,decay,rho, itermax,term_cond = 1,train_split = 0.7):
    x = data.get('X_train')
    M, N , K = x.shape
    b0 = init['b0']
    v0 = init['v0']
    c0 = init['c0']
    p0 = init['p0']
    M0 = init['M0']
    A0 = init['A0']
    Gamma_d0 = init['Gamma_d0']
    Gamma_o0 = init['Gamma_o0']
    Gamma_h0 = init['Gamma_h0']

    vz,vz_old = v0, v0
    bz, bz_old = b0, b0
    pz, pz_old = p0, p0
    cz, cz_old = c0, c0
    Az, Az_old = A0, A0
    Mz, Mz_old = M0, M0
    Gamma_dz, Gamma_dz_old = Gamma_d0, Gamma_d0
    Gamma_oz, Gamma_oz_old = Gamma_o0, Gamma_o0
    Gamma_hz, Gamma_hz_old = Gamma_h0, Gamma_h0

    funcVal = [compute_grads(x, mask, A0, decay, v0, M0, b0, 1, p0, c0, Gamma_d0,
                             Gamma_o0, Gamma_h0, out_grad=True)[-1]]

    t, t_old = 1,0

    iter = 0
    gamma = 250
    gamma_inc = 5
    while iter < itermax:
        print(iter)
        a = (t_old - 1) / t
        # print(a)
        vs = (1 + a) * vz - a * vz_old
        bs = (1 + a) * bz - a * bz_old
        ps = (1 + a) * pz - a * pz_old
        cs = (1 + a) * cz - a * cz_old

        vs[vs <= 0] = 0.1
        bs[bs >= 1] = 0.99
        bs[bs <= 0] = 0.1

        As = (1 + a) * Az - a * Az_old
        Ms = (1 + a) * Mz - a * Mz_old
        Gamma_ds = (1 + a) * Gamma_dz - a * Gamma_dz_old
        Gamma_os = (1 + a) * Gamma_oz - a * Gamma_oz_old
        Gamma_hs = (1 + a) * Gamma_hz - a * Gamma_hz_old

        As[As <= 0] = 0.001
        As[As >= 1] = 0.99
        Ms[Ms <= 0] = 0.001
        # Ms[Ms >= 100] = 100

        Gamma_ds[Gamma_ds < 0] = 0.01
        Gamma_os[Gamma_os < 0] = 0.01
        Gamma_hs[Gamma_hs < 0] = 0.01


        dalphas, dms, dvs, dbs, dps, dcs, dgammads, dgammaos, dgammahs, Fs_matrix, Fs \
            = compute_grads(x, mask, As, decay, vs, Ms, bs, 1, ps, cs, Gamma_ds,
                              Gamma_os, Gamma_hs, out_grad=True)
        inner_iter = 0
        while True:
            if inner_iter<100:
                # print(inner_iter)
                Azp, Azp_tn = trace_proj((As - dalphas / gamma/50).astype(float), 3, 0.1,mask)
                Azp[Azp <= 0] = 0.001

                Mzp, Mzp_tn = trace_proj((Ms - dms / gamma).astype(float), 3, 1,mask)
                Mzp[Mzp <= 0] = 0.001
                # Mzp[Mzp >= 100] = 100
                Gamma_dzp, Gamma_dzp_tn = trace_proj((Gamma_ds - dgammads / gamma / 20).astype(float), 3, 1,mask)
                Gamma_ozp, Gamma_ozp_tn = trace_proj((Gamma_os - dgammaos / gamma / 20).astype(float), 3, 1,mask)
                Gamma_hzp, Gamma_hzp_tn = trace_proj((Gamma_hs - dgammahs / gamma / 20).astype(float), 3, 1,mask)

                pzp, _ = trace_proj((ps - dps /gamma).astype(float),1,1,mask)
                bzp, _ = trace_proj((bs - dbs / gamma).astype(float),1,1,mask)
                czp, _ = trace_proj((cs - dcs / gamma/20).astype(float), 1, 1,mask)
                vzp, _ = trace_proj((vs - dvs / gamma).astype(float), 1,1,mask)
                bzp = np.array([np.mean(bzp,axis = 1),]*N).T
                czp= np.array([np.mean(czp,axis = 1),]*N).T
                vzp= np.array([np.mean(vzp,axis = 1),]*N).T

                vzp[vzp <= 0] = 0.1
                bzp[bzp >= 1] = 0.99
                bzp[bzp <= 0] = 0.1
                czp[czp <= 1] = 1.001


                Fzp = compute_grads(x, mask, Azp, decay, vzp, Mzp, bzp, 1, pzp, czp, Gamma_dzp,
                                 Gamma_ozp, Gamma_hzp,out_grad=True)[-1]
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
                inner_iter+=1
                # print(f'r sum: {rsum}')
                if rsum < 0.001:
                    bflag = 1
                    break
                if Fzp < Fzp_gamma:
                    # print('Fzp< Fzp_gamma')
                    # print(Fzp,Fzp_gamma)
                    break
                elif gamma<10**5:
                    gamma = gamma * gamma_inc
            else:
                break

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

    est = {'A':A, 'M': M,'Gammad':Gamma_d,"Gammao":Gamma_o,"Gammah":Gamma_h,
           'b':b,'v':v,'p':p,'c':c}
    return est,funcVal

