import numpy as np
import pandas as pd
import scipy.special

'''parameters:
mu, M * N array: location of reversed log-normal that models ddl. 
v, M * N array: scale of reversed log-normal that models ddl
m, M * N array: offset of reversed log-normal that models ddl
b, M * N array: base of the exponential function that models open time
gammas, M * N arrays: coefficients
X, K * M * N  array, time stamps
x, 1 * nx array: time stamps of a student-assignment pair
xk: the last time stamp

'''

def compute_grads(x,mask,alpha,beta,v,m,b,s,p,c,gammad, gammao, gammat,out_grad=True):

    def R_function(x):
        Rxnew = [0]
        for i in range(1,len(x)):
            Rprev = Rxnew[-1]
            Rcurr = (1+Rprev)*(np.exp(-beta*(x[i]-x[i-1])))
            Rxnew.append(Rcurr)
        return np.array(Rxnew)

    def mu_d(x,v,m,s):
        mud = 1/(np.sqrt(2*np.pi*v)*(m-x/s))*np.exp(-(np.log(m-x/s))**2/v)
        return np.nan_to_num(mud)

    def mu_o(x,b,s):
        muo = b**(x/s)
        return np.nan_to_num(muo)

    def mu_t(x,s,p,c):
        mut = np.sin(np.pi/24*(x/s + p)) + c
        return mut

    def Ustar_d(x,s,v,m):
        try:
            xkk = x[x<=m][-1]
            first = np.log(-(xkk - m*s)/s)/np.sqrt(v)
            second = np.log(m)/np.sqrt(v)
            Ud = -s/(2**1.5)*(scipy.special.erf(first) - scipy.special.erf(second))
        except: Ud = 0
        return Ud

    def Ustar_o(xK,s,b):
        Uo = s*(b**(xK/s) - 1)/np.log(b)
        return Uo

    def Ustar_t(xK,s,p,c):
        Ut = 1/np.pi*(-24*s*np.cos((np.pi*xK + np.pi*p*s)/(24*s)) + 24*s*np.cos(np.pi*p/24) + np.pi*c*xK)
        return Ut

    def Lamba_star(x,alpha,beta):
        Lamba = -alpha*np.sum(np.exp(-beta*(x[-1]-x))-1)
        return Lamba

    def intensity(x,gammad,gammao,gammat,mud,muo,mut,alpha,beta):
        base =  gammad*mud + gammao*muo + gammat*mut
        excitement = alpha*beta*R_function(x)
        _lambda = base + excitement
        return base,excitement

    def grad_par(grad_mu,gamma,grad_Ustar,_lambda):
        dpar = np.sum(np.nan_to_num(np.divide(1,_lambda)*gamma*grad_mu)) - gamma*grad_Ustar
        return dpar

    def grad_gamma(mu,_lambda,Ustar):
        dgamma = np.sum(np.divide(1,_lambda)*mu) - Ustar
        return dgamma

    def grad_mud(x,s,m,v):
        upper = s**2*np.exp(-np.log(m-x/s)**2/v)*(2*np.log(m-x/s)+v)
        lower = np.sqrt(2*np.pi)*v**1.5*(s*m-x)**2
        grad_m = np.nan_to_num(-np.divide(upper,lower))
        common = np.log(m-x/s)**2
        grad_v = s*(v - 2*common)*np.exp(-common/v)
        grad_v = np.nan_to_num(np.divide(grad_v, 2**1.5*np.sqrt(np.pi)*(x - m*s)*v**2.5))
        return grad_m,grad_v

    def grad_muo(x,s,b):
        grad_b = x*b**(x/s -1)/s
        return  grad_b

    def grad_mut(x,s,p,c):
        grad_p = np.pi*np.cos(np.pi*(p+x/s)/24)/24
        grad_c = 1
        return grad_p,grad_c

    def grad_Ustar_d(x,s,m,v,Ustar):
        try:
            xkk = x[x<= m][-1]
            common1 = np.log((s*m-xkk)/s)
            common2 = np.log(m)
            # grad_m = - s/2**1.5*(np.nan_to_num(2*s*np.exp(-common1**2/v)/ (np.sqrt(np.pi*v)*(s*m-xK))) - 2*np.exp(-common2**2/v)/(np.sqrt(np.pi*v)*m))
            # grad_v = - s*(common2*np.exp(-common2**2/v) - np.nan_to_num(common1*np.exp(-common1**2/v)))/(2**1.5*np.sqrt(np.pi)*v**1.5)
            grad_m = - s / 2 ** 1.5 * (2 * s * np.exp(-common1 ** 2 / v) / (np.sqrt(np.pi * v) * (s * m - xkk)) - 2 * np.exp(
                    -common2 ** 2 / v) / (np.sqrt(np.pi * v) * m))
            grad_v = - s * (common2 * np.exp(-common2 ** 2 / v) - common1 * np.exp(-common1 ** 2 / v)) / (
                        2 ** 1.5 * np.sqrt(np.pi) * v ** 1.5)
        except:
            grad_m = 0
            grad_v = 0
        return grad_m,grad_v

    def grad_Ustar_o(xK,s,b):
        grad_b = xK*b**(xK/s)*np.log(b) - s*b**(xK/s) + s
        grad_b = grad_b/(b*np.log(b)*np.log(b))
        return grad_b
        # return xK*b**(xK/s-1)/np.log(b) - s*(b**xK/s-1)/(b*np.log(b)**2)

    def grad_Ustar_t(xK,s,p,c):
        grad_p = s*(np.sin((np.pi*s*p+np.pi*xK)/24/s) - np.sin(np.pi*p/24))
        grad_c = xK
        return grad_p,grad_c

    def grad_Alpha(x,R_value,_lambda):
        grad_alpha = np.sum(np.divide(beta*R_value,_lambda)+np.exp(-beta*(x[-1]-x))-1)
        return grad_alpha


    def nnl(_lambda,Ustar_d,Ustar_o,Ustar_t,Lambda,gammad,gammao,gammat):
        ld = gammad*Ustar_d
        lo = gammao*Ustar_o
        lt = gammat*Ustar_t
        l = np.sum(np.log(_lambda)) - ld -lo - lt - Lambda

        return -l


    def grad_bb(xi,_lambda,b,s,gammaoi):

        db = np.sum(np.divide(1,_lambda)*xi*gammaoi*b**(xi/s - 1)/s)
        upper = xi[-1]*b**(xi[-1]/s)*np.log(b) - s*b**(xi[-1]/s) + s
        lower = b*np.log(b)**2
        db = db - np.divide(upper,lower)
        return db



    M, N, K= x.shape
    # xK = np.array([[inner[-1] for inner in x0] for x0 in x])
    Dalpha,Dm,Dv,Db,Dp,Dc,Dgammad,Dgammao, Dgammah = [],[],[],[],[],[],[],[],[]
    Fs = []

    #print('initial m', m)
    for i in range(M):
        #print('i',i)
        for j in range(N):
            #print(i,j)
            if mask[i,j]==1:
                #print('xij',x[i,j] )
                #print('m',m)
                #xi,mi,vi,bi,pi,ci,alphai  = x[i,j],m[i,j],v[i,j],b[i,j],p[i,j],c[i,j],alpha[i,j]


                xi = x[i,j]
                mi = m[i, j]
                vi = v[i, j]
                bi =   b[i, j]
                pi =   p[i, j]
                ci = c[i, j]
                alphai = alpha[i, j]

                #xKi = xi[0][-1]
                xKi = xi[0]
                gammadi,gammaoi,gammati = gammad[i,j],gammao[i,j],gammat[i,j]
                R_value = R_function(xi)

                #R_value = R_function(x)

                mud = mu_d(xi,vi,mi,s)
                muo = mu_o(xi,bi,s)
                mut = mu_t(xi,s,pi,ci)
                Ud = Ustar_d(xi,s,vi,mi)
                Uo = Ustar_o(xKi,s,bi)
                Ut = Ustar_t(xKi,s,pi,ci)
                Lambda = Lamba_star(xi,alphai,beta)
                base,excite = intensity(xi,gammadi,gammaoi,gammati,mud,muo,mut,alphai,beta)
                _lambda = base+excite
                fs = nnl(_lambda, Ud, Uo, Ut, Lambda, gammadi, gammaoi, gammati)
                Fs.append(fs)
                grad_mu_m, grad_mu_v = grad_mud(xi,s,mi,vi)
                grad_mu_b = grad_muo(xi,s,bi)
                grad_mu_p,grad_mu_c = grad_mut(xi,s,pi,ci)
                grad_U_m,grad_U_v = grad_Ustar_d(xi,s,mi,vi,Ud)
                grad_U_b = grad_Ustar_o(xKi,s,bi)
                grad_U_p,grad_U_c = grad_Ustar_t(xKi,s,pi,ci)
                grad_m = grad_par(grad_mu_m,gammadi,grad_U_m,_lambda)
                grad_v = grad_par(grad_mu_v, gammadi, grad_U_v, _lambda)
                grad_b = grad_par(grad_mu_b,gammaoi,grad_U_b,_lambda)
                db =grad_bb(xi,_lambda,bi,s,gammaoi)
                grad_p = grad_par(grad_mu_p,gammati,grad_U_p,_lambda)
                grad_c = grad_par(grad_mu_c,gammati,grad_U_c,_lambda)
                grad_gammad = grad_gamma(mud,_lambda,Ud)
                grad_gammao = grad_gamma(muo,_lambda,Uo)
                grad_gammat = grad_gamma(mut,_lambda,Ut)
                grad_alpha = grad_Alpha(xi,R_value,_lambda)
                Dalpha.append(grad_alpha)
                Dm.append(grad_m)
                Dv.append(grad_v)
                Db.append(grad_b)
                Dp.append(grad_p)
                Dc.append(grad_c)
                Dgammad.append(grad_gammad)
                Dgammao.append(grad_gammao)
                Dgammah.append(grad_gammat)
            else:
                Dalpha.append(0)
                Dm.append(0)
                Dv.append(0)
                Db.append(0)
                Dp.append(0)
                Dc.append(0)
                Dgammad.append(0)
                Dgammao.append(0)
                Dgammah.append(0)
                Fs.append(0)

    Dalpha= np.array(Dalpha).reshape(M,N)
    Dm = np.array(Dm).reshape(M,N)
    Dv = np.array(Dv).reshape(M,N)
    Db = np.array(Db).reshape(M, N)
    Dp = np.array(Dp).reshape(M, N)
    Dc = np.array(Dc).reshape(M, N)
    Dgammad = np.array(Dgammad).reshape(M, N)
    Dgammao = np.array(Dgammao).reshape(M, N)
    Dgammah = np.array(Dgammah).reshape(M, N)
    Fs = np.array(Fs).reshape(M,N)
    return Dalpha,Dm,Dv,Db,Dp,Dc,Dgammad,Dgammao, Dgammah, Fs/np.sum(mask), np.sum(Fs)/np.sum(mask)
