import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal
from numpy.random import random
from scipy.stats import truncnorm
from scipy.stats import gamma
import matplotlib.pyplot as plt
import timeit

start = timeit.default_timer()



### DGP
N = 1000
T=10
NT = N * T
true_alpha = np.array([-2,4])
true_beta = np.array([0.5,0.3])
true_sigma = 1.
true_sigma_u =1.
epi = np.random.normal(0,1,N)
u = np.random.normal(0,true_sigma_u**2,T)
v = np.random.normal(0,true_sigma**2,NT)

X = np.concatenate([np.ones([NT,1]),np.random.uniform(0,1,[NT,1])],axis=1)
Z = np.concatenate([np.ones([N,1]),np.random.uniform(0,1,[N,1])],axis=1)

d_star = np.dot(Z, true_alpha) + epi
d_star = d_star >=0
d_star = d_star.astype(int)

y = np.dot(X, true_beta) + v
y[np.logical_or(np.kron(d_star, np.ones(T,)) == 0,y < 0)] = 0.
y = y.reshape(N,T)
X = X.reshape(N,T,2)

def simple_resample(particles, weights):
    N = len(particles)
    cumulative_sum = np.cumsum(weights)
    cumulative_sum[-1] = 1. # avoid round-off error
    indexes = np.searchsorted(cumulative_sum, random(N))

    # resample according to indexes
    particles[:] = particles[indexes]
    weights.fill(1.0 / N)
    return particles, weights

def RWMH(theta, i, K1, K2, K3, K4, Z, X, y):

    #step size
    step = 0.5
    # draw theta*
    cov = np.cov(theta[:,:K1].T)
    sig_factor = np.linalg.cholesky(cov).T
    theta_star_alpha = theta[:,:K1] + np.dot(np.random.normal(0,1,theta[:,:K1].shape), sig_factor * step)

    cov = np.cov(theta[:,K1:K1+K2].T)
    sig_factor = np.linalg.cholesky(cov).T
    theta_star_beta = theta[:,K1:K1+K2] + np.dot(np.random.normal(0,1,theta[:,K1:K1+K2].shape), sig_factor * step)

    theta_star_sig = 1/theta[:,K1+K2+K3-1] + np.random.normal(0,1,theta[:,K1+K2+K3-1].shape) * (1/theta[:,K1+K2+K3-1]).std() * 0.2
    while np.any(theta_star_sig < 0):
        iind = theta_star_sig < 0
        theta_star_sig[iind] = 1/theta[iind,K1+K2+K3-1] + np.random.normal(0,1,theta[iind,K1+K2+K3-1].shape) * (1/theta[:,K1+K2+K3-1]).std() * 0.2

    theta_star_sig_u = 1/theta[:,K1+K2+K3+K4-1] + np.random.normal(0,1,theta[:,K1+K2+K3+K4-1].shape) * (1/theta[:,K1+K2+K3+K4-1]).std() * 0.2
    while np.any(theta_star_sig < 0):
        iind = theta_star_sig_u < 0
        theta_star_sig_u[iind] = 1/theta[iind,K1+K2+K3+K4-1] + np.random.normal(0,1,theta[iind,K1+K2+K3+K4-1].shape) * (1/theta[:,K1+K2+K3+K4-1]).std() * 0.2
        
    theta_star = np.concatenate([theta_star_alpha, theta_star_beta, theta_star_sig.reshape(-1,1), theta_star_sig_u.reshape(-1,1)], axis=1)
    #evaluate log dist
    logp = cal_logp(theta, Z, X, y, i, K1, K2, K3, K4)
    logp_star = cal_logp(theta_star, Z, X, y, i, K1, K2, K3, K4)
    logu = np.log(np.random.uniform(0,1,[theta.shape[0],]))
    log_RW = logp_star - logp
    print("nan_number%f" %np.isnan(log_RW).sum())
    select = logu < log_RW
    theta[select,:] = theta_star[select,:] #change value if over u
    return theta

def cal_logp(theta, Z, X, y, N, K1, K2, K3, K4):
    logp = np.zeros([theta.shape[0],])
    T = X.shape[1]
    alpha = theta[:,:K1]
    beta = theta[:,K1:K1+K2]
    sig = theta[:,K1+K2+K3-1]**0.5
    sig_u = theta[:,K1+K2+K3+K4-1]**0.5
    #gpu parallel computation
    for i in range(N+1):
        ind1 = np.where(y[i,:]>0)[0]
        ind0 = np.where(y[i,:]==0)[0]
        u = np.random.normal(0,1,100)
        tot_s_pdf = []
        for s in range(100):
            pdf = np.ones([theta.shape[0],])
            if (y[i,:] == 0).all():
                for t in ind0:
                    pdf_ = 1 - norm.cdf(((X[i,t,:]*beta).sum(axis=1) + u[s] * sig_u)/sig)
                    pdf *= pdf_
                pdf = norm.cdf((Z[i,:]*alpha).sum(axis=1)) * pdf
                pdf = 1 - norm.cdf((Z[i,:]*alpha).sum(axis=1)) + pdf
            else:
                ### simulation maximization likelihood
                for t in range(T):
                    if y[i,t] == 0:
                        pdf_ = 1 - norm.cdf(((X[i,t,:]*beta).sum(axis=1) + u[s] * sig_u)/sig)
                    else:
                        pdf_ = norm.pdf((y[i,t]-(X[i,t,:]*beta).sum(axis=1)-u[s]*sig_u)/sig)/sig
                    pdf *= pdf_
                pdf = norm.cdf((Z[i,:]*alpha).sum(axis=1)) * pdf
            tot_s_pdf.append(pdf)
        sim_pdf = np.vstack(tot_s_pdf).mean(axis=0)
        logp += np.log(sim_pdf)
    return logp

def reweight(theta, w, y, i, X, Z, K1, K2, K3, K4):
    alpha = theta[:,:K1]
    beta = theta[:,K1:K1+K2]
    sig = theta[:,K1+K2+K3-1]**0.5
    sig_u = theta[:,K1+K2+K3+K4-1]**0.5
    T = X.shape[1]
    
    ind1 = np.where(y[i,:]>0)[0]
    ind0 = np.where(y[i,:]==0)[0]
    u = np.random.normal(0,1,100)
    tot_s_pdf = []
    for s in range(100):
        pdf = np.ones([theta.shape[0],])
        if (y[i,:] == 0).all():
            for t in ind0:
                pdf_ = 1 - norm.cdf(((X[i,t,:]*beta).sum(axis=1) + u[s] * sig_u)/sig)
                pdf *= pdf_
            pdf = norm.cdf((Z[i,:]*alpha).sum(axis=1)) * pdf
            pdf = 1 - norm.cdf((Z[i,:]*alpha).sum(axis=1)) + pdf
            tot_s_pdf.append(pdf)
        else:
            ### simulation maximization likelihood
            for t in range(T):
                if y[i,t] == 0:
                    pdf_ = 1 - norm.cdf(((X[i,t,:]*beta).sum(axis=1) + u[s] * sig_u)/sig)
                else:
                    pdf_ = norm.pdf((y[i,t]-(X[i,t,:]*beta).sum(axis=1)-u[s]*sig_u)/sig)/sig
                pdf *= pdf_
            pdf = norm.cdf((Z[i,:]*alpha).sum(axis=1)) * pdf
            tot_s_pdf.append(pdf)
    sim_pdf = np.vstack(tot_s_pdf).mean(axis=0)

    w = sim_pdf.reshape(-1,1) * w # unnormalized weight
    w = w / w.sum(axis=0) # normalized
    return w


H=10000
resample_type = 'multinomial'
K1 = 2
K2 = 2
K3 = 1
K4 = 1

### initialization
theta = np.concatenate([np.random.normal(0,5,[H,K1]), np.random.normal(0,5,[H,K2]),np.random.gamma(1,1,[H,K3]), np.random.gamma(1,1,[H,K4])], axis=1)
w = np.concatenate([norm.pdf(theta[:,:K1],0,5), norm.pdf(theta[:,K1:K1+K2],0,5), gamma.pdf(theta[:,K1+K2+K3-1],1).reshape(-1,1), gamma.pdf(theta[:,K1+K2+K3+K4-1],1).reshape(-1,1)],axis=1)
w = w / w.sum(axis=0) # ini_weights

### run sequential monte carlo
all_mu_theta = []
all_std_theta = []

for i in range(N):
    print(i)
    ### run sequential monte carlo
    ###reweight
    #parallel computing
    w = reweight(theta, w, y, i, X, Z, K1, K2, K3, K4)
    ### resampling-np.random.choice
    all_theta_ = []
    all_w_ = []
    for k in range(K1+K2+K3+K4): #cal k theta separately or together?
        new_theta, new_w = simple_resample(theta[:,k], w[:,k])
        all_theta_.append(new_theta)
        all_w_.append(new_w)
    theta = np.vstack(all_theta_).T
    w = np.vstack(all_w_).T

    ### move
    theta = RWMH(theta, i, K1, K2, K3, K4, Z, X, y)
    # acceptance rate?
    # Chopin (2002) proposal distribution...
#     theta = IndMH(theta, x, y, i)
#     theta = RWMH(theta, x, y, i)
    g_mu = theta.mean(axis=0)
    g_std = theta.std(axis=0)
    all_std_theta.append(g_std)
    all_mu_theta.append(g_mu)
    
#Your statements here

stop = timeit.default_timer()

print('Time: ', stop - start)  
pd.DataFrame(theta).to_csv('two_tier_tobit_theta')
pd.DataFrame(all_mu_theta).to_csv('two_tier_tobit_all_mu_theta')
pd.DataFrame(all_std_theta).to_csv('two_tier_tobit_all_std_theta')
