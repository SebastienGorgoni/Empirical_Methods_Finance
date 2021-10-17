import datetime as dt
from fitter import Fitter # pip install fitter
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
from numpy.random import randn
import os
import pandas as pd
import pandas_datareader.data as web
from numpy import size, log, pi, sum, array, zeros, diag, mat, asarray, sqrt, copy
from numpy.linalg import inv
from scipy.optimize import fmin_slsqp, minimize
from scipy.stats.distributions import chi2
import scipy.stats as sc
from scipy.stats import norm, gennorm, norminvgauss, laplace, johnsonsu, dgamma
import seaborn as sns


if not os.path.isdir('Output'):
    os.makedirs('Output')

if not os.path.isdir('Plot'):
    os.makedirs('Plot')

sns.set_theme(style='whitegrid')


"""DATA FOR PART 1"""
start = dt.datetime(2011,1,1)
end = dt.datetime(2021,5,1)
# S&P500 (^GSPC)
sp500 = web.DataReader('^GSPC', 'yahoo', start=start, end=end)
sp500_logreturns =  (log(sp500['Adj Close']) - log(sp500['Adj Close'].shift(1))).dropna()
# BCOM Commodity (^BCOM)
bcom = web.DataReader('^BCOM','yahoo',start=start,end=end)
bcom_logreturns =  (log(bcom['Adj Close']) - log(bcom['Adj Close'].shift(1))).dropna()
# Apple Inc. (AAPL)
apple = web.DataReader('AAPL','yahoo',start=start,end=end)
apple_logreturns =  (log(apple['Adj Close']) - log(apple['Adj Close'].shift(1))).dropna()



#################
######  PART 1
#################

"""GARCH"""
# out: to decide if we only want the loglik as output or more information
def garch_likelihood(parameters, data, sigma2, out=None):
    ''' Returns negative log-likelihood for GARCH(1,1) model.'''
    mu = parameters[0]
    omega = parameters[1]
    alpha = parameters[2]
    beta = parameters[3]
    
    T = size(data,0)
    eps = data - mu # residuals
    sigma2[0] = data.var()
    # Data and sigma2 are T by 1 vectors
    for t in range(1,T):
        # need to compute sigmas recursively
        sigma2[t] = (omega + alpha * eps[t-1]**2 
                     + beta * sigma2[t-1])
    
    logliks = 0.5*(log(2*pi) + log(sigma2) + eps**2/sigma2) # the negative log-likelihood
    loglik = sum(logliks)
    
    global vol_hist # store the volatilities
    vol_hist = sqrt(sigma2)
    
    if out is None:
        return loglik
    else:
        return loglik, logliks, copy(sigma2)
    
def garch_constraint(parameters, data, sigma2, out=None):
    ''' Constraint that alpha+beta<1'''
    alpha = parameters[2]
    beta= parameters[3]
    return array([1-alpha-beta])


def garch_estimates(logreturns):
    startingVals = array([logreturns.mean()*100, logreturns.var() * .01, .03, .90])

    finfo = np.finfo(np.float64)
    # 4 bounds ranges, for mu, omega, alpha, beta
    if logreturns.mean() < 0:
        bounds = [(10*logreturns.mean(), -10*logreturns.mean()),
                  (finfo.eps, 2*logreturns.var() ),
                  (0.0,1.0), (0.0,1.0)]
    else:
        bounds = [(-10*logreturns.mean(), 10*logreturns.mean()),
                  (finfo.eps, 2*logreturns.var() ),
                  (0.0,1.0), (0.0,1.0)]
    
    T = logreturns.shape[0]
    sigma2 = np.ones(T) * logreturns.var() # starting value = long-run variance of the series
    # Pass a NumPy array, not a pandas Series
    args = (np.asarray(logreturns*100), sigma2) # returns*100 (better working with percentage units for convergence)

    # f_ieqcons : function as a constraint (SLSQP : sequential least squares programming)
    estimates = fmin_slsqp(garch_likelihood, startingVals,
                            f_ieqcons=garch_constraint, bounds = bounds,
                            args = args)
    return estimates

# estimates for each series
sp500_garch_estimates = garch_estimates(sp500_logreturns)
sigma2_garch_sp500 = vol_hist
bcom_garch_estimates = garch_estimates(bcom_logreturns)
sigma2_garch_bcom = vol_hist
apple_garch_estimates = garch_estimates(apple_logreturns)
sigma2_garch_apple = vol_hist


### plots

# S&P500
plt.figure()
plt.plot(sp500_logreturns.index, sqrt(sigma2_garch_sp500)*sqrt(252))
plt.title('S&P500 Annualized Volatility - GARCH(1,1)')
# BCOM
plt.figure()
plt.plot(bcom_logreturns.index, sqrt(sigma2_garch_bcom)*sqrt(252))
plt.title('BCOM  Annualized Volatility - GARCH(1,1)')
# Apple
plt.figure()
plt.plot(apple_logreturns.index, sqrt(sigma2_garch_apple)*sqrt(252))
plt.title('Apple Annualized Volatility - GARCH(1,1)')




# Numerical approximation of the score function and final estimation results display
# f'(x) = lim_{h->0} (f(x+h)-f(x))/h
# in practice: (f(x+h/2)-f(x-h/2))/h approximates the derivative of the function for small h
# Fisher information matrix: I = E[gg'], where g is the gradient (BHHH)
def garch_significance(logreturns, garch_estimates):
    T = len(logreturns)
    nb_estimates = len(garch_estimates)
    step = 1e-5 * garch_estimates
    scores = zeros((T,nb_estimates))
    sigma2 = np.ones(T) * logreturns.var()
    for i in range(nb_estimates):
        h = step[i]
        delta = np.zeros(nb_estimates)
        delta[i] = h
        
        loglik, logliksplus, sigma2 = garch_likelihood(garch_estimates + delta, 
                                                       np.asarray(logreturns), 
                                                       sigma2, 
                                                       out=True)
        loglik, logliksminus, sigma2 = garch_likelihood(garch_estimates - delta,
                                                        np.asarray(logreturns), 
                                                        sigma2, 
                                                        out=True)
        scores[:,i] = (logliksplus - logliksminus)/(2*h)
    
    I = (scores.T @ scores)/T    
    vcv = mat(inv(I))/T # variance covariance matrix
    vcv = asarray(vcv)
   
    output = np.vstack((garch_estimates,sqrt(diag(vcv)),garch_estimates/sqrt(diag(vcv)))).T
    
    significance_summary = pd.DataFrame(data=output, 
                                        index=['mu','omega','alpha','beta'],
                                        columns=['Parameter','Std. Err.','T-stat'])
   
    return significance_summary

garch_signif_sp500 = garch_significance(sp500_logreturns, sp500_garch_estimates)
garch_signif_bcom = garch_significance(bcom_logreturns, bcom_garch_estimates)
garch_signif_apple = garch_significance(apple_logreturns, apple_garch_estimates)







"""EGARCH"""
def egarch_likelihood(parameters, data, sigma2, out=None):
    ''' Returns negative log-likelihood for EGARCH(1,1) model.'''
    mu = parameters[0]
    omega = parameters[1]
    alpha = parameters[2]
    gamma = parameters[3]
    beta = parameters[4]
    
    T = size(data,0)
    eps = data - mu
    z=eps
    sigma2[0] = data.var()
    # Data and sigma2 are T by 1 vectors
    for t in range(1,T):
        z[t-1]=(data[t-1] - mu)/(sigma2[t-1]**(0.5))
        temp=(omega + alpha * (abs(eps[t-1]) - (2*(pi)**.5)) 
              + gamma * eps[t-1] + beta * log(sigma2[t-1]))
        sigma2[t] = np.exp(temp)
    
    logliks = 0.5*(log(2*pi) + log(sigma2) + eps**2/sigma2)
    loglik = sum(logliks)
    
    global vol_hist
    vol_hist=sqrt(sigma2)
    
    if out is None:
        return loglik
    else:
        return loglik, logliks, copy(sigma2)
    
    
def egarch_constraint(parameters, data, sigma2, out=None):
    ''' Constraint that 1-beta>0'''
    beta = parameters[4]
    return array([1-beta])


def egarch_estimates(logreturns):
    startingVals = array([logreturns.mean(), 0, .01, 0, .70])
   
    if logreturns.mean() < 0:
        bounds = [(10*logreturns.mean(), -10*logreturns.mean()),
                  (-10, 10),
                  (0.0,1.0), (-1,1.0), (0.0,1.0)]
    else:
        bounds = [(-10*logreturns.mean(), 10*logreturns.mean()),
                  (-10, 10),
                  (0.0,1.0), (-1,1.0), (0.0,1.0)]
      
    T = logreturns.shape[0]
    sigma2 = np.ones(T) * logreturns.var()
    # Pass a NumPy array, not a pandas Series
    args = (np.asarray(logreturns*100), sigma2)
    estimates = fmin_slsqp(egarch_likelihood, startingVals,
                           f_ieqcons=egarch_constraint, bounds = bounds,
                           args = args)
    return estimates

# estimates for each series
sp500_egarch_estimates = egarch_estimates(sp500_logreturns)
sigma2_egarch_sp500 = vol_hist
bcom_egarch_estimates = egarch_estimates(bcom_logreturns)
sigma2_egarch_bcom = vol_hist
apple_egarch_estimates = egarch_estimates(apple_logreturns)
sigma2_egarch_apple = vol_hist

### plots

# S&P500
plt.figure()
plt.plot(sp500_logreturns.index, sqrt(sigma2_egarch_sp500)*sqrt(252))
plt.title('S&P500 Annualized Volatility - EGARCH(1,1)')
# BCOM
plt.figure()
plt.plot(bcom_logreturns.index, sqrt(sigma2_egarch_bcom)*sqrt(252))
plt.title('BCOM  Annualized Volatility - EGARCH(1,1)')
# Apple
plt.figure()
plt.plot(apple_logreturns.index, sqrt(sigma2_egarch_apple)*sqrt(252))
plt.title('Apple Annualized Volatility - EGARCH(1,1)')

# Numerical approximation of the score function and final estimation results display
# f'(x) = lim_{h->0} (f(x+h)-f(x))/h
# in practice: (f(x+h/2)-f(x-h/2))/h approximates the derivative of the function for small h
# Fisher information matrix: I = E[gg'], where g is the gradient (BHHH)
def egarch_significance(logreturns, egarch_estimates):
    T = len(logreturns)
    nb_estimates = len(egarch_estimates)
    step = 1e-5 * egarch_estimates
    scores = zeros((T,nb_estimates))
    sigma2 = np.ones(T) * logreturns.var()
    for i in range(nb_estimates):
        h = step[i]
        delta = np.zeros(nb_estimates)
        delta[i] = h
        
        loglik, logliksplus, sigma2 = egarch_likelihood(egarch_estimates + delta, 
                                                       np.asarray(logreturns), 
                                                       sigma2, 
                                                       out=True)
        loglik, logliksminus, sigma2 = egarch_likelihood(egarch_estimates - delta,
                                                        np.asarray(logreturns), 
                                                        sigma2, 
                                                        out=True)
        scores[:,i] = (logliksplus - logliksminus)/(2*h)
    
    I = (scores.T @ scores)/T    
    vcv = mat(inv(I))/T # variance covariance matrix
    vcv = asarray(vcv)
   
    output = np.vstack((egarch_estimates,sqrt(diag(vcv)),egarch_estimates/sqrt(diag(vcv)))).T
    
    significance_summary = pd.DataFrame(data=output, 
                                        index=['mu','omega','alpha','gamma','beta'],
                                        columns=['Parameter','Std. Err.','T-stat'])
   
    return significance_summary

egarch_signif_sp500 = egarch_significance(sp500_logreturns, sp500_egarch_estimates)
egarch_signif_bcom = egarch_significance(bcom_logreturns, bcom_egarch_estimates)
egarch_signif_apple = egarch_significance(apple_logreturns, apple_egarch_estimates)







"""GARCH-GJR"""
def gjr_garch_likelihood(parameters, data, sigma2, out=None):
    ''' Returns negative log-likelihood for GJR-GARCH(1,1) model.'''
    mu = parameters[0]
    omega = parameters[1]
    alpha = parameters[2]
    gamma = parameters[3]
    beta = parameters[4]
    
    
    T = size(data,0)
    eps = data - mu
    sigma2[0] = data.var()
    # Data and sigma2 are T by 1 vectors
    for t in range(1,T):
        sigma2[t] = (omega + alpha * eps[t-1]**2 
                     + gamma * eps[t-1]**2 * (eps[t-1]<0) + beta * sigma2[t-1])
    
    logliks = 0.5*(log(2*pi) + log(sigma2) + eps**2/sigma2)
    loglik = sum(logliks)
    
    global vol_hist
    vol_hist = sqrt(sigma2)
    
    if out is None:
        return loglik
    else:
        return loglik, logliks, copy(sigma2)
    
    
def gjr_constraint(parameters, data, sigma2, out=None):
    ''' Constraint that alpha+gamma/2+beta<1'''
    alpha = parameters[2]
    gamma = parameters[3]
    beta = parameters[4]
    return array([1-alpha-gamma/2-beta])


def gjr_estimates(logreturns):
    startingVals = array([logreturns.mean(), logreturns.var() * .01, .03, .09, .90])
    finfo = np.finfo(np.float64)
    if logreturns.mean() < 0:
        bounds = [(10*logreturns.mean(), -10*logreturns.mean()),
                  (finfo.eps, 2*logreturns.var() ),
                  (0.0,1.0), (0.0,1.0), (0.0,1.0)]
    else:
        bounds = [(-10*logreturns.mean(), 10*logreturns.mean()),
                  (finfo.eps, 2*logreturns.var() ),
                  (0.0,1.0), (0.0,1.0), (0.0,1.0)]
    
       
    T = logreturns.shape[0]
    sigma2 = np.ones(T) * logreturns.var()
    # Pass a NumPy array, not a pandas Series
    args = (np.asarray(logreturns*100), sigma2)
    estimates = fmin_slsqp(gjr_garch_likelihood, startingVals,
                       f_ieqcons=gjr_constraint, bounds = bounds,
                       args = args)
    return estimates

# estimates for each series
sp500_gjr_estimates = gjr_estimates(sp500_logreturns)
sigma2_gjr_sp500 = vol_hist
bcom_gjr_estimates = gjr_estimates(bcom_logreturns)
sigma2_gjr_bcom = vol_hist
apple_gjr_estimates = gjr_estimates(apple_logreturns)
sigma2_gjr_apple = vol_hist

### plots

# S&P500
plt.figure()
plt.plot(sp500_logreturns.index, sqrt(sigma2_gjr_sp500)*sqrt(252))
plt.title('S&P500 Annualized Volatility - GJR(1,1)')
# BCOM
plt.figure()
plt.plot(bcom_logreturns.index, sqrt(sigma2_gjr_bcom)*sqrt(252))
plt.title('BCOM  Annualized Volatility - GJR(1,1)')
# Apple
plt.figure()
plt.plot(apple_logreturns.index, sqrt(sigma2_gjr_apple)*sqrt(252))
plt.title('Apple Annualized Volatility - GJR(1,1)')


# Numerical approximation of the score function and final estimation results display
# f'(x) = lim_{h->0} (f(x+h)-f(x))/h
# in practice: (f(x+h/2)-f(x-h/2))/h approximates the derivative of the function for small h
# Fisher information matrix: I = E[gg'], where g is the gradient (BHHH)
def gjr_significance(logreturns, gjr_estimates):
    T = len(logreturns)
    nb_estimates = len(gjr_estimates)
    step = 1e-5 * gjr_estimates
    scores = zeros((T,nb_estimates))
    sigma2 = np.ones(T) * logreturns.var()
    for i in range(nb_estimates):
        h = step[i]
        delta = np.zeros(nb_estimates)
        delta[i] = h
        
        loglik, logliksplus, sigma2 = gjr_garch_likelihood(gjr_estimates + delta, 
                                                       np.asarray(logreturns), 
                                                       sigma2, 
                                                       out=True)
        loglik, logliksminus, sigma2 = gjr_garch_likelihood(gjr_estimates - delta,
                                                        np.asarray(logreturns), 
                                                        sigma2, 
                                                        out=True)
        scores[:,i] = (logliksplus - logliksminus)/(2*h)
    
    I = (scores.T @ scores)/T    
    vcv = mat(inv(I))/T # variance covariance matrix
    vcv = asarray(vcv)
   
    output = np.vstack((gjr_estimates,sqrt(diag(vcv)),gjr_estimates/sqrt(diag(vcv)))).T
    
    significance_summary = pd.DataFrame(data=output, 
                                        index=['mu','omega','alpha','gamma','beta'],
                                        columns=['Parameter','Std. Err.','T-stat'])
   
    return significance_summary

gjr_signif_sp500 = gjr_significance(sp500_logreturns, sp500_gjr_estimates)
gjr_signif_bcom = gjr_significance(bcom_logreturns, bcom_gjr_estimates)
gjr_signif_apple = gjr_significance(apple_logreturns, apple_gjr_estimates)




###
#   Q1.2
###

"""
LR test : 
    -> lr_stat = -2(loglik_reduced_model - loglik_full_model)
    -> lr_stat ~ chi_square(k) under the null
    -> k = number of coefficients that are set to zero in the full model to obtain the reduced model
"""
logliks_sp500_garch = -garch_likelihood(sp500_garch_estimates, sp500_logreturns, sigma2_garch_sp500, out=True)[1]
logliks_sp500_egarch = -egarch_likelihood(sp500_egarch_estimates, sp500_logreturns, sigma2_egarch_sp500, out=True)[1]
logliks_sp500_gjr = -gjr_garch_likelihood(sp500_gjr_estimates, sp500_logreturns, sigma2_gjr_sp500, out=True)[1]

logliks_bcom_garch = -garch_likelihood(bcom_garch_estimates, bcom_logreturns, sigma2_garch_bcom, out=True)[1]
logliks_bcom_egarch = -egarch_likelihood(bcom_egarch_estimates, bcom_logreturns, sigma2_egarch_bcom, out=True)[1]
logliks_bcom_gjr = -gjr_garch_likelihood(bcom_gjr_estimates, bcom_logreturns, sigma2_gjr_bcom, out=True)[1]

logliks_apple_garch = -garch_likelihood(apple_garch_estimates, apple_logreturns, sigma2_garch_apple, out=True)[1]
logliks_apple_egarch = -egarch_likelihood(apple_egarch_estimates, apple_logreturns, sigma2_egarch_apple, out=True)[1]
logliks_apple_gjr = -gjr_garch_likelihood(apple_gjr_estimates, apple_logreturns, sigma2_gjr_apple, out=True)[1]

def vuong_test(logliks1, logliks2):
    """Vuong likelihood ratio test."""
    # compute the components of the vuong statistic
    pointwise_log_likelihood = logliks1 - logliks2
    loglikelihood_ratio = pointwise_log_likelihood.sum()
    sigma = pointwise_log_likelihood.std()
    n = pointwise_log_likelihood.size

    # compute the vuong statistic and p-values
    vuong_statistic = loglikelihood_ratio / (n**0.5 * sigma)
    p1_sided = sc.norm.cdf(vuong_statistic)
    p2_sided = 2 * p1_sided if p1_sided < 0.5 else 2 * (1 - p1_sided)

    return vuong_statistic, p1_sided, p2_sided

vuong_sp500_garch_egarch = vuong_test(logliks_sp500_garch, logliks_sp500_egarch)
vuong_sp500_egarch_gjr = vuong_test(logliks_sp500_egarch, logliks_sp500_gjr)

vuong_bcom_garch_egarch = vuong_test(logliks_bcom_garch, logliks_bcom_egarch)
vuong_bcom_egarch_gjr = vuong_test(logliks_bcom_egarch, logliks_bcom_gjr)

vuong_apple_garch_egarch = vuong_test(logliks_apple_garch, logliks_apple_egarch)
vuong_apple_egarch_gjr = vuong_test(logliks_apple_egarch, logliks_apple_gjr)


def lr_test(loglik_reduced, loglik_full):
    lr_stat = -2*(loglik_reduced - loglik_full)
    return chi2.sf(lr_stat, 1)


# ### S&P500
loglik_sp500_garch = -garch_likelihood(sp500_garch_estimates, sp500_logreturns, sigma2_garch_sp500, out=None)
loglik_sp500_egarch = -egarch_likelihood(sp500_egarch_estimates, sp500_logreturns, sigma2_egarch_sp500, out=None)
loglik_sp500_gjr = -gjr_garch_likelihood(sp500_gjr_estimates, sp500_logreturns, sigma2_gjr_sp500, out=None)

# # LR test
#     # full: GJR, reduced: GARCH -> k = 1
lr_sp500_gjr_garch_pval = lr_test(loglik_sp500_garch,loglik_sp500_gjr)



# ### BCOM
loglik_bcom_garch = -garch_likelihood(bcom_garch_estimates, bcom_logreturns, sigma2_garch_bcom, out=None)
loglik_bcom_egarch = -garch_likelihood(bcom_egarch_estimates, bcom_logreturns, sigma2_egarch_bcom, out=None)
loglik_bcom_gjr = -gjr_garch_likelihood(bcom_gjr_estimates, bcom_logreturns, sigma2_gjr_bcom, out=None)

# # LR test
#     # full: GJR, reduced: GARCH -> k = 1
lr_bcom_gjr_garch_pval = lr_test(loglik_bcom_garch,loglik_bcom_gjr)



# ### Apple
loglik_apple_garch = -garch_likelihood(apple_garch_estimates, apple_logreturns, sigma2_garch_apple, out=None)
loglik_apple_egarch = -egarch_likelihood(apple_egarch_estimates, apple_logreturns, sigma2_egarch_apple, out=None)
loglik_apple_gjr = -gjr_garch_likelihood(apple_gjr_estimates, apple_logreturns, sigma2_gjr_apple, out=None)

# # LR test
#     # full: GJR, reduced: GARCH -> k = 1
lr_apple_gjr_garch_pval = lr_test(loglik_apple_garch,loglik_apple_gjr)


aic_snp500_garch = 2*len(sp500_garch_estimates) - 2*loglik_sp500_garch
aic_snp500_egarch = 2*len(sp500_egarch_estimates) - 2*loglik_sp500_egarch
aic_snp500_gjr = 2*len(sp500_gjr_estimates) - 2*loglik_sp500_gjr


aic_bcom_garch = 2*len(bcom_garch_estimates) - 2*loglik_bcom_garch
aic_bcom_egarch = 2*len(bcom_egarch_estimates) - 2*loglik_bcom_egarch
aic_bcom_gjr = 2*len(bcom_gjr_estimates) - 2*loglik_bcom_gjr

aic_apple_garch = 2*len(apple_garch_estimates) - 2*loglik_apple_garch
aic_apple_egarch = 2*len(apple_egarch_estimates) - 2*loglik_apple_egarch
aic_apple_gjr = 2*len(apple_gjr_estimates) - 2*loglik_apple_gjr

"""
S&P500 -> go for GARCH
BCOM -> go for GARCH
APPLE -> go for GARCH

"""

###
#   Q1.3
###

# Filter the residuals
filtered_sp500 = (sp500_logreturns - sp500_garch_estimates[0]) / sqrt(sigma2_garch_sp500)
filtered_bcom = (bcom_logreturns - bcom_garch_estimates[0]) / sqrt(sigma2_garch_bcom)
filtered_apple = (apple_logreturns - apple_garch_estimates[0]) / sqrt(sigma2_garch_apple)

# draw some histograms to have a first insight
plt.figure()
filtered_sp500.hist(bins=100)
plt.title('Filtered Residuals of S&P500 using GARCH(1,1)')

plt.figure()
filtered_bcom.hist(bins=100)
plt.title('Filtered Residuals of BCOM using GARCH(1,1)')

plt.figure()
filtered_apple.hist(bins=100)
plt.title('Filtered Residuals of Apple using GARCH(1,1)')

# Kolmogorov and Smirnov Test
def ks_test(series, distribution, args=None):
    """
    Performs a KS test to the given series.
    
    Parameters:
        series: the series we want to test
        distribution: string, the corresponding distribution
    """
    if args is None:
        return pd.Series(sc.kstest(series, distribution), index=['KS statistic','p-val'])
    else:
        return pd.Series(sc.kstest(series, distribution, args=args), index=['KS statistic','p-val'])
        
                         
ks_filtered_sp500 = ks_test(filtered_sp500, 'norm')
ks_filtered_bcom = ks_test(filtered_bcom, 'norm')
ks_filtered_apple = ks_test(filtered_apple, 'norm')



###
#   Q1.4
###


def loglikelihood_gennorm(parameters, X):
    """Returns the negative loglikelihood of a generalised normal distribution"""
    beta = parameters[0]
    loc = parameters[1] # loc
    scale = parameters[2] # scale
    likelihood = gennorm.pdf(X,beta,loc,scale)
    loglik = -sum(np.log(likelihood))
    return loglik

def loglikelihood_norminv(parameters, X):
    """Returns the negative loglikelihood of a normal inverse distribution"""
    alpha = parameters[0]
    beta = parameters[1]
    mu = parameters[2]
    delta = parameters[3]
    likelihood = norminvgauss.pdf(X, alpha, beta, mu, delta)
    loglik = -sum(np.log(likelihood))
    return loglik

def loglikelihood_laplace(parameters, X):
    """Returns the negative loglikelihood of a Laplace distribution"""
    mu = parameters[0]
    scale = parameters[1]
    likelihood = laplace.pdf(X,mu,scale)
    loglik = -sum(np.log(likelihood))
    return loglik

def loglikelihood_johnsonsu(parameters, X):
    """Returns the negative loglikelihood of a Johnson SU distribution"""
    a = parameters[0]
    b = parameters[1]
    loc = parameters[2]
    scale = parameters[3]
    likelihood = johnsonsu.pdf(X, a,b,loc,scale)
    loglik = -sum(np.log(likelihood))
    return loglik

def loglikelihood_dgamma(parameters, X):
    """Returns the negative loglikelihood of a double Gamma distribution"""
    a = parameters[0]
    loc = parameters[1]
    scale = parameters[2]
    likelihood = dgamma.pdf(X, a, loc, scale)
    loglik = -sum(np.log(likelihood))
    return loglik

"""bounds and constraints"""
bounds_gennorm = [(0.0,5.0), (-5.0, 5.0), (0.0,5.0)]
bounds_norminv = [(0.0,10.0), (-5.0, 10.0), (-5.0,5.0), (0.0,5.0)]
bounds_laplace = [(-5.0,5.0), (0.0, 5.0)]
bounds_johnsonsu = [(-5.0,10.0), (0.0, 5.0), (-5.0,5.0), (0.0,5.0)]
bounds_dgamma = [(0.0,5.0),(-5.0,5.0),(0.0,5.0)]

"""starting values"""
start_gennorm = np.array([1, 1, 1])
start_norminv = np.array([1,0.5,1,1])
start_laplace = np.array([1,1])
start_johnsonsu = np.array([1,1,1,1])
start_dgamma = np.array([1,1,1])

# S&P500
estimates_sp500_gennorm = fmin_slsqp(loglikelihood_gennorm, 
                                start_gennorm, 
                                f_ieqcons=None,
                                bounds = bounds_gennorm,
                                args = [filtered_sp500])
estimates_sp500_norminv = fmin_slsqp(loglikelihood_norminv, 
                                start_norminv, 
                                f_ieqcons=None,
                                bounds = bounds_norminv,
                                args = [filtered_sp500])
estimates_sp500_laplace = fmin_slsqp(loglikelihood_laplace, 
                                start_laplace, 
                                f_ieqcons=None,
                                bounds = bounds_laplace,
                                args = [filtered_sp500])
estimates_sp500_johnsonsu = fmin_slsqp(loglikelihood_johnsonsu, 
                                start_johnsonsu, 
                                f_ieqcons=None,
                                bounds = bounds_johnsonsu,
                                args = [filtered_sp500])
estimates_sp500_dgamma = fmin_slsqp(loglikelihood_dgamma, 
                                start_dgamma, 
                                f_ieqcons=None,
                                bounds = bounds_dgamma,
                                args = [filtered_sp500])

# BCOM 
estimates_bcom_gennorm = fmin_slsqp(loglikelihood_gennorm, 
                                start_gennorm, 
                                f_ieqcons=None,
                                bounds = bounds_gennorm,
                                args = [filtered_bcom])
estimates_bcom_norminv = fmin_slsqp(loglikelihood_norminv, 
                                start_norminv, 
                                f_ieqcons=None,
                                bounds = bounds_norminv,
                                args = [filtered_bcom])
estimates_bcom_laplace = fmin_slsqp(loglikelihood_laplace, 
                                start_laplace, 
                                f_ieqcons=None,
                                bounds = bounds_laplace,
                                args = [filtered_bcom])
estimates_bcom_johnsonsu = fmin_slsqp(loglikelihood_johnsonsu, 
                                start_johnsonsu, 
                                f_ieqcons=None,
                                bounds = bounds_johnsonsu,
                                args = [filtered_bcom])
estimates_bcom_dgamma = fmin_slsqp(loglikelihood_dgamma, 
                                start_dgamma, 
                                f_ieqcons=None,
                                bounds = bounds_dgamma,
                                args = [filtered_bcom])


# Apple 
estimates_apple_gennorm = fmin_slsqp(loglikelihood_gennorm, 
                                start_gennorm, 
                                f_ieqcons=None,
                                bounds = bounds_gennorm,
                                args = [filtered_apple])
estimates_apple_norminv = fmin_slsqp(loglikelihood_norminv, 
                                start_norminv, 
                                f_ieqcons=None,
                                bounds = bounds_norminv,
                                args = [filtered_apple])
estimates_apple_laplace = fmin_slsqp(loglikelihood_laplace, 
                                start_laplace, 
                                f_ieqcons=None,
                                bounds = bounds_laplace,
                                args = [filtered_apple])
estimates_apple_johnsonsu = fmin_slsqp(loglikelihood_johnsonsu, 
                                start_johnsonsu, 
                                f_ieqcons=None,
                                bounds = bounds_johnsonsu,
                                args = [filtered_apple])
estimates_apple_dgamma = fmin_slsqp(loglikelihood_dgamma, 
                                start_dgamma, 
                                f_ieqcons=None,
                                bounds = bounds_dgamma,
                                args = [filtered_apple])



dfgennorm=pd.DataFrame({'S&P500':estimates_sp500_gennorm,
                   'Apple':estimates_apple_gennorm,
                   'BBG':estimates_bcom_gennorm})
dfgennorm.to_latex('GenNorm.tex')

dfnorminv=pd.DataFrame({'S&P500':estimates_sp500_norminv,
                   'Apple':estimates_apple_norminv,
                   'BBG':estimates_bcom_norminv})
dfnorminv.to_latex('NormInv.tex')

dflaplace=pd.DataFrame({'S&P500':estimates_sp500_laplace,
                   'Apple':estimates_apple_laplace,
                   'BBG':estimates_bcom_laplace})
dflaplace.to_latex('Laplace.tex')

dfjohnsu=pd.DataFrame({'S\&P500':estimates_sp500_johnsonsu,
                   'Apple':estimates_apple_johnsonsu,
                   'BBG':estimates_bcom_johnsonsu})
dfjohnsu.to_latex('JohnSU.tex')

dfdgamma=pd.DataFrame({'S&P500':estimates_sp500_dgamma,
                   'Apple':estimates_apple_dgamma,
                   'BBG':estimates_bcom_dgamma})
dfdgamma.to_latex('DGamma.tex')


"""KS test"""
ks_filtered_sp500_gennorm = ks_test(filtered_sp500, 'gennorm', args = estimates_sp500_gennorm)
ks_filtered_sp500_norminv = ks_test(filtered_sp500, 'norminvgauss', args = estimates_sp500_norminv)
ks_filtered_sp500_laplace = ks_test(filtered_sp500, 'laplace', args = estimates_sp500_laplace)
ks_filtered_sp500_johnsonsu = ks_test(filtered_sp500, 'johnsonsu', args = estimates_sp500_johnsonsu)
ks_filtered_sp500_dgamma = ks_test(filtered_sp500, 'dgamma', args = estimates_sp500_dgamma)

KSSP=pd.DataFrame({'GenNorm':ks_filtered_sp500_gennorm,
                   'NormInvGauss':ks_filtered_sp500_norminv,
                   'Laplace':ks_filtered_sp500_laplace,
                   'JohnSU':ks_filtered_sp500_johnsonsu,
                   'DGamma':ks_filtered_sp500_dgamma})
KSSP.to_latex('KSSP.tex')


ks_filtered_bcom_gennorm = ks_test(filtered_bcom, 'gennorm', args = estimates_bcom_gennorm)
ks_filtered_bcom_norminv = ks_test(filtered_bcom, 'norminvgauss', args = estimates_bcom_norminv)
ks_filtered_bcom_laplace = ks_test(filtered_bcom, 'laplace', args = estimates_bcom_laplace)
ks_filtered_bcom_johnsonsu = ks_test(filtered_bcom, 'johnsonsu', args = estimates_bcom_johnsonsu)
ks_filtered_bcom_dgamma = ks_test(filtered_bcom, 'dgamma', args = estimates_bcom_dgamma)

KSBCOM=pd.DataFrame({'GenNorm':ks_filtered_bcom_gennorm,
                   'NormInvGauss':ks_filtered_bcom_norminv,
                   'Laplace':ks_filtered_bcom_laplace,
                   'JohnSU':ks_filtered_bcom_johnsonsu,
                   'DGamma':ks_filtered_bcom_dgamma})
KSBCOM.to_latex('KSBBG.tex')


ks_filtered_apple_gennorm = ks_test(filtered_apple, 'gennorm', args = estimates_apple_gennorm)
ks_filtered_apple_norminv = ks_test(filtered_apple, 'norminvgauss', args = estimates_apple_norminv)
ks_filtered_apple_laplace = ks_test(filtered_apple, 'laplace', args = estimates_apple_laplace)
ks_filtered_apple_johnsonsu = ks_test(filtered_apple, 'johnsonsu', args = estimates_apple_johnsonsu)
ks_filtered_apple_dgamma = ks_test(filtered_apple, 'dgamma', args = estimates_apple_dgamma)

KSAAPL=pd.DataFrame({'GenNorm':ks_filtered_apple_gennorm,
                   'NormInvGauss':ks_filtered_apple_norminv,
                   'Laplace':ks_filtered_apple_laplace,
                   'JohnSU':ks_filtered_apple_johnsonsu,
                   'DGamma':ks_filtered_apple_dgamma})
KSAAPL.to_latex('KSAAPL.tex')



# Use the distribution that minimizes the sum of squares and information criteria
"""S&P500"""
plt.figure()
f_sp500 = Fitter(filtered_sp500, distributions=["gennorm", "norminvgauss", "johnsonsu"])
f_sp500.fit()
f_sp500.summary().to_latex('sp500fitter.tex')
plt.title('Filtered Residuals S&P500 GARCH(1,1)')
plt.tight_layout()


"""BCOM"""
plt.figure()
f_bcom = Fitter(filtered_bcom, distributions=["gennorm", "norminvgauss", "johnsonsu"])
f_bcom.fit()
f_bcom.summary().to_latex('bbgfitter.tex')
plt.title('Filtered Residuals BCOM GARCH(1,1)')
plt.tight_layout()


"""Apple"""
plt.figure()
f_apple = Fitter(filtered_apple, distributions=["gennorm", "norminvgauss", "johnsonsu"])
f_apple.fit()
f_apple.summary().to_latex('applefitter.tex') 
plt.title('Filtered Residuals Apple GARCH(1,1)')
plt.tight_layout()



# # never reject H0 with the KS test for the infered distributions


#################
######  PART 2
#################


"""DATA FOR PART 2"""
start = dt.datetime(1989,12,31)
end = dt.datetime(2020,12,31)
sp500 = web.DataReader('^GSPC', 'yahoo', start=start, end=end)
sp500_logreturns =  (log(sp500['Adj Close']) - log(sp500['Adj Close'].shift(1))).dropna()


r = 0.0025 # risk-free rate

"""Question 1"""

def heston_nandi_loglik(parameters, data, out=None):
    omega, beta, alpha, gamma, lbda = parameters
    series = np.array(data)
    h = np.zeros(len(series))
    e = np.zeros(len(series))
    
    for i in range(0,len(series)):
        if i==0:
            h[0] = (omega + alpha) / (1 - alpha * gamma**2 - beta) # unconditional variance
            if h[0] < 0 or 1 - alpha * gamma**2 - beta < 1e-4: #1e-3
                return 1e50
        else:
            h[i] = omega + beta * h[i-1] + alpha * (e[i-1] - gamma * np.sqrt(h[i-1]))**2
            e[i] = (series[i] - r - lbda * h[i]) / np.sqrt(h[i])
    logliks = -0.5 * (-np.log(2*np.pi) - np.log(h[:]) - (e[:]**2)) # negative log-likelihood
    loglik = -0.5 * np.sum(-np.log(2*np.pi) - np.log(h[:]) - (e[:]**2))
    if out is None:
        return loglik
    else:
        return loglik, logliks, h

def heston_nandi_estimates(data):
    omega, beta, alpha, gamma, lbda = 9.765e-10, 0.90, 2.194e-06, 100.15, 10
    # omega, beta, alpha, gamma, lbda = 5.04e-7, 0.664, 1.496e-6, 462.32, 0.91 # as in slides
    x0 = np.array([omega, beta, alpha, gamma, lbda])
    cons = ({'type': 'ineq', 'fun': lambda x: 1-x[1]-x[2]*(x[3]**2)},
            {'type': 'ineq', 'fun': lambda x: np.array(x)})
    bounds = ((1e-120,1e-5),(1e-6,1),(1e-12,1e-5),(1e-10,1e3),(1e-10,100))
    res = minimize(heston_nandi_loglik, x0, bounds = bounds, constraints = cons, 
                   method = 'Nelder-Mead', args=(data, None))
    omega, beta, alpha, gamma, lbda = res.x
    print(res.success)
    return res.x
    
    

sp500_heston_nandi_estimates = heston_nandi_estimates(sp500_logreturns)
sigma2_heston_nandi_sp500 = heston_nandi_loglik(sp500_heston_nandi_estimates, sp500_logreturns, out=True)[2]

# S&P500
plt.figure()
plt.plot(sp500_logreturns.index, sqrt(sigma2_heston_nandi_sp500)*sqrt(252))
plt.title('S&P500 Annualized Volatility - Heston Nandi')
plt.savefig('Plot/Volat_HN.png')
plt.show()
plt.close()


# Numerical approximation of the score function and final estimation results display
# f'(x) = lim_{h->0} (f(x+h)-f(x))/h
# in practice: (f(x+h/2)-f(x-h/2))/h approximates the derivative of the function for small h
# Fisher information matrix: I = E[gg'], where g is the gradient (BHHH)
def heston_nandi_significance(logreturns, heston_nandi_estimates):
    T = len(logreturns)
    nb_estimates = len(heston_nandi_estimates)
    step = 1e-5 * heston_nandi_estimates
    scores = zeros((T,nb_estimates))
    for i in range(nb_estimates):
        h = step[i]
        delta = np.zeros(nb_estimates)
        delta[i] = h
        
        loglik, logliksplus, sigma2 = heston_nandi_loglik(heston_nandi_estimates + delta, 
                                                        np.asarray(logreturns), 
                                                        out=True)
        loglik, logliksminus, sigma2 = heston_nandi_loglik(heston_nandi_estimates - delta,
                                                        np.asarray(logreturns), 
                                                        out=True)
        scores[:,i] = (logliksplus - logliksminus)/(2*h)
    
    I = (scores.T @ scores)/T    
    vcv = mat(inv(I))/T # variance covariance matrix
    vcv = asarray(vcv)
   
    output = np.vstack((heston_nandi_estimates,sqrt(diag(vcv)),heston_nandi_estimates/sqrt(diag(vcv)))).T
    
    significance_summary = pd.DataFrame(data=output, 
                                        index=['omega','beta','alpha','gamma', 'lambda'],
                                        columns=['Parameter','Std. Err.','T-stat'])
   
    return significance_summary

heston_nandi_signif_sp500 = heston_nandi_significance(sp500_logreturns, sp500_heston_nandi_estimates)
heston_nandi_signif_sp500.to_latex('Output/HN_estimates.tex')




"""Question 2"""
N = 10000 # number of simulations
T = [63, 126, 252] # maturities 3m, 6m, 1y (in trading days)
period_price = sp500[sp500.index >= '2020-12-31']['Adj Close']

# first, simulate the Heston and Nandi returns for the relevant period
def returns_simulated(horizon, heston_nandi_estimates):
    """
    Generates a 10000-size sample of returns for the given time horizon (days) and given estimates of 
    parameters of a Heston Nandi process. For example, if horizon=30, it will give a sample of 10000 
    values of 1-month returns.
    
    Parameters:
        horizon: int, the number of days of the time horizon
        heston_nandi_estimates: array, estimates of the Heston Nandi model (omega, beta, alpha, gamma, lambda)
    """
    returns_sum_heston_nandi = np.ones(N)*0
    for i in range(0,N):
        z = randn(horizon)
        omega, beta, alpha, gamma, lbda = heston_nandi_estimates
        returns_heston_nandi = np.ones(horizon) * 0
        lt_vol_heston_nandi = (omega + alpha) / (1 - alpha * gamma**2 - beta)
        h = np.ones(horizon) * lt_vol_heston_nandi
    
        for t in range(1,horizon):
            h[t] = omega + beta * h[t-1] + alpha * (z[t-1] - gamma * np.sqrt(h[t-1]))**2
            returns_heston_nandi[t] = r + lbda*h[t] + np.sqrt(h[t]) * z[t]
        returns_sum_heston_nandi[i] = np.sum(returns_heston_nandi)
    return returns_sum_heston_nandi

# compute the simulated returns for each maturity
returns_heston_nandi_3m = returns_simulated(T[0], sp500_heston_nandi_estimates)
returns_heston_nandi_6m = returns_simulated(T[1], sp500_heston_nandi_estimates)
returns_heston_nandi_1y = returns_simulated(T[2], sp500_heston_nandi_estimates)

# Computation of Option prices
spot = period_price.iloc[0] # price on 2020-12-31
K = np.linspace(0.9*spot, 1.1*spot, 200) # moneyness between 0.9 and 1.1

# Computation of trajectories for each horizon
heston_nandi_prices_3m = spot*np.exp(returns_heston_nandi_3m)
heston_nandi_prices_6m = spot*np.exp(returns_heston_nandi_6m)
heston_nandi_prices_1y = spot*np.exp(returns_heston_nandi_1y)

# Option prices
strikes_3m = np.matlib.repmat(K,len(heston_nandi_prices_3m),1)
strikes_6m = np.matlib.repmat(K,len(heston_nandi_prices_6m),1)
strikes_1y = np.matlib.repmat(K,len(heston_nandi_prices_1y),1)

    #
    # 3m maturity
    #
heston_nandi_option_prices_3m = np.matlib.repmat(heston_nandi_prices_3m,len(K),1)-strikes_3m.T
heston_nandi_option_prices_3m[np.where(heston_nandi_option_prices_3m<0)] = 0
heston_nandi_calls_3m = np.exp(-r*T[0]/252)*np.mean(heston_nandi_option_prices_3m,1)
plt.figure(figsize=(15,7))
plt.plot(K/spot, heston_nandi_calls_3m, 'r', label='3 Months')

    #
    # 6m maturity
    #
heston_nandi_option_prices_6m = np.matlib.repmat(heston_nandi_prices_6m,len(K),1)-strikes_3m.T
heston_nandi_option_prices_6m[np.where(heston_nandi_option_prices_6m<0)] = 0
heston_nandi_calls_6m = np.exp(-r*T[1]/252)*np.mean(heston_nandi_option_prices_6m,1)
plt.plot(K/spot, heston_nandi_calls_6m, 'b', label='6 Months')

    #
    # 1y maturity
    #
heston_nandi_option_prices_1y = np.matlib.repmat(heston_nandi_prices_1y,len(K),1)-strikes_3m.T
heston_nandi_option_prices_1y[np.where(heston_nandi_option_prices_1y<0)] = 0
heston_nandi_calls_1y = np.exp(-r*T[2]/252)*np.mean(heston_nandi_option_prices_1y,1)
plt.plot(K/spot, heston_nandi_calls_1y, 'green', label='1 Year')
plt.title("Call price on strike price on 2020-12-31")
plt.ylabel("Call Price")
plt.xlabel("Moneyness")
plt.legend(loc='upper right')
plt.savefig('Plot/Call_price_MC.png')
plt.show()
plt.close()

# recover implied volatility
def black_scholes_call(sigma, spot, strike, riskfree_rate, maturity):
    """Computes the price of a call option under Black-Scholes framework"""
    d1 = (np.log(spot/strike) + (riskfree_rate + 0.5*(sigma**2))*maturity) / (sigma * np.sqrt(maturity))
    d2 = d1 - sigma*np.sqrt(maturity)
    call = spot * norm.cdf(d1) - strike * np.exp(-riskfree_rate * maturity) * norm.cdf(d2)
    return call
    
def implied_volatility_criterion(spot, strike, riskfree_rate, maturity, call):
    """Recovers the implied volatlity from a call price by brute forcing possible values"""
    sigma = np.linspace(0.01, 1, 1000) # 1000 to avoid sawtooth chart but very long to compute
    error = np.ones(len(sigma))
    
    for i in range(len(error)):
        error[i] = (black_scholes_call(sigma[i], spot, strike, riskfree_rate, maturity) - call) **2
    sigma_opt = sigma[np.where(error==min(error))] # collects the values of volatility that minimize the error
    if len(sigma_opt) > 1: # several candidates, retain the last one
        sigma_opt = sigma_opt[len(sigma_opt)-1]
    return np.array(sigma_opt)

def implied_volatility(heston_nandi_calls, horizon):
    IV_heston_nandi = np.ones(len(heston_nandi_calls))
    for i in range(0, len(heston_nandi_calls)):
        IV_heston_nandi[i] = implied_volatility_criterion(spot, K[i], r, horizon/252, heston_nandi_calls[i])
    return IV_heston_nandi

# 3m maturity implied volatilty
IV_heston_nandi_3m = implied_volatility(heston_nandi_calls_3m, T[0])
plt.figure(figsize=(15,7))
plt.plot(K/spot, IV_heston_nandi_3m, 'r', label='3 Months')

# 6m maturity implied volatility
IV_heston_nandi_6m = implied_volatility(heston_nandi_calls_6m, T[1])
plt.plot(K/spot, IV_heston_nandi_6m, 'b', label='6 Months')

# 1y maturity implied volatility
IV_heston_nandi_1y = implied_volatility(heston_nandi_calls_1y, T[2])
plt.plot(K/spot, IV_heston_nandi_1y, 'green', label = '1 Year')
plt.title("Implied Volatility - Heston Nandi")
plt.ylabel("Implied Volatility")
plt.xlabel("Moneyness")
plt.legend(loc='upper right')
plt.savefig('Plot/HN_IV.png')
plt.show()
plt.close()

"""
Question 3

Under the P-measure: r_t = r + lambda * h_t + sqrt(h_t) * epsilon_t
Under the Q-measure: r_t = r - (1/2) * h_t + sqrt(h_t) * epsilon_t
"""

"""

To avoid sawtooth charts modify the implied volatility criterion by adding 
more points for the possible sigma (1000 instead of 500), note that it slows down
the runtime substantially

"""

# first get the original estimates from the optimizer
omega, beta, alpha, gamma, lbda = sp500_heston_nandi_estimates

colors = ['r', 'b', 'green']

# make omega vary
omega_range = np.array([omega*0.90, omega*1.15, omega])

plt.figure(figsize=(15,7))
for i in range(len(omega_range)):
    returns_hn_3m = returns_simulated(T[0], np.array([omega_range[i], beta, alpha, gamma, lbda]))
    hn_prices_3m = spot*np.exp(returns_hn_3m)
    hn_option_prices_3m = np.matlib.repmat(hn_prices_3m,len(K),1)-strikes_3m.T
    hn_option_prices_3m[np.where(hn_option_prices_3m<0)] = 0
    hn_calls_3m = np.mean(hn_option_prices_3m,1)
    
    IV_hn_3m = implied_volatility(hn_calls_3m, T[0])
    plt.plot(K/spot, IV_hn_3m, colors[i], label = "omega = "+ str(omega_range[i]))
    
    if i == 2:
        plt.title("Implied Volatility 3m - Heston Nandi")
        plt.ylabel("Implied Volatility")
        plt.xlabel("Moneyness")
        plt.legend(loc='upper right')
        plt.savefig('Plot/IV_HN_omega')
        plt.show()
        plt.close()

# make beta vary
beta_range = np.array([beta*0.95, beta*0.90, beta]) # does not work to test higher values of beta...

plt.figure(figsize=(15,7))
for i in range(len(beta_range)):
    returns_hn_3m = returns_simulated(T[0], np.array([omega, beta_range[i], alpha, gamma, lbda]))
    hn_prices_3m = spot*np.exp(returns_hn_3m)
    hn_option_prices_3m = np.matlib.repmat(hn_prices_3m,len(K),1)-strikes_3m.T
    hn_option_prices_3m[np.where(hn_option_prices_3m<0)] = 0
    hn_calls_3m = np.mean(hn_option_prices_3m,1)
    
    IV_hn_3m = implied_volatility(hn_calls_3m, T[0])
    plt.plot(K/spot, IV_hn_3m, colors[i], label="beta = "+ str(beta_range[i]))
    
    if i == 2:
        plt.title("Implied Volatility 3m - Heston Nandi")
        plt.ylabel("Implied Volatility")
        plt.xlabel("Moneyness")
        plt.legend(loc='upper right')
        plt.savefig('Plot/IV_HN_beta')
        plt.show()
        plt.close()


# make alpha vary
alpha_range = np.array([alpha*0.95, alpha*1.05, alpha])

plt.figure(figsize=(15,7))
for i in range(len(alpha_range)):
    returns_hn_3m = returns_simulated(T[0], np.array([omega, beta, alpha_range[i], gamma, lbda]))
    hn_prices_3m = spot*np.exp(returns_hn_3m)
    hn_option_prices_3m = np.matlib.repmat(hn_prices_3m,len(K),1)-strikes_3m.T
    hn_option_prices_3m[np.where(hn_option_prices_3m<0)] = 0
    hn_calls_3m = np.mean(hn_option_prices_3m,1)
    
    IV_hn_3m = implied_volatility(hn_calls_3m, T[0])
    plt.plot(K/spot, IV_hn_3m, colors[i], label="alpha = "+ str(alpha_range[i]))

    if i == 2:
        plt.title("Implied Volatility 3m - Heston Nandi")
        plt.ylabel("Implied Volatility")
        plt.xlabel("Moneyness")
        plt.legend(loc='upper right')
        plt.savefig('Plot/IV_HN_alpha')
        plt.show()
        plt.close()


# make gamma vary
gamma_range = np.array([gamma*0.95, gamma*1.05, gamma])

plt.figure(figsize=(15,7))
for i in range(len(gamma_range)):
    returns_hn_3m = returns_simulated(T[0], np.array([omega, beta, alpha, gamma_range[i], lbda]))
    hn_prices_3m = spot*np.exp(returns_hn_3m)
    hn_option_prices_3m = np.matlib.repmat(hn_prices_3m,len(K),1)-strikes_3m.T
    hn_option_prices_3m[np.where(hn_option_prices_3m<0)] = 0
    hn_calls_3m = np.mean(hn_option_prices_3m,1)
    
    IV_hn_3m = implied_volatility(hn_calls_3m, T[0])
    plt.plot(K/spot, IV_hn_3m, colors[i], label="gamma = "+ str(gamma_range[i]))

    if i == 2:
        plt.title("Implied Volatility 3m - Heston Nandi")
        plt.ylabel("Implied Volatility")
        plt.xlabel("Moneyness")
        plt.legend(loc='upper right')
        plt.savefig('Plot/IV_HN_gamma')
        plt.show()
        plt.close()


# make lambda vary
lbda_range = np.array([lbda*0.95, lbda*1.05, lbda])

plt.figure(figsize=(15,7))
for i in range(len(lbda_range)):
    returns_hn_3m = returns_simulated(T[0], np.array([omega, beta, alpha, gamma, lbda_range[i]]))
    hn_prices_3m = spot*np.exp(returns_hn_3m)
    hn_option_prices_3m = np.matlib.repmat(hn_prices_3m,len(K),1)-strikes_3m.T
    hn_option_prices_3m[np.where(hn_option_prices_3m<0)] = 0
    hn_calls_3m = np.mean(hn_option_prices_3m,1)
    
    IV_hn_3m = implied_volatility(hn_calls_3m, T[0])
    plt.plot(K/spot, IV_hn_3m, colors[i], label="lambda = "+ str(lbda_range[i]))

    if i == 2:
        plt.title("Implied Volatility 3m - Heston Nandi")
        plt.ylabel("Implied Volatility")
        plt.xlabel("Moneyness")
        plt.legend(loc='upper right')
        plt.savefig('Plot/IV_HN_lambda')
        plt.show()
        plt.close()


###Risk Neutral Risk Distribution###

def heston_nandi_loglik_rn(parameters, data, out=None):
    omega, beta, alpha, gamma = parameters
    series = np.array(data)
    h = np.zeros(len(series))
    e = np.zeros(len(series))
    
    for i in range(0,len(series)):
        if i==0:
            h[0] = (omega + alpha) / (1 - alpha * gamma**2 - beta) # unconditional variance
            if h[0] < 0 or 1 - alpha * gamma**2 - beta < 1e-4: #1e-3
                return 1e50
        else:
            h[i] = omega + beta * h[i-1] + alpha * (e[i-1] - gamma * np.sqrt(h[i-1]))**2
            e[i] = (series[i] - r + 0.5 * h[i]) / np.sqrt(h[i]) 
    logliks = -0.5 * (-np.log(2*np.pi) - np.log(h[:]) - (e[:]**2)) # negative log-likelihood
    loglik = -0.5 * np.sum(-np.log(2*np.pi) - np.log(h[:]) - (e[:]**2))
    if out is None:
        return loglik
    else:
        return loglik, logliks, h

def heston_nandi_estimates_rn(data):
    omega, beta, alpha, gamma = 9.765e-10, 0.90, 2.194e-06, 100.15
    #omega, beta, alpha, gamma = 5.04e-7, 0.664, 1.496e-6, 462.32 # as in slides
    x0 = np.array([omega, beta, alpha, gamma])
    cons = ({'type': 'ineq', 'fun': lambda x: 1-x[1]-x[2]*((x[3])**2)},
            {'type': 'eq', 'fun': lambda x: x[3] - gamma - lbda - 0.5})
    # cons = ({'type': 'ineq', 'fun': lambda x: 1-x[1]-x[2]*((x[3]+x[4]+0.5)**2)},
    #         {'type': 'ineq', 'fun': lambda x: np.array(x)})
    bounds = ((-1,1),(-1,1),(-1,1),(0,1000))
    res = minimize(heston_nandi_loglik_rn, x0, bounds = bounds, constraints = cons, 
                   method = 'Nelder-Mead', args=(data, None))
    omega, beta, alpha, gamma = res.x
    print(res.success)
    return res.x



sp500_heston_nandi_estimates_rn = heston_nandi_estimates_rn(sp500_logreturns)

omega_rn, beta_rn, alpha_rn, gamma_rn = sp500_heston_nandi_estimates_rn
lbda_rn = -0.5
#gamma_rn = gamma + lbda + 0.5

# make omega vary
omega_range_rn = np.array([omega_rn*0.95, omega_rn*1.05, omega_rn])

plt.figure(figsize=(15,7))
for i in range(len(omega_range_rn)):
    returns_hn_3m = returns_simulated(T[0], np.array([omega_range_rn[i], beta_rn, alpha_rn, gamma_rn, lbda_rn]))
    hn_prices_3m = spot*np.exp(returns_hn_3m)
    hn_option_prices_3m = np.matlib.repmat(hn_prices_3m, len(K), 1)-strikes_3m.T
    hn_option_prices_3m[np.where(hn_option_prices_3m<0)] = 0
    hn_option_prices_3m = np.nan_to_num(hn_option_prices_3m)
    hn_calls_3m = np.mean(hn_option_prices_3m,1)
    
    IV_hn_3m = implied_volatility(hn_calls_3m, T[0])
    plt.plot(K/spot, IV_hn_3m, colors[i], label = "omega = "+ str(omega_range_rn[i]))
    
    if i == 2:
        plt.title("Implied Volatility 3m - Heston Nandi (Risk Neutral)")
        plt.ylabel("Implied Volatility")
        plt.xlabel("Moneyness")
        plt.legend(loc='upper right')
        plt.savefig('Plot/IV_HN_omega_rn')
        plt.show()
        plt.close()

# make beta vary
beta_range_rn = np.array([beta_rn*1.005, beta_rn*0.99, beta_rn]) # does not work to test higher values of beta...

plt.figure(figsize=(15,7))
for i in range(len(beta_range_rn)):
    returns_hn_3m = returns_simulated(T[0], np.array([omega_rn, beta_range_rn[i], alpha_rn, gamma_rn, lbda_rn]))
    hn_prices_3m = spot*np.exp(returns_hn_3m)
    hn_option_prices_3m = np.matlib.repmat(hn_prices_3m,len(K),1)-strikes_3m.T
    hn_option_prices_3m[np.where(hn_option_prices_3m<0)] = 0
    hn_option_prices_3m = np.nan_to_num(hn_option_prices_3m)
    hn_calls_3m = np.mean(hn_option_prices_3m,1)
    
    IV_hn_3m = implied_volatility(hn_calls_3m, T[0])
    plt.plot(K/spot, IV_hn_3m, colors[i], label="beta = "+ str(beta_range_rn[i]))
    
    if i == 2:
        plt.title("Implied Volatility 3m - Heston Nandi (Risk Neutral)")
        plt.ylabel("Implied Volatility")
        plt.xlabel("Moneyness")
        plt.legend(loc='upper right')
        plt.savefig('Plot/IV_HN_beta_rn')
        plt.show()
        plt.close()

# make alpha vary
alpha_range_rn = np.array([alpha_rn*0.95, alpha_rn*1.05, alpha_rn])

plt.figure(figsize=(15,7))
for i in range(len(alpha_range_rn)):
    returns_hn_3m = returns_simulated(T[0], np.array([omega_rn, beta_rn, alpha_range_rn[i], gamma_rn, lbda_rn]))
    hn_prices_3m = spot*np.exp(returns_hn_3m)
    hn_option_prices_3m = np.matlib.repmat(hn_prices_3m,len(K),1)-strikes_3m.T
    hn_option_prices_3m[np.where(hn_option_prices_3m<0)] = 0
    hn_option_prices_3m = np.nan_to_num(hn_option_prices_3m)
    hn_calls_3m = np.mean(hn_option_prices_3m,1)
    
    IV_hn_3m = implied_volatility(hn_calls_3m, T[0])
    plt.plot(K/spot, IV_hn_3m, colors[i], label="alpha = "+ str(alpha_range_rn[i]))

    if i == 2:
        plt.title("Implied Volatility 3m - Heston Nandi (Risk Neutral)")
        plt.ylabel("Implied Volatility")
        plt.xlabel("Moneyness")
        plt.legend(loc='upper right')
        plt.savefig('Plot/IV_HN_alpha_rn')
        plt.show()
        plt.close()

# make gamma vary
gamma_range_rn = np.array([gamma_rn*0.95, gamma_rn*1.05, gamma_rn])

plt.figure(figsize=(15,7))
for i in range(len(gamma_range_rn)):
    returns_hn_3m = returns_simulated(T[0], np.array([omega_rn, beta_rn, alpha_rn, gamma_range_rn[i], lbda_rn]))
    hn_prices_3m = spot*np.exp(returns_hn_3m)
    hn_option_prices_3m = np.matlib.repmat(hn_prices_3m,len(K),1)-strikes_3m.T
    hn_option_prices_3m[np.where(hn_option_prices_3m<0)] = 0
    hn_option_prices_3m = np.nan_to_num(hn_option_prices_3m)
    hn_calls_3m = np.mean(hn_option_prices_3m,1)
    
    IV_hn_3m = implied_volatility(hn_calls_3m, T[0])
    plt.plot(K/spot, IV_hn_3m, colors[i], label="gamma = "+ str(gamma_range_rn[i]))

    if i == 2:
        plt.title("Implied Volatility 3m - Heston Nandi (Risk Neutral)")
        plt.ylabel("Implied Volatility")
        plt.xlabel("Moneyness")
        plt.legend(loc='upper right')
        plt.savefig('Plot/IV_HN_gamma_rn')
        plt.show()
        plt.close()























