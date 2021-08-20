#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 16:53:26 2021

@author: Florian
"""

"""SET WORKING DIRECTORY HERE"""
import os
os.chdir("/Users/Florian/UNIL/Master Finance/1ère année/2ème Semestre/Empirical Methods in Finance/Assignments/Assignment 2/Data -20210504")


# import packages
import numpy as np
import numpy.matlib
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as ts
from statsmodels.tsa.api import VAR
from scipy.stats import multivariate_normal
from scipy.stats.distributions import chi2
from scipy.optimize import minimize
import seaborn as sns

# set theme for the charts
sns.set_theme(style='whitegrid')


##################
### PART 1
##################

### import data
bitcoin = pd.read_csv("Part1_BTCUSD_d.csv")
bitcoin['date'] = pd.to_datetime(bitcoin['date'])
bitcoin.set_index('date', inplace=True)

ethereum = pd.read_csv("Part1_ETHUSD_d.csv")
ethereum['date'] = pd.to_datetime(ethereum['date'])
ethereum.set_index('date', inplace=True)


monero = pd.read_csv("Part1_XMRUSD_d.csv")
monero['date'] = pd.to_datetime(monero['date'])
monero.set_index('date', inplace=True)


### Merge the datasets
df = pd.concat([bitcoin['close'], ethereum['close'], monero['close']], 1).dropna()
df.columns = ["bitcoin", "ethereum", "monero"]

# log prices
df_log = np.log(df)

###
#   TESTING FOR STATIONARITY
###

### 1.1 p_t = alpha + beta * p_{t-1} + u_t

def ols_pp(column):
    """
    Computes the regression on 1 lag values of the given series.
    pp stands for price against price regression
    """
    x = sm.add_constant(column.values[:-1])
    y = column.values[1:]
    results = sm.OLS(y,x).fit()
    return pd.Series(data = [results.params[1], results.tvalues[1]], index=['beta estimates','t-stat']) # returns the beta and the associated t-val only

dfOLS_pp = df_log.apply(ols_pp)
dfOLS_pp = dfOLS_pp.transpose()


### 1.3 r_t = alpha + beta * p_{t-1} + u_t, with r_t = p_t - p_{t-1}

def ols_rp(column):
    """
    Computes the regression for the column with the following model: r_t = alpha + beta * p_{t-1} + u_t, with r_t = p_t - p_{t-1}
    rp stands for returns against prices regression
    """
    x = sm.add_constant(column.values[:-1])
    y = column.values[1:] - column.values[:-1]
    results = sm.OLS(y,x).fit()
    return pd.Series(data = [results.params[1], results.tvalues[1]], index=['beta estimates','t-stat']) # returns the beta and the associated t-val only

dfOLS_rp = df_log.apply(ols_rp)
dfOLS_rp = dfOLS_rp.transpose()


### 1.1.1 CRITICAL VALUES

N = 10000 # number of samples (given)
T = len(df) # length of each series

t_values = [] # will contain the t-stats

# for-loop where each iteration corresponds to one simulation
for n in range(N):
    # 1: simulate the errors
    e = np.random.normal(0,1,T)
    # 2: create an empty vector for the random walk of prices
    p = np.zeros(T)
    for t in range(1,len(p)):
        # 3: fill element by element the vector of prices
        p[t] = p[t-1] + e[t]
    # 4: compute the first difference of the prices so that H0: beta = 0 (classic t-test)
    delta_p = np.diff(p)
    # 5: run a regression of delta_p_t on p_{t-1}
    x = sm.add_constant(p[:-1])
    results = sm.OLS(delta_p, x).fit()
    # 6: append the t-val of beta from the regression
    t_values += [results.tvalues[1]]

# plot the histogram
plt.figure()
sns.dark_palette("#69d", reverse=True, as_cmap=True)
sns.histplot(t_values).set_title("DF Distribution T=1'613 (10'000 iterations)")

# Extract the critical values
percentiles = [10,5,1] # critical levels (in percent)
critical_values = np.percentile(t_values,percentiles)
critical_values_DF_test = pd.DataFrame({'10%':[critical_values[0]],'5%':[critical_values[1]],
                                        '1%':[critical_values[2]]},index=['Critical Value'])


### 1.1.2 Testing non-stationarity

def df_test(column):
    """
    Performs a Dickey Fuller test to the column, that is, H0: the column is not stationary.
    Note that we use the formula given on slide 12 of the class on cointegration for the test statistic
    
    Parameters:
        column: a time series
    
    Returns:
        relevant information regarding the test
    """
    x = sm.add_constant(column.values[:-1])
    y = column.values[1:]
    results = sm.OLS(y,x).fit()
    
    alpha = results.params[0]
    beta = results.params[1]
    
    sum_squared_eps = 0
    for t in range(1,len(y)):
        sum_squared_eps += (y[t]-alpha-beta*y[t-1])**2
    var_eps = (1/(len(column)-1))*sum_squared_eps
    
    y_mean = np.mean(y)
    
    sum_squared_price_dev = 0
    for t in range(1,len(y)):
        sum_squared_price_dev += (y[t]-y_mean)**2
    
    std_beta = (var_eps/sum_squared_price_dev)**0.5
    
    tstat = (beta - 1) / std_beta
    return pd.Series(data = [beta, tstat], index=['beta','t-stat'])
    
df_log_DF_test = df_log.apply(df_test)

# visuals of each time series of log-prices
plt.figure()
df_log['bitcoin'].plot()
df_log['ethereum'].plot()
df_log['monero'].plot()
plt.title('Evolution of log-prices')
plt.legend(['BTC','ETH','XMR'])
plt.tight_layout()



#######
###     COINTEGRATION
#######

def cointegration_test(y,x):
    # regress y on x
    res_ols = sm.OLS(y,sm.add_constant(x)).fit()
    a = res_ols.params[0]
    b = res_ols.params[1]
    # then, obtain the residuals
    z = y - res_ols.predict()
    # difference in residuals
    delta_z = (z.shift(1)-z).dropna()
    # regress the difference in residuals on the residuals
    res_ols_z = sm.OLS(delta_z.values, sm.add_constant(z.values[:-1])).fit()
    t_beta = res_ols_z.tvalues[1]
    return pd.Series(data=[a,b,t_beta],index=['a','b','t-beta'])

adfuller_result_btc_eth = cointegration_test(df_log['bitcoin'], df_log['ethereum'])
adfuller_result_xmr_btc = cointegration_test(df_log['monero'], df_log['bitcoin'])
adfuller_result_xmr_eth = cointegration_test(df_log['monero'], df_log['ethereum']) # only pair cointegrated at size 5%


# def cointegration_test(y, x):
#     # Regression
#     ols_result = sm.OLS(y, sm.add_constant(x)).fit() 
#     a = ols_result.params[0] # get constant
#     b = ols_result.params[1] # get the beta
#     # ADF test on residuals of the regression
#     adf = ts.adfuller(ols_result.resid,maxlag=0)
#     return pd.Series(data=[a,b,adf[0],adf[4]['1%'],adf[4]['5%'],adf[4]['10%']], index=['a','b','t-stat','1% critical','5% critical','10% critical'])

# # check for each pair if it is cointegrated
# adfuller_result_btc_eth = cointegration_test(df_log['bitcoin'], df_log['ethereum'])
# adfuller_result_xmr_btc = cointegration_test(df_log['monero'], df_log['bitcoin'])
# adfuller_result_xmr_eth = cointegration_test(df_log['monero'], df_log['ethereum']) # only pair cointegrated at size 5%



#######
###     PAIR TRAIDING
#######

# first, compute the log-returns of ethereum (A) and monero (B)
r_eth = (df_log['ethereum'] - df_log['ethereum'].shift(1)).dropna()
r_xmr = (df_log['monero'] - df_log['monero'].shift(1)).dropna()

# compute the spread between the two
spread = r_xmr - r_eth
#spread = r_eth - r_xmr

# compute the zscore: zscore_t = frac{spread_t - mean(spread_t)}{std(spread_t)}
zscore = (spread - np.mean(spread)) / np.std(spread)

# thresholds
zscore_up = np.percentile(zscore,95)
zscore_low = np.percentile(zscore,5)
zscore_out = np.percentile(zscore,40)

# initial wealth
wealth_0 = 100

weights_eth = np.zeros(len(zscore)-1)
weights_xmr = np.zeros(len(zscore)-1)

# set the initial values
if zscore[0] > zscore_up:
    weights_xmr[0] = -0.5
    weights_eth[0] = 0.5
if zscore[0] < zscore_low:
    weights_eth[0] = -0.5
    weights_xmr[0] = 0.5

# update the positions for each date
for t in range(1,len(zscore)):
    if zscore[t] > zscore_up:
        # long ethereum and short monero
        weights_xmr[t] = -0.5 # in terms of expense for later purpose
        weights_eth[t] = 0.5
    if zscore[t] < zscore_low:
        # long monero and short ethereum
        weights_eth[t] = -0.5
        weights_xmr[t] = 0.5
    if ((zscore[t-1] < zscore_low) & (zscore[t] >= -zscore_out)) | ((zscore[t-1] > zscore_up) & (zscore[t] <= zscore_out)):
        # check whether long monero or long ethereum at previous period
        weights_eth[t] = 0 # close position
        weights_xmr[t] = 0 # close position

# returns of the portfolio
returns = np.zeros(len(weights_eth)+1)
returns[0] = 0
for t in range(0,len(weights_eth)):
    returns[t] = weights_eth[t] * r_eth[t+1] + weights_xmr[t] * r_xmr[t+1]

gross_returns = 1 + returns

av_ret_pt = np.mean(returns*252)
vol_ret_pt = np.std(returns) * np.power(252, 0.5)
SR_pt = av_ret_pt / vol_ret_pt # assuming Rf = 0


# evolution of wealth
cumulated_wealth = []
cumulated_wealth += [wealth_0]
for t in range(0,len(gross_returns)):
    cumulated_wealth += [cumulated_wealth[t]*gross_returns[t]]

# plot the wealth evolution
plt.figure()
plt.plot(df.index.values,cumulated_wealth)
plt.xticks(rotation=90)
plt.title("Evolution of Wealth (starting with 100)")

# plot the weights
fig, ax = plt.subplots(3,1,figsize=(10,8),dpi=600)
ax[0].plot(df.index.values[2:], weights_eth,'bo',markersize=2)
ax[0].set_title('ETH weights over time')

ax[1].plot(df.index.values[2:], weights_xmr,'bo',markersize=2)
ax[1].set_title('XMR weights over time')

ax[2].plot(df.index.values[2:], weights_eth,'bo',df.index.values[2:], weights_xmr,'go',markersize=2)
ax[2].set_title('Portfolio weigths over time')
ax[2].legend(['ETH','XMR'])
plt.tight_layout()


#######
###     PART 2: VAR MODELS
#######


df2 = pd.read_excel("Part2_dataset.xlsx",index_col=0)
df2.index = pd.to_datetime(df2.index)

df2_log = np.log(df2)

### Test the stationarity of each series

def adf_test(series):
    """
    Performs an augmented Dickey Fuller test to the argument to see if it is stationary
    
    Parameters:
        series: a time series
        
    Returns:
        a series with relevant information regarding the test
    """
    result = ts.adfuller(series)
    return pd.Series([result[0], result[1], result[4]['1%'], result[4]['5%'],result[4]['10%']],
                       index = ['adf-stat','p-val','critical 1%','critical 5%','critical 10%'])


"""Dollar"""
# ADF test, H0: not stationary
adf_dollar = adf_test(df2_log['Dollar'])
# KPSS test, H0: stationary
# kpss_dollar = kpss_test(df2_log['Dollar'])

"""Gold"""
# ADF test, H0: not stationary
adf_gold = adf_test(df2_log['Gold'])
# KPSS test, H0: stationary
# kpss_gold = kpss_test(df2_log['Gold'])

"""VIX"""
# ADF test, H0: not stationary
adf_vix = adf_test(df2_log['VIX'])
# KPSS test, H0: stationary
# kpss_vix = kpss_test(df2_log['VIX'])

"""Bitcoin"""
# ADF test, H0: not stationary
adf_bitcoin = adf_test(df2_log['BitCoin'])
# KPSS test, H0: stationary
# kpss_bictoin = kpss_test(df2_log['BitCoin'])


### Transformation of the non-stationary series into stationary series
df2_log_transformed = df2_log.apply(np.diff)
df2_log_transformed.index = df2_log.index.values[1:]

# check if stationary

    # Dollar
adf_dollar_transformed = adf_test(df2_log_transformed['Dollar'])
# kpss_dollar_transformed = kpss_test(df2_log_transformed['Dollar'])

    # Gold
adf_gold_transformed = adf_test(df2_log_transformed['Gold'])
# kpss_gold_transformed = kpss_test(df2_log_transformed['Gold'])   

    # VIX
adf_vix_transformed = adf_test(df2_log_transformed['VIX'])
# kpss_vix_transformed = kpss_test(df2_log_transformed['VIX'])   

    # Bitcoin
adf_bitcoin_transformed = adf_test(df2_log_transformed['BitCoin'])
# kpss_bitcoin_transformed = kpss_test(df2_log_transformed['BitCoin'])   


# confirm intuition by drawing the two series (untransformed and transformed)
  
    # untransformed
fig, ax = plt.subplots(4,1,figsize=(10,8))
ax[0].plot(df2_log.index, df2_log['Dollar'])
ax[0].set_title('Dollar (log)')
ax[1].plot(df2_log.index, df2_log['Gold'])
ax[1].set_title('Gold (log-price)')
ax[2].plot(df2_log.index, df2_log['VIX'])
ax[2].set_title('VIX (log)')
ax[3].plot(df2_log.index, df2_log['BitCoin'])
ax[3].set_title('Bitcoin (log-price)')
plt.tight_layout()

    # transformed
fig, ax = plt.subplots(4,1,figsize=(10,8))
ax[0].plot(df2_log_transformed.index, df2_log_transformed['Dollar'])
ax[0].set_title('Dollar (log)')
ax[1].plot(df2_log_transformed.index, df2_log_transformed['Gold'])
ax[1].set_title('Gold (log-price)')
ax[2].plot(df2_log_transformed.index, df2_log_transformed['VIX'])
ax[2].set_title('VIX (log)')
ax[3].plot(df2_log_transformed.index, df2_log_transformed['BitCoin'])
ax[3].set_title('Bitcoin (log-price)')
plt.tight_layout()


### VAR models

# ORDER SELECTION: here we check for up to 9 lags the values of different information criteria.
# we store those values in arrays that we subsequently merge into a dataframe, and check what number of
# lags is optimal according to each criterion

var_model = VAR(df2_log_transformed)
lags = np.linspace(1,9,9)
aic = []
bic = []
fpe = []
hqic = [] 
for i in [1,2,3,4,5,6,7,8,9]:
    result = var_model.fit(i)
    aic += [result.aic]
    bic += [result.bic]
    fpe += [result.fpe]
    hqic += [result.hqic]

lag_selection = pd.DataFrame({'AIC':aic,'BIC':bic,'FPE':fpe,'HQIC':hqic},index=lags)

# check for each column which lag is optimal (here lag is the index of the dataframe)
optimal_aic = lag_selection['AIC'][lag_selection['AIC']==lag_selection['AIC'].min()].index[0]
optimal_bic = lag_selection['BIC'][lag_selection['BIC']==lag_selection['BIC'].min()].index[0]
optimal_fpe = lag_selection['FPE'][lag_selection['FPE']==lag_selection['FPE'].min()].index[0]
optimal_hqic = lag_selection['HQIC'][lag_selection['HQIC']==lag_selection['HQIC'].min()].index[0]

# store the optimal lags for each criterion
optimal_lags = pd.Series([optimal_aic,optimal_bic,optimal_fpe,optimal_hqic],index=['AIC','BIC','FPE','HQIC'])


## CONSTRUCTION OF THE VAR(1) --> here we refer to HQIC criterion for the choice of lags
var_model_fitted = var_model.fit(1)
var_model_fitted.summary()

## IRF of a 1 standard deviation shock of Dollar, Gold and VIX on the Bitcoin
# recall that we deal with daily data
irf = var_model_fitted.irf(5)
irf.plot(orth=True)
plt.tight_layout()

# plot the specific graphs separately
for i in range(0,3):
    irf.plot(impulse=i,response=3,signif=0.05)

### Constrained vs Unconstrained VAR(1)

# first define a function that computes the log-likelihood for a set of parameters
def ml_var(theta,X):
    """
    Computes the log-likelihood of a VAR(1) model with 4 variables contained in x, given set of parameters theta
    
    Parameters:
        theta: array containing the values of Phi1 in the first 16 positions, and the values of Phi0 in the last 4 positions
        X: matrix of values for the 4 time series
    
    Returns:
        the negative log-likelihood for minimization purposes
    """
    # unbundle the parameters
    Phi1 = [[theta[0],theta[1],theta[2],theta[3]],
            [theta[4],theta[5],theta[6],theta[7]],
            [theta[8],theta[9],theta[10],theta[11]],
            [theta[12],theta[13],theta[14],theta[15]]]
    Phi0 = [[theta[16],theta[17],theta[18],theta[19]]]
    
    Y = X.iloc[1:np.size(X,0),:]; # Y is the matrix X starting at second row (size(X,0) is the number of rows of X)
    Z = X.iloc[0:(np.size(X,0)-1),:] # matrix X up to penultimate line
    Phi0_temp = np.matlib.repmat(Phi0, np.size(Z,0), 1) # np.size(Z,0) x 1 matrix with Phi0 at each row
    # SEE SLIDE 11 class 6/7 for VAR(1) model
    temp = np.matmul(Z,Phi1) + Phi0_temp
    res = np.subtract(Y,temp) # subtract the predictions from the observed dependent variable to get the residual matrix
    # we assume that residuals are i.i.d. multivariate standard normal processes (mean vector zero, identity variance-covariance matrix)
    loglik = multivariate_normal.logpdf(res, mean=[0,0,0,0], cov=np.identity(4)) # Log of the multivariate normal probability density function
    loglik = -np.sum(loglik) # becaause it will be minimized
    return loglik


# set the initial values
Phi1 = var_model_fitted.coefs;
Phi0 = var_model_fitted.coefs_exog;
theta = [*Phi1.flatten(),*Phi0.flatten()]

# define the constraint set, that is assume bitcoin has no influence on other variables
constraint_set = ({'type':'eq', 'fun': lambda x: x[3]},
                  {'type':'eq', 'fun': lambda x: x[7]},
                  {'type':'eq', 'fun': lambda x: x[11]})

    # Constrained
estimation_output_cons = minimize(ml_var, theta, method='SLSQP', args=(df2_log_transformed), constraints=constraint_set);
estimated_para_cons = estimation_output_cons.x # extract the parameters
loglikelihood_cons = -estimation_output_cons.fun # remember that ml_var returns the negative log-likelihood

    # Unconstrained
# reset initial values
Phi1 = var_model_fitted.coefs;
Phi0 = var_model_fitted.coefs_exog;
theta = [*Phi1.flatten(),*Phi0.flatten()]

estimation_output_uncons = minimize(ml_var, theta, method='SLSQP', args=(df2_log_transformed));
estimated_para_uncons = estimation_output_uncons.x # extract the parameters
loglikelihood_uncons = -estimation_output_uncons.fun # remember that ml_var returns the negative log-likelihood


### LR test, H0: bitcoin does not affect the other asset classes
lr_test_stat = 2*(loglikelihood_uncons - loglikelihood_cons) 
p_val_lr = chi2.sf(lr_test_stat, 3) # df = nb constraints = 3
















