#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 13:59:29 2021

@author: Florian
"""

import os
os.chdir("C:/Users/Antoine/Desktop/MScF/EmpiricalMethods/Ass1")

import pandas as pd
import scipy.stats as sc
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt 
import numpy as np

sns.set_theme(style="whitegrid")

# Load data
sp500Price = pd.read_excel("Data_assignment.xlsx") # daily prices

sp500Ret = sp500Price / sp500Price.shift(1) - 1 # daily simple returns for each stock
sp500Ret = sp500Ret.dropna(how='any'); # get rid of NaN
marketRet = sp500Ret['Index'] # store the market returns
marketRet.dropna(how='any', inplace=True)
sp500Ret.drop('Index', inplace=True, axis=1) # get rid of the column 'Index' for the first questions


m_range = range(0,21)

sp500RollingVariance = sp500Ret.copy()
sp500RollingVariance = abs(sp500RollingVariance*0)

for i in m_range:
    sp500RollingVariance += (sp500Ret.shift(i) - sp500Ret.mean())**2

######################################
sp500RollingVariance = (sp500RollingVariance/(len(m_range)-1))
sp500RetCurly = sp500Ret / (sp500RollingVariance**.5) # DataFrame containing the r_t / sigma_t

#20-Day Rolling Variance of S&P500 Index
marketRollingVariance = marketRet.copy()
marketRollingVariance = abs(marketRollingVariance*0)

for i in m_range:
    marketRollingVariance += (marketRet.shift(i) - marketRet.mean())**2

marketRollingVariance = (marketRollingVariance/20)

# clean data from NaN
sp500RetCurly.dropna(how='any',inplace=True)
sp500RollingVariance.dropna(how='any',inplace=True)
marketRollingVariance.dropna(how='any',inplace=True)


#Graphs for SP500 Index Price
plt.plot(marketRollingVariance**0.5, '-r', label='20-day Rolling Std')
plt.axhline(y=marketRet.std(), color='b', linestyle='-',label='Full Sample Std')
plt.xlabel('Time Unit')
plt.ylabel('Volatility of the S&P500 Index')
plt.title('20-day Rolling Volatility vs. Full Sample Volatility')
plt.legend(loc='upper left', frameon=False)
plt.show()

#Graph for Cumulative Returns of SP500 index (Same Interpretation as the Next Plot)
marketCumulRet = (marketRet + 1).cumprod() -1
plt.plot(marketCumulRet, '-b')
plt.xlabel('Time Unit')
plt.ylabel('S&P500 Index Cumulative Return')
plt.title('S&P500 Index Cumulative Return Over Time')
plt.show()

#Graphs for SP500 Price
plt.plot(sp500Price['Index'], '-r')
plt.xlabel('Time Unit')
plt.ylabel('S&P500 Index Price')
plt.title('The Evolution of the S&P500 Index Over Time')
plt.show()


##############################################################################
# QUESTION 2
##############################################################################

#Introduction Graphs Skewness (Adapted from: https://www.astroml.org/book_figures/chapter3/fig_kurtosis_skew.html)
x = np.linspace(-8, 8, 1000)
N = sc.norm(0, 1)

plt.plot(x, N.pdf(x), 'b', label='Normal Distribution')
plt.plot(-x, 0.5 * N.pdf(x) * (2 + x + 0.5 * (x * x - 1)), 'r', label='Positive Skewness')
plt.plot(x, 0.5 * N.pdf(x) * (2 + x + 0.5 * (x * x - 1)), 'g', label='Negative Skewness')
plt.legend(loc='upper right', frameon=False, fontsize=10)
plt.title('Skewness')
plt.show()


#Skewness of simple returns and returns adjusted to volatility
sp500RetSkew = sp500Ret.skew().to_frame(name='skewness')
sp500RetCurlySkew = sp500RetCurly.skew().to_frame(name='skewness')

sp500RetNegSkew = sp500RetSkew['skewness'].loc[sp500RetSkew['skewness'] < 0].count() / sp500RetSkew.shape[0]
sp500RetTildaNegSkew = sp500RetCurlySkew['skewness'].loc[sp500RetCurlySkew['skewness'] < 0].count() / sp500RetSkew.shape[0]

# share of constituents of S&P500 with negative skewness for r_t
print('share of constituents of S&P500 with negative skewness for r_t: ', sp500RetNegSkew,'\n')
# share of constituents of S&P500 with negative skewness for r_t_curly
print('share of constituents of S&P500 with negative skewness for r_t_curly: ', sp500RetTildaNegSkew,'\n')


label=['Simple Returns', 'Adjusted Returns']

#Plot the propositon obtained as a bar chart
sp500NegSkew = [sp500RetNegSkew, sp500RetTildaNegSkew]

colors = ['b', 'r']

plt.bar(label, sp500NegSkew, color=colors)
plt.title('Share of Constituents of S&P500 with Negative Skewness')

plt.figure(figsize=(20,10))


#Historgram of the number of observation per excess kurtosis
plt.subplot(121)
plt.hist(sp500RetSkew, color='b', bins=40)
plt.title('Skewness of Simple Return', fontsize=20)
plt.ylabel('Number of Observations', fontsize=20)
plt.xlabel('Skewness', fontsize=20)


plt.subplot(122)
plt.hist(sp500RetCurlySkew, color='r', bins=40)
plt.title('Skewness of Return Adjusted to Volatility', fontsize=20)
plt.ylabel('Number of Observations', fontsize=20)
plt.xlabel('Skewness', fontsize=20)

plt.show()

##############################################################################
# QUESTION 3
##############################################################################

#Introduction Graphs Skewness (Adapted from: https://www.astroml.org/book_figures/chapter3/fig_kurtosis_skew.html)
x = np.linspace(-5, 5, 1000)
plt.plot(x, sc.laplace(0, 1).pdf(x), 'r', label='Positive Excess Kurtosis')
plt.plot(x, sc.norm(0, 1).pdf(x), 'b', label='Normal Distribution')
plt.plot(x, sc.cosine(0, 1).pdf(x), 'g', label='Negative Excess Kurtosis')
plt.legend(loc='upper right', frameon=False, fontsize=10)
plt.title('Kurtosis')
plt.show()


#Kurtosis of simple returns and returns adjusted to volatility
sp500RetExcessKurt = sp500Ret.kurtosis().to_frame(name='excess kurtosis')
sp500RetCurlyExcessKurt = sp500RetCurly.kurtosis().to_frame(name='excess kurtosis')

#Proportion of stocks with a kurtosis above 3 (note that the command .kurtosis() compute directly the excess kurtosis)
sp500RetPosExcessKurt = sp500RetExcessKurt['excess kurtosis'].loc[sp500RetExcessKurt['excess kurtosis'] > 0].count() / sp500RetExcessKurt.shape[0]
sp500RetTildaPosExcessKurt = sp500RetCurlyExcessKurt['excess kurtosis'].loc[sp500RetCurlyExcessKurt['excess kurtosis'] > 0].count() / sp500RetCurlyExcessKurt.shape[0]

# proportion of constituents of S&P500 exhibiting r_t with kurtosis > 3
print('proportion of constituents of S&P500 exhibiting r_t with kurtosis > 3: ', sp500RetExcessKurt['excess kurtosis'].loc[sp500RetExcessKurt['excess kurtosis'] > 0].count() / sp500RetExcessKurt.shape[0],'\n')
# proportion of constituents of S&P500 exhibiting r_t_curly with kurtosis > 3
print('proportion of constituents of S&P500 exhibiting r_t_curly with kurtosis > 3: ', sp500RetCurlyExcessKurt['excess kurtosis'].loc[sp500RetCurlyExcessKurt['excess kurtosis'] > 0].count() / sp500RetCurlyExcessKurt.shape[0],'\n')

plt.figure(figsize=(20,10))

#Historgram of the number of observation per excess kurtosis
plt.subplot(121)
plt.hist(sp500RetExcessKurt, color='b', bins=40)
plt.title('Excess Kurtosis of Simple Return', fontsize=20)
plt.ylabel('Number of Observations', fontsize=20)
plt.xlabel('Excess Kurtosis', fontsize=20)

plt.subplot(122)
plt.hist(sp500RetCurlyExcessKurt, color='r', bins=40)
plt.title('Excess Kurtosis of Return Adjusted to Volatility', fontsize=20)
plt.ylabel('Number of Observations', fontsize=20)
plt.xlabel('Excess Kurtosis', fontsize=20)

plt.show()

##############################################################################
# QUESTION 4
##############################################################################

# Jarque Berra test for normality

# for r_t
T = sp500Ret.shape[0]
jbRet = T * ((sp500Ret.skew()**2)/6 + (sp500Ret.kurtosis()**2)/24)
jbRet = jbRet.to_frame(name='JB test statistic')
jbRet['p-val'] = 1.0 - sc.chi2.cdf(jbRet['JB test statistic'], 2)

# proportion of constituents of S&P500 whose r_t are not normal at 5% significance level
print('proportion of constituents of S&P500 whose r_t are not normal at 5% significance level: ', jbRet['p-val'].loc[jbRet['p-val'] < 0.05].count() / jbRet.shape[0],'\n')


# for r_t_curly
T = sp500RetCurly.shape[0]
jbRetCurly = T * ((sp500RetCurly.skew()**2)/6 + (sp500RetCurly.kurtosis()**2)/24)
jbRetCurly = jbRetCurly.to_frame(name='JB test statistic')
jbRetCurly['p-val'] = 1.0 - sc.chi2.cdf(jbRetCurly['JB test statistic'], 2)

# proportion of constituents of S&P500 whose r_t_curly are not normal at 5% significance level
print('proportion of constituents of S&P500 whose r_t_curly are not normal at 5% significance level: ', jbRetCurly['p-val'].loc[jbRetCurly['p-val'] < 0.05].count() / jbRetCurly.shape[0],'\n')


##############################################################################
# QUESTION 5
##############################################################################
# Kolmogorov and Smirnov Test
def ks_test(column):
    return pd.Series(sc.kstest(column, 'norm'), index=['KS statistic','p-val'])

# for r_t
ksRet = sp500Ret.apply(ks_test)
ksRet = ksRet.transpose()
# proportion of S&P500 constituents that reject Gaussian hypothesis for r_t at 5% significance
print('proportion of S&P500 constituents that reject Gaussian hypothesis for r_t at 5% significance: ', ksRet['p-val'].loc[ksRet['p-val'] < 0.05].count() / ksRet.shape[0],'\n')

# for r_t_curly
ksRetCurly = sp500RetCurly.apply(ks_test)
ksRetCurly = ksRetCurly.transpose()
# proportion of S&P500 constituents that reject Gaussian hypothesis for r_t_curly at 5% significance
print('proportion of S&P500 constituents that reject Gaussian hypothesis for r_t_curly at 5% significance: ', ksRetCurly['p-val'].loc[ksRetCurly['p-val'] < 0.05].count() / ksRetCurly.shape[0],'\n')

##############################################################################
# QUESTION 6
##############################################################################
# Ljung-Box test
def lb_test(column, p=10):
    T = column.shape[0]
    lbstats = 0
    for lag in range(1, p+1):
        lbstats += (column.autocorr(lag=lag) ** 2) / (T - lag)
    lbstats *= T * (T + 2)
    pvalue = 1 - sc.chi2.cdf(lbstats, p)
    return pd.Series(data=[lbstats, pvalue], index=['LB statistic', 'p-val'])


# for r_t
lbRet = sp500Ret.apply(lb_test)
lbRet = lbRet.transpose()
# proportion of S&P500 constituents exhibiting autocorrelated returns r_t at 5% significance
print('proportion of S&P500 constituents exhibiting autocorrelated returns r_t at 5% significance: ', lbRet['p-val'].loc[lbRet['p-val'] < 0.05].count() / lbRet.shape[0],'\n')

# for r_t_curly
lbRetCurly = sp500RetCurly.apply(lb_test)
lbRetCurly = lbRetCurly.transpose()
# proportion of S&P500 constituents exhibiting autocorrelated returns r_t_curly at 5% significance
print('proportion of S&P500 constituents exhibiting autocorrelated returns r_t_curly at 5% significance: ', lbRetCurly['p-val'].loc[lbRetCurly['p-val'] < 0.05].count() / lbRetCurly.shape[0],'\n')

# for abs_r_t
sp500AbsRet = abs(sp500Ret)
lbAbsRet = sp500AbsRet.apply(lb_test)
lbAbsRet = lbAbsRet.transpose()
# proportion of S&P500 constituents exhibiting autocorrelated absolute returns r_t at 5% significance
print('proportion of S&P500 constituents exhibiting autocorrelated absolute returns r_t at 5% significance: ', lbAbsRet['p-val'].loc[lbAbsRet['p-val'] < 0.05].count() / lbAbsRet.shape[0],'\n')

# for abs_r_t_curly
sp500AbsRetCurly = abs(sp500RetCurly)
lbAbsRetCurly = sp500AbsRetCurly.apply(lb_test)
lbAbsRetCurly = lbAbsRetCurly.transpose()
# proportion of S&P500 constituents exhibiting autocorrelated returns r_t_curly at 5% significance
print('proportion of S&P500 constituents exhibiting autocorrelated returns r_t_curly at 5% significance: ', lbAbsRetCurly['p-val'].loc[lbAbsRetCurly['p-val'] < 0.05].count() / lbAbsRetCurly.shape[0],'\n')


##############################################################################
# QUESTION 7
##############################################################################
#Linear regression
def ols_market_returns(column, market = marketRet):
    """Returns the results of a linear regression of @column on @market, by default marketRet"""
    x = sm.add_constant(market)
    y = column.values
    model = sm.OLS(y,x)
    results = model.fit()
    return pd.Series(data = [results.params[1], results.tvalues[1]], index=['beta estimates','t-stat']) # returns the beta and the associated t-val only

sp500RetOls = sp500Ret.apply(ols_market_returns)
sp500RetOls = sp500RetOls.transpose()


# proportion of stocks with beta_market > 1
print('proportion of stocks with beta_market > 1: ', sp500RetOls['beta estimates'].loc[sp500RetOls['beta estimates'] > 1].count() / sp500RetOls.shape[0],'\n')

# highest and lowest beta
print('the highest observed beta is: ', sp500RetOls['beta estimates'].max(), '\n')
print('the lowest observed beta is: ', sp500RetOls['beta estimates'].min(), '\n')






























