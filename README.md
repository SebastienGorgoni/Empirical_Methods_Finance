# Empirical Methods for Finance

Spring 2021 - HEC Lausanne

![alt text](https://camo.githubusercontent.com/c327657381291ed9f2e8866cb96ac4861431d9c244b7b14dcf4e1470cbf632da/68747470733a2f2f75706c6f61642e77696b696d656469612e6f72672f77696b6970656469612f636f6d6d6f6e732f7468756d622f612f61332f4845435f4c617573616e6e655f6c6f676f2e7376672f32393370782d4845435f4c617573616e6e655f6c6f676f2e7376672e706e67)

## Introduction

This repository contains all codes, papers and data for the projects achieved as part of the course "Empirical Methods for Finance".

## Project 1

This report aims to analyze the stock performance of the S&P500 index’s constituents. We will first briefly explain what the S&P500 stock market index is, and how its con- stituents enter and leave the index. Then, we will analyze the skewness and kurtosis of the individual companies to get a better idea of their distribution characteristics. Next, we will conduct various hypothesis tests. starting with the Jarque & Bera test and the Kolmogorov & Smirnov test to check for normality of distribution followed by the Ljung Box test to check for potential auto-correlations. Finally, we will run a regression on each individual stock, with the return on the market portfolio as the independent variable. Finally, we will briefly explain the significance of these empirical results in the context of investment strategies.

Our data set includes the prices of 452 out of the 500 constituents of the S&P500 index, as well as the value of the index itself, over an unspecified period of time, indexed at 100 in time 0. The data processing and analysis was conducted with Python, using various li- braries.

## Project 2


This report is divided into two parts. The first part aims to manipulate times series of various crypto-currencies (i.e. Bitcoin, Ethereum and Monero). We will first test for stationarity using the Dickey Fuller test, then check for co-integration among our crypto-currencies, and finally try to implement a pair-trading strategy using the re- sults found previously. The second part will aim to test again the stationarity of four new time series (i.e. US Dollar in trade weighted terms, Gold, the level of VIX index and Bitcoin), using the Augmented Dickey Fuller test and the Kwiatkowski Phillips Schmidt Shin (KPSS) test. After transforming the non-stationary series into stationary ones, we will try to build a Vector Auto-Regressive (VAR) model, by deciding the num- ber of lags using various information criteria model. Thereafter, we will determine the Impulse Response Function between our assets. Finally, we will estimate an uncon- strained and constrained VAR(1) model, and check whether there are any statistical differences between them. All the implementations will be done using Python.

## Project 3


This report consists of two different parts. In the first part, we discuss three different types of GARCH models (GARCH, EGARCH and GARCH-GJR) to estimate the time varying volatility of the log-returns of three different time series (S&P 500, BCOM Commodity index and Apple).
GARCH models allow us to perform a time-varying volatility estimation, as well as to take into account persistence in volatility. These are particularly interesting features of volatility that are present in empirical data. In addition to the standard GARCH model, we are using two extension, EGARCH and GARCH-GJR which allow us to take into account the asymmetry of volatility. In the second part, we are using the famous Heston and Nandi model to price call options on the S&P 500. This model allows us to take into account time varying volatility in the pricing of the calls. Using a time varying volatility model is important when it comes to option pricing because we know that empirically, volatility is indeed time varying. Thus, this kind of models offers a more precise estimation of the price of options than standards model like Black-Scholes or the Binomial Model who assume constant volatility.

In the first part of the report, we focus on an equity index, a commodity index, and a single stock. For all of them, we collect daily data of the last 10 years. In the second part of the assignment, we focus on the S&P500 and expand the time span as we collect the price series (daily) from 1989 until 2021. 
