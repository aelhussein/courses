# -*- coding: utf-8 -*-
"""
Created on Sat May 22 08:46:59 2021

@author: Ahmed Elhussein
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf#
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.arima_model import ARMA
import datetime

DJI = pd.read_csv('DJI.csv', parse_dates = ['Date'], index_col = 'Date' )
UFO = pd.read_csv('UFO.csv', parse_dates = ['Date'], index_col = 'Date')
levels = pd.merge(DJI, UFO, how  = 'inner', left_index = True, right_index = True)
levels.columns = ['DJI', 'UFO']
levels.head()
correlation1 = levels['DJI'].corr(levels['UFO'])
changes= levels.pct_change()
correlation2 = changes['DJI'].corr(changes['UFO'])


levels2 = sm.add_constant(levels)
results = sm.OLS(levels2['DJI'], levels2[['const','UFO']]).fit()
results.summary()

MSFT =pd.read_csv('MSFT.csv', parse_dates = ['Date'], index_col = 'Date')
MSFT.head()
MSFT = MSFT.resample(how = 'last', rule = 'W')
returns = MSFT.pct_change()
returns['Adj Close'].autocorr()

MSFT =pd.read_csv('MSFT.csv', parse_dates = ['Date'], index_col = 'Date')
MSFT_diff = MSFT.diff()
MSFT_diff['Adj Close'].autocorr()
MSFT = MSFT.resample(rule = 'W').last()
MSFT_diff = MSFT.diff()
MSFT_diff['Adj Close'].autocorr()

plot_acf(MSFT, lags = 20, alpha = 0.05)

HRB = pd.read_csv('HRB.csv', parse_dates = ['Quarter'], index_col = 'Quarter')
HRB.head()
acf_array = acf(HRB)
acf_array
plot_acf(HRB, lags = 20, alpha = 0.05)

MSFT =pd.read_csv('MSFT.csv', parse_dates = ['Date'], index_col = 'Date')
MSFT['Adj Close'].autocorr()
nobs = len(MSFT)
conf = 1.96/np.sqrt(nobs)
plot_acf(MSFT, alpha = 0.05, lags = 20)

returns = np.random.normal(loc = 0.02, scale =0.05, size = 1000 )
plot_acf(returns, lags = 20)


steps = np.random.normal(loc = 0, scale = 1, size = 500)
steps[0]=0
P = 100 + np.cumsum(steps)
sns.set()
plt.plot(P)
plt.show()

steps = np.random.normal(loc = 0.1/100, scale = 1/100, size = 500) + 1
steps[0]=1
P = 100*np.cumprod(steps)
plt.plot(P)

AMZN = pd.read_csv('AMZN.csv', parse_dates = ['Date'], index_col = 'Date')
AMZN.head()
results = adfuller(AMZN['Adj Close'])
results[0]
results[1]

AMZN_ret = AMZN.pct_change()
AMZN_ret = AMZN_ret.dropna()
adfuller(AMZN_ret['Adj Close'])

HRB = pd.read_csv('HRB.csv', parse_dates = ['Quarter'], index_col = 'Quarter')
HRBsa = HRB.diff(4)
HRBsa = HRBsa.dropna()
plot_acf(HRBsa)

ar = np.array([1,-0.1])
ma = np.array([1])
AR_object = ArmaProcess(ar,ma)
simulated_Data = AR_object.generate_sample(nsample=1000)
plt.plot(simulated_Data)

fig, ax = plt.subplots(2,1)
ar = np.array([1,-0.9])
ma = np.array([1])
AR_object = ArmaProcess(ar,ma)
simulated_Data = AR_object.generate_sample(nsample=1000)
ax[0] = plt.plot(simulated_Data)

ar = np.array([1,0.9])
ma = np.array([1])
AR_object = ArmaProcess(ar,ma)
simulated_Data = AR_object.generate_sample(nsample=1000)
ax[1] = plt.plot(simulated_Data)
plt.show()


mod = ARMA(simulated_Data, order = (1,0))
result = mod.fit()
result.summary()
result.plot_predict(start = 900, end = 1100)
plt.show()
result.params

DJI = pd.read_csv('DJI.csv', parse_dates = ['Date'], index_col = 'Date')
DJI.head()
result = ARMA(DJI, order = (1,0))
res = result.fit()
res.summary()
res.plot_predict(start = '1980', end = '2015')

fig, axes = plt.subplots(2,1)
fig = plot_acf(DJI, alpha = 1, lags = 12, ax = axes[0])
fig = plot_acf(simulated_Data, alpha = 1, lags = 12, ax = axes[1])
axes[0].set_title('DJI')
axes[1].set_title('Simulated Data')
plt.show()


plot_acf(simulated_Data, lags = 12, alpha = 0.05)


ma = np.array([1])
ar = np.array([1,-0.6])
AR_object = ArmaProcess(ar,ma)
simulated_data_1 = AR_object.generate_sample(nsample = 5000)
plot_pacf(simulated_data_1, lags = 20)

ma = np.array([1])
ar = np.array([1,-0.6, -0.3])
AR_object = ArmaProcess(ar,ma)
simulated_data_2 = AR_object.generate_sample(nsample = 5000)
plot_pacf(simulated_data_2, lags = 20)

BIC = np.zeros((6,1))
for i in range(0,len(BIC)):
    mod = ARMA(simulated_data_2, order= (i+1,0))
    result = mod.fit()
    BIC[i] = result.bic
sns.set()
plt.plot(range(1,7), BIC, marker = 'o')
plt.xlabel('Order')
plt.ylabel('BIC')
plt.show()


ma = np.array([1,0.9])
ar = np.array([1])
AR_object = ArmaProcess(ar,ma)
simulated_data_1 = AR_object.generate_sample(nsample = 5000)
plot_acf(simulated_data_1, lags = 20)

mod = ARMA(simulated_data_1, order = (0,1))
res = mod.fit()
res.summary()


intraday = pd.read_csv('Sprint_Intraday.txt', header = None)
intraday.head()
intraday = intraday.iloc[:,0:2]
intraday.columns = ['DATE','CLOSE']
intraday.iloc[0,0] = 0
intraday.dtypes
intraday['DATE'] = pd.to_numeric(intraday['DATE'])
intraday.set_index('DATE', inplace = True)

set_check = set(range(391))
set_intraday = set(intraday.index)
set_missing = set_check - set_intraday
intraday = intraday.reindex(range(391), method = 'ffill')
intraday.index = pd.date_range(start = '2017-09-01 9:30', end = '2017-09-01 16:00', freq = '1min')
intraday.head()
intraday.plot(grid = True)
plt.show()

returns = intraday.pct_change()
returns = returns.dropna()
plot_acf(returns, lags = 20)

mod = ARMA(returns, order = (0,1))
res = mod.fit()
res.params


ma = [0.8**i for i in range(30)]
ar = np.array([1])
AR_object = ArmaProcess(ar,ma)
simulated_data = AR_object.generate_sample(nsample = 5000)
plot_acf(simulated_data, lags = 30)

HO = pd.read_csv('CME_HO1.csv', parse_dates = ['Date'], index_col = 'Date')
HO.head()
NG = pd.read_csv('CME_ng1.csv', parse_dates = ['Date'], index_col = 'Date')
NG.head()
fig, ax = plt.subplots(2,1)
ax[0] = plt.plot(7.25*HO, label = 'Heating oil')
ax[0] = plt.plot(NG, label = 'Natural Gas')
plt.legend(loc = 'best', fontsize = 'small')

ax[1] = plt.plot(7.25*HO-NG, label = 'Spread')
plt.legend(loc = 'best', fontsize = 'small')
plt.axhline(y=0, linestyle = '--', color = 'k')
plt.show()


result_HO = adfuller(HO['Close'])
result_NG = adfuller(NG['Close'])
result_HO[1]
result_NG[1]

result_spread = adfuller(7.25*HO['Close'] - NG['Close'])
result_spread[1]

AMZN.head()
MSFT.head()
AMZN_1 = AMZN.loc['2012-08-6':]
MSFT_1 = MSFT.loc[:'2017-08-02']
MSFT_1 = sm.add_constant(MSFT_1)
result = sm.OLS(AMZN_1, MSFT_1).fit()
b = result.params[1]
adf_stats = adfuller(AMZN_1['Adj Close'] - b*MSFT_1['Adj Close'])
adf_stats[1]

temp_NY = pd.read_csv('NOAA_TAVG.csv', parse_dates = ['DATE'], index_col = 'DATE')
temp_NY.head()
temp_NY.index = pd.to_datetime(temp_NY.index, format = '%Y')
temp_NY.plot()
plt.show()
result= adfuller(temp_NY['TAVG'])
result[1]

chg_temp = temp_NY.diff()
chg_temp = chg_temp.dropna()

fig, axes = plt.subplots(2,1)
plot_acf(chg_temp, lags =20,  ax = axes[0])
plot_pacf(chg_temp, lags = 20, ax = axes[1])
plt.show()


mod_ar1 = ARMA(chg_temp, order = (1,0))
res_ar1 = mod_ar1.fit()
res_ar1.aic

mod_ar2 = ARMA(chg_temp, order = (2,0))
res_ar2 = mod_ar2.fit()
res_ar2.aic


mod_ar11 = ARMA(chg_temp, order = (1,1))
res_ar11 = mod_ar11.fit()
res_ar11.aic


mod = ARMA(temp_NY, order = (1,1,1))
res = mod.fit()
res.plot_predict(start='1872-01-01', end='2046-01-01')
plt.show()





































