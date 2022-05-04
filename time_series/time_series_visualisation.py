# -*- coding: utf-8 -*-
"""
Created on Fri May 28 10:16:45 2021

@author: Ahmed Elhussein
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics import tsaplots
import statsmodels.api as sm
from pylab import rcParams

df = pd.read_csv('ch1_discoveries.csv', parse_dates = ['date'], index_col = 'date')
df.head(5)
df.dtypes

sns.set()
plt.style.use('fivethirtyeight')
ax = df.plot(figsize =(12,5), linewidth = 2)
ax.set_xlabel('Date')
ax.set_ylabel('Number of great discoveries')
ax.set_title('Title')
ax.axvline(x = '1949-01-01', color = 'red')
ax.axhline(y = 6, color = 'green')
ax.axvspan('1939','1945', alpha = 0.2, color = 'red')
plt.show()


df = pd.read_csv('ch2_co2_levels.csv', parse_dates = ['datestamp'], index_col = 'datestamp')
df.head()
df.isnull().sum()
df = df.fillna(method = 'bfill')

co2_levels_mean = df.rolling(window = 52).mean()
ax = co2_levels_mean.plot()
ax.set_xlabel('Date')
ax.set_ylabel('CO2 levels')
ax.set_title('Rolling average')
plt.show()


co2_levels_month = df.groupby(df.index.month).mean()
co2_levels_month.plot()

df.boxplot()
plt.show()
df.plot(kind = 'hist')
df.plot(kind = 'density')

fig, ax = plt.subplots(2,1)
ax[0] = tsaplots.plot_acf(df['co2'], lags = 40)
ax[1] = tsaplots.plot_pacf(df['co2'], lags = 40)
fig.show()

rcParams['figure.figsize'] = 11,9
decomposition = sm.tsa.seasonal_decompose(df['co2'])
fig = decomposition.plot()
dir(decomposition)


df = pd.read_csv('ch3_airline_passengers.csv', parse_dates = ['Month'], index_col = 'Month')
df.head()
df.plot()

df = pd.read_csv('ch4_meat.csv', parse_dates = ['date'], index_col = 'date')
df.head()
plt.style.use('ggplot')
ax = df.plot(figsize = (12,4), fontsize = 14, colormap = 'Dark2')
df_summary = df.describe()
ax.table(cellText = df_summary.values,
         colWidths = [0.3]*len(df.columns),
         rowLabels = df_summary.index,
         colLabels = df_summary.columns,
         loc = 'top')
plt.show()
df.plot(subplots = True, sharex = False, sharey = False, layout = (2,4), figsize = (16,10))


corr_mat = df.corr(method = 'pearson')
sns.heatmap(corr_mat, annot = True, annot_kws= {'size':10})
plt.yticks(rotation = 0)
plt.xticks(rotation = 90)
sns.clustermap(corr_mat, annot = True)

df = pd.read_csv('ch5_employment.csv', parse_dates = ['datestamp'], index_col = 'datestamp')
df.head()
df.dtypes
df.isnull().sum()

df.boxplot(fontsize = 6, vert = False)
print(df.describe())

ax = df.plot(colormap = 'Dark2')
ax.axvline('2008-01-01', linestyle = '--', color = 'black')
plt.show()
df.index
df_month = df.groupby(df.index.month).mean()
ax = df_month.plot()
plt.legend(loc = 'best')
plt.show()

my_dict = {}
for ts in df.columns:
    my_dict[ts] = sm.tsa.seasonal_decompose(df[ts])

my_dict_trend = {}
for i in my_dict:
    my_dict_trend[i] = my_dict[i].trend
    
pd.DataFrame(my_dict_trend)
