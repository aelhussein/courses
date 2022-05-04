# -*- coding: utf-8 -*-
"""
Created on Mon May 10 14:17:57 2021

@author: Ahmed Elhussein
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
pd.set_option('display.max_columns', None)

df = pd.read_csv('1976-2020-president.csv')
df.head()
df_2020 = df[df['year'] == 2020]
df_2020_short = df_2020[['state', 'candidate','party_detailed','candidatevotes','totalvotes']]
candidates = ['BIDEN, JOSEPH R. JR', 'TRUMP, DONALD J.']
df_2020_BT = df_2020_short[df_2020_short.candidate.isin(candidates)]
df_2020_BT['vote_share'] = round(df_2020_BT['candidatevotes']/df_2020_BT['totalvotes']*100 ,2)
df_2020 = df_2020_BT[['state', 'party_detailed', 'vote_share']]
df_2020 = df_2020.set_index(['state', 'party_detailed'])


df_2016 = df[df['year'] == 2016]
df_2016_short = df_2016[['state', 'candidate','party_detailed','candidatevotes','totalvotes']]
candidates = ['CLINTON, HILLARY', 'TRUMP, DONALD J.']
df_2016_CT = df_2016_short[df_2016_short.candidate.isin(candidates)]
df_2016_CT['vote_share'] = round(df_2016_CT['candidatevotes']/df_2016_CT['totalvotes']*100 ,2)
df_2016 = df_2016_CT[['state', 'party_detailed', 'vote_share']]
df_2016 = df_2016.set_index(['state', 'party_detailed'])

df_2016_2020 = df_2016.merge(df_2020, left_index=True, right_index=True, suffixes = ('_2016', '_2020'))
df_2016_2020.reset_index(inplace = True)
ax = sns.lmplot(data = df_2016_2020, x = 'vote_share_2016', y = 'vote_share_2020', hue = 'party_detailed', fit_reg=False)
plt.plot((0,100),(0,100), color = 'r')


# =============================================================================
# Petal
# =============================================================================
#Wrangle data
df = pd.DataFrame(load_iris()['data'], columns = ['sepal length', 'sepal width', 'petal length', 'petal width'])
target = pd.Series(load_iris()['target'])
target_species = target.map({0:'setosa', 1:'versicolor', 2: 'virginica'})
df['target_species'] = target_species 

#Histogram
bins = int(np.sqrt(len(df['petal length'][target_species == 'versicolor'])))
sns.set()
_= plt.hist(df['petal length'][target_species == 'versicolor'], bins = bins)
_= plt.xlabel('petal length (cm)')
_= plt.ylabel('count')
plt.show()

#Swarmplot
_= sns.swarmplot(x = 'target_species', y= 'petal length', data = df)
_= plt.xlabel('target_species')
_= plt.ylabel('petal length')
plt.show()

#ECDF
def ECDF(data):
    n = len(data)
    x = np.sort(data)
    y = np.arange(1, n+1)/n
    return x, y

[df['target_species'].unique()]
x, y = ECDF(df['petal length'][target_species == 'versicolor'])
sns.set()
_= plt.plot(x,y, marker = '.', linestyle = 'none')
_= plt.xlabel('versicolor_petal_lenth')
_= plt.ylabel('ECDF')


#Summary stats
vers_petal_mean = df['petal length'][df['target_species']=='versicolor'].mean()

percentiles = np.array([2.5,25, 50, 75, 97.5])
ptiles_vers = np.percentile(df['petal length'][df['target_species']=='versicolor'], percentiles )

sns.set()
_= plt.plot(x,y, marker = '.', linestyle = 'none')
_= plt.xlabel('versicolor_petal_lenth')
_= plt.ylabel('ECDF')
_= plt.plot(ptiles_vers, percentiles/100, marker = 'D', color = 'red', linestyle = 'none')

sns.set()
_= sns.boxplot(data = df, x = 'target_species', y= 'petal length')
_= plt.xlabel('Species')
_= plt.ylabel('Petal length')
plt.show()

differences = df['petal length'][df['target_species']=='versicolor'] - vers_petal_mean
diff_sq = differences**2
diff_sq.mean()
np.var(df['petal length'][df['target_species']=='versicolor'])
np.std(df['petal length'][df['target_species']=='versicolor'])

df_vers = df[df['target_species']=='versicolor']

sns.set()
_= plt.plot('petal length', 'sepal length', data = df_vers, linestyle = 'none', marker = '.')
_= plt.xlabel('Petal length')
_=plt.ylabel('Sepal length')
plt.show()

np.cov(df_vers['petal length'], df_vers['petal width'])

def pearson_r(x,y):
    corr_mat = np.corrcoef(x,y)
    return corr_mat[0,1]
pearson_r(df_vers['petal length'], df_vers['petal width'])
   
#Simulation
def heads_sim(x,y):
    n_flips = 0
    tries=0
    for i in range(x):
        flips = np.random.random(y)
        tries +=1
        if np.all(flips >= 0.5):
            n_flips +=1
    return round(n_flips/tries,3)

heads_sim(1000,4)

random = np.random.random(1000)
sns.set()
_=plt.hist(random)

def perform_bernoulli_trials(n,p):
    n_success = 0
    tries = 0
    for i in range(n):
        random_num = np.random.random()
        tries +=1
        if random_num <= p:
            n_success+=1
    return round(n_success/tries,3)

np.random.seed(42)
sims = np.empty(1000)
for i in range(1000):
 sims[i] = perform_bernoulli_trials(100,0.05)
np.mean(sims)
_=plt.hist(sims)

x,y = ECDF(sims*100)
_ = plt.plot(x,y, marker = '.', linestyle = 'none')
_= plt.xlabel('defualts')
_= plt.ylabel('probability')
plt.show()

np.sum(sims>=0.1)/len(sims)

n_defaults = np.random.binomial(n = 100, p = 0.05, size = 10000)
bins = np.arange(0, max(n_defaults)+1)
_=plt.hist(n_defaults, bins = bins, normed = True)
x,y = ECDF(n_defaults)
_ = plt.plot(x,y, marker = '.', linestyle = 'none')
_= plt.xlabel('defualts')
_= plt.ylabel('probability')
plt.show()

#Poisson
samples_poisson  = np.random.poisson(10, size = 10000)
samples_poisson.std(), samples_poisson.mean()

n = [20,100,1000]
p = [0.5, 0.1, 0.01]
for i in range(3):
    samples_binomial = np.random.binomial(n[i],p[i], size = 10000)
    print(n[i], samples_binomial.mean(), samples_binomial.std())
    
n_nohitters = np.random.poisson(251/115, size = 10000)
n_large = np.sum(n_nohitters >= 7)
p_large = n_large/10000

#Normal distribution
samples_1 = np.random.normal(20,1, size = 10000)
samples_2 = np.random.normal(20,3, size = 10000)
samples_3 = np.random.normal(20,10,  size = 10000)

sns.set()
_=plt.hist(samples_1, normed = True, bins = 100, histtype = 'step')
_=plt.hist(samples_2, normed = True, bins = 100, histtype = 'step')
_=plt.hist(samples_3, normed = True, bins = 100, histtype = 'step')
_= plt.legend(['std=1','std=2', 'std=3'])
plt.ylim(-0.01,0.42)
plt.show()

x_std1, y_std1 = ECDF(samples_1)
x_std2, y_std2 = ECDF(samples_2)
x_std3, y_std3 = ECDF(samples_3)

_= plt.plot(x_std1, y_std1, marker = '.', linestyle = 'none')
_= plt.plot(x_std2, y_std2, marker = '.', linestyle = 'none')
_= plt.plot(x_std3, y_std3, marker = '.', linestyle = 'none')
_= plt.legend(['std=1','std=2', 'std=3'])
_=plt.xlabel('X')
_= plt.ylabel('Probability')
plt.show()

belmont_no_outliers = np.array([148.51, 146.65, 148.52, 150.7 , 150.42, 150.88, 151.57, 147.54,
       149.65, 148.74, 147.86, 148.75, 147.5 , 148.26, 149.71, 146.56,
       151.19, 147.88, 149.16, 148.82, 148.96, 152.02, 146.82, 149.97,
       146.13, 148.1 , 147.2 , 146.  , 146.4 , 148.2 , 149.8 , 147.  ,
       147.2 , 147.8 , 148.2 , 149.  , 149.8 , 148.6 , 146.8 , 149.6 ,
       149.  , 148.2 , 149.2 , 148.  , 150.4 , 148.8 , 147.2 , 148.8 ,
       149.6 , 148.4 , 148.4 , 150.2 , 148.8 , 149.2 , 149.2 , 148.4 ,
       150.2 , 146.6 , 149.8 , 149.  , 150.8 , 148.6 , 150.2 , 149.  ,
       148.6 , 150.2 , 148.2 , 149.4 , 150.8 , 150.2 , 152.2 , 148.2 ,
       149.2 , 151.  , 149.6 , 149.6 , 149.4 , 148.6 , 150.  , 150.6 ,
       149.2 , 152.6 , 152.8 , 149.6 , 151.6 , 152.8 , 153.2 , 152.4 ,
       152.2 ])

mu = np.mean(belmont_no_outliers)
sigma = np.std(belmont_no_outliers)

samples = np.random.normal(mu, sigma, size = 10000)
x_theor, y_theor = ECDF(samples)
x,y = ECDF(belmont_no_outliers)
_=plt.plot(x_theor, y_theor)
_=plt.plot(x,y, marker = '.', linestyle = 'none')
_ = plt.xlabel('Belmont winning time (sec.)')
_ = plt.ylabel('CDF')
plt.show()

prob = np.mean(samples<144)

#Exponential distribution
def successive_poisson(tau1, tau2, size = 1):
    t1 = np.random.exponential(tau1, size = size)
    t2 = np.random.exponential(tau2,size = size)
    return t1+t2

waiting_times = successive_poisson(764,715, size = 10000)
_= plt.hist(waiting_times, bins = 100, normed = True, histtype = 'step')
_ = plt.xlabel('Waiting times')
_ = plt.ylabel('PDF')
plt.show()





