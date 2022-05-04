# -*- coding: utf-8 -*-
"""
Created on Tue May 11 12:44:47 2021

@author: Ahmed Elhussein
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def ECDF(data):
    n = len(data)
    x = np.sort(data)
    y = np.arange(1,n+1)/n
    return x, y

def pearson_r(x,y):
    corr_mat = np.corrcoef(x,y)
    return corr_mat[0,1]

np.random.seet(42)

no_hitters = np.array([ 843, 1613, 1101,  215,  684,  814,  278,  324,  161,  219,  545,
        715,  966,  624,   29,  450,  107,   20,   91, 1325,  124, 1468,
        104, 1309,  429,   62, 1878, 1104,  123,  251,   93,  188,  983,
        166,   96,  702,   23,  524,   26,  299,   59,   39,   12,    2,
        308, 1114,  813,  887,  645, 2088,   42, 2090,   11,  886, 1665,
       1084, 2900, 2432,  750, 4021, 1070, 1765, 1322,   26,  548, 1525,
         77, 2181, 2752,  127, 2147,  211,   41, 1575,  151,  479,  697,
        557, 2267,  542,  392,   73,  603,  233,  255,  528,  397, 1529,
       1023, 1194,  462,  583,   37,  943,  996,  480, 1497,  717,  224,
        219, 1531,  498,   44,  288,  267,  600,   52,  269, 1086,  386,
        176, 2199,  216,   54,  675, 1243,  463,  650,  171,  327,  110,
        774,  509,    8,  197,  136,   12, 1124,   64,  380,  811,  232,
        192,  731,  715,  226,  605,  539, 1491,  323,  240,  179,  702,
        156,   82, 1397,  354,  778,  603, 1001,  385,  986,  203,  149,
        576,  445,  180, 1403,  252,  675, 1351, 2983, 1568,   45,  899,
       3260, 1025,   31,  100, 2055, 4043,   79,  238, 3931, 2351,  595,
        110,  215,    0,  563,  206,  660,  242,  577,  179,  157,  192,
        192, 1848,  792, 1693,   55,  388,  225, 1134, 1172, 1555,   31,
       1582, 1044,  378, 1687, 2915,  280,  765, 2819,  511, 1521,  745,
       2491,  580, 2072, 6450,  578,  745, 1075, 1103, 1549, 1520,  138,
       1202,  296,  277,  351,  391,  950,  459,   62, 1056, 1128,  139,
        420,   87,   71,  814,  603, 1349,  162, 1027,  783,  326,  101,
        876,  381,  905,  156,  419,  239,  119,  129,  467])

tau = np.mean(no_hitters)
inter_nohitter_time = np.random.exponential(tau, size = 10000)

sns.set()
_= plt.hist(inter_nohitter_time, bins = 100, normed = True, histtype = 'step')
_= plt.xlabel('no hitter time')
_= plt.ylabel('PDF')
plt.show()

x,y = ECDF(no_hitters)
x_theor, y_theor = ECDF(inter_nohitter_time)
_=plt.plot(x,y, linestyle = 'none',marker = '.')
_=plt.plot(x_theor,y_theor)
_= plt.xlabel('no hitter time')
_= plt.ylabel('CDF')
plt.margins(0.02)
plt.show()

#Linear regression
illiteracy = np.array([ 9.5, 49.2,  1. , 11.2,  9.8, 60. , 50.2, 51.2,  0.6,  1. ,  8.5,
        6.1,  9.8,  1. , 42.2, 77.2, 18.7, 22.8,  8.5, 43.9,  1. ,  1. ,
        1.5, 10.8, 11.9,  3.4,  0.4,  3.1,  6.6, 33.7, 40.4,  2.3, 17.2,
        0.7, 36.1,  1. , 33.2, 55.9, 30.8, 87.4, 15.4, 54.6,  5.1,  1.1,
       10.2, 19.8,  0. , 40.7, 57.2, 59.9,  3.1, 55.7, 22.8, 10.9, 34.7,
       32.2, 43. ,  1.3,  1. ,  0.5, 78.4, 34.2, 84.9, 29.1, 31.3, 18.3,
       81.8, 39. , 11.2, 67. ,  4.1,  0.2, 78.1,  1. ,  7.1,  1. , 29. ,
        1.1, 11.7, 73.6, 33.9, 14. ,  0.3,  1. ,  0.8, 71.9, 40.1,  1. ,
        2.1,  3.8, 16.5,  4.1,  0.5, 44.4, 46.3, 18.7,  6.5, 36.8, 18.6,
       11.1, 22.1, 71.1,  1. ,  0. ,  0.9,  0.7, 45.5,  8.4,  0. ,  3.8,
        8.5,  2. ,  1. , 58.9,  0.3,  1. , 14. , 47. ,  4.1,  2.2,  7.2,
        0.3,  1.5, 50.5,  1.3,  0.6, 19.1,  6.9,  9.2,  2.2,  0.2, 12.3,
        4.9,  4.6,  0.3, 16.5, 65.7, 63.5, 16.8,  0.2,  1.8,  9.6, 15.2,
       14.4,  3.3, 10.6, 61.3, 10.9, 32.2,  9.3, 11.6, 20.7,  6.5,  6.7,
        3.5,  1. ,  1.6, 20.5,  1.5, 16.7,  2. ,  0.9])
    
fertility = np.array([1.769, 2.682, 2.077, 2.132, 1.827, 3.872, 2.288, 5.173, 1.393,
       1.262, 2.156, 3.026, 2.033, 1.324, 2.816, 5.211, 2.1  , 1.781,
       1.822, 5.908, 1.881, 1.852, 1.39 , 2.281, 2.505, 1.224, 1.361,
       1.468, 2.404, 5.52 , 4.058, 2.223, 4.859, 1.267, 2.342, 1.579,
       6.254, 2.334, 3.961, 6.505, 2.53 , 2.823, 2.498, 2.248, 2.508,
       3.04 , 1.854, 4.22 , 5.1  , 4.967, 1.325, 4.514, 3.173, 2.308,
       4.62 , 4.541, 5.637, 1.926, 1.747, 2.294, 5.841, 5.455, 7.069,
       2.859, 4.018, 2.513, 5.405, 5.737, 3.363, 4.89 , 1.385, 1.505,
       6.081, 1.784, 1.378, 1.45 , 1.841, 1.37 , 2.612, 5.329, 5.33 ,
       3.371, 1.281, 1.871, 2.153, 5.378, 4.45 , 1.46 , 1.436, 1.612,
       3.19 , 2.752, 3.35 , 4.01 , 4.166, 2.642, 2.977, 3.415, 2.295,
       3.019, 2.683, 5.165, 1.849, 1.836, 2.518, 2.43 , 4.528, 1.263,
       1.885, 1.943, 1.899, 1.442, 1.953, 4.697, 1.582, 2.025, 1.841,
       5.011, 1.212, 1.502, 2.516, 1.367, 2.089, 4.388, 1.854, 1.748,
       2.978, 2.152, 2.362, 1.988, 1.426, 3.29 , 3.264, 1.436, 1.393,
       2.822, 4.969, 5.659, 3.24 , 1.693, 1.647, 2.36 , 1.792, 3.45 ,
       1.516, 2.233, 2.563, 5.283, 3.885, 0.966, 2.373, 2.663, 1.251,
       2.052, 3.371, 2.093, 2.   , 3.883, 3.852, 3.718, 1.732, 3.928])
    
_=plt.plot(illiteracy, fertility, marker = '.', linestyle = 'none')
plt.margins(0.02)
_ = plt.xlabel('percent illiterate')
_ = plt.ylabel('fertility')
plt.show()

pearson_r(illiteracy, fertility)

a, b = np.polyfit(illiteracy, fertility, deg = 1)
x = np.array([0,100])
y = a*x + b
_= plt.plot(x,y)
_=plt.plot(illiteracy, fertility, marker = '.', linestyle = 'none')
_=plt.margins(0.02)
_ = plt.xlabel('percent illiterate')
_ = plt.ylabel('fertility')
plt.show()

a_vals = np.linspace(0,0.1,200)
rss = np.empty_like(a_vals)
for i, a in enumerate(a_vals):
    rss[i] = np.sum((illiteracy*a-fertility+b)**2)
plt.plot(a_vals, rss, '-')
plt.xlabel('slope (children per woman / percent illiterate)')
plt.ylabel('sum of square of residuals')

plt.show()

anscombe_x = [np.array([10.,  8., 13.,  9., 11., 14.,  6.,  4., 12.,  7.,  5.]),
 np.array([10.,  8., 13.,  9., 11., 14.,  6.,  4., 12.,  7.,  5.]),
 np.array([10.,  8., 13.,  9., 11., 14.,  6.,  4., 12.,  7.,  5.]),
 np.array([ 8.,  8.,  8.,  8.,  8.,  8.,  8., 19.,  8.,  8.,  8.])]

anscombe_y = [np.array([ 8.04,  6.95,  7.58,  8.81,  8.33,  9.96,  7.24,  4.26, 10.84,
         4.82,  5.68]),
 np.array([9.14, 8.14, 8.74, 8.77, 9.26, 8.1 , 6.13, 3.1 , 9.13, 7.26, 4.74]),
 np.array([ 7.46,  6.77, 12.74,  7.11,  7.81,  8.84,  6.08,  5.39,  8.15,
         6.42,  5.73]),
 np.array([ 6.58,  5.76,  7.71,  8.84,  8.47,  7.04,  5.25, 12.5 ,  5.56,
         7.91,  6.89])]

for x,y in zip(anscombe_x, anscombe_y):
    a,b, = np.polyfit(x,y, deg = 1)
    print(a,b)

#Bootstrap
rainfall = np.array([ 875.5,  648.2,  788.1,  940.3,  491.1,  743.5,  730.1,  686.5,
        878.8,  865.6,  654.9,  831.5,  798.1,  681.8,  743.8,  689.1,
        752.1,  837.2,  710.6,  749.2,  967.1,  701.2,  619. ,  747.6,
        803.4,  645.6,  804.1,  787.4,  646.8,  997.1,  774. ,  734.5,
        835. ,  840.7,  659.6,  828.3,  909.7,  856.9,  578.3,  904.2,
        883.9,  740.1,  773.9,  741.4,  866.8,  871.1,  712.5,  919.2,
        927.9,  809.4,  633.8,  626.8,  871.3,  774.3,  898.8,  789.6,
        936.3,  765.4,  882.1,  681.1,  661.3,  847.9,  683.9,  985.7,
        771.1,  736.6,  713.2,  774.5,  937.7,  694.5,  598.2,  983.8,
        700.2,  901.3,  733.5,  964.4,  609.3, 1035.2,  718. ,  688.6,
        736.8,  643.3, 1038.5,  969. ,  802.7,  876.6,  944.7,  786.6,
        770.4,  808.6,  761.3,  774.2,  559.3,  674.2,  883.6,  823.9,
        960.4,  877.8,  940.6,  831.8,  906.2,  866.5,  674.1,  998.1,
        789.3,  915. ,  737.1,  763. ,  666.7,  824.5,  913.8,  905.1,
        667.8,  747.4,  784.7,  925.4,  880.2, 1086.9,  764.4, 1050.1,
        595.2,  855.2,  726.9,  785.2,  948.8,  970.6,  896. ,  618.4,
        572.4, 1146.4,  728.2,  864.2,  793. ])

for i in range(50):    
    bs_samples = np.random.choice(rainfall, size = len(rainfall))
    x,y = ECDF(bs_samples)
    _= plt.plot(x,y, linestyle = 'none', color = 'gray', alpha = 0.1, marker = '.')
x,y = ECDF(rainfall)
_=plt.plot(x,y, marker = '.')
plt.margins(0.02)
_ = plt.xlabel('yearly rainfall (mm)')
_ = plt.ylabel('ECDF')
plt.show()

def bootstrap_replicate_1d(data, func):
    bs_sample = np.random.choice(data, size = len(data))
    return func(bs_sample)

def draw_bs_reps(data, func, size =1):
    bs_replicates = np.empty(size)
    for i in range(size):
        bs_replicates[i] = bootstrap_replicate_1d(data,func)
    return bs_replicates

rainfall_bs = draw_bs_reps(rainfall, np.mean, size = 10000)
_=plt.hist(rainfall_bs, bins = 100, normed = True, histtype = "step")
_ = plt.xlabel('annual yearly rainfall (mm)')
_ = plt.ylabel('PDF')
plt.show()

np.std(rainfall_bs)
np.std(rainfall)/np.sqrt(len(rainfall))
np.percentile(rainfall_bs, [2.5,97.5])

rainfall_bs = draw_bs_reps(rainfall, np.var, size = 10000)
_=plt.hist(rainfall_bs/100, bins = 100, normed = True, histtype = "step")
_ = plt.xlabel('var yearly rainfall (mm)')
_ = plt.ylabel('PDF')
plt.show()

no_hitters_bs = draw_bs_reps(no_hitters, np.mean, size = 10000)
np.percentile(no_hitters_bs, [2.5,97.5])
_=plt.hist(no_hitters_bs, bins = 100, normed = True, histtype = "step")
_ = plt.xlabel('annual yearly rainfall (mm)')
_ = plt.ylabel('PDF')
plt.show()


#Boostrap pairs
def draw_bs_pairs_linreg(x,y, size = 1):
    ind = np.arange(0,len(x))
    slope = np.empty(size)
    intercept = np.empty(size)
    for i in range(size):
        c_ind = np.random.choice(ind, size = len(x))
        x_bs = x[c_ind]
        y_bs = y[c_ind]
        slope[i],intercept[i] = np.polyfit(x_bs, y_bs, deg = 1)
    return slope, intercept
    
fertility_slope_bs,fertility_intercept_bs = draw_bs_pairs_linreg(illiteracy,fertility, size = 10000)
sns.set()
_=plt.hist(fertility_slope_bs, bins = 50, normed = True)
plt.xlabel('Slope')
plt.ylabel('Count')
plt.show()
np.percentile(fertility_slope_bs, [2.5,97.5])

x = np.array([0,100])
for i in range(100):
    y = x*fertility_slope_bs[i] + fertility_intercept_bs[i]
    _=plt.plot(x,y, color = 'grey', alpha = 0.1)
_=plt.plot(illiteracy, fertility, marker = '.', linestyle = 'none')
plt.show()


#Hypothesis testing
def permutation_sample(data1, data2):
    joined = np.concatenate((data1,data2))
    joined = np.random.permutation(joined)
    perm_data1 = joined[:len(data1)]
    perm_data2 = joined[len(data1):]
    return perm_data1, perm_data2

rain_june = np.array([ 66.2,  39.7,  76.4,  26.5,  11.2,  61.8,   6.1,  48.4,  89.2,
       104. ,  34. ,  60.6,  57.1,  79.1,  90.9,  32.3,  63.8,  78.2,
        27.5,  43.4,  30.1,  17.3,  77.5,  44.9,  92.2,  39.6,  79.4,
        66.1,  53.5,  98.5,  20.8,  55.5,  39.6,  56. ,  65.1,  14.8,
        13.2,  88.1,   8.4,  32.1,  19.6,  40.4,   2.2,  77.5, 105.4,
        77.2,  38. ,  27.1, 111.8,  17.2,  26.7,  23.3,  77.2,  87.2,
        27.7,  50.6,  60.3,  15.1,   6. ,  29.4,  39.3,  56.3,  80.4,
        85.3,  68.4,  72.5,  13.3,  28.4,  14.7,  37.4,  49.5,  57.2,
        85.9,  82.1,  31.8, 126.6,  30.7,  41.4,  33.9,  13.5,  99.1,
        70.2,  91.8,  61.3,  13.7,  54.9,  62.5,  24.2,  69.4,  83.1,
        44. ,  48.5,  11.9,  16.6,  66.4,  90. ,  34.9, 132.8,  33.4,
       225. ,   7.6,  40.9,  76.5,  48. , 140. ,  55.9,  54.1,  46.4,
        68.6,  52.2, 108.3,  14.6,  11.3,  29.8, 130.9, 152.4,  61. ,
        46.6,  43.9,  30.9, 111.1,  68.5,  42.2,   9.8, 285.6,  56.7,
       168.2,  41.2,  47.8, 166.6,  37.8,  45.4,  43.2])
rain_nov = np.array([ 83.6,  30.9,  62.2,  37. ,  41. , 160.2,  18.2, 122.4,  71.3,
        44.2,  49.1,  37.6, 114.5,  28.8,  82.5,  71.9,  50.7,  67.7,
       112. ,  63.6,  42.8,  57.2,  99.1,  86.4,  84.4,  38.1,  17.7,
       102.2, 101.3,  58. ,  82. , 101.4,  81.4, 100.1,  54.6,  39.6,
        57.5,  29.2,  48.8,  37.3, 115.4,  55.6,  62. ,  95. ,  84.2,
       118.1, 153.2,  83.4, 104.7,  59. ,  46.4,  50. , 147.6,  76.8,
        59.9, 101.8, 136.6, 173. ,  92.5,  37. ,  59.8, 142.1,   9.9,
       158.2,  72.6,  28. , 112.9, 119.3, 199.2,  50.7,  44. , 170.7,
        67.2,  21.4,  61.3,  15.6, 106. , 116.2,  42.3,  38.5, 132.5,
        40.8, 147.5,  93.9,  71.4,  87.3, 163.7, 141.4,  62.6,  84.9,
        28.8, 121.1,  28.6,  32.4, 112. ,  50. ,  96.9,  81.8,  70.4,
       117.5,  41.2, 124.9,  78.2,  93. ,  53.5,  50.5,  42.6,  47.9,
        73.1, 129.1,  56.9, 103.3,  60.5, 134.3,  93.1,  49.5,  48.2,
       167.9,  27. , 111.1,  55.4,  36.2,  57.4,  66.8,  58.3,  60. ,
       161.6, 112.7,  37.4, 110.6,  56.6,  95.8, 126.8])


x,y = ECDF(rain_june)
x2,y2 = ECDF(rain_nov)
for i in range(50):
    sample_june, sample_nov = permutation_sample(rain_june,  rain_nov)
    x_sample, y_sample = ECDF(sample_june)
    x_sample2, y_sample2 = ECDF(sample_nov)
    _= plt.plot(x_sample, y_sample, marker = '.', linestyle = 'none', color = 'red', alpha = 0.05)
    _= plt.plot(x_sample2, y_sample2, marker = '.', linestyle = 'none',color = 'blue', alpha = 0.05)
_=plt.plot(x,y, 'red')
_=plt.plot(x2,y2, color = 'blue')
plt.xlabel('rainfull')
plt.ylabel('CDF')
plt.legend(['June','Nov'])
plt.show()

def draw_perm_reps(data1, data2, func, size = 1):
    perm_samples = np.empty(size)
    for i in range(size):
        sample1, sample2 = permutation_sample(data1,data2)
        stat = func(sample1,sample2)
        perm_samples[i]  = stat
    return perm_samples

dem_2016 = df_2016_2020['vote_share_2016'][df_2016_2020['party_detailed']=='DEMOCRAT']
dem_2020 = df_2016_2020['vote_share_2020'][df_2016_2020['party_detailed']=='DEMOCRAT']

_=plt.plot(dem_2016,dem_2020, linestyle = 'none', marker = '.')

def diff(data1,data2):
    return np.mean(data1-data2)
diff(dem_2016, dem_2020)

np.mean(draw_perm_reps(dem_2016, dem_2020, diff, size = 1000) <=diff(dem_2016, dem_2020))

dem_2012 = 52
dem_2020_shifted = dem_2020-np.mean(dem_2020)+ dem_2012
samples = draw_bs_reps(dem_2020_shifted, np.mean, size = 10000)
np.mean(samples>=np.mean(dem_2020))

dem_concat = np.concatenate((dem_2016, dem_2020))
dem_mean = np.mean(dem_concat)
dem_2020_shifted = dem_2020-np.mean(dem_2020)+ dem_mean
dem_2016_shifted = dem_2016-np.mean(dem_2016)+ dem_mean
bs_replicates_2020 = draw_bs_reps(dem_2020_shifted, np.mean, size = 10000)
bs_replicates_2016 = draw_bs_reps(dem_2016_shifted, np.mean, size = 10000)
bs_diff = bs_replicates_2020 - bs_replicates_2016
np.mean(bs_diff>=np.mean(dem_2020-dem_2016))

#A/B testing
dem = np.array([1]*153+[0]*91)
