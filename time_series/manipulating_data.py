# -*- coding: utf-8 -*-
"""
Created on Sun May 16 08:31:11 2021

@author: Ahmed Elhussein
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import norm


time_stamp = pd.Timestamp(datetime(2017,1,1))
time_stamp.year

period = pd.Period('2017-01')
period.asfreq('D')
period.to_timestamp().to_period('M')

index = pd.date_range(start = '2017-1-1', periods = 12, freq = 'M')
index.to_period()
df = pd.DataFrame({'date':index}).info()
data = np.random.random(size = (12,2))
pd.DataFrame(index = index, data = data)


seven_days = pd.date_range(start = '2017-1-1', periods = 7, freq = 'D')
for day in seven_days:
    print(day.dayofweek, day.weekday_name )
    

#google.date = pd.to_datetime(google.date)
#google.set_index('date', inplace = True)
#google['2015']
#    google['2015-3':'2016-2']
#    google.loc['2016-1-1', 'price']
#    google.asfreq('D') / google.asfreq('B'); google[google['price'].isnull()]
    
df = pd.read_csv('nyc.csv')
df.head()
df.info()
df.date = pd.to_datetime(df.date)
df.set_index('date', inplace = True)

sns.set()
df.plot(subplots = True)
plt.show()

fig, axis = plt.subplots(nrows = 3)
_ = axis[0].plot( df.ozone, color = 'blue', label = 'Ozone')
_=axis[1].plot(df.pm25, color = 'red', label = 'PM25')
_=axis[2].plot(df.co, color = 'green', label = 'CO')
axis.legend(loc = 'best')
plt.show()

df = pd.read_csv('yahoo.csv')
df.head()
df.info()
df.date = pd.to_datetime(df.date)
df.set_index('date', inplace = True)
prices = pd.DataFrame()
years = ['2013','2014','2015']
for year in years:
    prices_year = df.loc[year, ['price']].reset_index(drop = True)
    prices_year.rename(columns = {'price':year}, inplace = True)
    prices = pd.concat([prices, prices_year], axis = 1)

prices.plot()


co = pd.read_csv('co_cities.csv')
co.head()
co.date = pd.to_datetime(co.date)
co.set_index('date', inplace = True)
daily = co.asfreq('D')
montly = co.asfreq('M')
co.plot(subplots = True)
plt.show()
montly.plot(subplots = True)
plt.show()


#Manipulating data
google = pd.read_csv('google.csv', parse_dates =['date'], index_col = 'date', names = ['date', 'price'])
google.price = pd.to_numeric(google['price'], errors='coerce')

google['shifted'] = google.price.shift(periods = 1)
google['lagged'] = google.price.shift(periods = -1)
google['change'] = google.price/google.shifted
google['return'] = google.change.sub(1).mul(100)
google['diff'] = google.price.diff()
google.price.pct_change().mul(100)
#
google = pd.read_csv('google.csv')
google.info()
google.price = pd.to_numeric(google['price'], errors='coerce')
google.date = pd.to_datetime(google.date)
google.set_index('date', inplace = True)
google = google.asfreq('B')
google['shifted'] = google.price.shift(periods = 90)
google['lagged'] = google.price.shift(periods = -90)
google.plot()


yahoo = pd.read_csv('yahoo.csv', parse_dates = ['date'], index_col = 'date')
yahoo.head()
yahoo.info()
yahoo = yahoo.loc['2013':'2015']
yahoo = yahoo.asfreq('B')
yahoo['shifted_30'] = yahoo.price.shift(periods = 30)
yahoo['change_30'] = yahoo.price.sub(yahoo.shifted_30)
yahoo['diff_30'] = yahoo.price.diff(periods = 30)
yahoo.diff_30.sub(yahoo.change_30).value_counts()

google = pd.read_csv('google.csv', parse_dates = ['date'], index_col = 'date')
google.head()
google.info()
google = google.loc['2014':'2016']
google['daily_return'] = google.price.pct_change().mul(100)
google['monthly_return'] = google.price.pct_change(periods=30).mul(100)
google['yearly_return'] = google.price.pct_change(periods = 360).mul(100)
google.plot(subplots = True)

google = pd.read_csv('google.csv', parse_dates = ['date'], index_col = 'date')
first_price = google.price.iloc[0]
normalised = google.price.div(first_price).mul(100)
normalised.plot(title = 'Normalised google data')


prices = pd.read_csv('stock_data.csv', parse_dates = ['Date'], index_col = 'Date')
index = pd.read_csv('index.csv', parse_dates = ['Date'], index_col = 'Date')
index.rename(columns = {'Unnamed: 1': 'S&P'}, inplace = True)
index.head()
first_price = prices.iloc[0]
normalised = prices.div(first_price).mul(100)
normalised = pd.concat([normalised, index], axis =1).dropna()
normalised.plot()
plt.show()


prices = pd.read_csv('asset_classes.csv', parse_dates = ['DATE'], index_col = 'DATE')
prices.head()
first_prices = prices.iloc[0]
normalised = prices.div(first_prices).mul(100)
normalised.plot()
plt.show()

stocks = pd.read_csv('nyse.csv', parse_dates = ['date'], index_col = 'date')
dow_jones = pd.read_csv('dow_jones.csv', parse_dates = ['date'], index_col = 'date')
stocks.head()
dow_jones.head()
data = pd.concat([stocks, dow_jones], axis =1 )
fist_price = data.iloc[0]
normalised = data.div(fist_price).mul(100)
normalised.plot()

tickers = ['MSFT', 'AAPL']
stocks = pd.read_csv('msft_aapl.csv', parse_dates = ['date'], index_col = 'date')
sp500 = pd.read_csv('sp500.csv', parse_dates = ['date'], index_col = 'date')
stocks.head()
sp500.head()
data = pd.concat([stocks, sp500],axis = 1).dropna()
normalised = data.div(data.iloc[0]).mul(100)
normalised.head()
normalised[tickers].sub(normalised['SP500'], axis = 0).plot()


dates = pd.date_range(start = '2016-1-1', periods =4, freq = 'Q')
data = range(1,5)
quarterly = pd.Series(index = dates, data = data)
monthly = quarterly.asfreq('M')
monthly = monthly.to_frame('baseline')
monthly['ffill'] = quarterly.asfreq('M', method = 'ffill')
monthly['bfill'] = quarterly.asfreq('M', method = 'bfill')
monthly['value'] = quarterly.asfreq('M',fill_value = 0)

dates = pd.date_range(start = '2016-1-1', periods = 12, freq = 'M')
monthly.reindex(dates)

data = pd.read_csv('unrate.csv', parse_dates = ['DATE'], index_col = 'DATE')
data.asfreq('W', method = 'bfill')
data.head()

data.asfreq('MS').equals(data.resample('MS').asfreq())

gdp = pd.read_csv('gdp_growth.csv', parse_dates = ['date'], index_col = 'date')
gdp.head()
gdp.tail()
gdp.info()
gdp_1 = gdp.resample('MS').ffill().add_suffix('_ffill')
gdp_2 = gdp.resample('MS').interpolate().add_suffix('_inter')
pd.concat([gdp_1,gdp_2], axis = 1).loc['2015'].plot()
pd.concat([data, gdp], axis =1).interpolate().plot()
plt.show()

monthly = pd.read_csv('unrate.csv', parse_dates = ['DATE'], index_col = 'DATE')
weekly_dates = pd.date_range(start = min(monthly.index.values), end = max(monthly.index.values), freq = 'W')
weekly = monthly.reindex(weekly_dates)
weekly['ffill'] = weekly.UNRATE.ffill()
weekly['interpolated'] = weekly.UNRATE.interpolate()
weekly.plot()

data = pd.read_csv('debt_unemployment.csv', parse_dates = ['date'], index_col = 'date')
data.info()
interpolated = data.interpolate()
interpolated.plot(secondary_y = 'Unemployment')
plt.show()


ozone = pd.read_csv('ozone_nyc.csv', parse_dates  = ['date'], index_col = 'date')
ozone.head()
ozone = ozone.resample('D').asfreq()
ozone.resample('M').mean().head()
ozone.resample('M').agg(['mean','std']).head()
ozone = ozone.loc['2016':]
ax = ozone.plot()
monthly = ozone.resample('M').mean()
monthly.add_suffix('_monthly').plot(ax = ax)
plt.show()


data = pd.read_csv('ozone_nyla.csv', parse_dates = ['date'], index_col = 'date')
data = data.resample('D').asfreq()
data = data.loc['2014':]
ax = data.plot()
monthly = data.resample('M').mean()
monthly.add_suffix('_monthly').plot(ax=ax)

sp500 = pd.read_csv('sp500.csv', parse_dates = ['date'], index_col = 'date')
sp500.head()
daily_returns = sp500.squeeze().pct_change()
stats = daily_returns.resample('M').agg(['mean','median','std'])
stats.plot()
plt.show()


google = pd.read_csv('google.csv', parse_dates = ['date'], index_col = 'date')
google.price = pd.to_numeric(google['price'], errors='coerce')
google = google.asfreq('D').ffill()
r90 = google.rolling(window ='90D').mean()
google.join(r90.add_suffix('_90')).plot()
google['r90'] = r90
r360 = google.rolling(window = '360D').mean()
google['r360'] = r360
google.plot()
plt.show()
r = google.price.rolling('90D').agg(['mean','std'])
r.plot(subplots = True)
plt.show()
rolling = google.price.rolling('360D')
q10 = rolling.quantile(0.1).to_frame('q10')
median = rolling.median().to_frame('median')
q90 = rolling.quantile(0.9).to_frame('q90')
pd.concat([q10,median,q90], axis =1 ).plot()

ozone = pd.read_csv('ozone_nyc.csv', parse_dates  = ['date'], index_col = 'date')
ozone['90D'] = ozone.rolling('90D').mean()
ozone['360D'] = ozone.Ozone.rolling('360D').mean()
ozone.plot()
plt.show()

ozone = ozone.dropna()
rolling_stats = ozone['Ozone'].rolling('360D').agg(['mean','std'])
stats = ozone.join(rolling_stats)
stats.plot()

data = pd.read_csv('ozone_nyc.csv', parse_dates  = ['date'], index_col = 'date')
data = data.resample('D').interpolate()
rolling = data.rolling('360D')['Ozone']
data['q10'] = rolling.quantile(0.1)
data['q50'] = rolling.median()
data['q90'] = rolling.quantile(0.9)

data.plot()



data = pd.read_csv('sp500.csv', parse_dates = ['date'], index_col = 'date')
data.head()
data['returns']= data.SP500.pct_change() + 1
data.returns.cumprod() - 1
data['running_min'] = data.SP500.expanding().min()
data['running_max'] = data.SP500.expanding().max()
data.plot()

def multi_period_return(period_returns):
    return np.prod(period_returns+1) -1
pr = data.SP500.pct_change()
r = pr.rolling('360D').apply(multi_period_return)
data['Rolling 1yr return'] = r.mul(100)
data.plot(subplots = True)

google = pd.read_csv('google.csv', parse_dates = ['date'], index_col = 'date').dropna()
google.price = pd.to_numeric(google['price'], errors='coerce')
google = google.asfreq('D').dropna()
google.head()
differences = google.price.diff()
start_price = google.first('D')
cumulative_sum = google.price.cumsum()
start_price.append(differences)




stocks = pd.read_csv('msft_aapl.csv', parse_dates = ['date'], index_col = 'date')
investment = 1000
returns = stocks.pct_change()
returns_plus_one = returns.add(1)
cumulative_return = returns_plus_one.cumprod().sub(1)
cumulative_return.mul(investment).plot()


def multi_period_return(period_returns):
    return np.prod(period_returns+1)-1
stocks.info()
returns = stocks.pct_change()
rolling_returns = returns.rolling('360D').apply(multi_period_return)
rolling_returns.plot()
plt.show()


#random walk
np.random.seed(42)
random_returns = np.random.normal(loc = 0, scale = 0.01, size = 1000)
sns.distplot(random_returns, fit = norm, kde = False)
return_series = pd.Series(random_returns)
random_prices = return_series.add(1).cumprod().sub(1)
random_prices.mul(100).plot()
plt.show()


data = pd.read_csv('sp500.csv', parse_dates = ['date'], index_col = 'date').dropna()
data['returns'] = data.SP500.pct_change()
data.plot(subplots = True)
sns.distplot(data.returns.mul(100), fit = norm)
plt.show()
sample = data.returns.dropna()
n_obs = sample.count()
random_walk = np.random.choice(sample, size = n_obs+1)
random_walk = pd.Series(random_walk, index = data.index)
random_prices = random_walk.add(1).cumprod().sub(1)
random_prices.plot()
plt.show()

start = data.SP500.first('D')
random_walk = np.random.choice(sample, size = n_obs+1)
random_walk = pd.Series(random_walk, index = data.index)
sp500_random = start.append(random_walk.add(1))
data['sp500_random']=sp500_random.cumprod()


prices = pd.read_csv('asset_classes.csv', parse_dates = ['DATE'], index_col = 'DATE')
prices = prices.dropna()
daily_returns = prices.pct_change()
sns.jointplot(x = 'SP500', y = 'Bonds', data = daily_returns)
plt.show()

correlations = daily_returns.corr()
sns.heatmap(correlations, annot= True)

annual_change = prices.resample('A').last().pct_change()
sns.heatmap(annual_change.corr(), annot = True)

#Market cap
nyse= pd.read_excel('listings.xlsx', sheetname = 'nyse', na_values = 'n/a')
nyse.info()
nyse.set_index('Stock Symbol', inplace = True)
print(nyse.head())
nyse.dropna(subset = ['Sector'], inplace = True)
nyse['Market Capitalization'] /= 1e6

components = nyse.groupby(['Sector'])['Market Capitalization'].nlargest(1)
components.sort_values(ascending = False)
tickers = components.index.get_level_values('Stock Symbol').tolist()
columns = ['Company Name', 'Market Capitalization', 'Last Sale']
components_info = nyse.loc[tickers, columns]
pd.options.display.float_format = '{:,.2f}'.format

data = pd.read_csv('stock_data.csv', parse_dates = ['Date'], index_col ='Date').loc[:, tickers]
data.head()

shares = components['Market Capitalization'].div(components['Last Sale'])
market_cap_series = data.mul(no_shares)

market_cap_series = pd.read_csv('market_cap_series.csv', parse_dates =['Date'], index_col = 'Date')
pd.options.display.float_format = '{:,.2f}'.format
agg_market_cap = market_cap_series.sum(axis =1)
index = agg_market_cap.div(agg_market_cap.iloc[0]).mul(100)
index.plot()
agg_market_cap.iloc[-1] - agg_market_cap.iloc[0]  
change = market_cap_series.first('D').append(market_cap_series.last('D'))
change.diff().iloc[-1].sort_values()
market_cap = component['Market Capitalization']
weights = market_cap.div(market_cap.sum())
index_return = (index.iloc[-1]/index.iloc[0]-1)*100

data.index = data.index.date
with pd.ExcelWriter('stocl_data.xlsx') as writer:
    corr.to_excel(excel_writer = writer, sheet_name = 'correlations')
    data.to_excel(excel_writer = writer, sheet_name = 'prices')
    data.pct_change().to_excel(excel_writer = writer, sheet_name = 'returns')
