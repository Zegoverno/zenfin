import datetime as dt
import calendar
import pandas as pd
import numpy as np
from . import stats
from math import ceil
  
def drawdown_details(drawdown):
  """
  Calculates drawdown details, including start/end/valley dates,
  duration, max drawdown and max dd for 99% of the dd period
  for every drawdown period
  """
  def _drawdown_details(drawdown):
    # mark no drawdown
    no_dd = drawdown == 0

    # extract dd start dates
    starts = ~no_dd & no_dd.shift(1)
    starts = list(starts[starts].index)

    # extract end dates
    ends = no_dd & (~no_dd).shift(1)
    ends = list(ends[ends].index)

    # no drawdown :)
    if not starts:
        return pd.DataFrame(
            index=[], columns=('start', 'valley', 'end', 'days',
                                'max drawdown', '99% max drawdown'))

    # drawdown series begins in a drawdown
    if ends and starts[0] > ends[0]:
        starts.insert(0, drawdown.index[0])

    # series ends in a drawdown fill with last date
    if not ends or starts[-1] > ends[-1]:
        ends.append(drawdown.index[-1])

    # build dataframe from results
    data = []
    for i, _ in enumerate(starts):
        dd = drawdown[starts[i]:ends[i]]
        clean_dd = -utils.remove_outliers(-dd, .99)
        data.append((starts[i], dd.idxmin(), ends[i],
                      (ends[i] - starts[i]).days,
                      dd.min() * 100, clean_dd.min() * 100))

    df = pd.DataFrame(data=data,
                        columns=('start', 'valley', 'end', 'days',
                                'max drawdown',
                                '99% max drawdown'))
    df['days'] = df['days'].astype(int)
    df['max drawdown'] = df['max drawdown'].astype(float)
    df['99% max drawdown'] = df['99% max drawdown'].astype(float)

    df['start'] = df['start'].dt.strftime('%Y-%m-%d')
    df['end'] = df['end'].dt.strftime('%Y-%m-%d')
    df['valley'] = df['valley'].dt.strftime('%Y-%m-%d')

    return df

  if isinstance(drawdown, pd.DataFrame):
    _dfs = {}
    for col in drawdown.columns:
      _dfs[col] = _drawdown_details(drawdown[col])
    return pd.concat(_dfs, axis=1)
  return _drawdown_details(drawdown)

def to_pct (input, dec=1):
  if dec == 2:
    dec = '{:,.2f}%'
  elif dec == 3:
    dec = '{:,.3f}%'
  else:
    dec = '{:,.1f}%'
  return (input*100).map(dec.format)

def to_num (input, dec=1):
  if dec == 2:
    dec = '{:,.2f}'
  elif dec == 3:
    dec = '{:,.3f}'
  elif dec == 0:
    dec = '{:0.0f}'
  else:
    dec = '{:,.1f}'
  return input.map(dec.format)

# date manipulation
# the df.index should be datetime 
# dd/mm/yyyy is the default setting
# begin

def inc_day(date, inc):
  """increment date by some amount of days"""
  return date + dt.timedelta(days=inc)

def inc_month(date, inc):
  """increment date by some amount of months"""
  month = date.month - 1 + inc
  year = date.year + month // 12
  month = month % 12 + 1
  day = min(date.day, calendar.monthrange(year, month)[1])
  return dt.datetime(year, month, day)

def inc_year(date, inc):
  """increment date by some amount of years"""
  year = date.year + inc
  month = date.month
  day = min(date.day, calendar.monthrange(year, month)[1])
  return dt.datetime(year, month, day)

def month_end(date):
  """get last day of month from date given"""
  aux = inc_month(date,1)
  return dt.datetime(aux.year, aux.month , 1) - dt.timedelta(days=1)

def month_beg(date):
  """get first day of month from date given"""
  return dt.datetime(date.year, date.month, 1)

def year_end(date):
  """get last day of year from date given"""
  aux = inc_year(date,1)
  return dt.datetime(aux.year, 1 , 1) - dt.timedelta(days=1)

def year_beg(date):
  """get first day of year from date given"""
  return dt.datetime(date.year, 1 , 1)

def mtd(df, date=None):
  """month to date slice"""
  log_1 = df.index >= month_beg(date)
  log_2 = df.index <= date
  mask = np.logical_and(log_1, log_2)
  return df[mask]

def qtd(df, date):
  """quarter to date slice"""
  log_1 = df.index >= month_beg(inc_month(date, -((date.month -1 ) % 3)))
  log_2 = df.index <= date
  mask = np.logical_and(log_1, log_2)
  return df[mask]

def std(df, date):
  """semester to date slice"""
  log_1 = df.index >= month_beg(inc_month(date, -((date.month -1 ) % 6)))
  log_2 = df.index <= date
  mask = np.logical_and(log_1, log_2)
  return df[mask]

def ytd(df, date):
  """year to date slice"""
  log_1 = df.index >= date.strftime('%Y-01-01')
  log_2 = df.index <= date
  mask = np.logical_and(log_1, log_2)
  return df[mask]

def l3m(df, date):
  """year to date slice"""
  log_1 = df.index >= inc_month(date, -3)
  log_2 = df.index <= date
  mask = np.logical_and(log_1, log_2)
  return df[mask]

def l6m(df, date):
  """year to date slice"""
  log_1 = df.index >= inc_month(date, -6)
  log_2 = df.index <= date
  mask = np.logical_and(log_1, log_2)
  return df[mask]

def l1y(df, date):
  """year to date slice"""
  log_1 = df.index >= inc_year(date, -1)
  log_2 = df.index <= date
  mask = np.logical_and(log_1, log_2)
  return df[mask]

def l3y(df, date):
  """year to date slice"""
  log_1 = df.index >= inc_year(date, -3)
  log_2 = df.index <= date
  mask = np.logical_and(log_1, log_2)
  return df[mask]

def l5y(df, date):
  """year to date slice"""
  log_1 = df.index >= inc_year(date, -5)
  log_2 = df.index <= date
  mask = np.logical_and(log_1, log_2)
  return df[mask]

def l10y(df, date):
  """year to date slice"""
  log_1 = df.index >= inc_year(date, -10)
  log_2 = df.index <= date
  mask = np.logical_and(log_1, log_2)
  return df[mask]

# todo
# 1. calendarios DU brasil DI/B3
# 2. correct datas based on official callendar
# end

# results calculations
# default prices should be given
# default using daily returns
# view price, returns and cota should give the same results

# begin

def clean(data):
  """clean step to remove inconsistend data"""
  return data.copy().fillna(0).replace([np.inf, -np.inf, float('NaN')])

def to_quotes(returns, base=1e5):
  """converts returns series to quote prices"""
  return (1+ to_rolling_returns(returns))* base

def to_returns(prices):
  """simple arithmetic returns from a price series"""
  return prices.pct_change()

def to_rolling_returns(returns):
  """calculates rolling compounded returns"""
  return returns.add(1).cumprod() -1
  
def to_log_returns(returns, base='e'):
  """logarithm returns from a price series and base"""
  if base == '2':
    return np.log2(1+returns)
  elif base == '10':
    return np.log10(1+returns)
  else:
    return np.log(1+returns)

def rebase(prices, base=100):
  """rebase a series to a given initial base"""
  return prices / prices.iloc[0] * base

def to_excess_returns(returns, rf=0., periods=252):
  """Calculates excesss returns"""
  if periods is not None:
    rf = np.power(1 + rf, 1./ periods)  -1.
  if isinstance(returns, pd.DataFrame):
    result = pd.DataFrame()
    for c in returns.columns:
      result[c] = returns[c] - rf
    return result
  return returns - rf

def match_volatility(returns, benchmark): 
  return (returns / returns.std()) * benchmark.std()
  
def group_returns(returns, groupby, compounded=True):
  """Summarize returns
  group_returns(df, df.index.year)
  group_returns(df, [df.index.year, df.index.month])
  """
  if compounded:
      return returns.groupby(groupby).apply(stats.total_return)
  return returns.groupby(groupby).sum()

def aggregate_returns(returns, period=None, compounded=True):
  """Aggregates returns based on date periods"""
  if period is None or 'day' in period:
    return returns
  index = returns.index

  if 'month' in period:
    return group_returns(returns, index.month, compounded=compounded)

  if 'quarter' in period:
    return group_returns(returns, index.quarter, compounded=compounded)

  if period == "Y" or any(x in period for x in ['year', 'eoy', 'yoy']):
    return group_returns(returns, index.year, compounded=compounded)

  if 'week' in period:
    return group_returns(returns, index.isocalendar().week, compounded=compounded)

  if 'eow' in period or period == "W":
    return group_returns(returns, [index.year, index.isocalendar().week],
                            compounded=compounded)

  if 'eom' in period or period == "M":
    return group_returns(returns, [index.year, index.month],
                            compounded=compounded)

  if 'eoq' in period or period == "Q":
    return group_returns(returns, [index.year, index.quarter],
                            compounded=compounded)

  if not isinstance(period, str):
    return group_returns(returns, period, compounded)

  return returns

def count_consecutive(data):
  """Counts consecutive data"""
  if isinstance(data, pd.DataFrame):
    results = pd.DataFrame()
    for c in data:
      results[c] = data[c] * (data[c].groupby((data[c] != data[c].shift(1)).cumsum()).cumcount() + 1)
    return results

  return data * (data.groupby((data != data.shift(1)).cumsum()).cumcount() + 1)

def to_drawdown_series(returns):
    """Convert returns series to drawdown series"""
    prices = to_quotes(returns, 1)
    prices = clean(prices)
    dd = prices / np.maximum.accumulate(prices) - 1.
    return dd.replace([np.inf, -np.inf, -0], 0)

def remove_outliers(returns, quantile=.95):
  """Returns series of returns without the outliers"""
  return returns[returns < returns.quantile(quantile)]

def exposure(returns):
  """Returns the market exposure time (returns != 0)"""

  return 100 * returns[(returns != 0)].count() / returns.count() / 100

# end