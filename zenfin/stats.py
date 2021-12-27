import pandas as pd
import numpy as np
from math import ceil, sqrt
from scipy.stats import (norm, linregress )
from . import utils


# portfolio statistics
def rolling_returns(returns):
  """calculates rolling compounded returns"""
  return returns.add(1).cumprod() -1

def total_return(returns):
  """calculates total compounded returns"""
  return returns.add(1).prod() -1

def expected_return(returns, aggregate=None, compounded=True):
  """
  Returns the expected return for a given period
  by calculating the geometric holding period return
  """
  returns = utils.aggregate_returns(returns, aggregate, compounded)
  return np.product(1 + returns) ** (1 / len(returns)) - 1

def avg_return(returns, aggregate=None, compounded=True):
  """Calculates the average return/trade return for a period"""
  if aggregate:
    returns = utils.aggregate_returns(returns, aggregate, compounded)
  return returns[returns != 0].dropna().mean()

def cagr(returns, compounded=True, periods=252):
    """
    Calculates the communicative annualized growth return
    (CAGR%) of access returns
    """
    if compounded:
        total = total_return(returns)
    else:
        total = np.sum(returns)
    years = (returns.count()-1)/periods
    res = abs(total + 1.0) ** (1.0 / years) - 1
    return res

def volatility(returns, periods=252, annualize=True):
  """Calculates the volatility of returns for a period"""
  std = returns.std()
  if annualize:
    return std * np.sqrt(periods)
  return std

def rolling_volatility(returns, rolling_period=126, periods_per_year=252):
  return returns.rolling(rolling_period).std() * np.sqrt(periods_per_year)

def implied_volatility(returns, periods=252, annualize=True):
  """Calculates the implied volatility of returns for a period"""
  logret = utils.to_log_returns(returns)
  if annualize:
    return logret.rolling(periods).std() * np.sqrt(periods)
  return logret.std()

def exponential_stdev(returns, window=30, is_halflife=False):
  """Returns series representing exponential volatility of returns"""
  halflife = window if is_halflife else None
  return returns.ewm(com=None, span=window, halflife=halflife, min_periods=window).std()

def autocorr_penalty(returns):
  """Metric to account for auto correlation"""
  num = len(returns)
  coef = np.abs(np.corrcoef(returns[:-1], returns[1:])[0, 1])
  corr = [((num - x)/num) * coef ** x for x in range(1, num)]
  return np.sqrt(1 + 2 * np.sum(corr))

def sharpe(returns, rf, periods=252, annualize=True, smart=False):
  """
  Calculates the sharpe ratio of access returns
  """
  excess_returns = utils.to_excess_returns(returns, rf, periods)
  divisor = excess_returns.std(ddof=1)
  if smart:
    # penalize sharpe with auto correlation
    divisor = divisor * autocorr_penalty(returns)
  res = excess_returns.mean() / divisor
  if annualize:
    return res * np.sqrt(periods)
  return res

def rolling_sharpe(returns, rf, rolling_period=126, annualize=True, periods_per_year=252):
  excess_returns = utils.to_excess_returns(returns, rf, periods_per_year)
  res = excess_returns.rolling(rolling_period).mean() / returns.rolling(rolling_period).std(ddof=1)
  if annualize:
      res = res * np.sqrt(periods_per_year)
  return res

def sortino(returns, rf, periods=252, annualize=True, smart=False):
  """
  Calculates the sortino ratio of access returns
  If rf is non-zero, you must specify periods.
  In this case, rf is assumed to be expressed in yearly (annualized) terms
  Calculation is based on this paper by Red Rock Capital
  http://www.redrockcapital.com/Sortino__A__Sharper__Ratio_Red_Rock_Capital.pdf
  """
  excess_returns = utils.to_excess_returns(returns, rf, periods)
  downside = np.sqrt((excess_returns[excess_returns < 0] ** 2).sum() / len(excess_returns))
  if smart:
    # penalize sortino with auto correlation
    downside = downside * autocorr_penalty(returns)
  res = excess_returns.mean() / downside
  if annualize:
    return res * np.sqrt(periods)
  return res

def rolling_sortino(returns, rf, rolling_period=126, annualize=True, periods_per_year=252 ):
  excess_returns = utils.to_excess_returns(returns, rf, periods_per_year)
  downside = excess_returns.rolling(rolling_period).apply(lambda x: (x.values[x.values < 0]**2).sum()) / rolling_period
  res = excess_returns.rolling(rolling_period).mean() / np.sqrt(downside)
  if annualize:
    res = res * np.sqrt(periods_per_year)
  return res

def adjusted_sortino(returns, rf, periods=252, annualize=True, smart=False):
  """
  Jack Schwager's version of the Sortino ratio allows for
  direct comparisons to the Sharpe. See here for more info:
  https://archive.is/wip/2rwFW
  """
  data = sortino(returns, rf, periods=periods, annualize=annualize, smart=smart)
  return data / sqrt(2)

def rar(returns, rf=0., periods=252):
    """
    Calculates the risk-adjusted return of access returns
    (CAGR / exposure. takes time into account.)
    """
    excess_returns = utils.to_excess_returns(returns, rf, periods)
    cagrs = cagr(excess_returns)
    exs = utils.exposure(excess_returns)
    if isinstance(returns, pd.DataFrame):
      results = {}
      for c in returns:
        results[c] = cagrs[c] / exs[c]
      return results
    return cagr / ex

def omega(returns, required_returns=0., periods=252):
    """
    Determines the Omega ratio of a strategy.
    See https://en.wikipedia.org/wiki/Omega_ratio for more details.
    """
    returns_less_thresh = utils.to_excess_returns(returns, required_returns, periods)
    if isinstance(returns, pd.DataFrame):
      result = {}
      for c in returns.columns:
        numer = returns_less_thresh[c][returns_less_thresh[c] > 0.0].sum()
        denom = -1.0 * returns_less_thresh[c][returns_less_thresh[c] < 0.0].sum()
        if denom > 0.0:
          result[c] = numer / denom
        else: 
          result[c] = np.nan
      return result
      
    numer = returns_less_thresh[returns_less_thresh > 0.0].sum()
    denom = -1.0 * returns_less_thresh[returns_less_thresh < 0.0].sum()
    if denom > 0.0:
      return numer / denom
    return np.nan

def gain_to_pain_ratio(returns, resolution="D", periods=252):
    """
    Jack Schwager's GPR. See here for more info:
    https://archive.is/wip/2rwFW
    """
    returns = returns.resample(resolution).sum()
    downside = abs(returns[returns < 0].sum())
    return returns.sum() / downside

def outliers(returns, quantile=.95):
  """Returns series of outliers"""
  return returns[returns > returns.quantile(quantile)].dropna(how='all')

def best(returns, aggregate=None, compounded=True):
  """Returns the best day/month/week/quarter/year's return"""
  return utils.aggregate_returns(returns, aggregate, compounded).max()

def worst(returns, aggregate=None, compounded=True):
  """Returns the worst day/month/week/quarter/year's return"""
  return utils.aggregate_returns(returns, aggregate, compounded).min()

def consecutive_wins(returns, aggregate=None, compounded=True):
  """Returns the maximum consecutive wins by day/month/week/quarter/year"""
  returns = utils.aggregate_returns(returns, aggregate, compounded) > 0
  return utils.count_consecutive(returns).max()

def consecutive_losses(returns, aggregate=None, compounded=True):
  """Returns the maximum consecutive losses by day/month/week/quarter/year"""
  returns = utils.aggregate_returns(returns, aggregate, compounded) < 0
  return utils.count_consecutive(returns).max()

def win_rate(returns, aggregate=None, compounded=True):
  """Calculates the win ratio for a period"""
  if aggregate:
    returns = utils.aggregate_returns(returns, aggregate, compounded)
  return len(returns[returns > 0]) / len(returns[returns != 0])

def avg_win(returns, aggregate=None, compounded=True):
  """
  Calculates the average winning
  return/trade return for a period
  """
  if aggregate:
    returns = utils.aggregate_returns(returns, aggregate, compounded)
  return returns[returns > 0].dropna().mean()

def avg_loss(returns, aggregate=None, compounded=True, prepare_returns=True):
  """
  Calculates the average low if
  return/trade return for a period
  """
  if aggregate:
    returns = utils.aggregate_returns(returns, aggregate, compounded)
  return returns[returns < 0].dropna().mean()

def skew(returns):
    """
    Calculates returns' skewness
    (the degree of asymmetry of a distribution around its mean)
    """
    return returns.skew()

def kurtosis(returns):
    """
    Calculates returns' kurtosis
    (the degree to which a distribution peak compared to a normal distribution)
    """
    return returns.kurtosis()

def max_drawdown(returns):
    """Calculates the maximum drawdown"""
    ##can just get drawdown_series and take the minimun?
    prices = utils.to_quotes(returns, 1)
    return (prices / prices.expanding(min_periods=0).max()).min() - 1
  
def calmar(returns):
    """Calculates the calmar ratio (CAGR% / MaxDD%)"""
    cagr_ratio = cagr(returns)
    max_dd = max_drawdown(returns)
    return cagr_ratio / abs(max_dd)

def ulcer_index(returns):
    """Calculates the ulcer index score (downside risk measurment)"""
    dd = utils.to_drawdown_series(returns)
    return np.sqrt(np.divide((dd**2).sum(), returns.shape[0] - 1))

def ulcer_performance_index(returns, rf=0):
    """
    Calculates the ulcer index score
    (downside risk measurment)
    comp over exess return or comp of both rf and returns
    seemed correct if rf is fixed
    """
    return (total_return(returns)-rf) / ulcer_index(returns)

def value_at_risk(returns, sigma=1, confidence=0.95):
  """
  Calculats the daily value-at-risk
  (variance-covariance calculation with confidence n)
  """
  mu = returns.mean()
  sigma *= returns.std()

  if confidence > 1:
      confidence = confidence/100

  return norm.ppf(1-confidence, mu, sigma)

def conditional_value_at_risk(returns, sigma=1, confidence=0.95):
    """
    Calculats the conditional daily value-at-risk (aka expected shortfall)
    quantifies the amount of tail risk an investment
    """
    var = value_at_risk(returns, sigma, confidence)
    c_var = returns[returns < var].values.mean()
    return c_var if ~np.isnan(c_var) else var

def serenity_index(returns, rf=0):
  """
  Calculates the serenity index score
  (https://www.keyquant.com/Download/GetFile?Filename=%5CPublications%5CKeyQuant_WhitePaper_APT_Part1.pdf)
  """
  dd = utils.to_drawdown_series(returns)
  pitfall = - value_at_risk(dd) / returns.std()
  return (total_return(returns)-rf) / (ulcer_index(returns) * pitfall)

def risk_of_ruin(returns):
    """
    Calculates the risk of ruin
    (the likelihood of losing all one's investment capital)
    """
    wins = win_rate(returns)
    return ((1 - wins) / (1 + wins)) ** len(returns)

def tail_ratio(returns, cutoff=0.95):
    """
    Measures the ratio between the right
    (95%) and left tail (5%).
    """
    return abs(returns.quantile(cutoff) / returns.quantile(1-cutoff))

def payoff_ratio(returns):
    """Measures the payoff ratio (average win/average loss)"""
    return avg_win(returns) / abs(avg_loss(returns))

def profit_ratio(returns):
    """Measures the profit ratio (win ratio / loss ratio)"""
    wins = returns[returns >= 0]
    loss = returns[returns < 0]
    win_ratio = abs(wins.mean() / wins.count())
    loss_ratio = abs(loss.mean() / loss.count())
    try:
        return win_ratio / loss_ratio
    except Exception:
        return 0.

def profit_factor(returns):
    """Measures the profit ratio (wins/loss)"""
    return abs(returns[returns >= 0].sum() / returns[returns < 0].sum())

def cpc_index(returns):
    """
    Measures the cpc ratio
    (profit factor * win % * win loss ratio)
    """
    return profit_factor(returns) * win_rate(returns) * payoff_ratio(returns)

def common_sense_ratio(returns):
    """Measures the common sense ratio (profit factor * tail ratio)"""
    return profit_factor(returns) * tail_ratio(returns)

def outlier_win_ratio(returns, quantile=.99):
    """
    Calculates the outlier winners ratio
    99th percentile of returns / mean positive return
    """
    return returns.quantile(quantile).mean() / returns[returns >= 0].mean()

def outlier_loss_ratio(returns, quantile=.01):
    """
    Calculates the outlier losers ratio
    1st percentile of returns / mean negative return
    """
    return returns.quantile(quantile).mean() / returns[returns < 0].mean()

def recovery_factor(returns):
  """Measures how fast the strategy recovers from drawdowns"""
  max_dd = max_drawdown(returns)
  return total_return(returns) / abs(max_dd)

def risk_return_ratio(returns):
    """
    Calculates the return / risk ratio
    (sharpe ratio without factoring in the risk-free rate)
    """
    return returns.mean() / returns.std()

def kelly_criterion(returns):
  """
  Calculates the recommended maximum amount of capital that
  should be allocated to the given strategy, based on the
  Kelly Criterion (http://en.wikipedia.org/wiki/Kelly_criterion)
  """
  win_loss_ratio = payoff_ratio(returns)
  win_prob = win_rate(returns)
  lose_prob = 1 - win_prob

  return ((win_loss_ratio * win_prob) - lose_prob) / win_loss_ratio

def r_squared(returns, benchmark):
  """Measures the straight line fit of the equity curve"""
  # slope, intercept, r_val, p_val, std_err = linregress()
  _, _, r_val, _, _ = linregress(returns, benchmark)
  return r_val**2

def information_ratio(returns, benchmark):
  """
  Calculates the information ratio
  (basically the risk return ratio of the net profits)
  """
  diff_rets = returns - benchmark

  return diff_rets.mean() / diff_rets.std()

def beta(returns, benchmark ):
  """Calculates beta of the portfolio"""

  # find covariance
  matrix = np.cov(returns, benchmark)
  return matrix[0, 1] / matrix[1, 1]

def alpha(returns, benchmark, periods=252):
  """Calculates alpha of the portfolio"""

  # find beta
  _beta = beta(returns, benchmark)

  # calculates measures now
  alpha = returns.mean() - _beta * benchmark.mean()
  return alpha * periods

def rolling_greeks(returns, benchmark, periods=252):
  """Calculates rolling alpha and beta of the portfolio"""
  df = pd.DataFrame(data={
    "returns": returns,
    "benchmark": benchmark
  })
  df = df.fillna(0)
  corr = df.rolling(int(periods)).corr().unstack()['returns']['benchmark']
  std = df.rolling(int(periods)).std()
  beta = corr * std['returns'] / std['benchmark']

  alpha = df['returns'].mean() - beta * df['benchmark'].mean()

  return pd.DataFrame(index=returns.index, data={
    "beta": beta,
    "alpha": alpha
  }).fillna(0)