import pandas as pd
from . import utils
from . import stats
from tabulate import tabulate

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

def metrics (dr, bench, rf, periods=252):
  rfd = rf['dR'].rename('DI')
  rfy = rf['yR'].rename('DI')
  frames = [dr, bench]
  df = pd.concat(frames, axis=1)
  dd = utils.to_drawdown_series(df)
  dd_details = reports.drawdown_details(dd)
  display = pd.DataFrame()
  print('[Analysis info]')
  date_start = dr.index.strftime('%Y-%m-%d')[0]
  date_end = dr.index.strftime('%Y-%m-%d')[-1]
  display['Start Period'] = pd.Series(date_start)
  display['End Period'] = pd.Series(date_end)
  display['Working days'] = pd.Series(len(dr.index)) 
  display['Years'] = pd.Series("{:.1f}".format(len(dr.index)/ periods))
  print(tabulate(display.T, tablefmt='simple'))
  print('\n')
  display = pd.DataFrame()
  print('[Performance Metrics]')
  print('Return Info')
  frames = [dr, bench, rfd ]
  df = pd.concat(frames, axis=1)
  total_return = to_pct(stats.total_return(df),2)
  display['Total return'] = pd.Series(total_return)
  cagr = to_pct(stats.cagr(df),2)
  display['CAGR'] = pd.Series(cagr)
  d_r = to_pct(stats.expected_return(df),2)
  display['E[Daily Returns]'] = pd.Series(d_r)
  m_r = to_pct(stats.expected_return(df, aggregate='M'),2)
  display['E[Monthly Returns]'] = pd.Series(m_r)
  y_r = to_pct(stats.expected_return(df, aggregate='Y'),2)
  display['E[Yearly Returns]'] = pd.Series(y_r)
  r_mtd = to_pct(stats.total_return(utils.mtd(df, df.index[-1])),2)
  display['MTD'] = pd.Series(r_mtd)
  r_l3m = to_pct(stats.total_return(utils.l3m(df, df.index[-1])),2)
  display['3M'] = pd.Series(r_l3m)
  r_l6m = to_pct(stats.total_return(utils.l6m(df, df.index[-1])),2)
  display['6M'] = pd.Series(r_l6m)
  r_ytd = to_pct(stats.total_return(utils.ytd(df, df.index[-1])),2)
  display['YTD'] = pd.Series(r_ytd)
  r_l1y = to_pct(stats.total_return(utils.l1y(df, df.index[-1])),2)
  display['1Y'] = pd.Series(r_l1y)
  cagr_3y = to_pct(stats.cagr(utils.l3y(df, df.index[-3])),2)
  display['3Y (aa)'] = pd.Series(cagr_3y)
  cagr_5y = to_pct(stats.cagr(utils.l5y(df, df.index[-1])),2)
  display['5Y (aa)'] = pd.Series(cagr_5y)
  cagr_10y = to_pct(stats.cagr(utils.l10y(df, df.index[-1])),2)
  display['10Y (aa)'] = pd.Series(cagr_10y)
  display['Inception (aa)'] = pd.Series(cagr)
  print(tabulate(display.T, headers="keys", tablefmt='simple'))
  print('\n')
  display = pd.DataFrame()
  print('Performance Info')
  frames = [dr, bench]
  df = pd.concat(frames, axis=1)
  sharpe = to_num(stats.sharpe(df, rfy, smart=False),2)
  display['Sharpe'] = pd.Series(sharpe)
  smart_sharpe = to_num(stats.sharpe(df, rfy, smart=True),2)
  display['Smart Sharpe'] = pd.Series(smart_sharpe)
  sortino = to_num(stats.sortino(df, rfy, smart=False),2)
  display['Sortino'] = pd.Series(sortino)
  smart_sortino = to_num(stats.sortino(df, rfy, smart=True),2)
  display['Smart Sortino'] = pd.Series(smart_sortino)
  adj_sortino = to_num(stats.adjusted_sortino(df, rfy, smart=False),2)
  display['Sortino/√2'] = pd.Series(adj_sortino)
  adj_smart_sortino = to_num(stats.adjusted_sortino(df, rfy, smart=True),2)
  display['Smart Sortino/√2'] = pd.Series(adj_smart_sortino)
  omega = to_num(stats.omega(df, rfy),2)
  display['Omega'] = pd.Series(omega)
  beta = to_num(stats.beta(dr, bench),2)
  display['Beta'] = pd.Series(beta)
  alpha = to_num(stats.alpha(dr, bench),2)
  display['Alpha'] = pd.Series(alpha)
  print(tabulate(display.T, headers="keys", tablefmt='simple'))
  print('\n')
  display = pd.DataFrame()
  print('Risk Info')
  frames = [dr, bench]
  df = pd.concat(frames, axis=1)
  max_drawdown = to_pct(stats.max_drawdown(df),2)
  display['Max Drawdown'] = pd.Series(max_drawdown)
  longest_drawdown_days = to_num(stats.longest_drawdown_days(df),0)
  display['Longest DD Days'] = pd.Series(longest_drawdown_days)
  avg_drawdown = to_pct(stats.avg_drawdown(dd_details),2)
  display['Avg. Drawdown'] = pd.Series(avg_drawdown)
  avg_drawdown_days = to_num(stats.avg_drawdown_days(df),0)
  display['Avg. Drawdown Days'] = pd.Series(avg_drawdown_days)
  volatility = to_pct(stats.volatility(df, periods),2)
  display['Volatility (aa)'] = pd.Series(volatility)
  r_squared = to_num(stats.r_squared(dr, bench),2)
  display['R^2'] = pd.Series(r_squared)
  calmar = to_num(stats.calmar(df),2)
  display['Calmar'] = pd.Series(calmar)
  skew = to_num(stats.skew(df),2)
  display['Skew'] = pd.Series(skew)
  kurtosis = to_num(stats.kurtosis(df),2)
  display['Kurtosis'] = pd.Series(kurtosis)
  kelly_criterion = to_pct(stats.kelly_criterion(df),2)
  display['Kelly Criterion'] = pd.Series(kelly_criterion)
  risk_of_ruin = to_pct(stats.risk_of_ruin(df),2)
  display['Risk of Ruin'] = pd.Series(risk_of_ruin)
  value_at_risk = to_pct(stats.value_at_risk(df),2)
  display['Daily VaR'] = pd.Series(value_at_risk)
  conditional_value_at_risk = to_pct(stats.conditional_value_at_risk(df),2)
  display['Expected Shortfall (cVaR)'] = pd.Series(conditional_value_at_risk)
  recovery_factor = to_num(stats.recovery_factor(df),2)
  display['Recovery Factor'] = pd.Series(recovery_factor)
  ulcer_index = to_num(stats.ulcer_index(df),2)
  display['Ulcer Index'] = pd.Series(ulcer_index)
  serenity_index = to_num(stats.serenity_index(df, rfy),2)
  display['Serenity Index'] = pd.Series(serenity_index)
  print(tabulate(display.T, headers="keys", tablefmt='simple'))
  print('\n')
  display = pd.DataFrame()
  print('P&L Info')
  frames = [dr, bench]
  df = pd.concat(frames, axis=1)
  # showing based excess_returns! should show as returns?
  er = utils.to_excess_returns(df, rfy)
  gtp = to_num(stats.gain_to_pain_ratio(er),2)
  display['Gain/Pain Ratio'] = pd.Series(gtp)
  gtp_1m = to_num(stats.gain_to_pain_ratio(er, 'M'),2)
  display['Gain/Pain (1M)'] = pd.Series(gtp_1m)
  gtp_3m = to_num(stats.gain_to_pain_ratio(er, 'Q'),2)
  display['Gain/Pain (3M)'] = pd.Series(gtp_3m)
  gtp_6m = to_num(stats.gain_to_pain_ratio(er, 'BQ'),2)
  display['Gain/Pain (6M)'] = pd.Series(gtp_6m)
  gtp_1y = to_num(stats.gain_to_pain_ratio(er, 'A'),2)
  display['Gain/Pain (1Y)'] = pd.Series(gtp_1y)
  payoff_ratio = to_num(stats.payoff_ratio(df),2)
  display['Payoff Ratio'] = pd.Series(payoff_ratio)
  profit_factor = to_num(stats.profit_factor(df),2)
  display['Profit Factor'] = pd.Series(profit_factor)
  common_sense_ratio = to_num(stats.common_sense_ratio(df),2)
  display['Common Sense Ratio'] = pd.Series(common_sense_ratio)
  cpc_index = to_num(stats.cpc_index(df),2)
  display['CPC Index'] = pd.Series(cpc_index)
  tail_ratio = to_num(stats.tail_ratio(df),2)
  display['Tail Ratio'] = pd.Series(tail_ratio)
  outlier_win_ratio = to_num(stats.outlier_win_ratio(df),2)
  display['Outlier Win Ratio'] = pd.Series(outlier_win_ratio)
  outlier_loss_ratio = to_num(stats.outlier_loss_ratio(df),2)
  display['Outlier Loss Ratio'] = pd.Series(outlier_loss_ratio)
  avg_win = to_pct(stats.avg_win(df, aggregate='M'),2)
  display['Avg. Up Month'] = pd.Series(avg_win)
  avg_loss = to_pct(stats.avg_loss(df, aggregate='M'),2)
  display['Avg. Down Month'] = pd.Series(avg_loss)
  win_rate = to_pct(stats.win_rate(df),2)
  display['Win Days %'] = pd.Series(win_rate)
  win_rate_m = to_pct(stats.win_rate(df, aggregate='M'),2)
  display['Win Month %'] = pd.Series(win_rate_m)
  win_rate_q = to_pct(stats.win_rate(df, aggregate='Q'),2)
  display['Win Quarter %'] = pd.Series(win_rate_q)
  win_rate_y = to_pct(stats.win_rate(df, aggregate='Y'),2)
  display['Win Year %'] = pd.Series(win_rate_y)
  best = to_pct(stats.best(df),2)
  display['Best Day'] = pd.Series(best)
  worst = to_pct(stats.worst(df),2)
  display['Worst Day'] = pd.Series(worst)
  best_m = to_pct(stats.best(df, aggregate='M'),2)
  display['Best Month'] = pd.Series(best_m)
  worst_m = to_pct(stats.worst(df, aggregate='M'),2)
  display['Worst Month'] = pd.Series(worst_m)
  best_y = to_pct(stats.best(df, aggregate='Y'),2)
  display['Best Year'] = pd.Series(best_y)
  worst_y = to_pct(stats.worst(df, aggregate='Y'),2)
  display['Worst Year'] = pd.Series(worst_y)
  print(tabulate(display.T, headers="keys", tablefmt='simple'))
  print('\n')
  for c in dd_details.columns.get_level_values(0).unique():
    print('%s : 5 Worst Drawdowns' % c)
    print(tabulate(dd_details[c].sort_values(by='max drawdown', ascending=True)[:5], headers="keys", tablefmt='simple', showindex=False))
    print('\n')
  return None