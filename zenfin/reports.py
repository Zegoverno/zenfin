import pandas as pd
from . import utils
from . import stats
from . import plots
from tabulate import tabulate

def report_metrics(dr, bench, rf, periods=252):
  rfd = rf['dR'].rename('DI')
  rfy = rf['yR'].rename('DI')
  frames = [dr, bench]
  df = pd.concat(frames, axis=1)
  dd = utils.to_drawdown_series(df)
  dd_details = utils.drawdown_details(dd)
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
  total_return = utils.to_pct(stats.total_return(df),2)
  display['Total return'] = pd.Series(total_return)
  cagr = utils.to_pct(stats.cagr(df),2)
  display['CAGR'] = pd.Series(cagr)
  d_r = utils.to_pct(stats.expected_return(df),2)
  display['E[Daily Returns]'] = pd.Series(d_r)
  m_r = utils.to_pct(stats.expected_return(df, aggregate='M'),2)
  display['E[Monthly Returns]'] = pd.Series(m_r)
  y_r = utils.to_pct(stats.expected_return(df, aggregate='Y'),2)
  display['E[Yearly Returns]'] = pd.Series(y_r)
  r_mtd = utils.to_pct(stats.total_return(utils.mtd(df, df.index[-1])),2)
  display['MTD'] = pd.Series(r_mtd)
  r_l3m = utils.to_pct(stats.total_return(utils.l3m(df, df.index[-1])),2)
  display['3M'] = pd.Series(r_l3m)
  r_l6m = utils.to_pct(stats.total_return(utils.l6m(df, df.index[-1])),2)
  display['6M'] = pd.Series(r_l6m)
  r_ytd = utils.to_pct(stats.total_return(utils.ytd(df, df.index[-1])),2)
  display['YTD'] = pd.Series(r_ytd)
  r_l1y = utils.to_pct(stats.total_return(utils.l1y(df, df.index[-1])),2)
  display['1Y'] = pd.Series(r_l1y)
  cagr_3y = utils.to_pct(stats.cagr(utils.l3y(df, df.index[-3])),2)
  display['3Y (aa)'] = pd.Series(cagr_3y)
  cagr_5y = utils.to_pct(stats.cagr(utils.l5y(df, df.index[-1])),2)
  display['5Y (aa)'] = pd.Series(cagr_5y)
  cagr_10y = utils.to_pct(stats.cagr(utils.l10y(df, df.index[-1])),2)
  display['10Y (aa)'] = pd.Series(cagr_10y)
  display['Inception (aa)'] = pd.Series(cagr)
  print(tabulate(display.T, headers="keys", tablefmt='simple'))
  print('\n')
  display = pd.DataFrame()
  print('Performance Info')
  frames = [dr, bench]
  df = pd.concat(frames, axis=1)
  sharpe = utils.to_num(stats.sharpe(df, rfy, smart=False),2)
  display['Sharpe'] = pd.Series(sharpe)
  smart_sharpe = utils.to_num(stats.sharpe(df, rfy, smart=True),2)
  display['Smart Sharpe'] = pd.Series(smart_sharpe)
  sortino = utils.to_num(stats.sortino(df, rfy, smart=False),2)
  display['Sortino'] = pd.Series(sortino)
  smart_sortino = utils.to_num(stats.sortino(df, rfy, smart=True),2)
  display['Smart Sortino'] = pd.Series(smart_sortino)
  adj_sortino = utils.to_num(stats.adjusted_sortino(df, rfy, smart=False),2)
  display['Sortino/√2'] = pd.Series(adj_sortino)
  adj_smart_sortino = utils.to_num(stats.adjusted_sortino(df, rfy, smart=True),2)
  display['Smart Sortino/√2'] = pd.Series(adj_smart_sortino)
  omega = utils.to_num(stats.omega(df, rfy),2)
  display['Omega'] = pd.Series(omega)
  beta = utils.to_num(stats.beta(dr, bench),2)
  display['Beta'] = pd.Series(beta)
  alpha = utils.to_num(stats.alpha(dr, bench),2)
  display['Alpha'] = pd.Series(alpha)
  print(tabulate(display.T, headers="keys", tablefmt='simple'))
  print('\n')
  display = pd.DataFrame()
  print('Risk Info')
  frames = [dr, bench]
  df = pd.concat(frames, axis=1)
  max_drawdown = utils.to_pct(stats.max_drawdown(df),2)
  display['Max Drawdown'] = pd.Series(max_drawdown)
  longest_drawdown_days = utils.to_num(stats.longest_drawdown_days(df),0)
  display['Longest DD Days'] = pd.Series(longest_drawdown_days)
  avg_drawdown = utils.to_pct(stats.avg_drawdown(dd_details),2)
  display['Avg. Drawdown'] = pd.Series(avg_drawdown)
  avg_drawdown_days = utils.to_num(stats.avg_drawdown_days(df),0)
  display['Avg. Drawdown Days'] = pd.Series(avg_drawdown_days)
  volatility = utils.to_pct(stats.volatility(df, periods),2)
  display['Volatility (aa)'] = pd.Series(volatility)
  r_squared = utils.to_num(stats.r_squared(dr, bench),2)
  display['R^2'] = pd.Series(r_squared)
  calmar = utils.to_num(stats.calmar(df),2)
  display['Calmar'] = pd.Series(calmar)
  skew = utils.to_num(stats.skew(df),2)
  display['Skew'] = pd.Series(skew)
  kurtosis = utils.to_num(stats.kurtosis(df),2)
  display['Kurtosis'] = pd.Series(kurtosis)
  kelly_criterion = utils.to_pct(stats.kelly_criterion(df),2)
  display['Kelly Criterion'] = pd.Series(kelly_criterion)
  risk_of_ruin = utils.to_pct(stats.risk_of_ruin(df),2)
  display['Risk of Ruin'] = pd.Series(risk_of_ruin)
  value_at_risk = utils.to_pct(stats.value_at_risk(df),2)
  display['Daily VaR'] = pd.Series(value_at_risk)
  conditional_value_at_risk = utils.to_pct(stats.conditional_value_at_risk(df),2)
  display['Expected Shortfall (cVaR)'] = pd.Series(conditional_value_at_risk)
  recovery_factor = utils.to_num(stats.recovery_factor(df),2)
  display['Recovery Factor'] = pd.Series(recovery_factor)
  ulcer_index = utils.to_num(stats.ulcer_index(df),2)
  display['Ulcer Index'] = pd.Series(ulcer_index)
  serenity_index = utils.to_num(stats.serenity_index(df, rfy),2)
  display['Serenity Index'] = pd.Series(serenity_index)
  print(tabulate(display.T, headers="keys", tablefmt='simple'))
  print('\n')
  display = pd.DataFrame()
  print('P&L Info')
  frames = [dr, bench]
  df = pd.concat(frames, axis=1)
  # showing based excess_returns! should show as returns?
  er = utils.to_excess_returns(df, rfy)
  gtp = utils.to_num(stats.gain_to_pain_ratio(er),2)
  display['Gain/Pain Ratio'] = pd.Series(gtp)
  gtp_1m = utils.to_num(stats.gain_to_pain_ratio(er, 'M'),2)
  display['Gain/Pain (1M)'] = pd.Series(gtp_1m)
  gtp_3m = utils.to_num(stats.gain_to_pain_ratio(er, 'Q'),2)
  display['Gain/Pain (3M)'] = pd.Series(gtp_3m)
  gtp_6m = utils.to_num(stats.gain_to_pain_ratio(er, 'BQ'),2)
  display['Gain/Pain (6M)'] = pd.Series(gtp_6m)
  gtp_1y = utils.to_num(stats.gain_to_pain_ratio(er, 'A'),2)
  display['Gain/Pain (1Y)'] = pd.Series(gtp_1y)
  payoff_ratio = utils.to_num(stats.payoff_ratio(df),2)
  display['Payoff Ratio'] = pd.Series(payoff_ratio)
  profit_factor = utils.to_num(stats.profit_factor(df),2)
  display['Profit Factor'] = pd.Series(profit_factor)
  common_sense_ratio = utils.to_num(stats.common_sense_ratio(df),2)
  display['Common Sense Ratio'] = pd.Series(common_sense_ratio)
  cpc_index = utils.to_num(stats.cpc_index(df),2)
  display['CPC Index'] = pd.Series(cpc_index)
  tail_ratio = utils.to_num(stats.tail_ratio(df),2)
  display['Tail Ratio'] = pd.Series(tail_ratio)
  outlier_win_ratio = utils.to_num(stats.outlier_win_ratio(df),2)
  display['Outlier Win Ratio'] = pd.Series(outlier_win_ratio)
  outlier_loss_ratio = utils.to_num(stats.outlier_loss_ratio(df),2)
  display['Outlier Loss Ratio'] = pd.Series(outlier_loss_ratio)
  avg_win = utils.to_pct(stats.avg_win(df, aggregate='M'),2)
  display['Avg. Up Month'] = pd.Series(avg_win)
  avg_loss = utils.to_pct(stats.avg_loss(df, aggregate='M'),2)
  display['Avg. Down Month'] = pd.Series(avg_loss)
  win_rate = utils.to_pct(stats.win_rate(df),2)
  display['Win Days %'] = pd.Series(win_rate)
  win_rate_m = utils.to_pct(stats.win_rate(df, aggregate='M'),2)
  display['Win Month %'] = pd.Series(win_rate_m)
  win_rate_q = utils.to_pct(stats.win_rate(df, aggregate='Q'),2)
  display['Win Quarter %'] = pd.Series(win_rate_q)
  win_rate_y = utils.to_pct(stats.win_rate(df, aggregate='Y'),2)
  display['Win Year %'] = pd.Series(win_rate_y)
  best = utils.to_pct(stats.best(df),2)
  display['Best Day'] = pd.Series(best)
  worst = utils.to_pct(stats.worst(df),2)
  display['Worst Day'] = pd.Series(worst)
  best_m = utils.to_pct(stats.best(df, aggregate='M'),2)
  display['Best Month'] = pd.Series(best_m)
  worst_m = utils.to_pct(stats.worst(df, aggregate='M'),2)
  display['Worst Month'] = pd.Series(worst_m)
  best_y = utils.to_pct(stats.best(df, aggregate='Y'),2)
  display['Best Year'] = pd.Series(best_y)
  worst_y = utils.to_pct(stats.worst(df, aggregate='Y'),2)
  display['Worst Year'] = pd.Series(worst_y)
  print(tabulate(display.T, headers="keys", tablefmt='simple'))
  print('\n')
  for c in dd_details.columns.get_level_values(0).unique():
    print('%s : 5 Worst Drawdowns' % c)
    print(tabulate(dd_details[c].sort_values(by='max drawdown', ascending=True)[:5], headers="keys", tablefmt='simple', showindex=False))
    print('\n')
  return None

def report_plots(returns, benchmark, rf, grayscale=False, figsize=(8, 5), periods=252):
  rfd=rf['dR']
  rfy=rf['yR']
  plots.plt_returns(returns, benchmark, rfd,
              grayscale=grayscale,
              figsize=(figsize[0], figsize[0]*.6),
              show=True,
              ylabel=False)
  plots.plt_log_returns(returns, benchmark, rfd,
              grayscale=grayscale,
              figsize=(figsize[0], figsize[0]*.5),
              show=True,
              ylabel=False)
  plots.plt_volmatch_returns(returns, benchmark, rfd,
              grayscale=grayscale,
              figsize=(figsize[0], figsize[0]*.5),
              show=True,
              ylabel=False)
  plots.plt_yearly_returns(returns, benchmark, rfd,
              grayscale=grayscale,
              figsize=(figsize[0], figsize[0]*.5),
              show=True,
              ylabel=False)
  
  plots.plt_histogram(returns,
                grayscale=grayscale,
                figsize=(figsize[0], figsize[0]*.5),
                show=True,
                ylabel=False)
  
  plots.plt_daily_returns(returns,
                    grayscale=grayscale,
                    figsize=(figsize[0], figsize[0]*.3),
                    show=True,
                    ylabel=False)
  
  plots.plt_rolling_beta(returns, benchmark,
                   grayscale=grayscale,
                   window1=int(periods/2),
                   window2=periods,
                   figsize=(figsize[0], figsize[0]*.3),
                   show=True,
                   ylabel=False)
  
  plots.plt_rolling_volatility(returns, benchmark,
                         grayscale=grayscale,
                         figsize=(figsize[0], figsize[0]*.3),
                         show=True,
                         ylabel=False,
                         period=int(periods/2))
  
  plots.plt_rolling_sharpe(returns, benchmark, rfy,
                     grayscale=grayscale,
                     figsize=(figsize[0], figsize[0]*.3),
                     show=True,
                     ylabel=False,
                     period=int(periods/2))

  plots.plt_rolling_sortino(returns, benchmark, rfy,
                      grayscale=grayscale,
                      figsize=(figsize[0], figsize[0]*.3),
                      show=True,
                      ylabel=False,
                      period=int(periods/2))
  
  plots.plt_drawdowns_periods(returns, periods=5,
                        grayscale=grayscale,
                        figsize=(figsize[0], figsize[0]*.5),
                        show=True,
                        ylabel=False)

  plots.plt_drawdown(returns,
               grayscale=grayscale,
               figsize=(figsize[0], figsize[0]*.4),
               show=True,
               ylabel=False)

  plots.plt_monthly_heatmap(returns,
                      grayscale=grayscale,
                      figsize=(figsize[0], figsize[0]*.5),
                      show=True,
                      ylabel=False)

  plots.plt_distribution(returns,
                   grayscale=grayscale,
                   figsize=(figsize[0], figsize[0]*.5),
                   show=True, 
                   ylabel=False)
