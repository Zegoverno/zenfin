import warnings
import matplotlib.pyplot as _plt
from matplotlib.ticker import (
    StrMethodFormatter as StrMethodFormatter,
    FuncFormatter as FuncFormatter
)

import numpy as np
import pandas as pd
import seaborn as sns

from . import stats, _plot, utils, reports

def plt_returns(returns, benchmark, rf, 
                title='Cumulative returns',
                grayscale=False,
                figsize=(10, 6),
                fontname='Arial',
                lw=1.5,
                ylabel=False,
                subtitle=True,
                savefig=None,
                show=True):
  returns = utils.to_rolling_returns(returns)
  benchmark = utils.to_rolling_returns(benchmark)
  rf = utils.to_rolling_returns(rf)
  fig = _plot.timeseries(returns, benchmark, rf, title, ylabel=ylabel, lw=lw, figsize=figsize, fontname=fontname, grayscale=grayscale, subtitle=subtitle, savefig=savefig, show=show )
  if not show:
    return fig 
    
def plt_log_returns(returns, benchmark, rf, 
                title='Cumulative returns (Log scaled)',
                grayscale=False,
                figsize=(10, 6),
                fontname='Arial',
                lw=1.5,
                ylabel=False,
                subtitle=True,
                savefig=None,
                show=True):
  returns = utils.to_rolling_returns(returns)
  benchmark = utils.to_rolling_returns(benchmark)
  rf = utils.to_rolling_returns(rf)
  fig = _plot.timeseries(returns, benchmark, rf, title, log_scale=True, ylabel=ylabel, lw=lw, figsize=figsize, fontname=fontname, grayscale=grayscale, subtitle=subtitle, savefig=savefig, show=show )
  if not show:
    return fig

def plt_volmatch_returns(returns, benchmark, rf, 
                title='Cumulative returns (Volatility Matched)',
                grayscale=False,
                figsize=(10, 6),
                fontname='Arial',
                lw=1.5,
                ylabel=False,
                subtitle=True,
                savefig=None,
                show=True):
  returns = utils.match_volatility(returns, benchmark)
  returns = utils.to_rolling_returns(returns)
  benchmark = utils.to_rolling_returns(benchmark)
  rf = utils.to_rolling_returns(rf)
  fig = _plot.timeseries(returns, benchmark, rf, title, ylabel=ylabel, lw=lw, figsize=figsize, fontname=fontname, grayscale=grayscale, subtitle=subtitle, savefig=savefig, show=show )
  if not show:
    return fig 

def plt_yearly_returns(returns, benchmark, rf, 
                title='EOY Returns',
                grayscale=False,
                figsize=(10, 5),
                fontname='Arial',
                hlw=1.5,
                hlcolor='red',
                hllabel='',
                ylabel=False,
                subtitle=True,
                savefig=None,
                show=True):
  returns = returns.resample('A').apply(stats.total_return)
  benchmark = benchmark.resample('A').apply(stats.total_return)
  rf = rf.resample('A').apply(stats.total_return)
  fig = _plot.returns_bars(returns, benchmark, rf, title, ylabel=ylabel,hline=returns.mean(), hlw=hlw, hllabel=hllabel, hlcolor=hlcolor, figsize=figsize, fontname=fontname, grayscale=grayscale, subtitle=subtitle, savefig=savefig, show=show )
  if not show:
    return fig 

def plt_histogram(returns,
                  resample='M',
                  title='',
                  grayscale=False,
                  figsize=(10, 5),
                  fontname='Arial',
                  ylabel=False,
                  subtitle=True,
                  savefig=None,
                  show=True):
  if resample == 'W':
      title = "Weekly"
  elif resample == 'M':
      title = "Monthly"
  elif resample == 'Q':
      title = "Quarterly"
  elif resample == 'A':
      title = "Annual"
  else:
      title = ""
  returns = returns.resample(resample).apply(stats.total_return)
  fig = _plot.histogram(returns, title="Distribution of %sReturns" % title, ylabel=ylabel, figsize=figsize, fontname=fontname, grayscale=grayscale, subtitle=subtitle, savefig=savefig, show=show )
  if not show:
    return fig

def plt_daily_returns(returns,
                  title="Daily Returns",
                  grayscale=False,
                  figsize=(10, 4),
                  fontname='Arial',
                  lw=0.5,
                  log_scale=False,
                  ylabel=False,
                  subtitle=True,
                  savefig=None,
                  show=True):

    fig = _plot.timeseries(returns, None, 
                          title,
                          ylabel=ylabel,
                          log_scale=log_scale,
                          lw=lw,
                          figsize=figsize,
                          fontname=fontname,
                          grayscale=grayscale,
                          subtitle=subtitle,
                          savefig=savefig,
                          show=show)
    if not show:
        return fig

def plt_rolling_beta(returns, benchmark,
                 window1=126, window1_label="6-Months",
                 window2=252, window2_label="12-Months",
                 title="Rolling Beta to Benchmark",
                 lw=1.5,
                 fontname='Arial',
                 grayscale=False,
                 figsize=(10, 3),
                 ylabel=True,
                 subtitle=True,
                 savefig=None,
                 show=True):
    returns = pd.DataFrame(returns)
    beta1 = stats.rolling_greeks(returns, benchmark, window1)
    beta1 = beta1[beta1.columns.get_level_values(0)[0]]['beta']
    beta2= stats.rolling_greeks(returns, benchmark, window2)
    beta2 = beta2[beta2.columns.get_level_values(0)[0]]['beta']
    fig = _plot.rolling_beta(beta1, beta2,
                            window1_label=window1_label,
                            window2_label=window2_label,
                            title=title,
                            fontname=fontname,
                            grayscale=grayscale,
                            lw=lw,
                            figsize=figsize,
                            ylabel=ylabel,
                            subtitle=subtitle,
                            savefig=savefig,
                            show=show)
    if not show:
        return fig

def plt_rolling_volatility(returns, benchmark=None,
                       period=126, period_label="6-Months",
                       periods_per_year=252,
                       lw=1.5,
                       fontname='Arial',
                       grayscale=False,
                       figsize=(10, 3),
                       ylabel="Volatility",
                       subtitle=True,
                       savefig=None,
                       show=True):

    returns = stats.rolling_volatility(returns, period, periods_per_year)
    benchmark = stats.rolling_volatility(benchmark, period, periods_per_year)

    fig = _plot.rolling_stats(returns, benchmark,
                                   hline=returns.mean(),
                                   hlw=1.5,
                                   ylabel=ylabel,
                                   title='Rolling Volatility (%s)' % period_label,
                                   fontname=fontname,
                                   grayscale=grayscale,
                                   lw=lw,
                                   figsize=figsize,
                                   subtitle=subtitle,
                                   savefig=savefig, show=show)
    if not show:
        return fig

def plt_rolling_sharpe(returns, benchmark=None, rf=0.,
                       period=126, period_label="6-Months",
                       periods_per_year=252,
                       lw=1.25,
                       fontname='Arial',
                       grayscale=False,
                       figsize=(10, 3),
                       ylabel="Sharpe",
                       subtitle=True,
                       savefig=None,
                       show=True):

    returns = stats.rolling_sharpe(returns, rf, period, True, periods_per_year )

    if benchmark is not None:
        benchmark = stats.rolling_sharpe(benchmark, rf, period, True, periods_per_year)

    fig = _plot.rolling_stats(returns, benchmark,
                            hline=returns.mean(),
                            hlw=1.5,
                            ylabel=ylabel,
                            title='Rolling Sharpe (%s)' % period_label,
                            fontname=fontname,
                            grayscale=grayscale,
                            lw=lw,
                            figsize=figsize,
                            subtitle=subtitle,
                            savefig=savefig,
                            show=show)
    if not show:
        return fig


def plt_rolling_sortino(returns, benchmark=None, rf=0.,
                       period=126, period_label="6-Months",
                       periods_per_year=252,
                       lw=1.25,
                       fontname='Arial',
                       grayscale=False,
                       figsize=(10, 3),
                       ylabel="Sortino",
                       subtitle=True,
                       savefig=None,
                       show=True):

    returns = stats.rolling_sortino(returns, rf, period, True, periods_per_year )

    if benchmark is not None:
        benchmark = stats.rolling_sortino(benchmark, rf, period, True, periods_per_year)

    fig = _plot.rolling_stats(returns, benchmark,
                            hline=returns.mean(),
                            hlw=1.5,
                            ylabel=ylabel,
                            title='Rolling Sortino (%s)' % period_label,
                            fontname=fontname,
                            grayscale=grayscale,
                            lw=lw,
                            figsize=figsize,
                            subtitle=subtitle,
                            savefig=savefig,
                            show=show)
    if not show:
        return fig

def plt_drawdowns_periods(returns, periods=5,
                          lw=1.5,
                          log_scale=False,
                          fontname='Arial',
                          grayscale=False,
                          figsize=(10, 5),
                          ylabel=True,
                          subtitle=True,
                          savefig=None,
                          show=True):


    dd = utils.to_drawdown_series(returns)
    dddf = utils.drawdown_details(dd)
    longest_dd = dddf.sort_values(
        by='days', ascending=False, kind='mergesort')[:periods]

    returns = utils.to_rolling_returns(returns)

    fig = _plot.longest_drawdowns(returns, longest_dd, periods,
                                 lw=lw,
                                 log_scale=log_scale,
                                 fontname=fontname,
                                 grayscale=grayscale,
                                 figsize=figsize,
                                 ylabel=ylabel,
                                 subtitle=subtitle,
                                 savefig=savefig,
                                 show=show)
    if not show:
        return fig

def plt_drawdown(returns,
             title='Underwater Plot',
             grayscale=False,
             figsize=(10, 5),
             fontname='Arial',
             lw=1,
             log_scale=False,
             ylabel="Drawdown",
             subtitle=True,
             savefig=None,
             show=True):

    dd = utils.to_drawdown_series(returns)
    fig = _plot.timeseries(dd,
                          title=title,
                          hline=dd.mean(),
                          hlw=2,
                          hllabel="Average",
                          returns_label="Drawdown",
                          log_scale=log_scale,
                          fill=True,
                          lw=lw,
                          figsize=figsize,
                          ylabel=ylabel,
                          fontname=fontname,
                          grayscale=grayscale,
                          subtitle=subtitle,
                          savefig=savefig,
                          show=show)
    if not show:
        return fig

def plt_monthly_heatmap(returns,
             title='Monthly Returns',
             annot_size=10,
             figsize=(10, 5),
             cbar=True,
             square=False,
             eoy=False,
             grayscale=False,
             fontname='Arial',
             ylabel=False,
             savefig=None,
             show=True):

    returns = stats.monthly_returns(returns, eoy=eoy) * 100
    fig = _plot.heatmap(returns,
             title='Monthly Returns',
             grayscale=False,
             figsize=(10, 5),
             fontname='Arial',
             ylabel=False,
             savefig=None,
             show=True)
    
    if not show:
        return fig

def plt_distribution(returns,
                     fontname='Arial',
                     grayscale=False,
                     ylabel=True,
                     figsize=(10, 6),
                     subtitle=True,
                     savefig=None,
                     show=True):
  port = pd.DataFrame(returns.fillna(0))
  port.columns = ['Daily']

  port['Weekly'] = port['Daily'].resample(
      'W-MON').apply(stats.total_return)
  port['Weekly'].ffill(inplace=True)

  port['Monthly'] = port['Daily'].resample(
      'M').apply(stats.total_return)
  port['Monthly'].ffill(inplace=True)

  port['Quarterly'] = port['Daily'].resample(
      'Q').apply(stats.total_return)
  port['Quarterly'].ffill(inplace=True)

  port['Yearly'] = port['Daily'].resample(
      'A').apply(stats.total_return)
  port['Yearly'].ffill(inplace=True)

  fig = _plot.distribution(port,
                          fontname=fontname,
                          grayscale=grayscale,
                          figsize=figsize,
                          ylabel=ylabel,
                          subtitle=subtitle,
                          savefig=savefig,
                          show=show)
  if not show:
      return fig