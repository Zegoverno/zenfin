import warnings
import matplotlib.pyplot as _plt
from matplotlib.ticker import (
    StrMethodFormatter as StrMethodFormatter,
    FuncFormatter as FuncFormatter
)

import numpy as np
import pandas as pd
import seaborn as sns

from . import stats, _plot, utils

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
  returns = stats.rolling_returns(returns)
  benchmark = stats.rolling_returns(benchmark)
  rf = stats.rolling_returns(rf)
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
  returns = stats.rolling_returns(returns)
  benchmark = stats.rolling_returns(benchmark)
  rf = stats.rolling_returns(rf)
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
  returns = stats.rolling_returns(returns)
  benchmark = stats.rolling_returns(benchmark)
  rf = stats.rolling_returns(rf)
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
  returns = returns.resample('A').apply(stats.total_returns)
  benchmark = benchmark.resample('A').apply(stats.total_returns)
  rf = rf.resample('A').apply(stats.total_returns)
  fig = _plot.returns_bars(returns, benchmark, rf, title, ylabel=ylabel,hline=returns.mean(), hlw=hlw, hllabel=hllabel, hlcolor=hlcolor, figsize=figsize, fontname=fontname, grayscale=grayscale, subtitle=subtitle, savefig=savefig, show=show )
  if not show:
    return fig 