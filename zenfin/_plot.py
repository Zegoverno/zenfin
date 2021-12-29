import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FormatStrFormatter, FuncFormatter

import pandas as pd
import numpy as np

import seaborn as sns

sns.set(font_scale=1.1, rc={
    'figure.figsize': (10, 6),
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'grid.color': '#dddddd',
    'grid.linewidth': 0.5,
    "lines.linewidth": 1.5,
    'text.color': '#333333',
    'xtick.color': '#666666',
    'ytick.color': '#666666'
})

_FLATUI_COLORS = ["#fedd78", "#348dc1", "#af4b64",
                  "#4fa487", "#9b59b6", "#808080"]
_GRAYSCALE_COLORS = ['silver', '#222222', 'gray'] * 3

def _get_colors(grayscale):
    colors = _FLATUI_COLORS
    ls = '-'
    alpha = .8
    if grayscale:
        colors = _GRAYSCALE_COLORS
        ls = '-'
        alpha = 0.5
    return colors, ls, alpha

def format_pct_axis(x, _):
    x *= 100  # lambda x, loc: "{:,}%".format(int(x * 100))
    if x >= 1e12:
        res = '%1.1fT%%' % (x * 1e-12)
        return res.replace('.0T%', 'T%')
    if x >= 1e9:
        res = '%1.1fB%%' % (x * 1e-9)
        return res.replace('.0B%', 'B%')
    if x >= 1e6:
        res = '%1.1fM%%' % (x * 1e-6)
        return res.replace('.0M%', 'M%')
    if x >= 1e3:
        res = '%1.1fK%%' % (x * 1e-3)
        return res.replace('.0K%', 'K%')
    res = '%1.0f%%' % x
    return res.replace('.0%', '%')
  
def timeseries(returns, benchmark, rf,
                    title="Returns",
                    fill=False, returns_label="Strategy",
                    hline=None, hlw=None, hlcolor="red", hllabel="",
                    percent=True, log_scale=False,
                    lw=1.5, figsize=(10, 6), ylabel="",
                    grayscale=False, fontname="Arial",
                    subtitle=True, savefig=None, show=True):

    colors, ls, alpha = _get_colors(grayscale)

    fig, ax = plt.subplots(figsize=figsize)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    fig.suptitle(title+"\n", y=.99, fontweight="bold", fontname=fontname,
                 fontsize=14, color="black")

    if subtitle:
        ax.set_title("\n%s - %s                  " % (
            returns.index.date[:1][0].strftime('%e %b \'%y'),
            returns.index.date[-1:][0].strftime('%e %b \'%y')
        ), fontsize=12, color='gray')

    fig.set_facecolor('white')
    ax.set_facecolor('white')

    if isinstance(rf, pd.Series):
      ax.plot(rf, lw=lw, ls=ls, label="Risk-Free", color=colors[1])

    if isinstance(benchmark, pd.Series):
      ax.plot(benchmark, lw=lw, ls=ls, label="Benchmark", color=colors[2])

    alpha = .25 if grayscale else 1
    ax.plot(returns, lw=lw, label=returns_label, color=colors[3], alpha=alpha)

    if fill:
        ax.fill_between(returns.index, 0, returns, color=colors[1], alpha=.25)

    # rotate and align the tick labels so they look better
    fig.autofmt_xdate()

    # use a more precise date string for the x axis locations in the toolbar
    ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')

    if hline:
        ax.legend(fontsize=12)
        if grayscale:
            hlcolor = 'black'
        ax.axhline(hline, ls="--", lw=hlw, color=hlcolor,
                   label=hllabel, zorder=2)

    ax.axhline(0, ls="-", lw=1,
               color='gray', zorder=1)
    ax.axhline(0, ls="--", lw=1,
               color='white' if grayscale else 'black', zorder=2)

    plt.yscale("symlog" if log_scale else "linear")

    if percent:
        ax.yaxis.set_major_formatter(FuncFormatter(format_pct_axis))

    ax.set_xlabel('')
    if ylabel:
        ax.set_ylabel(ylabel, fontname=fontname,
                      fontweight='bold', fontsize=12, color="black")
    ax.yaxis.set_label_coords(-.1, .5)

    ax.legend(fontsize=12)
    
    try:
        plt.subplots_adjust(hspace=0, bottom=0, top=1)
    except Exception:
        pass

    try:
        fig.tight_layout()
    except Exception:
        pass

    if savefig:
        if isinstance(savefig, dict):
            plt.savefig(**savefig)
        else:
            plt.savefig(savefig)

    if show:
        plt.show(block=False)

    plt.close()

    if not show:
        return fig

    return None

def returns_bars(returns, benchmark, rf,
                      title="Returns",
                      fill=False,
                      returns_label="Strategy",
                      hline=None, hlw=None, hlcolor="red", hllabel="",
                      log_scale=False,
                      lw=1.5,
                      figsize=(10, 6),
                      ylabel="",
                      grayscale=False,
                      fontname="Arial",
                      subtitle=True,
                      savefig=None,
                      show=True):

    df = pd.DataFrame(index=returns.index, data={returns_label: returns,
                                                 'Benchmark': benchmark,
                                                 'Risk-free': rf
                                                 })
    colors, ls, alpha = _get_colors(grayscale)

    fig, ax = plt.subplots(figsize=figsize)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    fig.suptitle(title+"\n", y=.99, fontweight="bold", fontname=fontname,
                 fontsize=14, color="black")

    if subtitle:
        ax.set_title("\n%s - %s                  " % (
            returns.index.date[:1][0].strftime('%e %b \'%y'),
            returns.index.date[-1:][0].strftime('%e %b \'%y')
        ), fontsize=12, color='gray')

    df.plot(kind='bar', ax=ax, color=colors)

    fig.set_facecolor('white')
    ax.set_facecolor('white')

    try:
        ax.set_xticklabels(df.index.year)
        years = sorted(list(set(df.index.year)))
    except AttributeError:
        ax.set_xticklabels(df.index)
        years = sorted(list(set(df.index)))

    # rotate and align the tick labels so they look better
    fig.autofmt_xdate()

    if hline:
        if grayscale:
            hlcolor = 'gray'
        ax.axhline(hline, ls="--", lw=hlw, color=hlcolor,
                   label=hllabel, zorder=2)

    ax.axhline(0, ls="--", lw=1, color="#000000", zorder=2)

    ax.legend(fontsize=12)

    plt.yscale("symlog" if log_scale else "linear")

    ax.set_xlabel('')
    if ylabel:
        ax.set_ylabel("Returns", fontname=fontname,
                      fontweight='bold', fontsize=12, color="black")
        ax.yaxis.set_label_coords(-.1, .5)

    ax.yaxis.set_major_formatter(FuncFormatter(format_pct_axis))

    try:
        plt.subplots_adjust(hspace=0, bottom=0, top=1)
    except Exception:
        pass

    try:
        fig.tight_layout()
    except Exception:
        pass

    if savefig:
        if isinstance(savefig, dict):
            plt.savefig(**savefig)
        else:
            plt.savefig(savefig)

    if show:
        plt.show(block=False)

    plt.close()

    if not show:
        return fig

    return None