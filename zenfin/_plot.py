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

_FLATUI_COLORS = [ "#348dc1", "#4fa487", "#af4b64",
                  "#fedd78", "#9b59b6", "#808080"]
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
  
def timeseries(returns, benchmark=None, rf=0.,
                    title="Returns",
                    fill=False,
                    returns_label="Strategy",
                    hline=None,
                    hlw=None,
                    hlcolor="red",
                    hllabel="",
                    percent=True,
                    log_scale=False,
                    lw=1.5,
                    figsize=(10, 6),
                    ylabel="",
                    grayscale=False,
                    fontname="Arial",
                    subtitle=True,
                    savefig=None,
                    show=True):

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
      ax.plot(rf, lw=lw, ls=ls, label="Risk-Free", color=colors[2])

    if isinstance(benchmark, pd.Series):
      ax.plot(benchmark, lw=lw, ls=ls, label="Benchmark", color=colors[1])

    alpha = .25 if grayscale else 1
    ax.plot(returns, lw=lw, label=returns_label, color=colors[0], alpha=alpha)

    if fill:
        ax.fill_between(returns.index, 0, returns, color=colors[0], alpha=.25)

    fig.autofmt_xdate()

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

def histogram(returns,
                   bins=20,
                   fontname='Arial', 
                   grayscale=False,
                   title="Returns", 
                   kde=True, 
                   figsize=(10, 6),
                   ylabel=True,
                   subtitle=True,
                   savefig=None,
                   show=True):

    colors = ['#348dc1', '#003366', 'red']
    if grayscale:
        colors = ['silver', 'gray', 'black']

    fig, ax = plt.subplots(figsize=figsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    fig.suptitle(title+"\n", y=.99, fontweight="bold", fontname=fontname,
                 fontsize=14, color="black")

    if subtitle:
        ax.set_title("\n%s - %s                   " % (
            returns.index.date[:1][0].strftime('%Y'),
            returns.index.date[-1:][0].strftime('%Y')
        ), fontsize=12, color='gray')

    fig.set_facecolor('white')
    ax.set_facecolor('white')

    ax.axvline(returns.mean(), ls="--", lw=1.5,
               color=colors[2], zorder=2, label="Average")

    sns.histplot(returns, bins=bins,
                  color=colors[0],
                  alpha=1,
                  kde=kde,
                  stat="density",
                  ax=ax)
    
    sns.kdeplot(returns, color='black', linewidth=1.5)

    ax.xaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, loc: "{:,}%".format(int(x*100))))

    ax.axhline(0.01, lw=1, color="#000000", zorder=2)
    ax.axvline(0, lw=1, color="#000000", zorder=2)

    ax.set_xlabel('')
    if ylabel:
        ax.set_ylabel("Occurrences", fontname=fontname,
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

def rolling_beta(beta1, beta2,
                      window1_label="",
                      window2_label="",
                      title="", 
                      hlcolor="red",
                      figsize=(10, 6),
                      grayscale=False,
                      fontname="Arial",
                      lw=1.5,
                      ylabel=True,
                      subtitle=True,
                      savefig=None,
                      show=True):

    colors, _, _ = _get_colors(grayscale)

    fig, ax = plt.subplots(figsize=figsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    fig.suptitle(title+"\n", y=.99, fontweight="bold", fontname=fontname,
                 fontsize=14, color="black")

    if subtitle:
        ax.set_title("\n%s - %s                   " % (
            beta1.index.date[:1][0].strftime('%e %b \'%y'),
            beta1.index.date[-1:][0].strftime('%e %b \'%y')
        ), fontsize=12, color='gray')

    ax.plot(beta1, lw=lw, label=window1_label, color=colors[1])
    ax.plot(beta2,lw=lw, label=window2_label, color="gray", alpha=0.8)

    mmin = min([-100, int(beta1.min()*100)])
    mmax = max([100, int(beta1.max()*100)])
    step = 50 if (mmax-mmin) >= 200 else 100
    ax.set_yticks([x / 100 for x in list(range(mmin, mmax, step))])

    hlcolor = 'black' if grayscale else hlcolor
    ax.axhline(beta1.mean(), ls="--", lw=1.5,
               color=hlcolor, zorder=2)

    ax.axhline(0, ls="--", lw=1, color="#000000", zorder=2)

    fig.autofmt_xdate()

    ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')

    if ylabel:
        ax.set_ylabel("Beta", fontname=fontname,
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

def rolling_stats(returns, benchmark=None,
                       title="",
                       returns_label="Strategy",
                       hline=None,
                       hlw=None,
                       hlcolor="red",
                       hllabel="",
                       lw=1.5,
                       figsize=(10, 6),
                       ylabel="",
                       grayscale=False,
                       fontname="Arial",
                       subtitle=True,
                       savefig=None,
                       show=True):

    colors, _, _ = _get_colors(grayscale)

    fig, ax = plt.subplots(figsize=figsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    if isinstance(benchmark, pd.Series):
        ax.plot(benchmark, lw=lw, label="Benchmark", color=colors[1], alpha=.8)

    ax.plot(returns, lw=lw, label=returns_label, color=colors[0])

    fig.autofmt_xdate()

    fig.suptitle(title+"\n", y=.99, fontweight="bold", fontname=fontname,
                 fontsize=14, color="black")

    if subtitle:
        ax.set_title("\n%s - %s                   " % (
            returns.index.date[:1][0].strftime('%e %b \'%y'),
            returns.index.date[-1:][0].strftime('%e %b \'%y')
        ), fontsize=12, color='gray')

    if hline:
        if grayscale:
            hlcolor = 'black'
        ax.axhline(hline, ls="--", lw=hlw, color=hlcolor,
                   label=hllabel, zorder=2)

    ax.axhline(0, ls="--", lw=1, color="#000000", zorder=2)

    if ylabel:
        ax.set_ylabel(ylabel, fontname=fontname,
                      fontweight='bold', fontsize=12, color="black")
        ax.yaxis.set_label_coords(-.1, .5)

    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

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

def longest_drawdowns(returns, longest_dd, periods,
                           lw=1.5,
                           fontname='Arial',
                           grayscale=False,
                           log_scale=False,
                           figsize=(10, 6),
                           ylabel=True,
                           subtitle=True,
                           savefig=None, 
                           show=True):

    colors = ['#348dc1', '#003366', 'red']
    if grayscale:
        colors = ['#000000'] * 3

    fig, ax = plt.subplots(figsize=figsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    fig.suptitle("Worst %.0f Drawdown Periods\n" %
                 periods, y=.99, fontweight="bold", fontname=fontname,
                 fontsize=14, color="black")
    if subtitle:
        ax.set_title("\n%s - %s                   " % (
            returns.index.date[:1][0].strftime('%e %b \'%y'),
            returns.index.date[-1:][0].strftime('%e %b \'%y')
        ), fontsize=12, color='gray')

    fig.set_facecolor('white')
    ax.set_facecolor('white')
    ax.plot(returns, lw=lw, label="Backtest", color=colors[0])

    highlight = 'black' if grayscale else 'red'
    for _, row in longest_dd.iterrows():
        ax.axvspan(*mdates.datestr2num([str(row['start']), str(row['end'])]),
                   color=highlight, alpha=.1)

  
    fig.autofmt_xdate()

    ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')

    ax.axhline(0, ls="--", lw=1, color="#000000", zorder=2)
    plt.yscale("symlog" if log_scale else "linear")
    if ylabel:
        ax.set_ylabel("Cumulative Returns", fontname=fontname,
                      fontweight='bold', fontsize=12, color="black")
        ax.yaxis.set_label_coords(-.1, .5)

    ax.yaxis.set_major_formatter(FuncFormatter(format_pct_axis))

    fig.autofmt_xdate()

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

def distribution(port,
                      figsize=(10, 6),
                      fontname='Arial',
                      grayscale=False,
                      ylabel=True,
                      subtitle=True,
                      compounded=True,
                      savefig=None,
                      show=True):

    colors = _FLATUI_COLORS
    if grayscale:
        colors = ['#f9f9f9', '#dddddd', '#bbbbbb', '#999999', '#808080']
    # colors, ls, alpha = _get_colors(grayscale)

    
    fig, ax = plt.subplots(figsize=figsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    fig.suptitle("Return Quantiles\n", y=.99,
                 fontweight="bold", fontname=fontname,
                 fontsize=14, color="black")

    if subtitle:
        ax.set_title("\n%s - %s                   " % (
            port.index.date[:1][0].strftime('%e %b \'%y'),
            port.index.date[-1:][0].strftime('%e %b \'%y')
        ), fontsize=12, color='gray')

    fig.set_facecolor('white')
    ax.set_facecolor('white')

    sns.boxplot(data=port, ax=ax, palette=tuple(colors[:5]))

    ax.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, loc: "{:,}%".format(int(x*100))))

    if ylabel:
        ax.set_ylabel('Rerurns', fontname=fontname,
                      fontweight='bold', fontsize=12, color="black")
        ax.yaxis.set_label_coords(-.1, .5)

    fig.autofmt_xdate()

    try:
        plt.subplots_adjust(hspace=0)
    except Exception:
        pass
    try:
        fig.tight_layout(w_pad=0, h_pad=0)
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

def heatmap(returns,
                 title='',
                 annot_size=10,
                 figsize=(10, 5),
                 cbar=True,
                 square=False,
                 eoy=False,
                 grayscale=False,
                 fontname='Arial',
                 ylabel=True,
                 savefig=None,
                 show=True):

    cmap = 'gray' if grayscale else 'RdYlGn'
    fig_height = len(returns) / 3

    if figsize is None:
        size = list(plt.gcf().get_size_inches())
        figsize = (size[0], size[1])

    figsize = (figsize[0], max([fig_height, figsize[1]]))

    if cbar:
        figsize = (figsize[0]*1.04, max([fig_height, figsize[1]]))

    fig, ax = plt.subplots(figsize=figsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    fig.set_facecolor('white')
    ax.set_facecolor('white')

    ax.set_title(title, fontsize=14, y=.995,
                 fontname=fontname, fontweight='bold', color='black')

    ax = sns.heatmap(returns, ax=ax, annot=True, center=0,
                      annot_kws={"size": annot_size},
                      fmt="0.2f", linewidths=0.5,
                      square=square, cbar=cbar, cmap=cmap,
                      cbar_kws={'format': '%.0f%%'})

    if ylabel:
        ax.set_ylabel('Years', fontname=fontname,
                      fontweight='bold', fontsize=12)
        ax.yaxis.set_label_coords(-.1, .5)

    ax.tick_params(colors="#808080")
    plt.xticks(rotation=0, fontsize=annot_size*1.2)
    plt.yticks(rotation=0, fontsize=annot_size*1.2)

    try:
        plt.subplots_adjust(hspace=0, bottom=0, top=1)
    except Exception:
        pass
    try:
        fig.tight_layout(w_pad=0, h_pad=0)
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