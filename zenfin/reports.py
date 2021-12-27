import pandas as pd
from . import utils

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
