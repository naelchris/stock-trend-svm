import matplotlib as plt
import pandas as pd 
import numpy as np
import talib

produce_charts = False

if produce_charts:
    import matplotlib.pyplot as plt    

def SayHello():
    print("hello world")

def return_features(df):
    """
    Args:
        df: pandas.DataFrame, columns include at least ["date", "open", "high", "low", "close", "volume"]
    Returns:
        pandas.DataFrame
    """
    df["return"] = df["Close"] / df["Close"].shift(1)
    df["close_to_open"] = df["Close"] / df["Open"]
    df["close_to_high"] = df["Close"] / df["High"]
    df["close_to_low"] = df["Close"] / df["Low"]
    df = df.iloc[1:] # first first row: does not have a return value
    return df

def target_value(df):
    df["y"] = df["return"].shift(-1)
    df = df.iloc[:len(df)-1]
    return df


def macd(df):
    """
    Math reference: https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:moving_average_convergence_divergence_macd
    Args:
        df: pandas.DataFrame, columns include at least ["close"]
    Returns:
        pandas.DataFrame
    """
    ema_12_day = df["Close"].ewm(com=(12-1)/2).mean()
    ema_26_day = df["Close"].ewm(com=(26-1)/2).mean()
    df["macd_line"] = ema_12_day - ema_26_day
    df["macd_9_day"] = df["macd_line"].ewm(com=(9-1)/2).mean()
    df["macd_diff"] = df["macd_line"] - df["macd_9_day"]
    # print(df.tail(10)[["date", "close", "macd_line", "macd_9_day"]])
    if produce_charts:
        chart_macd(df)
    return df

def chart_macd(df):
    """
    Save chart to charts/macd
    Args:
        df: pandas.DataFrame, columns include at least ["date", "close", "macd_line", "macd_9_day"]
    Returns:
        None
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    fig.tight_layout()
    plt.suptitle("MSFT", fontsize=24)
    plt.subplots_adjust(left=0.1, top=0.9, hspace = 0.4)
    ax1 = axes[0]
    ax1.set_title("Price")
    ax1.set_ylabel("$")
    df.tail(300)[["date", "close"]].plot(x="date", kind="line", ax=ax1)
    ax2 = axes[1]
    ax2.set_title("MACD")
    df.tail(300)[["date", "macd_line", "macd_9_day"]].plot(x="date", kind="line", ax=ax2, secondary_y=False)
    # df.tail(300)[["date", "macd_diff"]].plot(x="date", kind="bar", ax=ax2, secondary_y=True)
    fig.savefig("charts/macd.png")


def ma(df):
    """
    Math reference: https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:moving_averages
    Args:
        df: pandas.DataFrame, columns include at least ["close"]
    Returns:
        pandas.DataFrame
    """
    df["ma_50_day"] = df["Close"].rolling(50).mean()
    df["ma_200_day"] = df["Close"].rolling(200).mean()
    df["ma_50_200"] = df["ma_50_day"] - df["ma_200_day"]
    # print(df.tail(10)[["date", "close", "ma_50_200"]])
    if produce_charts:
        chart_ma(df)
    return df

def chart_ma(df):
    """
    Save chart to charts/ma
    Args:
        df: pandas.DataFrame, columns include at least ["date", "close", "ma_50_200"]
    Returns:
        None
    """
    fig, axes = plt.subplots(1, 1, figsize=(10, 10))
    fig.tight_layout()
    plt.suptitle("MSFT Moving Average 50 day - 200 day", fontsize=24)
    plt.subplots_adjust(left=0.1, top=0.9, right=0.9, hspace=0.4)
    axes.set_ylabel("$")
    df.tail(1500)[["date", "Close"]].plot(x="date", kind="line", ax=axes, secondary_y=False)
    df.tail(1500)[["date", "ma_50_day"]].plot(x="date", kind="line", ax=axes, secondary_y=False)
    df.tail(1500)[["date", "ma_200_day"]].plot(x="date", kind="line", ax=axes, secondary_y=False)
    df.tail(1500)[["date", "ma_50_200"]].plot(x="date", kind="line", ax=axes, secondary_y=True)
    fig.savefig("charts/ma.png")

def parabolic_sar(df):
    """
    Math reference: https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:parabolic_sar
    Args:
        df: pandas.DataFrame, columns include at least ["close"]
    Returns:
        pandas.DataFrame
    """
    df["sar"] = np.nan
    step = 5
    acc_factor = 0.02
    uptrend = False
    prior_sar = max(df.loc[1:step, "Close"])
    extreme_point = min(df.loc[1:step, "Close"])
    for i, row in df.iloc[step:].iterrows():
        if uptrend:
            df.at[i, "sar"] = prior_sar + acc_factor*(extreme_point - prior_sar)
            if df.at[i, "Low"] < df.at[i, "sar"]:
                # reverse to downtrend
                uptrend = False
                prior_sar = max(df.loc[i-step:i, "Close"])
                extreme_point = min(df.loc[i-step:i, "Close"])
            else:
                # continue uptrend
                if df.at[i, "Close"] > extreme_point:
                    extreme_point = df.at[i, "Close"]
                    acc_factor = min(0.2, acc_factor+0.02)
        else:
            df.at[i, "sar"] = prior_sar - acc_factor*(prior_sar - extreme_point)
            if df.at[i, "High"] > df.at[i, "sar"]:
                # reverse to uptrend
                uptrend = True
                prior_sar = min(df.loc[i-step:i, "Close"])
                extreme_point = max(df.loc[i-step:i, "Close"])
            else:
                # continue downtrend
                if df.at[i, "Close"] < extreme_point:
                    extreme_point = df.at[i, "Close"]
                    acc_factor = min(0.2, acc_factor+0.02)
        prior_sar = df.at[i, "sar"]
    if produce_charts:
        chart_sar(df)
    return df

def chart_sar(df):
    """
    Save chart to charts/ma
    Args:
        df: pandas.DataFrame, columns include at least ["date", "close", "sar"]
    Returns:
        None
    """
    fig, axes = plt.subplots(1, 1, figsize=(10, 10))
    fig.tight_layout()
    plt.suptitle("MSFT SAR", fontsize=24)
    plt.subplots_adjust(left=0.1, top=0.9, right=0.9, hspace = 0.4)
    axes.set_ylabel("$")
    df.tail(100)[["date", "close"]].plot(x="date", kind="line", ax=axes, secondary_y=False)
    df.tail(100)[["date", "sar"]].plot(x="date", style=".", ax=axes, secondary_y=False)
    fig.savefig("charts/sar.png")


def trend_feature(df):
  df = macd(df)
  df = ma(df)
  # df = parabolic_sar(df)
  return df


def feature_indicator(df):
  """
  ['return', 'close_to_open', 'close_to_high', 'close_to_low', 'macd_diff', 
  'ma_50_200', 'sar', 'stochastic_oscillator', 'cci', 'rsi', '5d_volatility', 
  '21d_volatility', '60d_volatility', 'bollinger', 'atr', 'on_balance_volume',
   'chaikin_oscillator']
  """

  dataset = df
  dataset = dataset.dropna()
  dataset = dataset[['Open', 'High', 'Low', 'Close','Adj Close', 'Volume']]

  #var
  short_window = 40
  long_window = 100
  lags = [1,2,3,4,5]
  
  #momentum
  dataset['EMA_40'] = dataset['Close'].rolling(window=short_window, min_periods=1, center=False).mean()
  dataset['EMA_100'] = dataset['Close'].rolling(window=long_window, min_periods=1, center=False).mean()
  dataset['ADX'] = talib.ADX(dataset['High'].values, dataset['Low'].values, dataset['Close'].values, timeperiod=14)
  dataset['RSI'] = talib.RSI(dataset['Close'].values, timeperiod = 14)

  #volume
  dataset['OBV'] = talib.OBV(dataset['Close'], dataset['Volume'])

  #volatility
  dataset['ATR'] = talib.ATR(np.array(dataset['High']), np.array(dataset['Low']), np.array(dataset['Close']), timeperiod=14)
  dataset['NATR'] = talib.NATR(dataset['High'], dataset['Low'], dataset['Close'], timeperiod=14)
  dataset['TRANGE'] = talib.TRANGE(dataset['High'], dataset['Low'], dataset['Close'])

  #trend feature
  dataset = trend_feature(dataset)
  dataset['BBANDS_U'] = talib.BBANDS(dataset['Close'], timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[0] / \
                         talib.BBANDS(dataset['Close'], timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[0].mean()
  dataset['BBANDS_M'] = talib.BBANDS(dataset['Close'], timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[1] / \
                    talib.BBANDS(dataset['Close'], timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[1].mean()
  dataset['BBANDS_L'] = talib.BBANDS(dataset['Close'], timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[2] / \
                    talib.BBANDS(dataset['Close'], timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[2].mean()

  #direction ema 3 bars in the future
  dataset['signal_3'] = np.where( dataset['EMA_40'].shift(-3) > dataset['EMA_40'], 1.0, -1.0)

  #direction ema 7 bars in the future
  dataset['signal_7'] = np.where( dataset['EMA_40'].shift(-7) > dataset['EMA_40'], 1.0, -1.0)


  #direction ema 10 bars in the future
  dataset['signal_10'] = np.where( dataset['EMA_40'].shift(-10) > dataset['EMA_40'], 1.0, -1.0)



  dataset = dataset.dropna()


  return dataset
