from ast import Str
from tokenize import String
import pyfolio as pf
import backtrader as bt
from backtrader.feeds import PandasData
import warnings
import pandas as pd
import empyrical as ep

# STAT_FUNC_NAMES = {
#     'annual_return': 'Annual return',
#     'cum_returns_final': 'Cumulative returns',
#     'annual_volatility': 'Annual volatility',
#     'sharpe_ratio': 'Sharpe ratio',
#     'calmar_ratio': 'Calmar ratio',
#     'stability_of_timeseries': 'Stability',
#     'max_drawdown': 'Max drawdown',
#     'omega_ratio': 'Omega ratio',
#     'sortino_ratio': 'Sortino ratio',
#     'skew': 'Skew',
#     'kurtosis': 'Kurtosis',
#     'tail_ratio': 'Tail ratio',
#     'common_sense_ratio': 'Common sense ratio',
#     'value_at_risk': 'Daily value at risk',
#     'alpha': 'Alpha',
#     'beta': 'Beta',
# }



OHLCV = ['Open', 'High', 'Low', 'Close', 'Volume']

# class to define the columns we will provide
class SignalData(PandasData):
    """
    Define pandas DataFrame structure
    """
    cols = OHLCV + ['predicted']

    # create lines
    lines = tuple(cols)

    # define parameters
    params = {c: -1 for c in cols}
    params.update({'datetime': None})
    params = tuple(params.items())

# define backtesting strategy class
class MLStrategy(bt.Strategy):
    params = dict(
        stop_loss = 0.02,
        take_profit = 0.5
    )
    
    def __init__(self):
        # keep track of open, close prices and predicted value in the series
        self.data_predicted = self.datas[0].predicted
        self.data_open = self.datas[0].open
        self.data_close = self.datas[0].close
        
        # keep track of pending orders/buy price/buy commission
        self.order = None
        self.price = None
        self.comm = None

    # logging function
    def log(self, txt):
        '''Logging function'''
        dt = self.datas[0].datetime.date(0).isoformat()
        print(f'{dt}, {txt}')

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # order already submitted/accepted - no action required
            return

        # report executed order
        if order.status in [order.Completed]:
            if order.isbuy():
                #log to pandas
                self.log(f'BUY EXECUTED --- Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f},Commission: {order.executed.comm:.2f}'
                )
                self.price = order.executed.price
                self.comm = order.executed.comm
            else:
                self.log(f'SELL EXECUTED --- Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f},Commission: {order.executed.comm:.2f}'
                )

        # report failed order
        elif order.status in [order.Canceled, order.Margin, 
                              order.Rejected]:
            self.log('Order Failed')

        # set no pending order
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log(f'OPERATION RESULT --- Gross: {trade.pnl:.2f}, Net: {trade.pnlcomm:.2f}')

    # We have set cheat_on_open = True.This means that we calculated the signals on day t's close price, 
    # but calculated the number of shares we wanted to buy based on day t+1's open price.
    def next_open(self):
        if not self.position: # not in position then buy
            if self.data_predicted > 0:
                # calculate the max number of shares ('all-in')
                size = 100 #int(self.broker.getcash() / self.datas[0].close) * 0.1
                # buy order
                self.log(f'BUY CREATED --- Size: {size}, Cash: {self.broker.getcash():.2f}, Open: {self.data_open[0]}, Close: {self.data_close[0]} prediction: {self.data_predicted[0]}')
                self.buy(size=size)
        else:
            #calculate stop loss
            stop_price = self.position.price - (self.p.stop_loss * self.position.price)

            #calculate take profit
            take_profit = self.position.price + (self.p.take_profit * self.position.price)

            
            if stop_price > self.data_close : # stoploss triggered
                # sell order
                self.log(f'STOP LOSS SELL CREATED --- Size: {self.position.size} ---position: {self.position.price} ---stop price:{stop_price}')
                self.sell(size=self.position.size)

            elif self.data_close >= take_profit : #take profit triggered
                # sell order
                self.log(f'TAKE PROFIT SELL CREATED --- Size: {self.position.size} ---position: {self.position.price} ---take profit: {take_profit}')
                self.sell(size=self.position.size)

            elif self.data_predicted <= 0:
                # sell order
                self.log(f'SELL CREATED --- Size: {self.position.size} ---prediction: {self.data_predicted[0]}')
                self.sell(size=self.position.size)

class MLStrategyNoStopLoss(bt.Strategy):
    params = dict(
        stop_loss = 0.02,
        take_profit = 0.5
    )
    
    def __init__(self):
        # keep track of open, close prices and predicted value in the series
        self.data_predicted = self.datas[0].predicted
        self.data_open = self.datas[0].open
        self.data_close = self.datas[0].close
        
        # keep track of pending orders/buy price/buy commission
        self.order = None
        self.price = None
        self.comm = None

    # logging function
    def log(self, txt):
        '''Logging function'''
        dt = self.datas[0].datetime.date(0).isoformat()
        print(f'{dt}, {txt}')

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # order already submitted/accepted - no action required
            return

        # report executed order
        if order.status in [order.Completed]:
            if order.isbuy():
                #log to pandas
                self.log(f'BUY EXECUTED --- Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f},Commission: {order.executed.comm:.2f}'
                )
                self.price = order.executed.price
                self.comm = order.executed.comm
            else:
                self.log(f'SELL EXECUTED --- Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f},Commission: {order.executed.comm:.2f}'
                )

        # report failed order
        elif order.status in [order.Canceled, order.Margin, 
                              order.Rejected]:
            self.log('Order Failed')

        # set no pending order
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log(f'OPERATION RESULT --- Gross: {trade.pnl:.2f}, Net: {trade.pnlcomm:.2f}')

    # We have set cheat_on_open = True.This means that we calculated the signals on day t's close price, 
    # but calculated the number of shares we wanted to buy based on day t+1's open price.
    def next_open(self):
        if not self.position: # not in position then buy
            if self.data_predicted > 0:
                # calculate the max number of shares ('all-in')
                size = 100 #int(self.broker.getcash() / self.datas[0].close) * 0.1   
                # buy order
                self.log(f'BUY CREATED --- Size: {size}, Cash: {self.broker.getcash():.2f}, Open: {self.data_open[0]}, Close: {self.data_close[0]} prediction: {self.data_predicted[0]}')
                self.buy(size=size)
        else:

            if self.data_predicted <= 0:
                # sell order
                self.log(f'SELL CREATED --- Size: {self.position.size} ---prediction: {self.data_predicted[0]}')
                self.sell(size=self.position.size)



# define analyzer class
# Trade list similar to Amibroker output
class Trade_list(bt.Analyzer):

    def get_analysis(self):

        return self.trades


    def __init__(self):

        self.trades = []
        self.cumprofit = 0.0


    def notify_trade(self, trade):

        if trade.isclosed:

            brokervalue = self.strategy.broker.getvalue()

            dir = 'short'
            if trade.history[0].event.size > 0: dir = 'long'

            commision = trade.history[0].event.commission + trade.history[len(trade.history)-1].event.commission
            pricein = trade.history[len(trade.history)-1].status.price
            priceout = trade.history[len(trade.history)-1].event.price
            datein = bt.num2date(trade.history[0].status.dt)
            dateout = bt.num2date(trade.history[len(trade.history)-1].status.dt)
            if trade.data._timeframe >= bt.TimeFrame.Days:
                datein = datein.date()
                dateout = dateout.date()

            pcntchange = 100 * priceout / pricein - 100
            pnl = trade.history[len(trade.history)-1].status.pnlcomm
            pnlpcnt = 100 * pnl / brokervalue
            barlen = trade.history[len(trade.history)-1].status.barlen
            pbar = pnl / barlen
            self.cumprofit += pnl

            size = value = 0.0
            for record in trade.history:
                if abs(size) < abs(record.status.size):
                    size = record.status.size
                    value = record.status.value

            highest_in_trade = max(trade.data.high.get(ago=0, size=barlen+1))
            lowest_in_trade = min(trade.data.low.get(ago=0, size=barlen+1))
            hp = 100 * (highest_in_trade - pricein) / pricein
            lp = 100 * (lowest_in_trade - pricein) / pricein
            if dir == 'long':
                mfe = hp
                mae = lp
            if dir == 'short':
                mfe = -lp
                mae = -hp

            self.trades.append({'ref': trade.ref, 'ticker': trade.data._name, 'dir': dir,
                 'datein': datein, 'pricein': pricein, 'dateout': dateout, 'priceout': priceout,
                 'chng%': round(pcntchange, 2), 'pnl': pnl, 'pnl%': round(pnlpcnt, 2),
                 'size': size, 'value': value, 'cumpnl': self.cumprofit,
                 'nbars': barlen, 'pnl/bar': round(pbar, 2),
                 'mfe%': round(mfe, 2), 'mae%': round(mae, 2), 'commision': round(commision)})



def PrepareDataPredicition(stock_df:pd.DataFrame):
    prediction = stock_df[['Date','strategy_svm','Open','High','Low','Close','Volume']]
    prediction.rename(columns = {'strategy_svm':'predicted'}, inplace=True)
    prediction["Date"] = pd.to_datetime(prediction["Date"])
    prediction.set_index('Date', inplace=True)
    prediction = prediction.dropna()
    
    return prediction


def RunBacktesting(stock_df:pd.DataFrame, ticker:Str, isStopLoss:bool):
    '''
        predicition : DataFrame pandas, with column
    '''

    data = PrepareDataPredicition(stock_df)
    print(data)
    data = SignalData(dataname=data)

    #initiate cerebro
    # instantiate Cerebro, add strategy, data, initial cash, commission and pyfolio for performance analysis
    cerebro = bt.Cerebro(stdstats = True, cheat_on_open=True)

    if isStopLoss :
        cerebro.addstrategy(MLStrategy)
    else :
        cerebro.addstrategy(MLStrategyNoStopLoss)
        
    cerebro.adddata(data, name=ticker)
    cerebro.broker.setcash(1000000.0)
    cerebro.broker.setcommission(commission=0.0019)
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')
    cerebro.addanalyzer(Trade_list, _name='trade_list')

    #run cerebro
    backtest_result = cerebro.run(tradehistory=True)

    #get analyzer data
    trade_list = backtest_result[0].analyzers.trade_list.get_analysis()
    trade_df = pd.DataFrame(trade_list)

    #get pyfolio
    strat = backtest_result[0]
    pyfoliozer = strat.analyzers.getbyname('pyfolio')
    returns, positions, transactions, gross_lev = pyfoliozer.get_pf_items()
    returns.name = 'Strategy'
    
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

    return trade_df, returns

def BacktestingStatistics(returns):

    #initiate dict
    stats = {}

    #calculate annual return
    DAILY = 'daily'
    WEEKLY = 'weekly'
    MONTHLY = 'monthly'
    YEARLY = 'yearly'

    stats['annual_return'] = ep.annual_return(returns, period='daily') * 100

    #caluclate cumulative return
    stats['cumulative_return'] = ep.cum_returns_final(returns) * 100

    #calculate max drawdown
    stats['max_drawdown'] = ep.max_drawdown(returns) * 100

    #calculate shape ratio
    stats['sharpe_ratio'] = ep.sharpe_ratio(returns)
    #callculate stability
    stats['stability_of_timeseries'] = ep.stability_of_timeseries(returns)

    return stats