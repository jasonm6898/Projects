import numpy as np
import pandas as pd
import math
import statistics as s
import glob
from colorama import Fore, Style
from datetime import datetime
from matplotlib import pyplot as plt
from matplotlib import dates as mdates
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


def get_return(x: tuple):
    if (x[2] == 1 or x[2] == 10) and x[3] == 1:
        return x[0] / x[1] - 1
    if (x[2] == -1 or x[2] == -10) and x[3] == -1:
        return x[1] / x[0] - 1
    if x[2] == 0 or (x[2] == 1 and x[3] != 1) or (x[2] == -1 and x[3] != -1):
        return 0


class GetPerformance:

    def __init__(self, path: str, position: [], price: [], share: int, ticker: str, log_path: str,
                 initial=10000.00, commission=0, log=False, begin_time=False, end_time=False):
        if path.endswith('\\'):
            self._path = path
        else:
            self._path = path + '\\'
        if log_path.endswith('\\'):
            self._log_path = log_path
        else:
            self._log_path = log_path + '\\'
        self._ticker = ticker
        self._position = position
        self._commission = commission
        self._price = price
        self._log = log
        self._initial = initial
        if not begin_time:
            self._begin_time = list(self.get_records()['Date'])[0]
        else:
            self._begin_time = begin_time
        if not end_time:
            self._end_time = list(self.get_records()['Date'])[-1]
        else:
            self._end_time = end_time
        self._share = share

        if log:
            self.trade_log(out_put_dir=self._log_path)

    def get_records(self):
        path_name = glob.glob(f'{self._path}*{self._ticker}*.csv')
        if len(path_name) == 0:
            raise FileNotFoundError(f'The data of asset {self._ticker} is not found.')
        elif len(path_name) > 1:
            raise Exception(f'There are duplicate data for {self._ticker}.')
        else:
            return pd.read_csv(path_name[0], encoding='utf-8')

    def convergence_period(self):
        trades = self._position
        if -1 in trades and 1 in trades:
            cp = min(trades.index(1), trades.index(-1))
        elif -1 in trades and 1 not in trades:
            cp = trades.index(-1)
        elif 1 in trades and -1 not in trades:
            cp = trades.index(1)
        else:
            raise Exception('No trades happened during trading window.')
        return cp

    def benchmark(self, lb_period=1):
        df = self.get_records()
        start = list(df['Date']).index(self._begin_time)
        end = list(df['Date']).index(self._end_time)
        period_return = list(df['Adj Close'][start: end + 1].pct_change(lb_period))
        period_return[0] = 0
        temp = list(map(lambda p: p + 1, period_return))
        cp = self.convergence_period()
        return [period_return[cp:], list(np.cumprod(temp) - 1)[cp:]]

    def performance(self, lb_period=1):
        df = self.get_records()
        start = list(df['Date']).index(self._begin_time)
        end = list(df['Date']).index(self._end_time)
        price = list(df['Adj Close'][start: end + 1])
        cp = self.convergence_period()
        temp = self._position
        z = list(zip(price[lb_period:], price[:-lb_period], temp[lb_period:], temp[:-lb_period]))
        returns = list(map(get_return, z))
        temp = list(map(lambda p: p+1, returns))
        cum_return = [0] + list(np.cumprod(temp) - 1)
        returns = [0] + returns
        zip_daily_cum = [returns[cp:], cum_return[cp:]]
        return zip_daily_cum

    def p_stats(self):
        report = {}
        a = self.performance()[1]
        b = self.benchmark()[1]
        n = len(a)
        sharpe = (s.mean(a) - 0.0012)/math.sqrt(s.variance(a))
        adjusted_sharpe = 10*math.sqrt(n)*s.mean(list(np.subtract(a, b)))/math.sqrt(s.variance(np.subtract(a, b)))
        anu_return = (self.performance()[1][-1])/(n/360)
        anu_market = (self.benchmark()[1][-1])/(n/360)
        report['sharpe'] = f'Sharpe = {sharpe}'
        report['adjusted_sharpe'] = f'Adjusted sharpe against market = {adjusted_sharpe}'
        report['return'] = f'Annulized return = {anu_return}'
        report['market_return'] = f'Market Annulized return = {anu_market}'
        return report

    def performance_plot(self):
        df = self.get_records()
        start = list(df['Date']).index(self._begin_time)
        end = list(df['Date']).index(self._end_time)
        date_time = [datetime.strptime(x, '%Y-%m-%d') for x in list(df['Date'][start: end+1])]
        cp = self.convergence_period()
        years = mdates.YearLocator()
        months = mdates.MonthLocator()
        years_fmt = mdates.DateFormatter('%Y-%m')
        fig, ax = plt.subplots()
        line1, = ax.plot(date_time[cp:], pd.DataFrame(self.performance()[1]), marker='o',
                         markersize=1, color='red', linewidth=0.5)
        line2, = ax.plot(date_time[cp:], pd.DataFrame(self.benchmark()[1]), marker='o',
                         markersize=1, color='blue', linewidth=0.5)
        plt.legend([line1, (line1, line2)], ['Investment Return', 'Market Return'])
        plt.ylabel('Cumulative Return')
        plt.title('Investment Performance')
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(years_fmt)
        ax.xaxis.set_minor_locator(months)
        datemin = datetime.strftime(date_time[0], '%Y-%m')
        datemax = date_time[-1].replace(date_time[-1].year + 1)
        datemax = datetime.strftime(datemax, '%Y-%m')
        ax.set_xlim(datemin, datemax)
        ax.format_xdate = mdates.DateFormatter('%Y-%m-%d')
        ax.grid(True)
        fig.autofmt_xdate()
        plt.show()

    def trade_log(self, out_put_dir: str):
        log_name = out_put_dir + f'log_{self._ticker}.csv'
        log_file = open(log_name, 'w')
        log_file.write("Trades,Entry,Exit,Position,Gain(Loss),Balance,Comission \n")
        df = self.get_records()
        start = list(df['Date']).index(self._begin_time)
        end = list(df['Date']).index(self._end_time)
        df = df.iloc[start: end + 1, :]
        p = self._position
        r = len(self._position) + 1
        p = [0] + p + [0]
        entry = ''
        p_str = ''
        entry_price = 0
        trade_count = 1
        account = self._initial
        for i in range(1, r):
            if p[i] != p[i-1] and (p[i] == 1 or p[i] == -1):
                entry = df['Date'].iloc[i]
                entry_price = df['Adj Close'][i]
                if entry_price * self._share > account:
                    print(Fore.RED + f"You don't have enough cash for this investment!")
                    print(Style.RESET_ALL)
            elif (p[i] == p[i-1] and p[i] != p[i+1] and p[i] != 0) or p[i] == 10 or p[i] == -10:
                exits = df['Date'].iloc[i-1]
                gain_loss = (df['Adj Close'].iloc[i-1]-entry_price-self._commission) * self._share
                account += gain_loss
                if p[i-1] == 1:
                    p_str = 'Long'
                elif p[i-1] == -1:
                    p_str = 'Short'
                log_file.write(f"{trade_count},{entry},{exits},{p_str},"
                               f"{gain_loss},{account},{self._commission} \n")
                trade_count += 1
        log_file.close()


class StrategyBuilder:

    def __init__(self, directory: str, ticker: str, long_only=True, short_only=False, long_short=False):
        if directory.endswith('\\'):
            self._dir = directory
        else:
            self._dir = directory + '\\'
        self.ticker = ticker
        self.long = long_only
        self.short = short_only
        self.long_short = long_short

    def get_records(self):
        path_name = glob.glob(f'{self._dir}*{self.ticker}*.csv')
        if len(path_name) == 0:
            raise FileNotFoundError(f'The data of asset {self.ticker} is not found.')
        elif len(path_name) > 1:
            raise Exception(f'There are duplicate data for {self.ticker}.')
        else:
            return pd.read_csv(path_name[0], encoding='utf-8')

    def long_short_switch(self, pa):
        if self.long:
            pa = [0 if x == -1 or x == -10 else x for x in pa]
            return pa
        elif self.long_short:
            return pa
        else:
            pa = [0 if x == 1 or x == 10 else x for x in pa]
            return pa

    # def black-box(self, ):
        # User Defined Strategy Building Area

