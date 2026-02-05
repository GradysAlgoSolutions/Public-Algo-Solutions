import pandas as pd
import numpy as np
from datetime import datetime

class BacktestEngine:
    
    def __init__(self, initial_capital=10000, commission=0.001, base_slippage=0.0005):
        self.initial_capital = initial_capital
        self.commission = commission
        self.base_slippage = base_slippage
        
    def _calculate_realistic_slippage(self, data, index, is_buy):
        if 'TEMP_ATR' in data.columns:
            atr = data['TEMP_ATR'].iloc[index]
            close = data['Close'].iloc[index]
            if not pd.isna(atr) and close > 0:
                volatility_factor = (atr / close)
                slippage = self.base_slippage * (1 + volatility_factor * 10)
            else:
                slippage = self.base_slippage
        else:
            slippage = self.base_slippage
        
        slippage = min(slippage, 0.005)
        
        return slippage if is_buy else -slippage
        
    def run_backtest(self, data, signals):
        capital = self.initial_capital
        position = 0
        position_value = 0
        entry_price = 0
        
        trades = []
        equity_curve = []
        
        for i in range(len(data)):
            current_price = data['Close'].iloc[i]
            date = data.index[i]
            
            if signals['entry'].iloc[i] and position == 0:
                slippage = self._calculate_realistic_slippage(data, i, True)
                execution_price = current_price * (1 + slippage)
                
                shares_to_buy = int(capital / (execution_price * (1 + self.commission)))
                
                if shares_to_buy > 0:
                    entry_price = execution_price
                    cost = shares_to_buy * entry_price * (1 + self.commission)
                    
                    position = shares_to_buy
                    capital -= cost
                    position_value = position * current_price
                    
                    trades.append({
                        'date': date,
                        'type': 'BUY',
                        'price': entry_price,
                        'shares': shares_to_buy,
                        'value': cost
                    })
            
            elif signals['exit'].iloc[i] and position > 0:
                slippage = self._calculate_realistic_slippage(data, i, False)
                execution_price = current_price * (1 + slippage)
                proceeds = position * execution_price * (1 - self.commission)
                
                trades.append({
                    'date': date,
                    'type': 'SELL',
                    'price': execution_price,
                    'shares': position,
                    'value': proceeds,
                    'pnl': proceeds - (position * entry_price * (1 + self.commission)),
                    'pnl_pct': ((execution_price / entry_price) - 1) * 100
                })
                
                capital += proceeds
                position = 0
                position_value = 0
            
            if position > 0:
                position_value = position * current_price
            
            total_equity = capital + position_value
            equity_curve.append({
                'date': date,
                'equity': total_equity,
                'capital': capital,
                'position_value': position_value
            })
        
        if position > 0:
            final_price = data['Close'].iloc[-1]
            slippage = self._calculate_realistic_slippage(data, len(data)-1, False)
            execution_price = final_price * (1 + slippage)
            proceeds = position * execution_price * (1 - self.commission)
            
            trades.append({
                'date': data.index[-1],
                'type': 'SELL',
                'price': execution_price,
                'shares': position,
                'value': proceeds,
                'pnl': proceeds - (position * entry_price * (1 + self.commission)),
                'pnl_pct': ((execution_price / entry_price) - 1) * 100
            })
            
            capital += proceeds
        
        metrics = self._calculate_metrics(
            equity_curve, 
            trades, 
            self.initial_capital
        )
        
        return {
            'metrics': metrics,
            'trades': pd.DataFrame(trades),
            'equity_curve': pd.DataFrame(equity_curve)
        }
    
    def _calculate_metrics(self, equity_curve, trades, initial_capital):
        df_equity = pd.DataFrame(equity_curve)
        df_trades = pd.DataFrame(trades)
        
        if len(df_equity) == 0:
            return self._empty_metrics()
        
        final_equity = df_equity['equity'].iloc[-1]
        total_return = ((final_equity / initial_capital) - 1) * 100
        
        df_equity['returns'] = df_equity['equity'].pct_change()
        returns = df_equity['returns'].dropna()
        
        df_equity['peak'] = df_equity['equity'].cummax()
        df_equity['drawdown'] = (df_equity['equity'] - df_equity['peak']) / df_equity['peak'] * 100
        max_drawdown = df_equity['drawdown'].min()
        
        avg_drawdown = df_equity['drawdown'].mean()
        drawdown_duration = self._calculate_drawdown_duration(df_equity)
        
        if len(returns) > 0 and returns.std() > 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0 and downside_returns.std() > 0:
                sortino_ratio = (returns.mean() / downside_returns.std()) * np.sqrt(252)
            else:
                sortino_ratio = 0
            calmar_ratio = (total_return / abs(max_drawdown)) if max_drawdown != 0 else 0
        else:
            sharpe_ratio = 0
            sortino_ratio = 0
            calmar_ratio = 0
        
        ulcer_index = np.sqrt(np.mean(df_equity['drawdown'] ** 2))
        
        if len(df_trades) > 0:
            sell_trades = df_trades[df_trades['type'] == 'SELL'].copy()
            num_trades = len(sell_trades)
            
            if num_trades > 0:
                wins = sell_trades[sell_trades['pnl'] > 0]
                losses = sell_trades[sell_trades['pnl'] < 0]
                
                num_wins = len(wins)
                num_losses = len(losses)
                win_rate = (num_wins / num_trades) * 100
                
                avg_win = wins['pnl'].mean() if num_wins > 0 else 0
                avg_loss = losses['pnl'].mean() if num_losses > 0 else 0
                
                total_wins = wins['pnl'].sum() if num_wins > 0 else 0
                total_losses = abs(losses['pnl'].sum()) if num_losses > 0 else 0
                
                profit_factor = total_wins / total_losses if total_losses != 0 else 0
                payoff_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
                
                expectancy = (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss)
                
                max_win = wins['pnl'].max() if num_wins > 0 else 0
                max_loss = losses['pnl'].min() if num_losses > 0 else 0
                
                buy_dates = df_trades[df_trades['type'] == 'BUY']['date'].values
                sell_dates = sell_trades['date'].values
                if len(buy_dates) > 0 and len(sell_dates) > 0:
                    trade_durations = []
                    for i in range(min(len(buy_dates), len(sell_dates))):
                        duration = (sell_dates[i] - buy_dates[i]).astype('timedelta64[D]').astype(int)
                        trade_durations.append(duration)
                    avg_trade_duration = np.mean(trade_durations) if trade_durations else 0
                    max_trade_duration = np.max(trade_durations) if trade_durations else 0
                    min_trade_duration = np.min(trade_durations) if trade_durations else 0
                else:
                    avg_trade_duration = 0
                    max_trade_duration = 0
                    min_trade_duration = 0
                
                consecutive_wins = self._calculate_consecutive(sell_trades, 'pnl', lambda x: x > 0)
                consecutive_losses = self._calculate_consecutive(sell_trades, 'pnl', lambda x: x < 0)
                max_consecutive_wins = max(consecutive_wins) if consecutive_wins else 0
                max_consecutive_losses = max(consecutive_losses) if consecutive_losses else 0
                
                recovery_factor = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
                
                if win_rate < 100 and payoff_ratio > 0:
                    kelly_pct = (win_rate/100 - (1 - win_rate/100) / payoff_ratio) * 100 if payoff_ratio > 0 else 0
                else:
                    kelly_pct = 0
                
                time_in_market = sum(trade_durations) if 'trade_durations' in locals() else 0
                total_days_traded = len(df_equity)
                exposure_time = (time_in_market / total_days_traded * 100) if total_days_traded > 0 else 0
                
            else:
                num_wins = 0
                num_losses = 0
                win_rate = 0
                avg_win = 0
                avg_loss = 0
                profit_factor = 0
                payoff_ratio = 0
                expectancy = 0
                max_win = 0
                max_loss = 0
                avg_trade_duration = 0
                max_trade_duration = 0
                min_trade_duration = 0
                max_consecutive_wins = 0
                max_consecutive_losses = 0
                recovery_factor = 0
                kelly_pct = 0
                exposure_time = 0
        else:
            num_trades = 0
            num_wins = 0
            num_losses = 0
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            payoff_ratio = 0
            expectancy = 0
            max_win = 0
            max_loss = 0
            avg_trade_duration = 0
            max_trade_duration = 0
            min_trade_duration = 0
            max_consecutive_wins = 0
            max_consecutive_losses = 0
            recovery_factor = 0
            kelly_pct = 0
            exposure_time = 0
        
        total_days = len(df_equity)
        annual_return = ((1 + total_return/100) ** (252/total_days) - 1) * 100 if total_days > 0 else 0
        
        return {
            'total_return_pct': round(float(total_return), 2),
            'annual_return_pct': round(float(annual_return), 2),
            'final_equity': round(float(final_equity), 2),
            'sharpe_ratio': round(float(sharpe_ratio), 2),
            'sortino_ratio': round(float(sortino_ratio), 2),
            'calmar_ratio': round(float(calmar_ratio), 2),
            'max_drawdown_pct': round(float(max_drawdown), 2),
            'avg_drawdown_pct': round(float(avg_drawdown), 2),
            'max_drawdown_duration': int(drawdown_duration),
            'ulcer_index': round(float(ulcer_index), 2),
            'recovery_factor': round(float(recovery_factor), 2),
            'num_trades': int(num_trades),
            'num_wins': int(num_wins),
            'num_losses': int(num_losses),
            'win_rate_pct': round(float(win_rate), 2),
            'avg_win': round(float(avg_win), 2),
            'avg_loss': round(float(avg_loss), 2),
            'max_win': round(float(max_win), 2),
            'max_loss': round(float(max_loss), 2),
            'profit_factor': round(float(profit_factor), 2),
            'payoff_ratio': round(float(payoff_ratio), 2),
            'expectancy': round(float(expectancy), 2),
            'avg_trade_duration_days': round(float(avg_trade_duration), 2),
            'max_trade_duration_days': int(max_trade_duration),
            'min_trade_duration_days': int(min_trade_duration),
            'max_consecutive_wins': int(max_consecutive_wins),
            'max_consecutive_losses': int(max_consecutive_losses),
            'kelly_pct': round(float(kelly_pct), 2),
            'exposure_time_pct': round(float(exposure_time), 2)
        }
    
    def _calculate_consecutive(self, trades, column, condition):
        streaks = []
        current_streak = 0
        for value in trades[column]:
            if condition(value):
                current_streak += 1
            else:
                if current_streak > 0:
                    streaks.append(current_streak)
                current_streak = 0
        if current_streak > 0:
            streaks.append(current_streak)
        return streaks
    
    def _calculate_drawdown_duration(self, df_equity):
        in_drawdown = False
        max_duration = 0
        current_duration = 0
        for dd in df_equity['drawdown']:
            if dd < 0:
                if not in_drawdown:
                    in_drawdown = True
                    current_duration = 1
                else:
                    current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                in_drawdown = False
                current_duration = 0
        return max_duration
    
    def _empty_metrics(self):
        return {
            'total_return_pct': 0,
            'annual_return_pct': 0,
            'final_equity': self.initial_capital,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'calmar_ratio': 0,
            'max_drawdown_pct': 0,
            'avg_drawdown_pct': 0,
            'max_drawdown_duration': 0,
            'ulcer_index': 0,
            'recovery_factor': 0,
            'num_trades': 0,
            'num_wins': 0,
            'num_losses': 0,
            'win_rate_pct': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'max_win': 0,
            'max_loss': 0,
            'profit_factor': 0,
            'payoff_ratio': 0,
            'expectancy': 0,
            'avg_trade_duration_days': 0,
            'max_trade_duration_days': 0,
            'min_trade_duration_days': 0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0,
            'kelly_pct': 0,
            'exposure_time_pct': 0
        }
