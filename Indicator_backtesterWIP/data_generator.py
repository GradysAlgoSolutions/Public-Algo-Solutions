import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class HistoricalDataGenerator:
    
    def __init__(self, seed=42):
        np.random.seed(seed)
    
    def generate_ohlcv(self, symbol='SPY', start_date='2020-01-01', 
                       end_date='2024-12-31', initial_price=300):
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        date_range = pd.bdate_range(start=start, end=end)
        n_days = len(date_range)
        
        drift = 0.0003
        volatility = 0.015
        
        regime_changes = np.random.randint(50, 150, size=n_days//100 + 1)
        regime_boundaries = np.cumsum(regime_changes)
        regime_boundaries = regime_boundaries[regime_boundaries < n_days]
        
        returns = np.zeros(n_days)
        current_idx = 0
        
        for regime_end in np.append(regime_boundaries, n_days):
            regime_length = regime_end - current_idx
            
            regime_drift = np.random.normal(drift, drift/2)
            regime_vol = np.random.uniform(volatility * 0.7, volatility * 1.3)
            
            regime_returns = np.random.normal(regime_drift, regime_vol, regime_length)
            
            ar_coef = np.random.uniform(0.05, 0.15)
            for i in range(1, regime_length):
                regime_returns[i] += ar_coef * regime_returns[i-1]
            
            returns[current_idx:regime_end] = regime_returns
            current_idx = regime_end
        
        closes = initial_price * np.exp(np.cumsum(returns))
        
        intraday_vol = volatility * 0.4
        
        data = []
        for i, close in enumerate(closes):
            
            if i == 0:
                open_price = initial_price
            else:
                gap = np.random.normal(0, volatility * 0.3)
                open_price = closes[i-1] * (1 + gap)
            
            intraday_return = np.random.normal(0, intraday_vol)
            intraday_range = abs(np.random.lognormal(np.log(intraday_vol), 0.5)) * close
            
            high = max(open_price, close) + np.random.uniform(0, intraday_range * 0.5)
            low = min(open_price, close) - np.random.uniform(0, intraday_range * 0.5)
            
            high = max(high, open_price, close)
            low = min(low, open_price, close)
            
            base_volume = 80000000
            volume_mult = np.random.lognormal(0, 0.6)
            range_factor = (high - low) / close
            volume = int(base_volume * volume_mult * (1 + range_factor * 5))
            
            data.append({
                'Date': date_range[i],
                'Open': round(open_price, 2),
                'High': round(high, 2),
                'Low': round(low, 2),
                'Close': round(close, 2),
                'Volume': volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('Date', inplace=True)
        df['Symbol'] = symbol
        
        return df
    
    def save_to_csv(self, df, filename='historical_data.csv'):
        df.to_csv(filename)
        print(f"Data saved to {filename}")
        return filename
