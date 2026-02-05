import pandas as pd
import numpy as np

class TechnicalIndicators:
    
    @staticmethod
    def calculate_rsi(data, period=14):
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_sma(data, period=20):
        return data['Close'].rolling(window=period).mean()
    
    @staticmethod
    def calculate_ema(data, period=12):
        return data['Close'].ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_macd(data, fast=12, slow=26, signal=9):
        ema_fast = data['Close'].ewm(span=fast, adjust=False).mean()
        ema_slow = data['Close'].ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        
        return pd.DataFrame({
            'MACD': macd,
            'Signal': signal_line,
            'Histogram': histogram
        })
    
    @staticmethod
    def calculate_bbands(data, period=20, std_dev=2):
        sma = data['Close'].rolling(window=period).mean()
        std = data['Close'].rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return pd.DataFrame({
            'BB_Upper': upper_band,
            'BB_Middle': sma,
            'BB_Lower': lower_band
        })
    
    @staticmethod
    def calculate_stochastic(data, k_period=14, d_period=3):
        low_min = data['Low'].rolling(window=k_period).min()
        high_max = data['High'].rolling(window=k_period).max()
        
        k = 100 * ((data['Close'] - low_min) / (high_max - low_min))
        d = k.rolling(window=d_period).mean()
        
        return pd.DataFrame({
            'Stoch_K': k,
            'Stoch_D': d
        })
    
    @staticmethod
    def calculate_atr(data, period=14):
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    @staticmethod
    def add_all_indicators(data):
        df = data.copy()
        
        df['TEMP_RSI'] = TechnicalIndicators.calculate_rsi(df)
        df['TEMP_SMA_20'] = TechnicalIndicators.calculate_sma(df, 20)
        df['TEMP_SMA_50'] = TechnicalIndicators.calculate_sma(df, 50)
        df['TEMP_EMA_12'] = TechnicalIndicators.calculate_ema(df, 12)
        
        macd_data = TechnicalIndicators.calculate_macd(df)
        df['TEMP_MACD'] = macd_data['MACD']
        df['TEMP_MACD_Signal'] = macd_data['Signal']
        df['TEMP_MACD_Hist'] = macd_data['Histogram']
        
        bb_data = TechnicalIndicators.calculate_bbands(df)
        df['TEMP_BB_Upper'] = bb_data['BB_Upper']
        df['TEMP_BB_Middle'] = bb_data['BB_Middle']
        df['TEMP_BB_Lower'] = bb_data['BB_Lower']
        
        stoch_data = TechnicalIndicators.calculate_stochastic(df)
        df['TEMP_Stoch_K'] = stoch_data['Stoch_K']
        df['TEMP_Stoch_D'] = stoch_data['Stoch_D']
        
        df['TEMP_ATR'] = TechnicalIndicators.calculate_atr(df)
        
        return df
