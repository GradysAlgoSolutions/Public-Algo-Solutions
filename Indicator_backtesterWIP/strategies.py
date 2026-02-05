import pandas as pd

class StrategyGenerator:
    
    @staticmethod
    def TEMP_strategy_rsi_only(data):
        signals = pd.DataFrame(index=data.index)
        signals['entry'] = (data['TEMP_RSI'] > 30) & (data['TEMP_RSI'].shift(1) <= 30)
        signals['exit'] = (data['TEMP_RSI'] < 70) & (data['TEMP_RSI'].shift(1) >= 70)
        return signals
    
    @staticmethod
    def TEMP_strategy_macd_only(data):
        signals = pd.DataFrame(index=data.index)
        signals['entry'] = (data['TEMP_MACD'] > data['TEMP_MACD_Signal']) & \
                          (data['TEMP_MACD'].shift(1) <= data['TEMP_MACD_Signal'].shift(1))
        signals['exit'] = (data['TEMP_MACD'] < data['TEMP_MACD_Signal']) & \
                         (data['TEMP_MACD'].shift(1) >= data['TEMP_MACD_Signal'].shift(1))
        return signals
    
    @staticmethod
    def TEMP_strategy_sma_crossover(data):
        signals = pd.DataFrame(index=data.index)
        signals['entry'] = (data['TEMP_SMA_20'] > data['TEMP_SMA_50']) & \
                          (data['TEMP_SMA_20'].shift(1) <= data['TEMP_SMA_50'].shift(1))
        signals['exit'] = (data['TEMP_SMA_20'] < data['TEMP_SMA_50']) & \
                         (data['TEMP_SMA_20'].shift(1) >= data['TEMP_SMA_50'].shift(1))
        return signals
    
    @staticmethod
    def TEMP_strategy_rsi_macd_combo(data):
        signals = pd.DataFrame(index=data.index)
        
        rsi_bullish = data['TEMP_RSI'] > 30
        macd_cross_up = (data['TEMP_MACD'] > data['TEMP_MACD_Signal']) & \
                       (data['TEMP_MACD'].shift(1) <= data['TEMP_MACD_Signal'].shift(1))
        
        rsi_bearish = data['TEMP_RSI'] > 70
        macd_cross_down = (data['TEMP_MACD'] < data['TEMP_MACD_Signal']) & \
                         (data['TEMP_MACD'].shift(1) >= data['TEMP_MACD_Signal'].shift(1))
        
        signals['entry'] = rsi_bullish & macd_cross_up
        signals['exit'] = rsi_bearish | macd_cross_down
        return signals
    
    @staticmethod
    def TEMP_strategy_bbands_rsi(data):
        signals = pd.DataFrame(index=data.index)
        signals['entry'] = (data['Close'] <= data['TEMP_BB_Lower']) & (data['TEMP_RSI'] < 35)
        signals['exit'] = (data['Close'] >= data['TEMP_BB_Upper']) | (data['TEMP_RSI'] > 65)
        return signals
    
    @staticmethod
    def TEMP_strategy_stochastic_only(data):
        signals = pd.DataFrame(index=data.index)
        signals['entry'] = (data['TEMP_Stoch_K'] > 20) & (data['TEMP_Stoch_K'].shift(1) <= 20)
        signals['exit'] = (data['TEMP_Stoch_K'] < 80) & (data['TEMP_Stoch_K'].shift(1) >= 80)
        return signals
    
    @staticmethod
    def TEMP_strategy_triple_confirmation(data):
        signals = pd.DataFrame(index=data.index)
        
        rsi_ok = data['TEMP_RSI'] > 30
        macd_bullish = data['TEMP_MACD'] > data['TEMP_MACD_Signal']
        macd_was_bearish = data['TEMP_MACD'].shift(1) <= data['TEMP_MACD_Signal'].shift(1)
        stoch_ok = data['TEMP_Stoch_K'] > 20
        
        rsi_exit = data['TEMP_RSI'] > 70
        macd_bearish = data['TEMP_MACD'] < data['TEMP_MACD_Signal']
        
        signals['entry'] = rsi_ok & macd_bullish & macd_was_bearish & stoch_ok
        signals['exit'] = rsi_exit | macd_bearish
        return signals
    
    @staticmethod
    def get_all_strategies():
        return {
            'TEMP_RSI_Only': StrategyGenerator.TEMP_strategy_rsi_only,
            'TEMP_MACD_Only': StrategyGenerator.TEMP_strategy_macd_only,
            'TEMP_SMA_Crossover': StrategyGenerator.TEMP_strategy_sma_crossover,
            'TEMP_RSI_MACD_Combo': StrategyGenerator.TEMP_strategy_rsi_macd_combo,
            'TEMP_BBands_RSI': StrategyGenerator.TEMP_strategy_bbands_rsi,
            'TEMP_Stochastic_Only': StrategyGenerator.TEMP_strategy_stochastic_only,
            'TEMP_Triple_Confirmation': StrategyGenerator.TEMP_strategy_triple_confirmation
        }
    
    @staticmethod
    def get_strategy_description(strategy_name):
        descriptions = {
            'TEMP_RSI_Only': 'TEMPORARY - Uses: RSI (14)',
            'TEMP_MACD_Only': 'TEMPORARY - Uses: MACD (12,26,9)',
            'TEMP_SMA_Crossover': 'TEMPORARY - Uses: SMA(20), SMA(50)',
            'TEMP_RSI_MACD_Combo': 'TEMPORARY - Uses: RSI (14), MACD (12,26,9)',
            'TEMP_BBands_RSI': 'TEMPORARY - Uses: Bollinger Bands (20,2), RSI (14)',
            'TEMP_Stochastic_Only': 'TEMPORARY - Uses: Stochastic (14,3)',
            'TEMP_Triple_Confirmation': 'TEMPORARY - Uses: RSI (14), MACD (12,26,9), Stochastic (14,3)'
        }
        return descriptions.get(strategy_name, 'No description available')
