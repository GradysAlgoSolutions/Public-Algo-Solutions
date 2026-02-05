import json
import os
from datetime import datetime
import pandas as pd

class ResultsLogger:
    
    def __init__(self, log_file='backtest_results.json'):
        self.log_file = log_file
        self._ensure_log_exists()
    
    def _ensure_log_exists(self):
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                json.dump([], f)
    
    def save_result(self, strategy_name, metrics, data_info=None):
        results = self.load_all_results()
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'strategy_name': strategy_name,
            'metrics': metrics,
            'data_info': data_info or {}
        }
        
        results.append(result)
        
        with open(self.log_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✓ Results saved for: {strategy_name}")
    
    def load_all_results(self):
        try:
            with open(self.log_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []
    
    def get_results_by_strategy(self, strategy_name):
        all_results = self.load_all_results()
        return [r for r in all_results if r['strategy_name'] == strategy_name]
    
    def get_unique_strategies(self):
        all_results = self.load_all_results()
        return list(set(r['strategy_name'] for r in all_results))
    
    def get_summary_dataframe(self):
        all_results = self.load_all_results()
        
        if not all_results:
            return pd.DataFrame()
        
        rows = []
        for result in all_results:
            row = {
                'timestamp': result['timestamp'],
                'strategy': result['strategy_name'],
            }
            row.update(result['metrics'])
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    
    def compare_strategies(self, metric='total_return_pct'):
        df = self.get_summary_dataframe()
        
        if df.empty:
            print("No results to compare yet.")
            return pd.DataFrame()
        
        summary = df.groupby('strategy')[metric].agg(['mean', 'std', 'count', 'min', 'max'])
        summary = summary.sort_values('mean', ascending=False)
        summary.columns = ['avg', 'std_dev', 'num_tests', 'min', 'max']
        
        return summary
    
    def get_best_strategy(self, metric='total_return_pct', min_tests=1):
        comparison = self.compare_strategies(metric)
        
        if comparison.empty:
            return None, None
        
        comparison = comparison[comparison['num_tests'] >= min_tests]
        
        if comparison.empty:
            return None, None
        
        best_strategy = comparison.index[0]
        best_value = comparison.iloc[0]['avg']
        
        return best_strategy, best_value
    
    def clear_results(self):
        with open(self.log_file, 'w') as f:
            json.dump([], f)
        print("All results cleared.")
    
    def export_to_csv(self, filename='backtest_results.csv'):
        df = self.get_summary_dataframe()
        if not df.empty:
            df.to_csv(filename, index=False)
            print(f"Results exported to {filename}")
        else:
            print("No results to export.")
