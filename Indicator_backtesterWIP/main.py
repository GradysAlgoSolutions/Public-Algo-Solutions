import sys
from data_generator import HistoricalDataGenerator
from indicators import TechnicalIndicators
from strategies import StrategyGenerator
from backtest_engine import BacktestEngine
from results_logger import ResultsLogger
import pandas as pd

class BacktestingSystem:
    
    def __init__(self):
        self.data_generator = HistoricalDataGenerator()
        self.backtest_engine = BacktestEngine(initial_capital=10000)
        self.logger = ResultsLogger()
        self.historical_data = None
        self.data_with_indicators = None
        
    def setup(self):
        print("\n" + "="*60)
        print("TRADING STRATEGY BACKTEST FRAMEWORK")
        print("="*60)
        print("\nInitializing system...")
        
        print("Generating historical price data (SPY, 2020-2024)...")
        self.historical_data = self.data_generator.generate_ohlcv(
            symbol='SPY',
            start_date='2020-01-01',
            end_date='2024-12-31',
            initial_price=300
        )
        
        print("Calculating technical indicators...")
        self.data_with_indicators = TechnicalIndicators.add_all_indicators(
            self.historical_data
        )
        
        self.data_with_indicators = self.data_with_indicators.dropna()
        
        print(f"✓ Data ready: {len(self.data_with_indicators)} trading days")
        print(f"  Date range: {self.data_with_indicators.index[0].date()} to {self.data_with_indicators.index[-1].date()}")
        print(f"  Price range: ${self.data_with_indicators['Close'].min():.2f} - ${self.data_with_indicators['Close'].max():.2f}")
    
    def display_available_strategies(self):
        strategies = StrategyGenerator.get_all_strategies()
        
        print("\n" + "="*60)
        print("AVAILABLE STRATEGIES")
        print("="*60)
        
        for i, (name, _) in enumerate(strategies.items(), 1):
            desc = StrategyGenerator.get_strategy_description(name)
            print(f"{i}. {name}")
            print(f"   {desc}")
        
        print("\n" + "="*60)
    
    def select_strategies(self):
        strategies = StrategyGenerator.get_all_strategies()
        strategy_names = list(strategies.keys())
        
        print("\nSelect strategies to backtest:")
        print("  - Enter numbers separated by commas (e.g., 1,3,5)")
        print("  - Enter 'all' to test all strategies")
        print("  - Enter 'q' to quit")
        
        while True:
            choice = input("\nYour selection: ").strip().lower()
            
            if choice == 'q':
                return None
            
            if choice == 'all':
                return strategy_names
            
            try:
                indices = [int(x.strip()) for x in choice.split(',')]
                selected = [strategy_names[i-1] for i in indices if 1 <= i <= len(strategy_names)]
                
                if selected:
                    return selected
                else:
                    print("Invalid selection. Please try again.")
            except (ValueError, IndexError):
                print("Invalid input. Please enter numbers separated by commas.")
    
    def run_backtest(self, strategy_name):
        strategies = StrategyGenerator.get_all_strategies()
        strategy_func = strategies[strategy_name]
        
        print(f"\n{'='*60}")
        print(f"RUNNING: {strategy_name}")
        print(f"{'='*60}")
        
        signals = strategy_func(self.data_with_indicators)
        
        results = self.backtest_engine.run_backtest(
            self.data_with_indicators,
            signals
        )
        
        metrics = results['metrics']
        print("\nPerformance Metrics:")
        print(f"  Total Return:              {metrics['total_return_pct']:>8.2f}%")
        print(f"  Annual Return:             {metrics['annual_return_pct']:>8.2f}%")
        print(f"  Final Equity:              ${metrics['final_equity']:>8,.2f}")
        print(f"  Sharpe Ratio:              {metrics['sharpe_ratio']:>8.2f}")
        print(f"  Sortino Ratio:             {metrics['sortino_ratio']:>8.2f}")
        print(f"  Calmar Ratio:              {metrics['calmar_ratio']:>8.2f}")
        print(f"  Max Drawdown:              {metrics['max_drawdown_pct']:>8.2f}%")
        print(f"  Avg Drawdown:              {metrics['avg_drawdown_pct']:>8.2f}%")
        print(f"  Max DD Duration (days):    {metrics['max_drawdown_duration']:>8}")
        print(f"  Ulcer Index:               {metrics['ulcer_index']:>8.2f}")
        print(f"  Recovery Factor:           {metrics['recovery_factor']:>8.2f}")
        print(f"  Number of Trades:          {metrics['num_trades']:>8}")
        print(f"  Winning Trades:            {metrics['num_wins']:>8}")
        print(f"  Losing Trades:             {metrics['num_losses']:>8}")
        print(f"  Win Rate:                  {metrics['win_rate_pct']:>8.2f}%")
        print(f"  Avg Win:                   ${metrics['avg_win']:>8.2f}")
        print(f"  Avg Loss:                  ${metrics['avg_loss']:>8.2f}")
        print(f"  Max Win:                   ${metrics['max_win']:>8.2f}")
        print(f"  Max Loss:                  ${metrics['max_loss']:>8.2f}")
        print(f"  Profit Factor:             {metrics['profit_factor']:>8.2f}")
        print(f"  Payoff Ratio:              {metrics['payoff_ratio']:>8.2f}")
        print(f"  Expectancy:                ${metrics['expectancy']:>8.2f}")
        print(f"  Avg Trade Duration:        {metrics['avg_trade_duration_days']:>8.2f} days")
        print(f"  Max Trade Duration:        {metrics['max_trade_duration_days']:>8} days")
        print(f"  Min Trade Duration:        {metrics['min_trade_duration_days']:>8} days")
        print(f"  Max Consecutive Wins:      {metrics['max_consecutive_wins']:>8}")
        print(f"  Max Consecutive Losses:    {metrics['max_consecutive_losses']:>8}")
        print(f"  Kelly Criterion:           {metrics['kelly_pct']:>8.2f}%")
        print(f"  Exposure Time:             {metrics['exposure_time_pct']:>8.2f}%")
        
        data_info = {
            'symbol': 'SPY',
            'start_date': str(self.data_with_indicators.index[0].date()),
            'end_date': str(self.data_with_indicators.index[-1].date()),
            'num_days': len(self.data_with_indicators)
        }
        self.logger.save_result(strategy_name, metrics, data_info)
        
        return results
    
    def run_multiple_backtests(self, strategy_names):
        results = {}
        
        for strategy_name in strategy_names:
            result = self.run_backtest(strategy_name)
            results[strategy_name] = result
            print()
        
        print("\n" + "="*60)
        print("ALL BACKTESTS COMPLETED")
        print("="*60)
        self.display_comparison(strategy_names)
    
    def display_comparison(self, strategy_names=None):
        print("\nQUICK COMPARISON (Current Run):")
        print("-" * 60)
        
        comparison_data = []
        for strategy_name in strategy_names or []:
            results = self.logger.get_results_by_strategy(strategy_name)
            if results:
                latest = results[-1]['metrics']
                comparison_data.append({
                    'Strategy': strategy_name,
                    'Return%': latest['total_return_pct'],
                    'Sharpe': latest['sharpe_ratio'],
                    'Sortino': latest['sortino_ratio'],
                    'Trades': latest['num_trades'],
                    'Win%': latest['win_rate_pct'],
                    'MaxDD%': latest['max_drawdown_pct'],
                    'ProfitFactor': latest['profit_factor']
                })
        
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            df = df.sort_values('Return%', ascending=False)
            print(df.to_string(index=False))
        else:
            print("No results to compare yet.")
    
    def analyze_all_results(self):
        print("\n" + "="*60)
        print("HISTORICAL ANALYSIS - ALL SAVED RESULTS")
        print("="*60)
        
        df = self.logger.get_summary_dataframe()
        
        if df.empty:
            print("\nNo historical results found. Run some backtests first!")
            return
        
        print(f"\nTotal tests in database: {len(df)}")
        print(f"Strategies tested: {', '.join(self.logger.get_unique_strategies())}")
        
        metrics_to_compare = [
            ('total_return_pct', 'Total Return %'),
            ('sharpe_ratio', 'Sharpe Ratio'),
            ('sortino_ratio', 'Sortino Ratio'),
            ('win_rate_pct', 'Win Rate %'),
            ('profit_factor', 'Profit Factor'),
            ('max_drawdown_pct', 'Max Drawdown %')
        ]
        
        for metric, label in metrics_to_compare:
            print(f"\n{'='*60}")
            print(f"RANKING BY: {label}")
            print(f"{'='*60}")
            comparison = self.logger.compare_strategies(metric)
            if not comparison.empty:
                print(comparison.to_string())
        
        print(f"\n{'='*60}")
        print("BEST STRATEGY OVERALL")
        print(f"{'='*60}")
        best_strategy, best_value = self.logger.get_best_strategy('total_return_pct')
        
        if best_strategy:
            print(f"\nBest performing strategy: {best_strategy}")
            print(f"Average total return: {best_value:.2f}%")
            
            best_results = self.logger.get_results_by_strategy(best_strategy)
            print(f"\nAll test runs for {best_strategy}:")
            for i, result in enumerate(best_results, 1):
                metrics = result['metrics']
                timestamp = result['timestamp'][:19]
                print(f"  {i}. {timestamp} - Return: {metrics['total_return_pct']:.2f}%, "
                      f"Sharpe: {metrics['sharpe_ratio']:.2f}, "
                      f"Trades: {metrics['num_trades']}")
        else:
            print("\nNot enough data to determine best strategy yet.")
    
    def main_menu(self):
        while True:
            print("\n" + "="*60)
            print("MAIN MENU")
            print("="*60)
            print("1. Run new backtests")
            print("2. Analyze all historical results")
            print("3. Export results to CSV")
            print("4. Clear all saved results")
            print("5. Exit")
            
            choice = input("\nSelect option (1-5): ").strip()
            
            if choice == '1':
                self.display_available_strategies()
                selected = self.select_strategies()
                if selected:
                    self.run_multiple_backtests(selected)
            
            elif choice == '2':
                self.analyze_all_results()
            
            elif choice == '3':
                self.logger.export_to_csv()
            
            elif choice == '4':
                confirm = input("Are you sure you want to clear all results? (yes/no): ").strip().lower()
                if confirm == 'yes':
                    self.logger.clear_results()
            
            elif choice == '5':
                print("\nThank you for using the Backtesting Framework!")
                break
            
            else:
                print("Invalid option. Please try again.")

def main():
    system = BacktestingSystem()
    system.setup()
    system.main_menu()

if __name__ == '__main__':
    main()
