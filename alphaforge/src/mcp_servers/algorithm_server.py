# src/mcp_servers/algorithm_server.py
import numpy as np
import pandas as pd
from scipy import stats
from arch import arch_model
import warnings
warnings.filterwarnings('ignore')

class AlgorithmServer:
    def __init__(self):
        self.technical_indicators = {}
        self.risk_models = {}
    
    async def calculate_technical_indicators(self, price_data: pd.DataFrame) -> dict:
        """Calculate comprehensive technical indicators"""
        df = price_data.copy()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Price trends
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['Trend'] = np.where(df['SMA_20'] > df['SMA_50'], 1, -1)
        
        return df.to_dict('records')
    
    async def calculate_fundamental_ratios(self, financial_data: dict) -> dict:
        """Calculate fundamental analysis ratios"""
        try:
            info = financial_data.get('info', {})
            
            ratios = {
                'valuation_ratios': {
                    'pe_ratio': info.get('trailingPE', 0),
                    'forward_pe': info.get('forwardPE', 0),
                    'price_to_book': info.get('priceToBook', 0),
                    'price_to_sales': info.get('priceToSalesTrailing12Months', 0),
                    'ev_to_ebitda': info.get('enterpriseToEbitda', 0)
                },
                'profitability_ratios': {
                    'roa': info.get('returnOnAssets', 0),
                    'roe': info.get('returnOnEquity', 0),
                    'gross_margin': info.get('grossMargins', 0),
                    'operating_margin': info.get('operatingMargins', 0),
                    'net_margin': info.get('profitMargins', 0)
                },
                'liquidity_ratios': {
                    'current_ratio': info.get('currentRatio', 0),
                    'quick_ratio': info.get('quickRatio', 0)
                },
                'efficiency_ratios': {
                    'asset_turnover': 0,  # Would calculate from financials
                    'inventory_turnover': 0
                }
            }
            
            return ratios
        except Exception as e:
            return {'error': str(e)}
    
    async def calculate_risk_metrics(self, returns: list) -> dict:
        """Calculate comprehensive risk metrics"""
        returns_array = np.array(returns)
        
        # Basic statistics
        volatility = np.std(returns_array) * np.sqrt(252)  # Annualized
        sharpe = np.mean(returns_array) / np.std(returns_array) * np.sqrt(252) if np.std(returns_array) > 0 else 0
        
        # Value at Risk (Historical)
        var_95 = np.percentile(returns_array, 5)
        var_99 = np.percentile(returns_array, 1)
        
        # Expected Shortfall
        es_95 = returns_array[returns_array <= var_95].mean()
        
        # Maximum Drawdown
        cumulative = (1 + returns_array).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # GARCH volatility
        try:
            garch_model = arch_model(returns_array * 100, vol='Garch', p=1, q=1)
            garch_fit = garch_model.fit(disp='off')
            garch_vol = garch_fit.conditional_volatility[-1] / 100
        except:
            garch_vol = volatility
        
        return {
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'var_95': var_95,
            'var_99': var_99,
            'expected_shortfall_95': es_95,
            'max_drawdown': max_drawdown,
            'garch_volatility': garch_vol
        }
    
    async def monte_carlo_simulation(self, initial_price: float, mu: float, sigma: float, 
                                   days: int = 252, simulations: int = 10000) -> dict:
        """Monte Carlo simulation for price forecasting"""
        dt = 1/252
        prices = np.zeros((days, simulations))
        prices[0] = initial_price
        
        for t in range(1, days):
            shock = np.random.normal(mu * dt, sigma * np.sqrt(dt), simulations)
            prices[t] = prices[t-1] * np.exp(shock)
        
        final_prices = prices[-1]
        
        return {
            'expected_price': np.mean(final_prices),
            'confidence_interval': {
                '5th_percentile': np.percentile(final_prices, 5),
                '95th_percentile': np.percentile(final_prices, 95)
            },
            'probability_profit': np.mean(final_prices > initial_price),
            'simulation_paths': prices.tolist()  # First 100 paths for visualization
        }