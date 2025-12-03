# src/mcp_servers/algorithm_server.py
import numpy as np
import pandas as pd
from scipy import stats
from arch import arch_model
import warnings
import logging
from typing import Dict, Any, List
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlgorithmServer:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=2)
        logger.info("Algorithm server initialized")

    async def calculate_technical_indicators(self, price_data: List[Dict[str, Any]]) -> dict:
        """Calculate comprehensive technical indicators asynchronously"""
        try:
            logger.info("Calculating technical indicators")

            def calculate_sync(data_list):
                # Convert to DataFrame
                df = pd.DataFrame(data_list)
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                    df.set_index('Date', inplace=True)

                # Ensure required columns
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                if not all(col in df.columns for col in required_cols):
                    raise ValueError(f"Missing required columns. Found: {df.columns.tolist()}")

                # RSI
                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df['RSI'] = 100 - (100 / (1 + rs))

                # MACD
                exp1 = df['Close'].ewm(span=12, adjust=False).mean()
                exp2 = df['Close'].ewm(span=26, adjust=False).mean()
                df['MACD'] = exp1 - exp2
                df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
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

                # Reset index for JSON serialization
                df.reset_index(inplace=True)
                return df.to_dict('records')

            # Run calculation in thread pool
            result = await asyncio.get_event_loop().run_in_executor(
                self.executor, calculate_sync, price_data
            )

            return {
                'indicators': result,
                'last_updated': datetime.now().isoformat(),
                'count': len(result)
            }

        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return {'error': str(e)}

    async def calculate_fundamental_ratios(self, financial_data: dict) -> dict:
        """Calculate fundamental analysis ratios"""
        try:
            logger.info("Calculating fundamental ratios")

            info = financial_data.get('info', {})

            ratios = {
                'valuation_ratios': {
                    'pe_ratio': float(info.get('trailingPE', 0)),
                    'forward_pe': float(info.get('forwardPE', 0)),
                    'price_to_book': float(info.get('priceToBook', 0)),
                    'price_to_sales': float(info.get('priceToSalesTrailing12Months', 0)),
                    'ev_to_ebitda': float(info.get('enterpriseToEbitda', 0))
                },
                'profitability_ratios': {
                    'roa': float(info.get('returnOnAssets', 0)),
                    'roe': float(info.get('returnOnEquity', 0)),
                    'gross_margin': float(info.get('grossMargins', 0)),
                    'operating_margin': float(info.get('operatingMargins', 0)),
                    'net_margin': float(info.get('profitMargins', 0))
                },
                'liquidity_ratios': {
                    'current_ratio': float(info.get('currentRatio', 0)),
                    'quick_ratio': float(info.get('quickRatio', 0))
                },
                'efficiency_ratios': {
                    'asset_turnover': 0.0,  # Would calculate from financials
                    'inventory_turnover': 0.0
                },
                'timestamp': datetime.now().isoformat()
            }

            return ratios

        except Exception as e:
            logger.error(f"Error calculating fundamental ratios: {e}")
            return {'error': str(e)}

    async def calculate_risk_metrics(self, returns: list) -> dict:
        """Calculate comprehensive risk metrics asynchronously"""
        try:
            logger.info("Calculating risk metrics")

            def calculate_risk_sync(returns_list):
                returns_array = np.array(returns_list, dtype=np.float64)

                if len(returns_array) < 2:
                    return {
                        'error': 'Insufficient data for risk calculation',
                        'count': len(returns_array)
                    }

                # Basic statistics
                volatility = np.std(returns_array, ddof=1) * np.sqrt(252)  # Annualized

                if np.std(returns_array) > 0:
                    sharpe = np.mean(returns_array) / np.std(returns_array) * np.sqrt(252)
                else:
                    sharpe = 0.0

                # Value at Risk (Historical)
                var_95 = np.percentile(returns_array, 5)
                var_99 = np.percentile(returns_array, 1)

                # Expected Shortfall
                returns_below_var = returns_array[returns_array <= var_95]
                if len(returns_below_var) > 0:
                    es_95 = float(np.mean(returns_below_var))
                else:
                    es_95 = 0.0

                # Maximum Drawdown
                cumulative = (1 + returns_array).cumprod()
                running_max = np.maximum.accumulate(cumulative)
                drawdown = (cumulative - running_max) / running_max
                max_drawdown = float(drawdown.min())

                # GARCH volatility (simplified)
                garch_vol = volatility
                try:
                    if len(returns_array) > 50:
                        # Scale returns for better numerical stability
                        scaled_returns = returns_array * 100
                        garch_model = arch_model(scaled_returns, vol='Garch', p=1, q=1)
                        garch_fit = garch_model.fit(disp='off')
                        garch_vol = float(garch_fit.conditional_volatility[-1] / 100)
                except Exception as e:
                    logger.warning(f"GARCH calculation failed: {e}")
                    garch_vol = volatility

                return {
                    'volatility': float(volatility),
                    'sharpe_ratio': float(sharpe),
                    'var_95': float(var_95),
                    'var_99': float(var_99),
                    'expected_shortfall_95': float(es_95),
                    'max_drawdown': float(max_drawdown),
                    'garch_volatility': float(garch_vol)
                }

            # Run in thread pool
            result = await asyncio.get_event_loop().run_in_executor(
                self.executor, calculate_risk_sync, returns
            )

            result['timestamp'] = datetime.now().isoformat()
            return result

        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {'error': str(e)}

    async def monte_carlo_simulation(self, initial_price: float, mu: float, sigma: float,
                                     days: int = 252, simulations: int = 1000,
                                     return_sample_paths: bool = False) -> dict:
        """Monte Carlo simulation for price forecasting asynchronously"""
        try:
            logger.info(f"Running Monte Carlo simulation: {simulations} simulations, {days} days")

            def run_simulation_sync(initial, mu_val, sigma_val, days_val, sims):
                dt = 1 / 252
                prices = np.zeros((days_val, sims))
                prices[0] = initial

                # Vectorized simulation
                for t in range(1, days_val):
                    shock = np.random.normal(mu_val * dt, sigma_val * np.sqrt(dt), sims)
                    prices[t] = prices[t - 1] * np.exp(shock)

                final_prices = prices[-1]

                result = {
                    'expected_price': float(np.mean(final_prices)),
                    'confidence_interval': {
                        '5th_percentile': float(np.percentile(final_prices, 5)),
                        '95th_percentile': float(np.percentile(final_prices, 95))
                    },
                    'probability_profit': float(np.mean(final_prices > initial)),
                    'simulation_count': sims,
                    'days': days_val
                }

                # Only return sample paths if requested
                if return_sample_paths:
                    n_sample = min(100, sims)  # Limit to 100 sample paths
                    result['sample_paths'] = prices[:, :n_sample].tolist()

                return result

            # Limit parameters for safety
            days = min(max(days, 1), 1000)  # 1-1000 days
            simulations = min(max(simulations, 100), 10000)  # 100-10000 simulations

            # Run in thread pool
            result = await asyncio.get_event_loop().run_in_executor(
                self.executor, run_simulation_sync,
                initial_price, mu, sigma, days, simulations
            )

            result['timestamp'] = datetime.now().isoformat()
            return result

        except Exception as e:
            logger.error(f"Error in Monte Carlo simulation: {e}")
            return {'error': str(e)}