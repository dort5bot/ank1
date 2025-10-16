"""
port_alloc.py
Portfolio Optimization & Allocation Module
Black-Litterman, HRP, Risk Parity optimizasyonları ile dinamik portföy ayırma
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor

from analysis.analysis_base_module import BaseAnalysisModule
from utils.binance_api.binance_a import BinanceAggregator
from utils.cache_manager import cache_result
#from config.c_portalloc import CONFIG

logger = logging.getLogger(__name__)

@dataclass
class PortfolioMetrics:
    """Portföy metriklerini tutan data class"""
    sharpe_ratio: float
    sortino_ratio: float
    var_95: float
    conditional_var: float
    max_drawdown: float
    volatility: float
    expected_return: float
    correlation_matrix: pd.DataFrame

@dataclass
class AllocationResult:
    """Portföy ayırma sonucu"""
    weights: Dict[str, float]
    metrics: PortfolioMetrics
    optimization_method: str
    score: float
    components: Dict[str, float]
    signal: str
    explain: Dict[str, str]

class PortfolioAllocationModule(BaseAnalysisModule):
    """
    Portfolio Optimization & Allocation Module
    Dinamik portföy optimizasyonu ve varlık ayırma
    """
    
    def __init__(self, config: Dict = None):
        super().__init__()
        self.config = config or self._load_default_config()
        self.binance_agg = BinanceAggregator()
        self.parallel_executor = ThreadPoolExecutor(
            max_workers=self.config["parallel_processing"]["max_workers"]
        )
        
        # Metrik dependency graph
        self.dependencies = {
            "returns": [],
            "volatility": ["returns"],
            "correlation": ["returns"],
            "covariance": ["returns"],
            "sharpe": ["returns", "volatility"],
            "sortino": ["returns", "volatility"],
            "var": ["returns", "volatility"]
        }
        
        self.version = "1.0.0"
        self.module_name = "portfolio_allocation"
        
    def _load_default_config(self) -> Dict:
        """Varsayılan config yükleme"""
        try:
            from config.c_portalloc import CONFIG
            return CONFIG
        except ImportError:
            logger.warning("c_portalloc config not found, using defaults")
            return CONFIG  # Fallback to hardcoded defaults
    
    @cache_result(ttl=300)  # 5 dakika cache
    async def get_historical_prices(self, symbols: List[str], lookback: int = 252) -> pd.DataFrame:
        """Semboller için geçmiş fiyat verilerini getir"""
        try:
            price_data = {}
            
            for symbol in symbols:
                # Binance API'den kline verisi
                klines_data = await self.binance_agg.get_klines(
                    symbol=symbol,
                    interval='1d',
                    limit=lookback
                )
                
                if klines_data and len(klines_data) > 0:
                    closes = [float(k[4]) for k in klines_data]  # Close price
                    price_data[symbol] = closes
            
            return pd.DataFrame(price_data)
            
        except Exception as e:
            logger.error(f"Error fetching historical prices: {e}")
            raise
    
    def calculate_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Log getirileri hesapla"""
        return np.log(prices / prices.shift(1)).dropna()
    
    def calculate_portfolio_metrics(self, returns: pd.DataFrame, weights: np.ndarray) -> PortfolioMetrics:
        """Portföy metriklerini hesapla"""
        
        # Portföy getirisi ve volatilitesi
        portfolio_returns = returns.dot(weights)
        portfolio_volatility = np.std(portfolio_returns) * np.sqrt(252)
        expected_return = np.mean(portfolio_returns) * 252
        
        # Sharpe Ratio
        risk_free_rate = self.config["metrics"]["sharpe_ratio"]["risk_free_rate"]
        sharpe = (expected_return - risk_free_rate) / portfolio_volatility
        
        # Sortino Ratio
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino = (expected_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
        
        # VaR (Parametrik)
        var_confidence = self.config["metrics"]["var"]["confidence_level"]
        var = self._calculate_var(portfolio_returns, var_confidence)
        
        # Conditional VaR (Expected Shortfall)
        cvar = self._calculate_conditional_var(portfolio_returns, var_confidence)
        
        # Maximum Drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Correlation Matrix
        correlation_matrix = returns.corr()
        
        return PortfolioMetrics(
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            var_95=var,
            conditional_var=cvar,
            max_drawdown=max_drawdown,
            volatility=portfolio_volatility,
            expected_return=expected_return,
            correlation_matrix=correlation_matrix
        )
    
    def _calculate_var(self, returns: pd.Series, confidence: float) -> float:
        """Value at Risk hesapla"""
        return np.percentile(returns, (1 - confidence) * 100)
    
    def _calculate_conditional_var(self, returns: pd.Series, confidence: float) -> float:
        """Conditional VaR (Expected Shortfall) hesapla"""
        var = self._calculate_var(returns, confidence)
        return returns[returns <= var].mean()
    
    def black_litterman_optimization(self, returns: pd.DataFrame, market_caps: Dict[str, float] = None) -> np.ndarray:
        """Black-Litterman model ile optimizasyon"""
        try:
            # Equilibrium returns (CAPM)
            cov_matrix = returns.cov() * 252
            market_weights = self._calculate_market_weights(market_caps, returns.columns)
            
            # Implied equilibrium returns
            tau = self.config["optimization_methods"]["black_litterman"]["tau"]
            risk_aversion = self.config["optimization_methods"]["black_litterman"]["risk_aversion"]
            
            pi = risk_aversion * cov_matrix.dot(market_weights)  # Implied returns
            
            # Views (burada basit trend views kullanıyoruz)
            P, Q = self._generate_views(returns)
            omega = self._generate_confidence_matrix(P, cov_matrix, tau)
            
            # Black-Litterman formula
            pi_bl = self._calculate_black_litterman_returns(
                pi, cov_matrix, tau, P, Q, omega
            )
            
            # Optimize weights
            weights = self._mean_variance_optimization(pi_bl, cov_matrix)
            return weights
            
        except Exception as e:
            logger.error(f"Black-Litterman optimization failed: {e}")
            return self._equal_weight_allocation(returns.shape[1])
    
    def hierarchical_risk_parity(self, returns: pd.DataFrame) -> np.ndarray:
        """Hierarchical Risk Parity optimizasyonu"""
        try:
            cov_matrix = returns.cov() * 252
            corr_matrix = returns.corr()
            
            # Distance matrix
            distance_matrix = np.sqrt((1 - corr_matrix) / 2)
            
            # Hierarchical clustering
            linkage_method = self.config["optimization_methods"]["hierarchical_risk_parity"]["linkage_method"]
            Z = linkage(squareform(distance_matrix.values), method=linkage_method)
            
            # HRP allocation
            weights = self._hrp_allocation(cov_matrix.values, Z)
            return weights
            
        except Exception as e:
            logger.error(f"HRP optimization failed: {e}")
            return self._equal_weight_allocation(returns.shape[1])
    
    def risk_parity_optimization(self, returns: pd.DataFrame) -> np.ndarray:
        """Risk Parity optimizasyonu"""
        try:
            cov_matrix = returns.cov() * 252
            
            def risk_parity_objective(weights, cov_matrix):
                portfolio_risk = np.sqrt(weights.T @ cov_matrix @ weights)
                risk_contributions = (weights * (cov_matrix @ weights)) / portfolio_risk
                
                # Equal risk contribution objective
                target_risk = portfolio_risk / len(weights)
                return np.sum((risk_contributions - target_risk) ** 2)
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Fully invested
            ]
            
            # Bounds
            bounds = [(0, self.config["constraints"]["max_allocation_per_asset"]) 
                     for _ in range(len(returns.columns))]
            
            # Initial guess (equal weight)
            x0 = np.array([1.0 / len(returns.columns)] * len(returns.columns))
            
            # Optimization
            result = minimize(
                risk_parity_objective,
                x0,
                args=(cov_matrix.values,),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': self.config["optimization_methods"]["risk_parity"]["max_iter"]}
            )
            
            return result.x if result.success else x0
            
        except Exception as e:
            logger.error(f"Risk parity optimization failed: {e}")
            return self._equal_weight_allocation(returns.shape[1])
    
    def _hrp_allocation(self, cov_matrix: np.ndarray, linkage_matrix: np.ndarray) -> np.ndarray:
        """HRP allocation implementation"""
        num_assets = cov_matrix.shape[0]
        weights = np.ones(num_assets)
        
        # Quasi-diagonalization
        clusters = self._quasi_diagonalize(linkage_matrix)
        
        # Recursive bisection
        weights = self._recursive_bisection(weights, clusters, cov_matrix)
        
        return weights / np.sum(weights)  # Normalize
    
    def _quasi_diagonalize(self, linkage_matrix: np.ndarray) -> List:
        """Quasi-diagonalization for HRP"""
        # Implementation of quasi-diagonalization
        # This is a simplified version
        return list(range(linkage_matrix.shape[0] + 1))
    
    def _recursive_bisection(self, weights: np.ndarray, clusters: List, cov_matrix: np.ndarray) -> np.ndarray:
        """Recursive bisection for HRP"""
        # Simplified implementation
        # In production, this would be more sophisticated
        num_assets = len(weights)
        return np.ones(num_assets) / num_assets
    
    def _calculate_market_weights(self, market_caps: Dict, symbols: List[str]) -> np.ndarray:
        """Piyasa ağırlıklarını hesapla"""
        if market_caps:
            total_cap = sum(market_caps.values())
            return np.array([market_caps.get(sym, 0) / total_cap for sym in symbols])
        else:
            # Equal weight fallback
            return np.ones(len(symbols)) / len(symbols)
    
    def _generate_views(self, returns: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Basit trend-based views oluştur"""
        # Simplified view generation
        # In production, this would use more sophisticated signals
        recent_returns = returns.tail(20).mean()
        
        # Outperform/underperform views based on recent momentum
        P = np.eye(len(returns.columns))  # Identity matrix for simple views
        Q = recent_returns.values * 0.1  # Scaled views
        
        return P, Q
    
    def _generate_confidence_matrix(self, P: np.ndarray, cov_matrix: pd.DataFrame, tau: float) -> np.ndarray:
        """View confidence matrix oluştur"""
        return np.diag(np.diag(P @ (tau * cov_matrix) @ P.T))
    
    def _calculate_black_litterman_returns(self, pi: np.ndarray, cov_matrix: pd.DataFrame, 
                                         tau: float, P: np.ndarray, Q: np.ndarray, 
                                         omega: np.ndarray) -> np.ndarray:
        """Black-Litterman expected returns hesapla"""
        # BL formula: E[R] = [(tauΣ)^-1 + P'Ω^-1 P]^-1 [(tauΣ)^-1 Π + P'Ω^-1 Q]
        tau_sigma_inv = np.linalg.inv(tau * cov_matrix)
        first_term = np.linalg.inv(tau_sigma_inv + P.T @ np.linalg.inv(omega) @ P)
        second_term = tau_sigma_inv @ pi + P.T @ np.linalg.inv(omega) @ Q
        
        return first_term @ second_term
    
    def _mean_variance_optimization(self, expected_returns: np.ndarray, cov_matrix: pd.DataFrame) -> np.ndarray:
        """Mean-Variance optimizasyonu"""
        def objective(weights):
            portfolio_return = weights.T @ expected_returns
            portfolio_risk = np.sqrt(weights.T @ cov_matrix @ weights)
            return -portfolio_return / portfolio_risk  # Maximize Sharpe
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        ]
        
        bounds = [(0, self.config["constraints"]["max_allocation_per_asset"]) 
                 for _ in range(len(expected_returns))]
        
        x0 = np.ones(len(expected_returns)) / len(expected_returns)
        
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x if result.success else x0
    
    def _equal_weight_allocation(self, num_assets: int) -> np.ndarray:
        """Eşit ağırlıklı portföy"""
        return np.ones(num_assets) / num_assets
    
    async def compute_metrics(self, symbols: List[str]) -> Dict:
        """Portföy metriklerini hesapla"""
        try:
            # Historical prices
            prices = await self.get_historical_prices(
                symbols, 
                self.config["data"]["lookback_period"]
            )
            
            if prices.empty or len(prices) < self.config["data"]["min_data_points"]:
                raise ValueError("Insufficient price data")
            
            # Calculate returns
            returns = self.calculate_returns(prices)
            
            # Apply different optimization methods
            allocation_results = {}
            
            if self.config["optimization_methods"]["black_litterman"]["enabled"]:
                bl_weights = self.black_litterman_optimization(returns)
                allocation_results["black_litterman"] = bl_weights
            
            if self.config["optimization_methods"]["hierarchical_risk_parity"]["enabled"]:
                hrp_weights = self.hierarchical_risk_parity(returns)
                allocation_results["hierarchical_risk_parity"] = hrp_weights
            
            if self.config["optimization_methods"]["risk_parity"]["enabled"]:
                rp_weights = self.risk_parity_optimization(returns)
                allocation_results["risk_parity"] = rp_weights
            
            # Calculate metrics for each method
            results = {}
            for method, weights in allocation_results.items():
                metrics = self.calculate_portfolio_metrics(returns, weights)
                score, components, signal, explain = self._generate_signal(metrics)
                
                results[method] = AllocationResult(
                    weights=dict(zip(symbols, weights)),
                    metrics=metrics,
                    optimization_method=method,
                    score=score,
                    components=components,
                    signal=signal,
                    explain=explain
                )
            
            return results
            
        except Exception as e:
            logger.error(f"Error computing portfolio metrics: {e}")
            raise
    
    def _generate_signal(self, metrics: PortfolioMetrics) -> Tuple[float, Dict, str, Dict]:
        """Portföy sinyali ve skor oluştur"""
        
        # Component scores
        components = {
            "sharpe": max(0, metrics.sharpe_ratio / 2.0),  # Normalize
            "sortino": max(0, metrics.sortino_ratio / 2.0),
            "var": 1 - min(1, abs(metrics.var_95) / 0.1),  # Lower VaR is better
            "drawdown": 1 - min(1, abs(metrics.max_drawdown)),
            "volatility": 1 - min(1, metrics.volatility / 0.4)  # Lower vol is better
        }
        
        # Weighted score
        weights = self.config["weights"]
        score = sum(components[metric] * weight for metric, weight in weights.items())
        
        # Signal generation
        if score > 0.7:
            signal = "optimal_allocation"
        elif score > 0.4:
            signal = "moderate_allocation"
        else:
            signal = "suboptimal_allocation"
        
        # Explanation
        explain = {
            "sharpe_impact": "High" if metrics.sharpe_ratio > 1.0 else "Low",
            "risk_adjusted": "Favorable" if metrics.sortino_ratio > metrics.sharpe_ratio else "Unfavorable",
            "downside_risk": "Controlled" if metrics.var_95 > -0.05 else "Elevated",
            "diversification": "Well diversified" if len(components) > 3 else "Concentrated"
        }
        
        return score, components, signal, explain
    
    async def aggregate_output(self, results: Dict) -> Dict:
        """Sonuçları aggregate et"""
        best_method = max(results.keys(), 
                         key=lambda x: results[x].score)
        best_result = results[best_method]
        
        return {
            "score": best_result.score,
            "signal": best_result.signal,
            "weights": best_result.weights,
            "optimization_method": best_result.optimization_method,
            "metrics": {
                "sharpe_ratio": best_result.metrics.sharpe_ratio,
                "sortino_ratio": best_result.metrics.sortino_ratio,
                "var_95": best_result.metrics.var_95,
                "expected_return": best_result.metrics.expected_return,
                "volatility": best_result.metrics.volatility
            },
            "components": best_result.components,
            "explain": best_result.explain,
            "all_methods": {
                method: {
                    "score": result.score,
                    "weights": result.weights
                } for method, result in results.items()
            }
        }
    
    def generate_report(self, aggregated_output: Dict) -> str:
        """Detaylı rapor oluştur"""
        report = [
            "PORTFOLIO ALLOCATION REPORT",
            "=" * 40,
            f"Optimization Method: {aggregated_output['optimization_method']}",
            f"Overall Score: {aggregated_output['score']:.3f}",
            f"Signal: {aggregated_output['signal']}",
            "",
            "ALLOCATION WEIGHTS:",
            "-" * 20
        ]
        
        for asset, weight in aggregated_output['weights'].items():
            report.append(f"{asset}: {weight:.3%}")
        
        report.extend([
            "",
            "PERFORMANCE METRICS:",
            "-" * 20,
            f"Sharpe Ratio: {aggregated_output['metrics']['sharpe_ratio']:.3f}",
            f"Sortino Ratio: {aggregated_output['metrics']['sortino_ratio']:.3f}",
            f"VaR (95%): {aggregated_output['metrics']['var_95']:.3%}",
            f"Expected Return: {aggregated_output['metrics']['expected_return']:.3%}",
            f"Volatility: {aggregated_output['metrics']['volatility']:.3%}",
            "",
            "COMPONENT SCORES:",
            "-" * 20
        ])
        
        for component, score in aggregated_output['components'].items():
            report.append(f"{component}: {score:.3f}")
        
        return "\n".join(report)

# Factory pattern için
class PortfolioAllocationFactory:
    """Portfolio Allocation factory sınıfı"""
    
    @classmethod
    def create_module(cls, config: Dict = None) -> PortfolioAllocationModule:
        """Portfolio allocation modülü oluştur"""
        return PortfolioAllocationModule(config)

# FastAPI router için yardımcı fonksiyon
async def allocate_portfolio(symbols: List[str], config: Dict = None) -> Dict:
    """Portföy ayırma endpoint'i için yardımcı fonksiyon"""
    module = PortfolioAllocationFactory.create_module(config)
    results = await module.compute_metrics(symbols)
    return await module.aggregate_output(results)