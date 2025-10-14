"""
config/c_portalloc.py
Portfolio Allocation Configuration Module
Black-Litterman, HRP, Risk Parity optimizasyonları için parametreler
"""

PortfolioAllocConfig = {
    "optimization_methods": {
        "black_litterman": {
            "enabled": True,
            "tau": 0.05,
            "risk_aversion": 2.5,
            "view_confidence": 0.75
        },
        "hierarchical_risk_parity": {
            "enabled": True,
            "linkage_method": "ward",
            "covariance_estimator": "ledoit_wolf"
        },
        "risk_parity": {
            "enabled": True,
            "max_iter": 1000,
            "tolerance": 1e-8
        }
    },
    
    "metrics": {
        "sharpe_ratio": {"risk_free_rate": 0.02},
        "var": {"confidence_level": 0.95, "time_horizon": 1},
        "sortino_ratio": {"target_return": 0.0},
        "max_drawdown": {"window": 252}
    },
    
    "constraints": {
        "max_allocation_per_asset": 0.3,
        "min_allocation_per_asset": 0.02,
        "total_leverage": 1.0,
        "target_return": 0.15
    },
    
    "data": {
        "lookback_period": 252,  # 1 year daily
        "min_data_points": 100,
        "correlation_threshold": 0.7
    },
    
    "weights": {
        "sharpe": 0.3,
        "sortino": 0.25,
        "var": 0.2,
        "drawdown": 0.15,
        "correlation": 0.1
    },
    
    "parallel_processing": {
        "enabled": True,
        "max_workers": 4,
        "batch_size": 10
    },
    
    "lifecycle": {
        "stage": "development",
        "version": "1.0.0",
        "dependencies": ["volatility", "correlation"]
    }
}

# Parametre açıklamaları
PARAM_DESCRIPTIONS = {
    "black_litterman": {
        "tau": "Beklentilerin güven seviyesi",
        "risk_aversion": "Riskten kaçınma katsayısı",
        "view_confidence": "Portföy görüş güven seviyesi"
    },
    "hierarchical_risk_parity": {
        "linkage_method": "Hiyerarşik kümeleme yöntemi",
        "covariance_estimator": "Kovaryans matrisi tahmin yöntemi"
    }
}