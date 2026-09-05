import threading
import numpy as np
from typing import Any

class InvestmentAnalyzer:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance.data = {}
        return cls._instance
    
    def __init__(self):  
        if not hasattr(self, 'initialized'):  
            self.initialized = True 
            self.data = {}  
    
    def record_score(self, date:int, investor_type: int, investor_id: int, stocks: dict[str, int]):
        with threading.Lock():
            if date not in self.data:
                self.data[date] = {}
            if investor_type not in self.data[date]:
                self.data[date][investor_type] = {}
            for (key, value) in stocks.items():
                if key not in self.data[date][investor_type]:
                    self.data[date][investor_type][key] = {
                        "score": 0,
                        "num_investors": 0
                    }
                self.data[date][investor_type][key]["score"] += value
                self.data[date][investor_type][key]["num_investors"] += 1
    
    def calculate_stock_disagreement_score(self,
                                           date: int,
                                           stock_pool: list[str], 
                                           agent_distributions:dict[int, float] ,
                                           alpha:float = 0.5,
                                           disagreement_diagnostics: dict[str, dict[str, Any]] | None = None):
        res = self.collect_stock_opinions(date, stock_pool, agent_distributions)
        for stock_code, values in res.items():
            scores = np.array(values["scores"], dtype=float)
            weights = np.array(values["weights"], dtype=float)
            if weights.sum() <= 0:
                mean_value = 0.0
                std = 0.0
            else:
                mean_value = float(np.sum(scores * weights) / np.sum(weights))
                std = float(np.sqrt(np.average((scores - mean_value) ** 2, weights=weights)))
            signal = alpha * mean_value - (1 - alpha) * std
            diagnostics = disagreement_diagnostics.get(stock_code, {}) if disagreement_diagnostics else {}
            exposure_control = float(diagnostics.get("exposure_control", 1.0))
            final_signal = exposure_control * signal
            res[stock_code] = [
                final_signal,
                mean_value,
                -std,
                exposure_control,
                float(diagnostics.get("severity", std)),
                str(diagnostics.get("type", "aggregate")),
            ]
        return res

    def collect_stock_opinions(self,
                               date: int,
                               stock_pool: list[str],
                               agent_distributions: dict[int, float]) -> dict[str, dict[str, list[float]]]:
        res: dict[str, dict[str, list[float]]] = {}
        distributions = []
        with threading.Lock():
            for investor_type in self.data[date]:
                if investor_type in agent_distributions:
                    distribution = agent_distributions[investor_type]
                else:
                    distribution = 0.0
                    raise ValueError(f"agent type {investor_type} does not exist!")
                for stock in stock_pool:
                    if stock not in self.data[date][investor_type]:
                        self.data[date][investor_type][stock] = {
                            "score": 0,
                            "num_investors": 0
                    }
                for stock_code in self.data[date][investor_type]:
                    stock_data = self.data[date][investor_type][stock_code]
                    total_scores = stock_data["score"]
                    total_investors = stock_data["num_investors"]
                    if stock_code not in res:
                        res[stock_code] = {"scores": [], "weights": [], "agent_types": []}
                    res[stock_code]["scores"].append((total_scores / total_investors) if total_investors != 0 else 0)
                    res[stock_code]["weights"].append(distribution)
                    res[stock_code]["agent_types"].append(investor_type)
                distributions.append(distribution)
        return res
