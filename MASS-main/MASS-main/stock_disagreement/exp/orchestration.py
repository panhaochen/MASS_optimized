from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MarketState:
    date: int
    volatility: float
    breadth: float
    industry_dispersion: float
    event_intensity: float
    complexity: float


class MarketStateEncoder:
    def __init__(
        self,
        stock_pool: pd.DataFrame,
        stock_labels: pd.DataFrame,
        news_relationship: pd.DataFrame | None = None,
    ) -> None:
        self.stock_pool = stock_pool
        self.stock_labels = stock_labels
        self.news_relationship = news_relationship if news_relationship is not None else pd.DataFrame()

    def encode(self, date: int) -> MarketState:
        labels = self.stock_labels[self.stock_labels["Date"] == date]
        returns = labels["1_15_labelB"] if "1_15_labelB" in labels else pd.Series(dtype=float)
        volatility = self._safe_float(returns.std())
        breadth = self._safe_float((returns > 0).mean()) if len(returns) else 0.5

        pool = self.stock_pool[self.stock_pool["Date"] == date]
        merged = pool.merge(labels[["Stock", "1_15_labelB"]], on="Stock", how="left") if len(labels) else pool
        if "Industry" in merged and "1_15_labelB" in merged:
            industry_ret = merged.groupby("Industry")["1_15_labelB"].mean()
            industry_dispersion = self._safe_float(industry_ret.std())
        else:
            industry_dispersion = 0.0

        if not self.news_relationship.empty and "Date" in self.news_relationship:
            event_intensity = float(len(self.news_relationship[self.news_relationship["Date"] == date]))
            event_intensity = event_intensity / max(float(len(pool)), 1.0)
        else:
            event_intensity = 0.0

        complexity = self._normalize(volatility) * 0.4
        complexity += abs(breadth - 0.5) * 0.6
        complexity += self._normalize(industry_dispersion) * 0.3
        complexity += min(event_intensity, 1.0) * 0.2
        complexity = min(max(complexity, 0.0), 1.0)
        return MarketState(date, volatility, breadth, industry_dispersion, event_intensity, complexity)

    @staticmethod
    def _normalize(value: float) -> float:
        if not np.isfinite(value):
            return 0.0
        return float(min(abs(value), 1.0))

    @staticmethod
    def _safe_float(value: Any) -> float:
        if value is None or not np.isfinite(value):
            return 0.0
        return float(value)


class DiversityCostAwareRouter:
    def __init__(
        self,
        top_k: int = 0,
        min_agents: int = 1,
        diversity_lambda: float = 0.35,
        cost_lambda: float = 0.05,
    ) -> None:
        self.top_k = top_k
        self.min_agents = max(1, min_agents)
        self.diversity_lambda = diversity_lambda
        self.cost_lambda = cost_lambda

    def select_agents(self, agents: list[Any], market_state: MarketState) -> list[Any]:
        if not agents:
            return []
        budget = self._budget(len(agents), market_state)
        selected: list[Any] = []
        remaining = list(agents)
        while remaining and len(selected) < budget:
            best_agent = max(
                remaining,
                key=lambda agent: self._selection_score(agent, selected, market_state),
            )
            selected.append(best_agent)
            remaining.remove(best_agent)
        return selected

    def active_distributions(self, agents: list[Any], base_distributions: dict[int, float]) -> dict[int, float]:
        active_types = {agent.modality for agent in agents}
        distributions = {
            agent_type: weight
            for agent_type, weight in base_distributions.items()
            if agent_type in active_types and weight > 0
        }
        total = sum(distributions.values())
        if total <= 0 and active_types:
            return {agent_type: 1.0 / len(active_types) for agent_type in active_types}
        return {agent_type: weight / total for agent_type, weight in distributions.items()}

    def _budget(self, agent_count: int, market_state: MarketState) -> int:
        if self.top_k > 0:
            return min(max(self.min_agents, self.top_k), agent_count)
        dynamic_budget = int(np.ceil(self.min_agents + market_state.complexity * (agent_count - self.min_agents)))
        return min(max(self.min_agents, dynamic_budget), agent_count)

    def _selection_score(self, agent: Any, selected: list[Any], market_state: MarketState) -> float:
        modality_count = max(int(agent.modality).bit_count(), 1)
        utility = 1.0 + 0.15 * modality_count + 0.25 * market_state.complexity
        diversity_penalty = sum(self._similarity(agent, other) for other in selected)
        cost = float(getattr(agent, "estimated_cost", 1.0))
        return utility - self.diversity_lambda * diversity_penalty - self.cost_lambda * cost

    @staticmethod
    def _similarity(agent_a: Any, agent_b: Any) -> float:
        a = int(agent_a.modality)
        b = int(agent_b.modality)
        union = (a | b).bit_count()
        if union == 0:
            return 0.0
        return (a & b).bit_count() / union


class StructuredDisagreementDiagnoser:
    def __init__(
        self,
        severity_threshold: float = 0.35,
        abstention_threshold: float = 0.45,
    ) -> None:
        self.severity_threshold = severity_threshold
        self.abstention_threshold = abstention_threshold

    def diagnose(
        self,
        stock_opinions: dict[str, dict[str, list[float]]],
        market_state: MarketState,
    ) -> dict[str, dict[str, Any]]:
        diagnostics = {}
        for stock, values in stock_opinions.items():
            scores = np.array(values["scores"], dtype=float)
            weights = np.array(values["weights"], dtype=float)
            if len(scores) == 0:
                pred_dispersion = 0.0
                confidence_dispersion = 0.0
            else:
                pred_dispersion = float(np.std(scores))
                confidence_dispersion = float(np.std(np.abs(scores - scores.mean())))
            horizon_gap = float(len(set(values.get("agent_types", []))) > 1)
            severity = pred_dispersion * 0.6 + confidence_dispersion * 0.2
            severity += horizon_gap * 0.1 + market_state.complexity * 0.1
            exposure_control = max(0.0, 1.0 - severity)
            if severity >= self.abstention_threshold:
                action = "reduce_exposure"
            elif severity >= self.severity_threshold:
                action = "diagnose"
            else:
                action = "aggregate"
            diagnostics[stock] = {
                "type": action,
                "severity": severity,
                "pred_dispersion": pred_dispersion,
                "confidence_dispersion": confidence_dispersion,
                "horizon_gap": horizon_gap,
                "market_complexity": market_state.complexity,
                "exposure_control": exposure_control,
            }
        return diagnostics
