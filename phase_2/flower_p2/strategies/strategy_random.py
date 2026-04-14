"""Random selection strategy (paper null hypothesis).


This baseline samples uniformly from all vehicles currently in coverage.
It intentionally uses no position, speed, channel, or history signal.
"""

from __future__ import annotations

from .strategy_base import MobilityAwareStrategyBase


class RandomStrategy(MobilityAwareStrategyBase):
    """Uniform random selection over all eligible in-coverage vehicles."""

    strategy_name = "random"

    def select_nodes(
        self,
        *,
        eligible_nodes: list[int],
        local_epochs: int,
    ) -> list[int]:
        del local_epochs
        sample_size = min(self.selection_size, len(eligible_nodes))
        return self._rng.sample(eligible_nodes, k=sample_size)

    def selection_summary(self) -> str:
        return "uniform random sampling over all eligible nodes"
