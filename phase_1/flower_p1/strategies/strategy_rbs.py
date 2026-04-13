"""Remain-time based selection baseline from the MAVFL paper."""

from __future__ import annotations

from .strategy_base import MobilityAwareStrategyBase


class RBSStrategy(MobilityAwareStrategyBase):
    """Select vehicles with the longest remaining time in coverage."""

    strategy_name = "rbs"

    def select_nodes(
        self,
        *,
        eligible_nodes: list[int],
        local_epochs: int,
    ) -> list[int]:
        del local_epochs
        positions = self.mobility.positions(eligible_nodes)
        speeds = self.mobility.speeds(eligible_nodes)
        ranked = sorted(
            eligible_nodes,
            key=lambda node_id: (
                -self.mobility.time_to_exit(positions[node_id], speeds[node_id]),
                positions[node_id],
                node_id,
            ),
        )
        pool = ranked[: self.selection_size * 2]
        sample_size = min(self.selection_size, len(pool))
        return self._rng.sample(pool, k=sample_size)

    def selection_summary(self) -> str:
        return "longest-remaining-time ranking over all eligible nodes"
