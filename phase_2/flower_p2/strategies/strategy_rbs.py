"""Remain-time based selection baseline from the MAVFL paper.

Phase 2 change: dwell_remaining(node_id) instead of time_to_exit(pos, speed).
"""

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
        # Phase 2: ground-truth dwell from trace
        ranked = sorted(
            eligible_nodes,
            key=lambda sid: (
                -self.mobility.dwell_remaining(sid),
                sid,
            ),
        )
        pool = ranked[: self.selection_size * 2]
        sample_size = min(self.selection_size, len(pool))
        return self._rng.sample(pool, k=sample_size)

    def selection_summary(self) -> str:
        return "longest-remaining-time ranking over all eligible nodes"
