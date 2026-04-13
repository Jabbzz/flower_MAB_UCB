"""Communication-based selection baseline from the MAVFL paper.

Phase 2 change: distance_to_bs(node_id) instead of distance_to_bs(position_m).
"""

from __future__ import annotations

from .strategy_base import MobilityAwareStrategyBase


class CBSStrategy(MobilityAwareStrategyBase):
    """Select vehicles nearest to the base station each round."""

    strategy_name = "cbs"

    def select_nodes(
        self,
        *,
        eligible_nodes: list[int],
        local_epochs: int,
    ) -> list[int]:
        del local_epochs
        # Phase 2: mobility.distance_to_bs takes shard_id directly
        ranked = sorted(
            eligible_nodes,
            key=lambda sid: (
                self.mobility.distance_to_bs(sid),
                sid,
            ),
        )
        pool = ranked[: self.selection_size * 2]
        sample_size = min(self.selection_size, len(pool))
        return self._rng.sample(pool, k=sample_size)

    def selection_summary(self) -> str:
        return "nearest-to-BS ranking over all eligible nodes"
