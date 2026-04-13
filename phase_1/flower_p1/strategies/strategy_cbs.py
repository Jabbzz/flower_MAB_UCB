"""Communication-based selection baseline from the MAVFL paper."""

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
        positions = self.mobility.positions(eligible_nodes)
        ranked = sorted(
            eligible_nodes,
            key=lambda node_id: (
                self.mobility.distance_to_bs(positions[node_id]),
                positions[node_id],
                node_id,
            ),
        )
        pool = ranked[: self.selection_size * 2]
        sample_size = min(self.selection_size, len(pool))
        return self._rng.sample(pool, k=sample_size)

    def selection_summary(self) -> str:
        return "nearest-to-BS ranking over all eligible nodes"
