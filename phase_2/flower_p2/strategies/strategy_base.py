"""Shared mobility-aware strategy base for Phase 2 selection policies.

Key differences from Phase 1:
    - Node ID ↔ shard ID mapping (Flower assigns arbitrary IDs)
    - Server-side dropout (dwell_remaining < T_k) instead of client-side
    - Simplified client config (no mobility snapshot fields)
    - Channel/delay formulas delegated to helpers.py free functions
"""

from __future__ import annotations

import math
import random
from abc import abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from logging import INFO

from flwr.app import ConfigRecord, Message, RecordDict
from flwr.common import ArrayRecord, MessageType, MetricRecord, log
from flwr.server import Grid
from flwr.serverapp.strategy import FedAvg
from flwr.serverapp.strategy.strategy_utils import (
    aggregate_arrayrecords,
    aggregate_metricrecords,
)

from ..helpers import (
    DelayParams,
    compute_time_s,
    delivery_time_bounds,
    delivery_time_s,
    round_delay_s,
)
from ..mobility_sumo import SUMOMobilityTrace


@dataclass
class RoundState:
    """Per-round mobility and selection metadata."""

    all_nodes: list[int]          # shard IDs
    selected_nodes: list[int]     # shard IDs
    eligible_nodes: list[int]     # shard IDs
    per_node_time_s: dict[int, float]
    round_delay_s: float
    compute_time_s: float
    local_epochs: int = 1
    dropout_nodes: set[int] = field(default_factory=set)  # shard IDs that will drop out


class MobilityAwareStrategyBase(FedAvg):
    """Common Flower strategy lifecycle for mobility-aware selection policies.

    Phase 2: uses SUMOMobilityTrace with server-side dropout.
    """

    strategy_name = "mobility-aware"

    def __init__(
        self,
        *args,
        mobility: SUMOMobilityTrace,
        delay_params: DelayParams,
        selection_size: int,
        alpha: float,
        model_size_bits: float,
        data_bits_per_client: float,
        seed: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.mobility = mobility
        self.delay_params = delay_params
        self.selection_size = selection_size
        self.alpha = alpha
        self.model_size_bits = model_size_bits
        self.data_bits_per_client = data_bits_per_client
        self._rng = random.Random(seed)
        self._round_state: RoundState | None = None

        # Geometry bounds from mobility (phase 2: circular coverage)
        self._dist_min: float = mobility.dist_min
        self._dist_max: float = mobility.dist_max

        # Node ID ↔ shard ID mapping — built on first configure_train call
        self._node_to_shard: dict[int, int] | None = None
        self._shard_to_node: dict[int, int] | None = None

    def _ensure_id_mapping(self, grid: Grid) -> None:
        """Build Flower node ID ↔ shard ID mapping on first call.

        Deterministic: sorted Flower IDs map to shard 0, 1, 2, ...
        """
        if self._node_to_shard is not None:
            return
        sorted_nodes = sorted(grid.get_node_ids())
        self._node_to_shard = {nid: i for i, nid in enumerate(sorted_nodes)}
        self._shard_to_node = {i: nid for i, nid in enumerate(sorted_nodes)}

    def _to_shard(self, flower_id: int) -> int:
        """Convert Flower node ID to shard ID."""
        return self._node_to_shard[flower_id]

    def _to_flower(self, shard_id: int) -> int:
        """Convert shard ID to Flower node ID."""
        return self._shard_to_node[shard_id]

    def _compute_round_state(
        self,
        all_shards: list[int],
        eligible_shards: list[int],
        selected_shards: list[int],
        local_epochs: int,
    ) -> RoundState:
        """Compute per-node delivery times, round delay, and dropout set."""
        if not selected_shards:
            return RoundState(all_shards, selected_shards, eligible_shards, {}, 0.0, 0.0)

        num_selected = len(selected_shards)
        comp_time = compute_time_s(
            self.data_bits_per_client, local_epochs, self.delay_params,
        )

        # Get 3D distances from mobility, then compute delivery times via helpers
        distances = {
            sid: self.mobility.distance_to_bs(sid) for sid in selected_shards
        }
        per_node_time, delay = round_delay_s(
            distances, self.model_size_bits, self.data_bits_per_client,
            num_selected, local_epochs, self.delay_params,
        )

        # Server-side dropout: vehicle drops if dwell_remaining < T_k
        dropout = set()
        for sid in selected_shards:
            t_k = per_node_time.get(sid, float("inf"))
            dwell = self.mobility.dwell_remaining(sid)
            if dwell < t_k:
                dropout.add(sid)

        return RoundState(
            all_shards,
            selected_shards,
            eligible_shards,
            per_node_time,
            delay,
            comp_time,
            local_epochs,
            dropout,
        )

    def _utility(
        self,
        success_ratio: float,
        round_delay: float,
        num_selected: int,
        local_epochs: int,
    ) -> float:
        """Compute paper utility: alpha * p^r - (1-alpha) * normalised_delay."""
        t_min, t_max = delivery_time_bounds(
            self._dist_min, self._dist_max,
            self.model_size_bits, self.data_bits_per_client,
            num_selected,
            local_epochs,
            self.delay_params,
        )
        normalised_delay = (round_delay - t_min) / max(t_max - t_min, 1e-9)
        normalised_delay = max(0.0, min(1.0, normalised_delay))
        return self.alpha * success_ratio - (1.0 - self.alpha) * normalised_delay

    @abstractmethod
    def select_nodes(
        self,
        *,
        eligible_nodes: list[int],
        local_epochs: int,
    ) -> list[int]:
        """Return the selected shard IDs for one round."""

    def selection_summary(self) -> str:
        """Short strategy-specific summary for logs."""
        return "mobility-aware selection"

    def on_round_end(
        self,
        *,
        server_round: int,
        state: RoundState,
        success_ratio: float,
        utility: float,
        responded_nodes: set[int] | None = None,
    ) -> None:
        """Hook for strategies that update history after a realized round outcome."""
        del server_round, state, success_ratio, utility, responded_nodes

    def configure_train(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
    ) -> list[Message]:
        self._ensure_id_mapping(grid)

        all_flower_ids = list(grid.get_node_ids())
        if len(all_flower_ids) < self.min_available_nodes:
            log(INFO, "configure_train [%s]: waiting for min available nodes", self.strategy_name)
            return []

        all_shards = [self._to_shard(nid) for nid in all_flower_ids]
        eligible_shards = self.mobility.eligible_nodes(all_shards)
        if not eligible_shards:
            wait_s = 15.0
            self._round_state = RoundState(all_shards, [], [], {}, wait_s, 0.0)
            log(
                INFO,
                "configure_train [%s]: no eligible nodes in coverage (pool: %s); "
                "advancing %.0fs then skipping round %d",
                self.strategy_name,
                self.mobility.pool_summary(),
                wait_s,
                server_round,
            )
            return []

        local_epochs = int(config["local-epochs"])
        selected_shards = self.select_nodes(
            eligible_nodes=eligible_shards,
            local_epochs=local_epochs,
        )
        config["server-round"] = server_round
        config["selected-count"] = len(selected_shards)
        config["eligible-count"] = len(eligible_shards)

        self._round_state = self._compute_round_state(
            all_shards, eligible_shards, selected_shards, local_epochs,
        )

        log(
            INFO,
            "configure_train [%s]: selected %d/%d eligible (%d total, pool: %s, dropout: %d)",
            self.strategy_name,
            len(selected_shards),
            len(eligible_shards),
            len(all_flower_ids),
            self.mobility.pool_summary(),
            len(self._round_state.dropout_nodes),
        )

        # Build per-client messages — simplified config (no mobility snapshot)
        messages: list[Message] = []
        for shard_id in selected_shards:
            node_config = ConfigRecord(dict(config))
            node_config["shard-id"] = shard_id
            record = RecordDict(
                {self.arrayrecord_key: arrays, self.configrecord_key: node_config}
            )
            flower_id = self._to_flower(shard_id)
            messages.extend(
                self._construct_messages(record, [flower_id], MessageType.TRAIN)
            )
        return messages

    def configure_evaluate(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
    ) -> list[Message]:
        self._ensure_id_mapping(grid)

        all_flower_ids = list(grid.get_node_ids())
        all_shards = [self._to_shard(nid) for nid in all_flower_ids]
        eligible_shards = self.mobility.eligible_nodes(all_shards)
        if not eligible_shards:
            return []

        log(
            INFO,
            "configure_evaluate [%s]: selected %d eligible nodes (%d total, pool: %s)",
            self.strategy_name,
            len(eligible_shards),
            len(all_flower_ids),
            self.mobility.pool_summary(),
        )

        # Per-node messages with shard-id (mirrors configure_train)
        messages: list[Message] = []
        for shard_id in eligible_shards:
            node_config = ConfigRecord(dict(config))
            node_config["shard-id"] = shard_id
            record = RecordDict(
                {self.arrayrecord_key: arrays, self.configrecord_key: node_config}
            )
            flower_id = self._to_flower(shard_id)
            messages.extend(
                self._construct_messages(record, [flower_id], MessageType.EVALUATE)
            )
        return messages

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        replies_list = list(replies)
        valid_replies, _ = self._check_and_log_replies(replies_list, is_train=True)
        state = self._round_state
        if state is None:
            return None, None

        # Server-side dropout filtering: exclude replies from dropout nodes
        if state.dropout_nodes and self._node_to_shard is not None:
            valid_replies = [
                msg for msg in valid_replies
                if self._to_shard(msg.metadata.src_node_id) not in state.dropout_nodes
            ]

        responded_shards = set()
        if self._node_to_shard is not None:
            responded_shards = {
                self._to_shard(msg.metadata.src_node_id)
                for msg in valid_replies
            }

        success_ratio = len(valid_replies) / max(len(state.selected_nodes), 1)
        utility = self._utility(
            success_ratio,
            state.round_delay_s,
            len(state.selected_nodes),
            state.local_epochs,
        )
        self.on_round_end(
            server_round=server_round,
            state=state,
            success_ratio=success_ratio,
            utility=utility,
            responded_nodes=responded_shards,
        )

        arrays: ArrayRecord | None = None
        metrics: MetricRecord | None = MetricRecord(
            {
                "mavfl-round-delay-s": state.round_delay_s,
                "mavfl-success-ratio": success_ratio,
                "mavfl-utility": utility,
                "mavfl-round": float(server_round),
                "mavfl-compute-time-s": state.compute_time_s,
                "mavfl-dropout-count": float(len(state.dropout_nodes)),
            }
        )

        if valid_replies:
            reply_contents = [msg.content for msg in valid_replies]
            for rc in reply_contents:
                for mr in rc.metric_records.values():
                    mr[self.weighted_by_key] = 1.0
            arrays = aggregate_arrayrecords(reply_contents, self.weighted_by_key)
            aggregated_metrics = aggregate_metricrecords(reply_contents, self.weighted_by_key)
            for key, value in aggregated_metrics.items():
                metrics[key] = value

        if state.round_delay_s > 0.0:
            self.mobility.advance(state.round_delay_s, state.all_nodes)

        return arrays, metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> MetricRecord | None:
        replies_list = list(replies)
        valid_replies, _ = self._check_and_log_replies(replies_list, is_train=False)
        if not valid_replies:
            return None
        reply_contents = [msg.content for msg in valid_replies]
        return aggregate_metricrecords(reply_contents, self.weighted_by_key)

    def summary(self) -> None:
        super().summary()
        log(INFO, "\t├──> %s:", self.__class__.__name__)
        log(INFO, "\t│\t├── selection_size: %d", self.selection_size)
        log(INFO, "\t│\t├── alpha (utility weight): %.2f", self.alpha)
        log(INFO, "\t│\t└── %s", self.selection_summary())
