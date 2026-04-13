"""Shared mobility-aware strategy base for Phase 1 selection policies."""

from __future__ import annotations

import math
import random
from abc import abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from logging import INFO

from flwr.app import ConfigRecord, Message, RecordDict
from flwr.common import ArrayRecord, MessageType, MetricRecord, log
from flwr.server import Grid
from flwr.serverapp.strategy import FedAvg
from flwr.serverapp.strategy.strategy_utils import (
    aggregate_arrayrecords,
    aggregate_metricrecords,
)

from ..helpers import DelayParams, uplink_rate_bps
from ..mobility import IDMRoadMobility


@dataclass
class RoundState:
    """Per-round mobility and selection metadata."""

    all_nodes: list[int]
    selected_nodes: list[int]
    eligible_nodes: list[int]
    per_node_time_s: dict[int, float]
    round_delay_s: float
    compute_time_s: float


class MobilityAwareStrategyBase(FedAvg):
    """Common Flower strategy lifecycle for mobility-aware selection policies."""

    strategy_name = "mobility-aware"

    def __init__(
        self,
        *args,
        mobility: IDMRoadMobility,
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
        self._dist_min: float = mobility.distance_to_bs(mobility.road_length_m / 2.0)
        self._dist_max: float = mobility.distance_to_bs(0.0)

    def _compute_round_state(
        self,
        all_nodes: list[int],
        eligible_nodes: list[int],
        selected_nodes: list[int],
        local_epochs: int,
    ) -> RoundState:
        """Compute per-node delivery times and round timing metadata."""
        if not selected_nodes:
            return RoundState(all_nodes, selected_nodes, eligible_nodes, {}, 0.0, 0.0)

        num_selected = len(selected_nodes)
        bandwidth_per_node = self.delay_params.bandwidth_hz / max(num_selected, 1)
        positions = self.mobility.positions(selected_nodes)
        per_node_time: dict[int, float] = {}
        compute_time_s = self.mobility.compute_time_s(self.data_bits_per_client, local_epochs)

        for node_id in selected_nodes:
            pos = positions.get(node_id)
            if pos is None:
                per_node_time[node_id] = float("inf")
                continue

            distance_m = self.mobility.distance_to_bs(pos)
            uplink_rate = uplink_rate_bps(
                distance_m=distance_m,
                bandwidth_hz=bandwidth_per_node,
                tx_power_dbm=self.delay_params.tx_power_dbm,
                noise_power_dbm=self.delay_params.noise_power_dbm,
                bs_antenna_gain_db=self.delay_params.bs_antenna_gain_db,
                path_loss_a=self.delay_params.path_loss_a,
                path_loss_b=self.delay_params.path_loss_b,
            )
            comm_time_s = self.model_size_bits / max(uplink_rate, 1e-9)
            per_node_time[node_id] = comm_time_s + compute_time_s

        finite_times = [t for t in per_node_time.values() if math.isfinite(t)]
        round_delay = max(finite_times) if finite_times else 0.0
        return RoundState(
            all_nodes,
            selected_nodes,
            eligible_nodes,
            per_node_time,
            round_delay,
            compute_time_s,
        )

    def _utility(
        self,
        success_ratio: float,
        round_delay_s: float,
        num_selected: int,
        compute_time_s: float,
    ) -> float:
        """Compute paper utility alpha*p^r - (1-alpha)*normalised_delay."""
        bw_per_node = self.delay_params.bandwidth_hz / max(num_selected, 1)
        tmin = (
            self.model_size_bits
            / max(
                uplink_rate_bps(
                    self._dist_min,
                    bw_per_node,
                    self.delay_params.tx_power_dbm,
                    self.delay_params.noise_power_dbm,
                    self.delay_params.bs_antenna_gain_db,
                    self.delay_params.path_loss_a,
                    self.delay_params.path_loss_b,
                ),
                1e-9,
            )
            + compute_time_s
        )
        tmax = (
            self.model_size_bits
            / max(
                uplink_rate_bps(
                    self._dist_max,
                    bw_per_node,
                    self.delay_params.tx_power_dbm,
                    self.delay_params.noise_power_dbm,
                    self.delay_params.bs_antenna_gain_db,
                    self.delay_params.path_loss_a,
                    self.delay_params.path_loss_b,
                ),
                1e-9,
            )
            + compute_time_s
        )
        normalised_delay = (round_delay_s - tmin) / max(tmax - tmin, 1e-9)
        normalised_delay = max(0.0, min(1.0, normalised_delay))
        return self.alpha * success_ratio - (1.0 - self.alpha) * normalised_delay

    @abstractmethod
    def select_nodes(
        self,
        *,
        eligible_nodes: list[int],
        local_epochs: int,
    ) -> list[int]:
        """Return the selected node IDs for one round."""

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
        all_nodes = list(grid.get_node_ids())
        if len(all_nodes) < self.min_available_nodes:
            log(INFO, "configure_train [%s]: waiting for min available nodes", self.strategy_name)
            return []

        eligible_nodes = self.mobility.eligible_nodes(all_nodes)
        if not eligible_nodes:
            wait_s = 15.0
            self._round_state = RoundState(all_nodes, [], [], {}, wait_s, 0.0)
            log(
                INFO,
                "configure_train [%s]: no eligible nodes on road (pool: %s); "
                "advancing %.0fs then skipping round %d",
                self.strategy_name,
                self.mobility.pool_summary(),
                wait_s,
                server_round,
            )
            return []

        local_epochs = int(config["local-epochs"])
        selected_nodes = self.select_nodes(
            eligible_nodes=eligible_nodes,
            local_epochs=local_epochs,
        )
        config["server-round"] = server_round
        config["selected-count"] = len(selected_nodes)
        config["eligible-count"] = len(eligible_nodes)

        self._round_state = self._compute_round_state(
            all_nodes,
            eligible_nodes,
            selected_nodes,
            local_epochs,
        )

        log(
            INFO,
            "configure_train [%s]: selected %d/%d eligible (%d total, pool: %s)",
            self.strategy_name,
            len(selected_nodes),
            len(eligible_nodes),
            len(all_nodes),
            self.mobility.pool_summary(),
        )

        messages: list[Message] = []
        for node_id in selected_nodes:
            node_config = ConfigRecord(dict(config))
            snapshot = self.mobility.snapshot_for_node(node_id, all_nodes)
            if snapshot is not None:
                node_config["mobility-position-m"] = snapshot.position_m
                node_config["mobility-speed-mps"] = snapshot.speed_mps
                node_config["mobility-leader-gap-m"] = snapshot.leader_gap_m
                node_config["mobility-leader-speed-mps"] = snapshot.leader_speed_mps
            node_config["mobility-road-length-m"] = self.mobility.road_length_m
            node_config["mobility-num-zones"] = self.mobility.num_zones
            node_config["mobility-bs-height-m"] = self.mobility.bs_height_m
            node_config["mobility-desired-speed-kmh"] = self.mobility.desired_speed_mps * 3.6
            node_config["mobility-time-step-s"] = self.mobility.time_step_s
            node_config["mobility-idm-max-accel-mps2"] = self.mobility.max_accel_mps2
            node_config["mobility-idm-comfort-decel-mps2"] = self.mobility.comfort_decel_mps2
            node_config["mobility-idm-min-gap-m"] = self.mobility.min_gap_m
            node_config["mobility-idm-time-headway-s"] = self.mobility.time_headway_s
            node_config["mobility-idm-accel-exponent"] = self.mobility.accel_exponent
            node_config["mobility-compute-time-s"] = self._round_state.compute_time_s
            record = RecordDict(
                {self.arrayrecord_key: arrays, self.configrecord_key: node_config}
            )
            messages.extend(
                self._construct_messages(record, [node_id], MessageType.TRAIN)
            )
        return messages

    def configure_evaluate(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
    ) -> list[Message]:
        all_nodes = list(grid.get_node_ids())
        eligible_nodes = self.mobility.eligible_nodes(all_nodes)
        if not eligible_nodes:
            return []
        record = RecordDict({self.arrayrecord_key: arrays, self.configrecord_key: config})
        log(
            INFO,
            "configure_evaluate [%s]: selected %d eligible nodes (%d total, pool: %s)",
            self.strategy_name,
            len(eligible_nodes),
            len(all_nodes),
            self.mobility.pool_summary(),
        )
        return list(self._construct_messages(record, eligible_nodes, MessageType.EVALUATE))

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

        responded_nodes = {msg.metadata.src_node_id for msg in valid_replies}
        success_ratio = len(valid_replies) / max(len(state.selected_nodes), 1)
        utility = self._utility(
            success_ratio,
            state.round_delay_s,
            len(state.selected_nodes),
            state.compute_time_s,
        )
        self.on_round_end(
            server_round=server_round,
            state=state,
            success_ratio=success_ratio,
            utility=utility,
            responded_nodes=responded_nodes,
        )
        arrays: ArrayRecord | None = None
        metrics: MetricRecord | None = MetricRecord(
            {
                "mavfl-round-delay-s": state.round_delay_s,
                "mavfl-success-ratio": success_ratio,
                "mavfl-utility": utility,
                "mavfl-round": float(server_round),
                "mavfl-compute-time-s": state.compute_time_s,
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
