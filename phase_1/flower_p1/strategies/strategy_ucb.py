"""Paper-grounded discounted UCB vehicle-selection strategy for MAVFL."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..helpers import uplink_rate_bps
from .strategy_base import MobilityAwareStrategyBase, RoundState


@dataclass
class UCBAuditRecord:
    """Round-level audit snapshot for UCB selection behavior."""

    server_round: int
    eligible_nodes: list[int]
    selected_nodes: list[int]
    scores: dict[int, float]
    success_ratio: float | None = None
    utility: float | None = None
    round_delay_s: float | None = None
    mu0: float | None = None


class UCBStrategy(MobilityAwareStrategyBase):
    """Discounted UCB with random cold-start"""

    strategy_name = "ucb"

    def __init__(
        self,
        *args,
        ucb_discount: float,
        ucb_exploration: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not 0.0 <= ucb_discount <= 1.0:
            raise ValueError("ucb_discount must be between 0 and 1")

        self.ucb_discount = ucb_discount
        self.ucb_exploration = ucb_exploration
        self._discounted_counts: dict[int, float] = {}
        self._discounted_rewards: dict[int, float] = {}
        self._discounted_total_count = 0.0
        self._has_completed_round = False
        self._current_server_round = 0
        self._last_audit_record: UCBAuditRecord | None = None
        self.audit_trail: list[UCBAuditRecord] = []

    def _avg_utility(self, node_id: int) -> float:
        count = self._discounted_counts[node_id]
        return self._discounted_rewards[node_id] / count

    def _seen_nodes(self) -> list[int]:
        return [node_id for node_id, count in self._discounted_counts.items() if count > 0.0]

    def _bootstrap_mu0(self) -> float | None:
        seen_nodes = self._seen_nodes()
        if not seen_nodes:
            return None
        return max(self._avg_utility(node_id) for node_id in seen_nodes)

    def _seen_ucb_score(self, node_id: int) -> float:
        count = self._discounted_counts[node_id]
        avg_reward = self._avg_utility(node_id)
        total = max(self._discounted_total_count, 1.0)
        explore = self.ucb_exploration * math.sqrt(2.0 * math.log(total) / count)
        return avg_reward + explore

    def _bootstrap_score(self, mu0: float) -> float:
        total = max(self._discounted_total_count, 1.0)
        return mu0 + self.ucb_exploration * math.sqrt(2.0 * math.log(total) / 1.0)

    def _per_node_utility(
        self, node_id: int, state: RoundState, num_selected: int,
    ) -> float:
        """Compute per-node utility: success=1 weighted by individual delay."""
        node_time = state.per_node_time_s.get(node_id, state.round_delay_s)
        bw_per_node = self.delay_params.bandwidth_hz / max(num_selected, 1)
        tmin = (
            self.model_size_bits
            / max(
                uplink_rate_bps(
                    self._dist_min, bw_per_node,
                    self.delay_params.tx_power_dbm,
                    self.delay_params.noise_power_dbm,
                    self.delay_params.bs_antenna_gain_db,
                    self.delay_params.path_loss_a,
                    self.delay_params.path_loss_b,
                ), 1e-9,
            )
            + state.compute_time_s
        )
        tmax = (
            self.model_size_bits
            / max(
                uplink_rate_bps(
                    self._dist_max, bw_per_node,
                    self.delay_params.tx_power_dbm,
                    self.delay_params.noise_power_dbm,
                    self.delay_params.bs_antenna_gain_db,
                    self.delay_params.path_loss_a,
                    self.delay_params.path_loss_b,
                ), 1e-9,
            )
            + state.compute_time_s
        )
        norm_delay = (node_time - tmin) / max(tmax - tmin, 1e-9)
        norm_delay = max(0.0, min(1.0, norm_delay))
        return self.alpha * 1.0 - (1.0 - self.alpha) * norm_delay

    def select_nodes(
        self,
        *,
        eligible_nodes: list[int],
        local_epochs: int,
    ) -> list[int]:
        del local_epochs
        sample_size = min(self.selection_size, len(eligible_nodes))
        if sample_size == 0:
            self._last_audit_record = UCBAuditRecord(
                server_round=self._current_server_round,
                eligible_nodes=[],
                selected_nodes=[],
                scores={},
            )
            return []

        if not self._has_completed_round:
            selected = self._rng.sample(eligible_nodes, k=sample_size)
            self._last_audit_record = UCBAuditRecord(
                server_round=self._current_server_round,
                eligible_nodes=list(eligible_nodes),
                selected_nodes=list(selected),
                scores={},
            )
            return selected

        seen_eligible = [node_id for node_id in eligible_nodes if node_id in self._discounted_counts]
        unseen_eligible = [node_id for node_id in eligible_nodes if node_id not in self._discounted_counts]

        if seen_eligible and unseen_eligible:
            mu0 = self._bootstrap_mu0()
            if mu0 is None:
                selected = self._rng.sample(eligible_nodes, k=sample_size)
                self._last_audit_record = UCBAuditRecord(
                    server_round=self._current_server_round,
                    eligible_nodes=list(eligible_nodes),
                    selected_nodes=list(selected),
                    scores={},
                )
                return selected
            scores = {node_id: self._seen_ucb_score(node_id) for node_id in seen_eligible}
            bootstrap_score = self._bootstrap_score(mu0)
            for node_id in unseen_eligible:
                scores[node_id] = bootstrap_score
        elif seen_eligible:
            mu0 = self._bootstrap_mu0()
            scores = {node_id: self._seen_ucb_score(node_id) for node_id in seen_eligible}
        else:
            selected = self._rng.sample(eligible_nodes, k=sample_size)
            self._last_audit_record = UCBAuditRecord(
                server_round=self._current_server_round,
                eligible_nodes=list(eligible_nodes),
                selected_nodes=list(selected),
                scores={},
            )
            return selected

        shuffled_nodes = list(eligible_nodes)
        self._rng.shuffle(shuffled_nodes)
        ranked = sorted(shuffled_nodes, key=lambda node_id: scores[node_id], reverse=True)
        selected = ranked[:sample_size]
        self._last_audit_record = UCBAuditRecord(
            server_round=self._current_server_round,
            eligible_nodes=list(eligible_nodes),
            selected_nodes=list(selected),
            scores=dict(scores),
            mu0=mu0,
        )
        return selected

    def configure_train(self, server_round, arrays, config, grid):
        self._current_server_round = server_round
        return super().configure_train(server_round, arrays, config, grid)

    def on_round_end(
        self,
        *,
        server_round: int,
        state: RoundState,
        success_ratio: float,
        utility: float,
        responded_nodes: set[int] | None = None,
    ) -> None:
        self._discounted_total_count *= self.ucb_discount
        for node_id in list(self._discounted_counts):
            self._discounted_counts[node_id] *= self.ucb_discount
            self._discounted_rewards[node_id] *= self.ucb_discount

        if responded_nodes is None:
            responded_nodes = set()

        for node_id in state.selected_nodes:
            if node_id in responded_nodes:
                node_reward = self._per_node_utility(
                    node_id, state, len(state.selected_nodes),
                )
            else:
                node_reward = -1.0
            self._discounted_counts[node_id] = self._discounted_counts.get(node_id, 0.0) + 1.0
            self._discounted_rewards[node_id] = self._discounted_rewards.get(node_id, 0.0) + node_reward

        self._discounted_total_count += len(state.selected_nodes)
        self._discounted_total_count = max(self._discounted_total_count, 1.0)
        self._has_completed_round = True

        audit = self._last_audit_record
        if audit is None:
            audit = UCBAuditRecord(
                server_round=server_round,
                eligible_nodes=list(state.eligible_nodes),
                selected_nodes=list(state.selected_nodes),
                scores={},
            )
        audit.success_ratio = success_ratio
        audit.utility = utility
        audit.round_delay_s = state.round_delay_s
        self.audit_trail.append(audit)
        self._last_audit_record = None

    def selection_summary(self) -> str:
        return (
            "paper-grounded discounted UCB with random cold start, "
            f"lambda={self.ucb_discount:.2f}"
        )
