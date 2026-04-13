"""Paper-grounded discounted UCB vehicle-selection strategy for MAVFL.

Phase 2 change: UCB reward uses round-level utility (alpha * p^r - (1-alpha) * T~)
instead of per-node utility.  The same reward value is assigned to every successful
node in the round.  Dropout nodes receive -1.0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

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
    """Discounted UCB with random cold-start and finite optimistic bootstrap."""

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

        seen_eligible = [sid for sid in eligible_nodes if sid in self._discounted_counts]
        unseen_eligible = [sid for sid in eligible_nodes if sid not in self._discounted_counts]

        mu0 = None
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
            scores = {sid: self._seen_ucb_score(sid) for sid in seen_eligible}
            bootstrap_score = self._bootstrap_score(mu0)
            for sid in unseen_eligible:
                scores[sid] = bootstrap_score
        elif seen_eligible:
            mu0 = self._bootstrap_mu0()
            scores = {sid: self._seen_ucb_score(sid) for sid in seen_eligible}
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
        ranked = sorted(shuffled_nodes, key=lambda sid: scores[sid], reverse=True)
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
        # Decay all existing counts and rewards
        self._discounted_total_count *= self.ucb_discount
        for node_id in list(self._discounted_counts):
            self._discounted_counts[node_id] *= self.ucb_discount
            self._discounted_rewards[node_id] *= self.ucb_discount

        if responded_nodes is None:
            responded_nodes = set()

        # Phase 2 change: round-level utility for all successful nodes,
        # -1.0 for dropout nodes.  Same reward for every successful node.
        for node_id in state.selected_nodes:
            if node_id in responded_nodes:
                node_reward = utility  # round-level utility, not per-node
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
