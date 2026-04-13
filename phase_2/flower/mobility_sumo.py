"""SUMO trace-based mobility model for Phase 2 vehicular FL.

Replaces Phase 1's IDMRoadMobility with a pre-generated CSV trace replay.
The trace (produced by ``generate_trace.py``) contains per-second snapshots
of vehicles inside a circular coverage zone, with exact ``dwell_remaining_s``
computed via post-processing.

Key differences from Phase 1:
    - 2D urban geometry (circular coverage) instead of 1D road segment
    - Ground-truth dwell from trace instead of IDM estimation
    - No client-side rollout; dropout is server-side
    - ``distance_to_bs(node_id)`` takes a node_id, not a position
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from typing import NamedTuple


class TraceExhaustedError(Exception):
    """Raised when the simulation clock exceeds the trace duration."""


class VehicleRecord(NamedTuple):
    """Per-vehicle state at one timestep, read from trace CSV."""

    x_m: float
    y_m: float
    speed_mps: float
    distance_to_bs_m: float
    dwell_remaining_s: int


class SUMOMobilityTrace:
    """Replay-based mobility from a pre-generated SUMO trace.

    Exposes an interface compatible with Phase 1's ``IDMRoadMobility`` so
    strategies can call the same method names.  Geometry is 2D (circular
    coverage zone) with a base station at the network centre.

    Parameters
    ----------
    trace_csv : str
        Path to ``dublin_morning.csv``.
    vehicle_map_json : str
        Path to ``vehicle_id_map.json`` (metadata).
    coverage_radius_m : float
        Radius of the circular coverage zone in metres.
    bs_height_m : float
        Base-station antenna height in metres.
    num_zones : int
        Number of concentric ring zones for reporting.
    """

    def __init__(
        self,
        *,
        trace_csv: str,
        vehicle_map_json: str,
        coverage_radius_m: float,
        bs_height_m: float,
        num_zones: int,
    ) -> None:
        self.coverage_radius_m = coverage_radius_m
        self.bs_height_m = bs_height_m
        self.num_zones = num_zones

        # Load metadata
        with open(vehicle_map_json) as f:
            meta = json.load(f)
        self.num_shards: int = meta["num_shards"]
        self._bs_position: tuple[float, float] = tuple(meta["bs_position"])

        # Load trace: _trace[time_s][shard_id] -> VehicleRecord
        self._trace: dict[int, dict[int, VehicleRecord]] = {}
        self._all_times: list[int] = []
        self._load_trace(trace_csv)

        # Start at the first timestep that has vehicles in coverage
        self._current_time_s: int = self._all_times[0] if self._all_times else 0
        self._max_time_s: int = self._all_times[-1] if self._all_times else 0

        # Geometry bounds for delay normalisation
        # dist_min: vehicle directly under BS (horizontal=0)
        # dist_max: vehicle at coverage edge
        self._dist_min: float = bs_height_m
        self._dist_max: float = math.sqrt(coverage_radius_m**2 + bs_height_m**2)

    def _load_trace(self, trace_csv: str) -> None:
        """Parse the trace CSV into the internal lookup dict."""
        times_set: set[int] = set()
        with open(trace_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                t = int(row["time_s"])
                sid = int(row["shard_id"])
                rec = VehicleRecord(
                    x_m=float(row["x_m"]),
                    y_m=float(row["y_m"]),
                    speed_mps=float(row["speed_mps"]),
                    distance_to_bs_m=float(row["distance_to_bs_m"]),
                    dwell_remaining_s=int(row["dwell_remaining_s"]),
                )
                if t not in self._trace:
                    self._trace[t] = {}
                self._trace[t][sid] = rec
                times_set.add(t)
        self._all_times = sorted(times_set)

    # ------------------------------------------------------------------
    # Public API — mirrors IDMRoadMobility interface
    # ------------------------------------------------------------------

    @property
    def current_time_s(self) -> int:
        """Current simulation clock (read-only)."""
        return self._current_time_s

    def advance(self, duration_s: float, node_ids: list[int]) -> None:
        """Advance simulation clock by duration_s (rounded to nearest second).

        Parameters
        ----------
        duration_s : float
            Time to advance (typically the synchronous round delay).
        node_ids : list[int]
            All node IDs (unused — kept for interface compatibility).

        Raises
        ------
        TraceExhaustedError
            If the clock would exceed the trace duration.
        """
        del node_ids  # unused; trace is pre-generated
        self._current_time_s += round(duration_s)
        if self._current_time_s > self._max_time_s:
            raise TraceExhaustedError(
                f"Trace exhausted at {self._current_time_s}s "
                f"(max={self._max_time_s}s). "
                f"Reduce num-server-rounds or use a longer trace."
            )

    def eligible_nodes(self, node_ids: list[int]) -> list[int]:
        """Return shard IDs present in coverage at the current timestep.

        Parameters
        ----------
        node_ids : list[int]
            All known node IDs (Flower node IDs, 0..num_shards-1).
            Used to filter: only return shard IDs that are in node_ids.

        Returns
        -------
        list[int]
            Shard IDs of vehicles currently in the coverage zone.
        """
        snapshot = self._trace.get(self._current_time_s, {})
        node_set = set(node_ids)
        return [sid for sid in snapshot if sid in node_set]

    def distance_to_bs(self, node_id: int) -> float:
        """3D distance from vehicle to base station (for channel model).

        Phase 2 change: takes node_id (shard_id), not position.
        Looks up 2D position from trace, returns sqrt(d_2d^2 + h^2).

        Returns
        -------
        float
            3D Euclidean distance in metres.

        Raises
        ------
        KeyError
            If node_id is not in coverage at the current timestep.
        """
        rec = self._trace[self._current_time_s][node_id]
        return math.sqrt(rec.distance_to_bs_m**2 + self.bs_height_m**2)

    def dwell_remaining(self, node_id: int) -> int:
        """Exact remaining time (seconds) before vehicle exits coverage.

        Ground truth from trace — not an estimate.

        Returns
        -------
        int
            Seconds remaining in coverage. 0 means the vehicle leaves
            at the next timestep.
        """
        return self._trace[self._current_time_s][node_id].dwell_remaining_s

    def zone_index(self, node_id: int) -> int:
        """Concentric ring zone index for a vehicle (for reporting).

        Zones are equal-width rings from BS outward.
        """
        rec = self._trace[self._current_time_s][node_id]
        ring_width = self.coverage_radius_m / self.num_zones
        return min(self.num_zones - 1, int(rec.distance_to_bs_m / ring_width))

    def positions(self, node_ids: list[int]) -> dict[int, tuple[float, float]]:
        """Return (x, y) positions for given node_ids at current time.

        Nodes not in coverage are silently omitted.
        """
        snapshot = self._trace.get(self._current_time_s, {})
        result: dict[int, tuple[float, float]] = {}
        for nid in node_ids:
            rec = snapshot.get(nid)
            if rec is not None:
                result[nid] = (rec.x_m, rec.y_m)
        return result

    def speeds(self, node_ids: list[int]) -> dict[int, float]:
        """Return speeds for given node_ids at current time."""
        snapshot = self._trace.get(self._current_time_s, {})
        return {
            nid: snapshot[nid].speed_mps
            for nid in node_ids
            if nid in snapshot
        }

    def pool_summary(self) -> dict[str, int]:
        """Summary of current trace state (for logging compatibility)."""
        snapshot = self._trace.get(self._current_time_s, {})
        return {
            "in_coverage": len(snapshot),
            "time_s": self._current_time_s,
            "trace_remaining_s": self._max_time_s - self._current_time_s,
        }

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    @property
    def dist_min(self) -> float:
        """Minimum 3D distance to BS (vehicle directly under BS)."""
        return self._dist_min

    @property
    def dist_max(self) -> float:
        """Maximum 3D distance to BS (vehicle at coverage edge)."""
        return self._dist_max
