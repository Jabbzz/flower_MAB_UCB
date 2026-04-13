"""Shared helpers for phase2_flower.

Channel model, physical-layer utilities, and delay functions.

Extends phase 1's helpers with standalone channel/delay functions extracted
from IDMRoadMobility.  In phase 2 the mobility class owns geometry (where is
the vehicle?) and these helpers own RF (given a distance, what's the rate/time?).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class DelayParams:
    """Physical-layer channel and computation parameters (paper §4-5).

    Groups constants that define the wireless channel between vehicles and
    the base station, and the on-vehicle computation model.
    """

    bandwidth_hz: float          # B -- total uplink bandwidth (paper: 3 MHz)
    tx_power_dbm: float          # P_k -- vehicle transmission power
    noise_power_dbm: float       # N_0 -- noise power (paper: -114 dBm)
    bs_antenna_gain_db: float    # G_v -- BS antenna gain (paper: 6 dBi)
    path_loss_a: float           # intercept in l_p = A + B*log10(d_km) (paper: 128.1)
    path_loss_b: float           # slope (paper: 37.6)
    gpu_frequency_ghz: float     # f_k -- GPU frequency (paper: 1.3 GHz)
    gpu_cycles_per_bit: float    # g_k -- GPU cycles per bit of data
    compute_normalization: float  # c_k -- normalization factor


def db_to_linear(db_val: float) -> float:
    """Convert a dB value to linear scale."""
    return 10.0 ** (db_val / 10.0)


def uplink_rate_bps(
    distance_m: float,
    bandwidth_hz: float,
    tx_power_dbm: float,
    noise_power_dbm: float,
    bs_antenna_gain_db: float,
    path_loss_a: float,
    path_loss_b: float,
) -> float:
    """Shannon uplink capacity for a single vehicle-BS link (bits/s).

    Implements the paper's channel model (section 4):
        path_loss = A + B*log10(d_km)
        q_k = B_k * log2(1 + SNR)
    """
    distance_km = max(distance_m / 1000.0, 1e-6)
    path_loss_db = path_loss_a + path_loss_b * math.log10(distance_km)
    channel_gain_linear = db_to_linear(bs_antenna_gain_db - path_loss_db)
    tx_power_mw = db_to_linear(tx_power_dbm)
    noise_mw = db_to_linear(noise_power_dbm)
    snr = max(tx_power_mw * channel_gain_linear / max(noise_mw, 1e-12), 1e-12)
    return bandwidth_hz * math.log2(1.0 + snr)


# ---------------------------------------------------------------------------
# Extracted channel/delay functions (phase 2 only)
#
# In phase 1 these live as methods on IDMRoadMobility.  In phase 2 they are
# standalone functions that take distance_to_bs (metres) as input.  The
# formulas are identical.
# ---------------------------------------------------------------------------


def uplink_rate(distance_to_bs: float, num_selected: int, params: DelayParams) -> float:
    """Per-vehicle Shannon uplink rate (bits/s) with OFDMA equal split.

    Paper §4: each selected vehicle gets B_k = B / num_selected bandwidth.

    Parameters
    ----------
    distance_to_bs : float
        3D distance from vehicle to BS in metres.
    num_selected : int
        Number of vehicles selected this round (for bandwidth splitting).
    params : DelayParams
        Channel/compute parameters.
    """
    bw_per_node = params.bandwidth_hz / max(num_selected, 1)
    return uplink_rate_bps(
        distance_to_bs, bw_per_node,
        params.tx_power_dbm, params.noise_power_dbm,
        params.bs_antenna_gain_db, params.path_loss_a, params.path_loss_b,
    )


def comm_time_s(
    distance_to_bs: float,
    model_size_bits: float,
    num_selected: int,
    params: DelayParams,
) -> float:
    """Upload time T_{k,c} = M / q_k (paper §4).

    Parameters
    ----------
    distance_to_bs : float
        3D distance from vehicle to BS in metres.
    model_size_bits : float
        Size of the model to upload (M in the paper).
    num_selected : int
        Number of vehicles selected this round.
    params : DelayParams
        Channel/compute parameters.
    """
    rate = uplink_rate(distance_to_bs, num_selected, params)
    return model_size_bits / max(rate, 1e-9)


def compute_time_s(
    data_bits: float,
    local_epochs: int,
    params: DelayParams,
) -> float:
    """Local computation time T_{k,p} = |D_k|·g_k / (c_k·f_k) (paper §5).

    Homogeneous across all vehicles for fixed data and hardware.

    Parameters
    ----------
    data_bits : float
        Per-client training data size in bits.
    local_epochs : int
        Number of local training epochs.
    params : DelayParams
        Channel/compute parameters.
    """
    gpu_freq_hz = params.gpu_frequency_ghz * 1e9
    return (
        local_epochs
        * data_bits
        * params.gpu_cycles_per_bit
        / max(params.compute_normalization * gpu_freq_hz, 1e-9)
    )


def delivery_time_s(
    distance_to_bs: float,
    model_size_bits: float,
    data_bits: float,
    num_selected: int,
    local_epochs: int,
    params: DelayParams,
) -> float:
    """Total round time for a node: T_{k,c} + T_{k,p} (paper §6).

    Parameters
    ----------
    distance_to_bs : float
        3D distance from vehicle to BS in metres.
    model_size_bits : float
        Size of the model to upload.
    data_bits : float
        Per-client training data size in bits.
    num_selected : int
        Number of vehicles selected this round.
    local_epochs : int
        Number of local training epochs.
    params : DelayParams
        Channel/compute parameters.
    """
    return (
        comm_time_s(distance_to_bs, model_size_bits, num_selected, params)
        + compute_time_s(data_bits, local_epochs, params)
    )


def delivery_time_bounds(
    dist_min: float,
    dist_max: float,
    model_size_bits: float,
    data_bits: float,
    num_selected: int,
    local_epochs: int,
    params: DelayParams,
) -> tuple[float, float]:
    """(T_min, T_max) delivery-time bounds for utility normalisation (paper Eq. 6).

    T_min at closest distance to BS, T_max at farthest.

    Parameters
    ----------
    dist_min : float
        Minimum 3D distance to BS (vehicle directly under BS).
    dist_max : float
        Maximum 3D distance to BS (vehicle at coverage edge).
    model_size_bits : float
        Size of the model to upload.
    data_bits : float
        Per-client training data size in bits.
    num_selected : int
        Number of vehicles selected this round.
    local_epochs : int
        Number of local training epochs.
    params : DelayParams
        Channel/compute parameters.
    """
    t_min = delivery_time_s(
        dist_min, model_size_bits, data_bits,
        num_selected, local_epochs, params,
    )
    t_max = delivery_time_s(
        dist_max, model_size_bits, data_bits,
        num_selected, local_epochs, params,
    )
    return t_min, t_max


def round_delay_s(
    distances: dict[int, float],
    model_size_bits: float,
    data_bits: float,
    num_selected: int,
    local_epochs: int,
    params: DelayParams,
) -> tuple[dict[int, float], float]:
    """Compute per-node delivery times and synchronous round delay.

    Paper §6: the round duration under synchronous aggregation is
    max_k(T_{k,c} + T_{k,p}) over all selected vehicles.

    Parameters
    ----------
    distances : dict[int, float]
        Mapping of node_id -> 3D distance to BS in metres.
    model_size_bits : float
        Size of the model to upload.
    data_bits : float
        Per-client training data size in bits.
    num_selected : int
        Number of vehicles selected this round.
    local_epochs : int
        Number of local training epochs.
    params : DelayParams
        Channel/compute parameters.

    Returns
    -------
    per_node_time : dict[int, float]
        Delivery time for each node.
    delay : float
        Synchronous round duration (max over all nodes).
    """
    if not distances:
        return {}, 0.0

    per_node_time: dict[int, float] = {}
    for node_id, dist in distances.items():
        per_node_time[node_id] = delivery_time_s(
            dist, model_size_bits, data_bits,
            num_selected, local_epochs, params,
        )

    delay = max(per_node_time.values()) if per_node_time else 0.0
    return per_node_time, delay
