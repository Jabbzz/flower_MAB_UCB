"""Shared helpers for phase1_flower.

Channel model and physical-layer utilities used by both the mobility
simulation and FL strategies.
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
