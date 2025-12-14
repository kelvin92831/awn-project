# env/channel.py
import math
from typing import Literal

import numpy as np

from . import config
from .nodes import Node, GroundUser, UAV, GroundBS, Satellite

LinkType = Literal["A2A", "A2G", "G2G", "SAT"]


def distance_3d(a: Node, b: Node) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def free_space_pathloss_linear(d: float, freq: float, n: float) -> float:
    """
    回傳 channel power gain h (線性)
    基本型式 ~ d^{-n}，含自由空間常數。
    """
    if d == 0:
        return 1.0
    k0 = (config.C / (4 * math.pi * freq)) ** 2
    return k0 * (d ** (-n))


def los_probability(a: Node, b: Node) -> float:
    """
    非常簡化版的 A2G LoS 機率模型。
    """
    d = distance_3d(a, b)
    horizontal = math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)
    if d == 0 or horizontal == 0:
        return 1.0
    theta = math.degrees(math.asin(abs(a.z - b.z) / d))
    b1 = config.B1
    b2 = config.B2
    return 1 / (1 + b1 * math.exp(-b2 * (theta - b1)))


def channel_gain(a: Node, b: Node, link_type: LinkType) -> float:
    d = distance_3d(a, b)
    if d == 0:
        return 1.0

    if link_type == "A2A":
        return free_space_pathloss_linear(d, config.CARRIER_FREQ, config.PATH_LOSS_EXP_A2A)
    elif link_type == "G2G":
        return free_space_pathloss_linear(d, config.CARRIER_FREQ, config.PATH_LOSS_EXP_G2G)
    elif link_type == "SAT":
        # UAV/GBS <-> Satellite：近似自由空間
        return free_space_pathloss_linear(d, config.CARRIER_FREQ, config.PATH_LOSS_EXP_SAT)
    elif link_type == "A2G":
        # LoS / NLoS 混合 A2G 模型
        p_los = los_probability(a, b)
        base = free_space_pathloss_linear(d, config.CARRIER_FREQ, n=2.0)
        eta_los = 10 ** (-config.ETA_LOS_DB / 10.0)
        eta_nlos = 10 ** (-config.ETA_NLOS_DB / 10.0)
        return base * (p_los * eta_los + (1 - p_los) * eta_nlos)
    else:
        raise ValueError(f"Unknown link type: {link_type}")


def snr_linear(h: float) -> float:
    return config.TX_POWER_WATT * h / config.NOISE_POWER_WATT


def snr_db(h: float) -> float:
    snr_lin = snr_linear(h)
    if snr_lin <= 0:
        return -200.0
    return 10 * math.log10(snr_lin)


def capacity_bps(h: float) -> float:
    snr_lin = snr_linear(h)
    return config.BANDWIDTH_HZ * math.log2(1 + snr_lin)


def infer_link_type(a: Node, b: Node) -> LinkType:
    """根據節點型態猜測 link 類型（只用在 backhaul）"""
    # 與衛星互連
    if isinstance(a, Satellite) or isinstance(b, Satellite):
        return "SAT"

    # UAV-UAV or UAV-GBS or GBS-UAV → 都當成 A2A 類型處理
    if (isinstance(a, UAV) and isinstance(b, UAV)) or \
       (isinstance(a, UAV) and isinstance(b, GroundBS)) or \
       (isinstance(a, GroundBS) and isinstance(b, UAV)):
        return "A2A"

    # 其他地面之間（如果有）
    return "G2G"
