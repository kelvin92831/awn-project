# env/network.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from . import config
from .nodes import GroundUser, UAV, GroundBS, Satellite, Node
from . import channel


@dataclass
class NetworkState:
    # 節點列表
    gus: List[GroundUser]
    uavs: List[UAV]
    gbss: List[GroundBS]
    sats: List[Satellite]

    # Access: GU -> 服務 UAV 的 mapping & capacity
    gu_to_uav: np.ndarray          # shape = (num_gus,), value in [0, num_uavs)
    access_capacity: np.ndarray    # shape = (num_gus,)

    # Backhaul: 所有 BS 節點 (UAV + GBS + Sat)
    bs_nodes: List[Node]           # index: 0..num_bs-1
    bh_capacity: np.ndarray        # shape = (num_bs, num_bs), 0 表示無 link


class NetworkEnv:
    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    # ---------- 隨機座標 ----------
    def _random_xy(self) -> Tuple[float, float]:
        x = self.rng.uniform(config.AREA_X_MIN, config.AREA_X_MAX)
        y = self.rng.uniform(config.AREA_Y_MIN, config.AREA_Y_MAX)
        return x, y

    # ---------- 建立隨機拓樸 ----------
    def init_random_state(self) -> NetworkState:
        node_id_counter = 0

        # Ground Users
        gus: List[GroundUser] = []
        for _ in range(config.NUM_GUS):
            x, y = self._random_xy()
            is_urgent = bool(self.rng.random() < config.URGENT_USER_RATIO)

            if is_urgent:
                r_req = config.REQ_RATE_URGENT_BPS
                v_s = config.V_URGENT
            else:
                r_req = config.REQ_RATE_NORMAL_BPS
                v_s = config.V_NORMAL

            gus.append(
                GroundUser(
                    id=node_id_counter,
                    x=x,
                    y=y,
                    z=0.0,
                    is_urgent=is_urgent,
                    r_req=r_req,
                    v_s=v_s,
                )
            )
            node_id_counter += 1

        # Ground BS
        gbss: List[GroundBS] = []
        for _ in range(config.NUM_GBSS):
            x, y = self._random_xy()
            gbss.append(
                GroundBS(
                    id=node_id_counter,
                    x=x,
                    y=y,
                    z=config.GBS_HEIGHT,
                )
            )
            node_id_counter += 1

        # UAVs
        uavs: List[UAV] = []
        for _ in range(config.NUM_UAVS):
            x, y = self._random_xy()
            uavs.append(
                UAV(
                    id=node_id_counter,
                    x=x,
                    y=y,
                    z=config.UAV_HEIGHT,
                )
            )
            node_id_counter += 1

        # Satellites（簡化：放在區域中心上空）
        sats: List[Satellite] = []
        if config.NUM_SATS > 0:
            center_x = 0.5 * (config.AREA_X_MIN + config.AREA_X_MAX)
            center_y = 0.5 * (config.AREA_Y_MIN + config.AREA_Y_MAX)
            for _ in range(config.NUM_SATS):
                sats.append(
                    Satellite(
                        id=node_id_counter,
                        x=center_x,
                        y=center_y,
                        z=config.SAT_HEIGHT,
                    )
                )
                node_id_counter += 1

        # ---------- Access: GU -> 服務 UAV ----------
        num_gus = len(gus)
        num_uavs = len(uavs)
        gu_to_uav = np.zeros(num_gus, dtype=int)
        access_capacity = np.zeros(num_gus, dtype=float)

        for gi, gu in enumerate(gus):
            best_uav_idx = 0
            best_c = 0.0
            for ui, uav in enumerate(uavs):
                h_su = channel.channel_gain(gu, uav, link_type="A2G")
                c_su = channel.capacity_bps(h_su)
                if c_su > best_c:
                    best_c = c_su
                    best_uav_idx = ui
            gu_to_uav[gi] = best_uav_idx
            access_capacity[gi] = best_c

        # ---------- Backhaul: UAV / GBS / Sat ----------
        bs_nodes: List[Node] = []
        bs_nodes.extend(uavs)
        bs_nodes.extend(gbss)
        bs_nodes.extend(sats)

        num_bs = len(bs_nodes)
        bh_capacity = np.zeros((num_bs, num_bs), dtype=float)

        for i in range(num_bs):
            for j in range(num_bs):
                if i == j:
                    continue
                a = bs_nodes[i]
                b = bs_nodes[j]
                link_type = channel.infer_link_type(a, b)
                h_ij = channel.channel_gain(a, b, link_type)
                snr_db = channel.snr_db(h_ij)
                if snr_db < config.MIN_SNR_DB_BACKHAUL:
                    continue  # 略過太差的 link
                c_ij = channel.capacity_bps(h_ij)
                bh_capacity[i, j] = c_ij

        return NetworkState(
            gus=gus,
            uavs=uavs,
            gbss=gbss,
            sats=sats,
            gu_to_uav=gu_to_uav,
            access_capacity=access_capacity,
            bs_nodes=bs_nodes,
            bh_capacity=bh_capacity,
        )
