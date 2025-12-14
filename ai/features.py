# ai/features.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import math
import numpy as np

from env.network import NetworkState
from env.paths import Path


@dataclass
class PathFeatures:
    phi: np.ndarray  # shape = (4,)
    description: str


def extract_path_features(
    state: NetworkState,
    path: Path,
) -> PathFeatures:
    """
    為單一候選路徑抽特徵向量 phi ∈ R^4：
      phi[0] = log10(1 + bottleneck_backhaul_Mbps)
      phi[1] = access_capacity / r_req
      phi[2] = 1.0 if uses_satellite else 0.0
      phi[3] = 1.0 if (urgent AND uses_satellite) else 0.0
    """

    user_idx = path.user_index
    user = state.gus[user_idx]

    # --- access 相關 ---
    r_acc = state.access_capacity[user_idx]  # bps
    r_req = max(user.r_req, 1e-6)
    access_ratio = r_acc / r_req

    # --- backhaul 瓶頸 ---
    bottleneck_bps = float("inf")
    if not path.backhaul_hops:
        bottleneck_bps = 0.0
    else:
        for (i, j) in path.backhaul_hops:
            c_ij = state.bh_capacity[i, j]
            if c_ij <= 0:
                bottleneck_bps = 0.0
                break
            if c_ij < bottleneck_bps:
                bottleneck_bps = c_ij

        if bottleneck_bps == float("inf"):
            bottleneck_bps = 0.0

    bottleneck_mbps = bottleneck_bps / 1e6
    # log-scale 避免數字太大
    f0 = math.log10(1.0 + bottleneck_mbps)

    # access ratio
    f1 = access_ratio

    # 是否使用衛星
    uses_sat = 1.0 if path.uses_satellite else 0.0
    f2 = uses_sat

    # urgent AND satellite 的交互項
    is_urgent = 1.0 if user.is_urgent else 0.0
    f3 = uses_sat * is_urgent

    phi = np.array([f0, f1, f2, f3], dtype=float)

    desc = (
        f"bottleneck_mbps={bottleneck_mbps:.2f}, "
        f"access_ratio={access_ratio:.2f}, "
        f"uses_sat={int(uses_sat)}, "
        f"urgent={int(is_urgent)}"
    )
    return PathFeatures(phi=phi, description=desc)


def get_default_theta() -> np.ndarray:
    """
    預設權重向量 theta，用於線性打分：
      score = theta · phi

    設計理念：
      - theta[0] (瓶頸容量): 正，偏好容量大的路徑
      - theta[1] (access ratio): 正，access 足夠較好
      - theta[2] (uses_sat): 負，一般情況下懲罰走衛星（延遲/成本）
      - theta[3] (urgent & sat): 正，對 urgent user 使用衛星時給一點補償加分
    """
    theta = np.array(
        [
            +1.0,   # bottleneck capacity 1
            +0.5,   # access adequacy 0.5
            -0.5,   # satellite penalty -0.5
            +0.8,   # urgent & satellite bonus 0.8
        ],
        dtype=float,
    )
    return theta
