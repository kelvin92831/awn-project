# baseline/bandwidth_allocation.py
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from env.network import NetworkState
from env.paths import Path
from env import qos


def equal_share_backhaul_allocation(
    state: NetworkState,
    chosen_paths: Dict[int, Path | None],
) -> np.ndarray:
    """
    對每條 backhaul link (i,j) 做 equal-share 分配：
        beta[s, i, j] = 1 / (#users that use link i->j)
    若某 link 沒有任何 user 使用，則不分配 (beta=0)。

    回傳：
        beta: shape = (num_gus, num_bs, num_bs)
    """
    num_gus = len(state.gus)
    num_bs = len(state.bs_nodes)

    beta = np.zeros((num_gus, num_bs, num_bs), dtype=float)

    # 先建立 link -> user list 的 mapping
    link_to_users: Dict[Tuple[int, int], List[int]] = {}

    for s in range(num_gus):
        path = chosen_paths.get(s)
        if path is None:
            continue
        for (i, j) in path.backhaul_hops:
            # 若這條 link 沒 capacity，skip
            if state.bh_capacity[i, j] <= 0:
                continue
            key = (i, j)
            if key not in link_to_users:
                link_to_users[key] = []
            link_to_users[key].append(s)

    # 對每條 backhaul link 做 equal-share
    for (i, j), users in link_to_users.items():
        k = len(users)
        if k == 0:
            continue
        share = 1.0 / k
        for s in users:
            beta[s, i, j] = share

    return beta


def priority_weighted_backhaul_allocation(
    state: NetworkState,
    chosen_paths: Dict[int, Path | None],
) -> np.ndarray:
    """
    Priority-aware 頻寬分配：
      - 對每條 backhaul link (i,j)，找所有經過的 users：S_ell
      - 對每個 user s ∈ S_ell 給一個權重 ω_{s,ell}：
            這邊先用簡單版：urgent → ω=2, non-urgent → ω=1，
            實作上直接使用 qos.user_weight(user)
      - 再做 normalized：
            beta[s, i, j] = ω_{s,ell} / Σ_{j∈S_ell} ω_{j,ell}

    這樣 urgent flows 在壅塞 link 上會拿到比較多頻寬。
    """
    num_gus = len(state.gus)
    num_bs = len(state.bs_nodes)

    beta = np.zeros((num_gus, num_bs, num_bs), dtype=float)

    # 建立 link -> user list 的 mapping
    link_to_users: Dict[Tuple[int, int], List[int]] = {}

    for s in range(num_gus):
        path = chosen_paths.get(s)
        if path is None:
            continue
        for (i, j) in path.backhaul_hops:
            if state.bh_capacity[i, j] <= 0:
                continue
            key = (i, j)
            if key not in link_to_users:
                link_to_users[key] = []
            link_to_users[key].append(s)

    # 對每條 backhaul link 做 priority-weighted 分配
    for (i, j), users in link_to_users.items():
        if not users:
            continue

        weights = []
        for s in users:
            user = state.gus[s]
            w = qos.user_weight(user) 
            weights.append(w)

        sum_w = float(sum(weights))
        if sum_w <= 0:
            continue

        for s, w in zip(users, weights):
            beta[s, i, j] = w / sum_w

    return beta
