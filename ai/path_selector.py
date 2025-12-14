# ai/path_selector.py
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from env.network import NetworkState
from env.paths import Path
from env import qos
from .features import extract_path_features, get_default_theta


def select_paths_ai(
    state: NetworkState,
    candidate_paths: Dict[int, List[Path]],
    theta: np.ndarray | None = None,
    lambda_cong: float = 0.3,
    debug: bool = False,
) -> Dict[int, Optional[Path]]:
    """
    Congestion-aware, AI-assisted 路徑選擇：

      1) 先決定 user 的處理順序：urgent 先選路，其次 non-urgent
      2) 對於每個 user s 及其候選 path k：
           - 抽特徵 phi_{s,k}
           - base_score = theta · phi_{s,k}
           - congestion_metric = 目前 path 上所有 backhaul link 的負載總和
           - score = base_score - lambda_cong * congestion_metric
      3) 選 score 最大的 path，並將該 user 的權重加到對應 link 的負載中

    link_load[i,j] 代表 link (i->j) 上目前累積的 "加權使用者數"
    （這裡用 qos.user_weight(user)，讓 urgent 的「壅塞影響力」更大）
    """
    if theta is None:
        theta = get_default_theta()
    theta = np.asarray(theta, dtype=float)
    assert theta.shape[0] == 4, "theta 維度應與 features 相同 (4)"

    num_gus = len(state.gus)
    num_bs = len(state.bs_nodes)

    # 初始化回傳結構 & link 負載矩陣
    chosen: Dict[int, Optional[Path]] = {s: None for s in range(num_gus)}
    link_load = np.zeros((num_bs, num_bs), dtype=float)

    # --------- 決定 user 處理順序：urgent 先 ---------
    urgent_indices = [i for i, u in enumerate(state.gus) if u.is_urgent]
    nonurgent_indices = [i for i, u in enumerate(state.gus) if not u.is_urgent]

    user_order = urgent_indices + nonurgent_indices

    if debug:
        print("[AI routing] user order (urgent first):", user_order)

    # --------- 逐個 user 選 path ---------
    for s in user_order:
        user = state.gus[s]
        paths = candidate_paths.get(s, [])
        if not paths:
            chosen[s] = None
            if debug:
                print(f"\nUser {s}: no candidate paths.")
            continue

        best_score = -1e18
        best_path: Optional[Path] = None

        if debug:
            print(f"\n[AI routing] User {s} (urgent={user.is_urgent}):")

        for p in paths:
            feats = extract_path_features(state, p)
            base_score = float(np.dot(theta, feats.phi))

            congestion_metric = 0.0
            for (i, j) in p.backhaul_hops:
                congestion_metric += link_load[i, j]

            score = base_score - lambda_cong * congestion_metric

            if debug:
                print(
                    f"  Path uses_sat={p.uses_satellite}, "
                    f"access_uav={p.access_uav_index}, "
                    f"base_score={base_score:.3f}, "
                    f"cong={congestion_metric:.3f}, "
                    f"score={score:.3f}, "
                    f"{feats.description}"
                )

            if score > best_score:
                best_score = score
                best_path = p

        chosen[s] = best_path

        if best_path is not None:
            w_s = qos.user_weight(user)
            for (i, j) in best_path.backhaul_hops:
                link_load[i, j] += w_s

    if debug:
        print("\n[AI routing] final link_load:")
        print(link_load)

    return chosen
