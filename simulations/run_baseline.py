# simulations/baseline_eval.py
from __future__ import annotations

import numpy as np

from env.network import NetworkEnv, NetworkState
from env.paths import build_all_candidate_paths, Path
from env import qos
from baseline.path_selection import select_paths_baseline
from baseline.bandwidth_allocation import equal_share_backhaul_allocation


def compute_user_rate(
    state: NetworkState,
    beta: np.ndarray,
    user_idx: int,
    path: Path | None,
) -> float:
    """
    計算單一 user 的 end-to-end rate:
        r_s = min( r_access, min_{backhaul link} beta[s, ell] * C_ell )
    若 path 為 None，代表完全無路，回傳 0。
    """
    # 接入速率
    r_acc = state.access_capacity[user_idx]

    if path is None or len(path.backhaul_hops) == 0:
        # 沒有回程 path，視為無法接上核心網路
        return 0.0

    # 回程瓶頸速率
    bottleneck = float("inf")
    for (i, j) in path.backhaul_hops:
        cap_ij = state.bh_capacity[i, j]
        if cap_ij <= 0:
            bottleneck = 0.0
            break
        r_ij = beta[user_idx, i, j] * cap_ij
        if r_ij < bottleneck:
            bottleneck = r_ij

    if bottleneck == float("inf"):
        # 理論上不會發生，但保險起見
        bottleneck = 0.0

    return min(r_acc, bottleneck)


def run_single_snapshot(seed: int | None = None) -> None:
    env = NetworkEnv(seed=seed)
    state = env.init_random_state()

    num_gus = len(state.gus)
    num_bs = len(state.bs_nodes)

    print(f"#GUs = {num_gus}, #UAVs = {len(state.uavs)}, "
          f"#GBSs = {len(state.gbss)}, #Sats = {len(state.sats)}")
    print("Backhaul capacity matrix shape:", state.bh_capacity.shape)

    # 1) 建立 candidate paths
    candidate_paths = build_all_candidate_paths(state)

    # 2) Baseline path selection
    chosen_paths = select_paths_baseline(state, candidate_paths)

    # 3) Equal-share backhaul allocation
    beta = equal_share_backhaul_allocation(state, chosen_paths)
    assert beta.shape == (num_gus, num_bs, num_bs)

    # 4) 計算每個 user 的 end-to-end rate & utility
    rates = np.zeros(num_gus, dtype=float)
    utilities = np.zeros(num_gus, dtype=float)
    weights = np.zeros(num_gus, dtype=float)
    urgent_flags = np.zeros(num_gus, dtype=bool)

    for s in range(num_gus):
        user = state.gus[s]
        path = chosen_paths.get(s)
        r_s = compute_user_rate(state, beta, s, path)
        u_s = qos.user_utility(user, r_s)
        w_s = qos.user_weight(user)

        rates[s] = r_s
        utilities[s] = u_s
        weights[s] = w_s
        urgent_flags[s] = user.is_urgent

    # 5) 系統層級 metrics
    # coverage：這裡先用「r_s >= r_req 的比例」作為 coverage
    coverage_flags = rates >= np.array([u.r_req for u in state.gus])
    coverage_ratio = coverage_flags.mean()

    weighted_sum_utility = float((weights * utilities).sum())
    avg_utility = float(utilities.mean())
    avg_rate_mbps = float(rates.mean() / 1e6)

    # urgent / non-urgent 分開統計
    if urgent_flags.any():
        urgent_util_avg = float(utilities[urgent_flags].mean())
        urgent_cov = float(coverage_flags[urgent_flags].mean())
    else:
        urgent_util_avg = float("nan")
        urgent_cov = float("nan")

    if (~urgent_flags).any():
        nonurgent_util_avg = float(utilities[~urgent_flags].mean())
        nonurgent_cov = float(coverage_flags[~urgent_flags].mean())
    else:
        nonurgent_util_avg = float("nan")
        nonurgent_cov = float("nan")

    # 6) 印結果
    print("\n=== Baseline Snapshot Result ===")
    print(f"Avg end-to-end rate     : {avg_rate_mbps:.2f} Mbps")
    print(f"Avg utility (unweighted): {avg_utility:.3f}")
    print(f"Weighted sum utility     : {weighted_sum_utility:.3f}")
    print(f"Coverage ratio (r >= r_req): {coverage_ratio:.3f}")

    print("\n  Urgent users:")
    print(f"    Avg utility : {urgent_util_avg:.3f}")
    print(f"    Coverage    : {urgent_cov:.3f}")

    print("  Non-urgent users:")
    print(f"    Avg utility : {nonurgent_util_avg:.3f}")
    print(f"    Coverage    : {nonurgent_cov:.3f}")

    # 也可以印出前幾個 user 的細節，方便 debug
    print("\nSample users:")
    for s in range(min(5, num_gus)):
        user = state.gus[s]
        print(
            f"  User {s} (urgent={user.is_urgent}) "
            f"r_s={rates[s]/1e6:.2f} Mbps, "
            f"r_req={user.r_req/1e6:.2f} Mbps, "
            f"U_s={utilities[s]:.3f}"
        )


if __name__ == "__main__":
    run_single_snapshot(seed=0)
