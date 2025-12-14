# simulations/run_scenarios_baseline.py
from __future__ import annotations

import numpy as np

from typing import Literal

from env.network import NetworkEnv, NetworkState
from env.paths import build_all_candidate_paths, Path
from env import qos
from baseline.path_selection import select_paths_baseline
from baseline.bandwidth_allocation import (
    equal_share_backhaul_allocation,
    priority_weighted_backhaul_allocation,
)
from simulations.run_baseline import compute_user_rate  # 重用 Stage 2 的函式

AllocationMode = Literal["equal_share", "priority"]


def evaluate_snapshot(
    state: NetworkState,
    allocation_mode: AllocationMode = "equal_share",
) -> dict:
    """
    對單一 snapshot 做 baseline 評估，allocation_mode 決定要用哪一種回程頻寬策略：
      - "equal_share"：平均分配
      - "priority"   ：urgent flows 權重大

    回傳一組 metrics dict。
    """
    num_gus = len(state.gus)
    num_bs = len(state.bs_nodes)

    # 1) candidate paths
    candidate_paths = build_all_candidate_paths(state)

    # 2) baseline path selection（目前 routing 一樣）
    chosen_paths = select_paths_baseline(state, candidate_paths)

    # 3) backhaul allocation
    if allocation_mode == "equal_share":
        beta = equal_share_backhaul_allocation(state, chosen_paths)
    elif allocation_mode == "priority":
        beta = priority_weighted_backhaul_allocation(state, chosen_paths)
    else:
        raise ValueError(f"Unknown allocation_mode: {allocation_mode}")

    assert beta.shape == (num_gus, num_bs, num_bs)

    # 4) per-user rate & utility
    rates = np.zeros(num_gus, dtype=float)
    utilities = np.zeros(num_gus, dtype=float)
    weights = np.zeros(num_gus, dtype=float)
    urgent_flags = np.zeros(num_gus, dtype=bool)

    for s in range(num_gus):
        user = state.gus[s]
        path: Path | None = chosen_paths.get(s)
        r_s = compute_user_rate(state, beta, s, path)
        u_s = qos.user_utility(user, r_s)
        w_s = qos.user_weight(user)

        rates[s] = r_s
        utilities[s] = u_s
        weights[s] = w_s
        urgent_flags[s] = user.is_urgent

    # coverage：r_s >= r_req
    r_req_arr = np.array([u.r_req for u in state.gus])
    coverage_flags = rates >= r_req_arr
    coverage_ratio = coverage_flags.mean()

    weighted_sum_utility = float((weights * utilities).sum())
    avg_utility = float(utilities.mean())
    avg_rate_mbps = float(rates.mean() / 1e6)

    metrics = {
        "avg_rate_mbps": avg_rate_mbps,
        "avg_utility": avg_utility,
        "weighted_sum_utility": weighted_sum_utility,
        "coverage_ratio": float(coverage_ratio),
    }

    # urgent / non-urgent 分開
    if urgent_flags.any():
        metrics["urgent_avg_utility"] = float(utilities[urgent_flags].mean())
        metrics["urgent_coverage"] = float(coverage_flags[urgent_flags].mean())
    else:
        metrics["urgent_avg_utility"] = float("nan")
        metrics["urgent_coverage"] = float("nan")

    if (~urgent_flags).any():
        metrics["nonurgent_avg_utility"] = float(utilities[~urgent_flags].mean())
        metrics["nonurgent_coverage"] = float(coverage_flags[~urgent_flags].mean())
    else:
        metrics["nonurgent_avg_utility"] = float("nan")
        metrics["nonurgent_coverage"] = float("nan")

    return metrics


def run_multi_scenarios(num_scenarios: int = 50, base_seed: int = 0) -> None:
    """
    重複跑多個隨機拓樸，分別用：
      - equal-share allocation
      - priority-weighted allocation
    統計兩種策略的平均表現，並印出比較結果。
    """
    all_metrics_equal = []
    all_metrics_priority = []

    for i in range(num_scenarios):
        seed = base_seed + i
        env = NetworkEnv(seed=seed)
        state = env.init_random_state()

        m_equal = evaluate_snapshot(state, allocation_mode="equal_share")
        m_prio = evaluate_snapshot(state, allocation_mode="priority")

        all_metrics_equal.append(m_equal)
        all_metrics_priority.append(m_prio)

    keys = list(all_metrics_equal[0].keys())

    arr_equal = {k: np.array([m[k] for m in all_metrics_equal], dtype=float) for k in keys}
    arr_prio = {k: np.array([m[k] for m in all_metrics_priority], dtype=float) for k in keys}

    print(f"=== Baseline vs Priority-weighted over {num_scenarios} scenarios ===\n")

    print(">> Equal-share allocation:")
    for k in keys:
        vals = arr_equal[k]
        mean = float(np.nanmean(vals))
        std = float(np.nanstd(vals))
        print(f"  {k:25s}: mean = {mean:.4f}, std = {std:.4f}")

    print("\n>> Priority-weighted allocation:")
    for k in keys:
        vals = arr_prio[k]
        mean = float(np.nanmean(vals))
        std = float(np.nanstd(vals))
        print(f"  {k:25s}: mean = {mean:.4f}, std = {std:.4f}")

    print("\n>> Difference (Priority - Equal-share):")
    for k in keys:
        diff_mean = float(
            np.nanmean(arr_prio[k] - arr_equal[k])
        )
        print(f"  {k:25s}: Δmean = {diff_mean:+.4f}")


if __name__ == "__main__":
    run_multi_scenarios(num_scenarios=50, base_seed=0)
