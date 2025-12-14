# simulations/run_scenarios_ai_routing.py
from __future__ import annotations

from typing import Literal, Dict

import numpy as np

from env.network import NetworkEnv, NetworkState
from env.paths import build_all_candidate_paths, Path
from env import qos
from baseline.path_selection import select_paths_baseline
from baseline.bandwidth_allocation import priority_weighted_backhaul_allocation
from simulations.run_baseline import compute_user_rate

from ai.path_selector import select_paths_ai

PathMode = Literal["baseline", "ai"]


def select_paths(
    state: NetworkState,
    candidate_paths,
    mode: PathMode = "baseline",
) -> Dict[int, Path | None]:
    """封裝一層，方便切換 routing 策略。"""
    if mode == "baseline":
        return select_paths_baseline(state, candidate_paths)
    elif mode == "ai":
        return select_paths_ai(state, candidate_paths)
    else:
        raise ValueError(f"Unknown path mode: {mode}")


def evaluate_snapshot_routing(
    state: NetworkState,
    path_mode: PathMode = "baseline",
) -> dict:
    """
    在固定 priority-weighted backhaul allocation 的前提下，
    比較不同 routing 策略的效能，並統計 SAT 使用比例。
    """
    num_gus = len(state.gus)
    num_bs = len(state.bs_nodes)

    # 1) candidate paths
    candidate_paths = build_all_candidate_paths(state)

    # 2) path selection（baseline vs ai）
    chosen_paths = select_paths(state, candidate_paths, mode=path_mode)

    # 3) backhaul allocation：固定用 priority-weighted
    from baseline.bandwidth_allocation import priority_weighted_backhaul_allocation

    beta = priority_weighted_backhaul_allocation(state, chosen_paths)
    assert beta.shape == (num_gus, num_bs, num_bs)

    # 4) per-user rate & utility
    rates = np.zeros(num_gus, dtype=float)
    utilities = np.zeros(num_gus, dtype=float)
    weights = np.zeros(num_gus, dtype=float)
    urgent_flags = np.zeros(num_gus, dtype=bool)
    sat_flags = np.zeros(num_gus, dtype=bool)

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
        sat_flags[s] = (path is not None and path.uses_satellite)

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
        metrics["urgent_sat_ratio"] = float(sat_flags[urgent_flags].mean())
    else:
        metrics["urgent_avg_utility"] = float("nan")
        metrics["urgent_coverage"] = float("nan")
        metrics["urgent_sat_ratio"] = float("nan")

    if (~urgent_flags).any():
        metrics["nonurgent_avg_utility"] = float(utilities[~urgent_flags].mean())
        metrics["nonurgent_coverage"] = float(coverage_flags[~urgent_flags].mean())
        metrics["nonurgent_sat_ratio"] = float(sat_flags[~urgent_flags].mean())
    else:
        metrics["nonurgent_avg_utility"] = float("nan")
        metrics["nonurgent_coverage"] = float("nan")
        metrics["nonurgent_sat_ratio"] = float("nan")

    # overall SAT 使用比例
    metrics["sat_ratio_overall"] = float(sat_flags.mean())

    return metrics



def run_multi_scenarios(num_scenarios: int = 50, base_seed: int = 100) -> None:
    """
    多場景比較：
      - baseline routing + priority allocation
      - AI routing      + priority allocation
    """
    all_metrics_baseline = []
    all_metrics_ai = []

    for i in range(num_scenarios):
        seed = base_seed + i
        env = NetworkEnv(seed=seed)
        state = env.init_random_state()

        m_base = evaluate_snapshot_routing(state, path_mode="baseline")
        m_ai = evaluate_snapshot_routing(state, path_mode="ai")

        all_metrics_baseline.append(m_base)
        all_metrics_ai.append(m_ai)

    keys = list(all_metrics_baseline[0].keys())

    arr_base = {k: np.array([m[k] for m in all_metrics_baseline], dtype=float) for k in keys}
    arr_ai = {k: np.array([m[k] for m in all_metrics_ai], dtype=float) for k in keys}

    print(f"=== Routing: Baseline vs AI-assisted over {num_scenarios} scenarios ===\n")

    print(">> Baseline routing + priority allocation:")
    for k in keys:
        vals = arr_base[k]
        mean = float(np.nanmean(vals))
        std = float(np.nanstd(vals))
        print(f"  {k:25s}: mean = {mean:.4f}, std = {std:.4f}")

    print("\n>> AI routing + priority allocation:")
    for k in keys:
        vals = arr_ai[k]
        mean = float(np.nanmean(vals))
        std = float(np.nanstd(vals))
        print(f"  {k:25s}: mean = {mean:.4f}, std = {std:.4f}")

    print("\n>> Difference (AI - Baseline):")
    for k in keys:
        diff_mean = float(
            np.nanmean(arr_ai[k] - arr_base[k])
        )
        print(f"  {k:25s}: Δmean = {diff_mean:+.4f}")


if __name__ == "__main__":
    run_multi_scenarios(num_scenarios=50, base_seed=0)
