# simulations/run_scenarios_all_schemes.py
from __future__ import annotations

from typing import Dict, Literal

import numpy as np

from env.network import NetworkEnv, NetworkState
from env.paths import build_all_candidate_paths, Path
from env import qos
from baseline.path_selection import select_paths_baseline
from baseline.bandwidth_allocation import (
    equal_share_backhaul_allocation,
    priority_weighted_backhaul_allocation,
)
from simulations.run_baseline import compute_user_rate
from ai.path_selector import select_paths_ai

SchemeName = Literal["equal_share", "priority", "ai", "ai_learned"]


def evaluate_scheme(
    state: NetworkState,
    candidate_paths: Dict[int, list[Path]],
    scheme: SchemeName,
    theta_learned: np.ndarray | None = None,
) -> dict:
    """
    在同一個 NetworkState / candidate_paths 下，評估多個方案之一：

      - "equal_share":
          routing  = baseline
          alloc    = equal-share backhaul

      - "priority":
          routing  = baseline
          alloc    = priority-weighted backhaul

      - "ai":
          routing  = AI-assisted (heuristic theta, 由 select_paths_ai 內部預設)
          alloc    = priority-weighted backhaul

      - "ai_learned":
          routing  = AI-assisted (使用離線訓練得到的 learned theta)
          alloc    = priority-weighted backhaul
    """
    num_gus = len(state.gus)
    num_bs = len(state.bs_nodes)

    # ---------- Path selection ----------
    if scheme in ("equal_share", "priority"):
        chosen_paths: Dict[int, Path | None] = select_paths_baseline(
            state, candidate_paths
        )
    elif scheme == "ai":
        # 使用 heuristic 版本的 theta（ai/path_selector.py 裡的預設參數）
        chosen_paths = select_paths_ai(state, candidate_paths)
    elif scheme == "ai_learned":
        if theta_learned is None:
            raise ValueError(
                "theta_learned is None，請確認 ai/learned_theta_batch.npy 已經訓練並存在。"
            )
        chosen_paths = select_paths_ai(state, candidate_paths, theta=theta_learned)
    else:
        raise ValueError(f"Unknown scheme: {scheme}")

    # ---------- Backhaul allocation ----------
    if scheme == "equal_share":
        beta = equal_share_backhaul_allocation(state, chosen_paths)
    else:
        beta = priority_weighted_backhaul_allocation(state, chosen_paths)

    assert beta.shape == (num_gus, num_bs, num_bs)

    # ---------- Per-user rate & utility ----------
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


def run_multi_scenarios_all_schemes(
    num_scenarios: int = 50,
    base_seed: int = 0,
    theta_learned: np.ndarray | None = None,
) -> None:
    """
    多場景實驗：一次比較四個方案

      1) equal_share baseline:
           - baseline routing
           - equal-share allocation

      2) priority baseline:
           - baseline routing
           - priority-weighted allocation

      3) ai-assisted (heuristic theta):
           - AI routing (congestion-aware, heuristic theta)
           - priority-weighted allocation

      4) ai-assisted (learned theta):
           - AI routing (congestion-aware, learned theta from RL)
           - priority-weighted allocation
    """
    schemes: list[SchemeName] = ["equal_share", "priority", "ai"]
    scheme_names = {
        "equal_share": "Equal-share baseline",
        "priority": "Priority-weighted baseline",
        "ai": "AI-assisted (heuristic θ)",
        "ai_learned": "AI-assisted (learned θ)",
    }

    # 如果有提供 learned theta，就把 ai_learned 也加入比較
    if theta_learned is not None:
        schemes.append("ai_learned")

    # 收集每個方案在多場景的 metrics
    all_metrics = {s: [] for s in schemes}

    for i in range(num_scenarios):
        seed = base_seed + i
        env = NetworkEnv(seed=seed)
        state = env.init_random_state()

        candidate_paths = build_all_candidate_paths(state)

        for s in schemes:
            m = evaluate_scheme(
                state,
                candidate_paths,
                scheme=s,
                theta_learned=theta_learned,
            )
            all_metrics[s].append(m)

    # 轉成 numpy array
    keys = list(all_metrics[schemes[0]][0].keys())
    arr = {
        s: {k: np.array([m[k] for m in all_metrics[s]], dtype=float) for k in keys}
        for s in schemes
    }

    print(f"=== Comparison over {num_scenarios} scenarios ===\n")

    # 逐方案印 mean/std
    for s in schemes:
        print(f">> {scheme_names[s]}:")
        for k in keys:
            vals = arr[s][k]
            mean = float(np.nanmean(vals))
            std = float(np.nanstd(vals))
            print(f"  {k:25s}: mean = {mean:.4f}, std = {std:.4f}")
        print()

    # 相對於 equal_share 的差異
    base = "equal_share"
    for s in schemes:
        if s == base:
            continue
        print(f">> Difference ({scheme_names[s]} - {scheme_names[base]}):")
        for k in keys:
            diff_mean = float(
                np.nanmean(arr[s][k] - arr[base][k])
            )
            print(f"  {k:25s}: Δmean = {diff_mean:+.4f}")
        print()

    # Priority 作為 baseline，比較兩種 AI 方案（如果有）
    base2 = "priority"
    for s in schemes:
        if s not in ("ai", "ai_learned"):
            continue
        print(f">> Difference ({scheme_names[s]} - {scheme_names[base2]}):")
        for k in keys:
            diff_mean = float(
                np.nanmean(arr[s][k] - arr[base2][k])
            )
            print(f"  {k:25s}: Δmean = {diff_mean:+.4f}")
        print()


if __name__ == "__main__":
    # 嘗試載入 batch RL 訓練得到的 theta
    try:
        learned_theta = np.load("ai/learned_theta_finetune.npy")
        print("Loaded learned theta from ai/learned_theta_finetune.npy")
    except FileNotFoundError:
        learned_theta = None
        print("Warning: ai/learned_theta_batch.npy not found, "
              "will skip 'AI-assisted (learned θ)' scheme.")

    run_multi_scenarios_all_schemes(
        num_scenarios=50,
        base_seed=0,
        theta_learned=learned_theta,
    )
