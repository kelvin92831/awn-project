# simulations/plot_figures.py
from __future__ import annotations

"""
繪製實驗圖表：
  圖1：三種方案的 weighted_sum_utility 柱狀圖
  圖2：Urgent / Non-urgent coverage & avg utility grouped bar
  圖4：不同 k_paths 下 AI-assisted + priority 的 urgent_avg_utility / urgent_coverage 折線圖

執行方式：
  python -m simulations.plot_figures

會在 ./figures/ 底下輸出 PNG 檔。
"""

import os
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt

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

import env.paths as paths_module  # 用於圖4動態調整 K_PATHS

SchemeName = str  # "equal_share" / "priority" / "ai"


def evaluate_scheme(
    state: NetworkState,
    candidate_paths: Dict[int, List[Path]],
    scheme: SchemeName,
) -> dict:
    """
    與 run_scenarios_all_schemes.py 中邏輯一致：
      - "equal_share": baseline routing + equal-share allocation
      - "priority"   : baseline routing + priority-weighted allocation
      - "ai"         : AI routing + priority-weighted allocation
    """
    num_gus = len(state.gus)
    num_bs = len(state.bs_nodes)

    # Path selection
    if scheme in ("equal_share", "priority"):
        chosen_paths: Dict[int, Path | None] = select_paths_baseline(
            state, candidate_paths
        )
    elif scheme == "ai":
        chosen_paths = select_paths_ai(state, candidate_paths)
    else:
        raise ValueError(f"Unknown scheme: {scheme}")

    # Backhaul allocation
    if scheme == "equal_share":
        beta = equal_share_backhaul_allocation(state, chosen_paths)
    else:
        beta = priority_weighted_backhaul_allocation(state, chosen_paths)

    assert beta.shape == (num_gus, num_bs, num_bs)

    # Per-user metrics
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


def collect_metrics_all_schemes(
    num_scenarios: int = 50,
    base_seed: int = 0,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    跑多場景，收集三方案的 metrics：
      schemes = ["equal_share", "priority", "ai"]
    回傳：
      arr[scheme][metric_name] -> np.ndarray (length = num_scenarios)
    """
    schemes = ["equal_share", "priority", "ai"]

    all_metrics = {s: [] for s in schemes}

    for i in range(num_scenarios):
        seed = base_seed + i
        env = NetworkEnv(seed=seed)
        state = env.init_random_state()
        candidate_paths = build_all_candidate_paths(state)

        for s in schemes:
            m = evaluate_scheme(state, candidate_paths, scheme=s)
            all_metrics[s].append(m)

    keys = list(all_metrics[schemes[0]][0].keys())
    arr = {
        s: {k: np.array([m[k] for m in all_metrics[s]], dtype=float) for k in keys}
        for s in schemes
    }
    return arr


def collect_metrics_ai_vs_priority_for_kpaths(
    k_values: list[int],
    num_scenarios: int = 50,
    base_seed: int = 100,
) -> Dict[int, Dict[str, float]]:
    """
    圖4用：
      對於每個 k_paths 值：
        - 設定 paths_module.K_PATHS = k
        - 跑多場景，但只評估 "ai" scheme（AI routing + priority allocation）
      回傳：
        results[k]["weighted_sum_utility"] = mean over scenarios (deprecated for plot but kept)
        results[k]["urgent_avg_utility"] = mean over scenarios
        results[k]["urgent_coverage"] = mean over scenarios
    """
    results: Dict[int, Dict[str, float]] = {}

    original_k = getattr(paths_module, "K_PATHS", None)

    for k in k_values:
        paths_module.K_PATHS = k
        metrics_list = []

        for i in range(num_scenarios):
            seed = base_seed + i
            env = NetworkEnv(seed=seed)
            state = env.init_random_state()
            candidate_paths = build_all_candidate_paths(state)
            m = evaluate_scheme(state, candidate_paths, scheme="ai")
            metrics_list.append(m)

        ws_array = np.array([m["weighted_sum_utility"] for m in metrics_list], dtype=float)
        urg_cov_array = np.array([m["urgent_coverage"] for m in metrics_list], dtype=float)
        urg_util_array = np.array([m["urgent_avg_utility"] for m in metrics_list], dtype=float)

        results[k] = {
            "weighted_sum_utility": float(np.nanmean(ws_array)),
            "urgent_coverage": float(np.nanmean(urg_cov_array)),
            "urgent_avg_utility": float(np.nanmean(urg_util_array)),
        }

    # 還原原本 K_PATHS
    if original_k is not None:
        paths_module.K_PATHS = original_k

    return results


# ---------- 繪圖函式們 ----------

def plot_fig1_weighted_utility(
    arr: Dict[str, Dict[str, np.ndarray]],
    outdir: str,
):
    schemes = ["equal_share", "priority", "ai"]
    labels = [
        "Equal-share baseline",
        "Priority-weighted baseline",
        "AI-assisted + priority",
    ]

    means = []
    # stds = [] # Removed as requested
    for s in schemes:
        vals = arr[s]["weighted_sum_utility"]
        means.append(float(np.nanmean(vals)))
        # stds.append(float(np.nanstd(vals)))

    x = np.arange(len(schemes))

    plt.figure()
    plt.bar(x, means) # Removed yerr=stds
    plt.xticks(x, labels, rotation=20)
    plt.ylabel("Weighted sum utility")
    plt.ylim(50, 60) # Changed y-axis limit
    plt.title("Figure 1: Weighted sum utility across schemes")
    plt.tight_layout()

    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, "fig1_weighted_sum_utility.png"), dpi=300)
    plt.close()


def plot_fig2_tradeoff(
    arr: Dict[str, Dict[str, np.ndarray]],
    outdir: str,
):
    schemes = ["equal_share", "priority", "ai"]
    scheme_labels = [
        "Equal-share",
        "Priority",
        "AI-assisted",
    ]
    x = np.arange(len(schemes))
    width = 0.35

    # 2(a) coverage
    urgent_cov_means = []
    nonurgent_cov_means = []
    for s in schemes:
        urgent_cov_means.append(float(np.nanmean(arr[s]["urgent_coverage"])))
        nonurgent_cov_means.append(float(np.nanmean(arr[s]["nonurgent_coverage"])))

    plt.figure()
    plt.bar(x - width / 2, urgent_cov_means, width, label="Urgent")
    plt.bar(x + width / 2, nonurgent_cov_means, width, label="Non-urgent")
    plt.xticks(x, scheme_labels)
    plt.ylabel("Coverage ratio")
    plt.title("Figure 2(a): Coverage trade-off (urgent vs non-urgent)")
    plt.legend()
    plt.tight_layout()

    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, "fig2a_coverage_tradeoff.png"), dpi=300)
    plt.close()

    # 2(b) average utility
    urgent_util_means = []
    nonurgent_util_means = []
    for s in schemes:
        urgent_util_means.append(float(np.nanmean(arr[s]["urgent_avg_utility"])))
        nonurgent_util_means.append(float(np.nanmean(arr[s]["nonurgent_avg_utility"])))

    plt.figure()
    plt.bar(x - width / 2, urgent_util_means, width, label="Urgent")
    plt.bar(x + width / 2, nonurgent_util_means, width, label="Non-urgent")
    plt.xticks(x, scheme_labels)
    plt.ylabel("Average utility")
    plt.title("Figure 2(b): Utility trade-off (urgent vs non-urgent)")
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(outdir, "fig2b_utility_tradeoff.png"), dpi=300)
    plt.close()


def plot_fig4_kpaths(
    results_k: Dict[int, Dict[str, float]],
    outdir: str,
):
    """
    圖4：折線圖
      x-axis: k_paths
      y-axis: urgent_avg_utility / urgent_coverage
    """
    k_values = sorted(results_k.keys())

    # Changed from weighted_sum_utility to urgent_avg_utility
    urg_util_means = [results_k[k]["urgent_avg_utility"] for k in k_values]
    urg_cov_means = [results_k[k]["urgent_coverage"] for k in k_values]

    plt.figure()
    plt.plot(k_values, urg_util_means, marker="o", label="Urgent avg utility") # Changed label
    plt.plot(k_values, urg_cov_means, marker="s", label="Urgent coverage")
    plt.xlabel("Number of candidate mesh paths (k_paths)")
    plt.ylabel("Value")
    plt.title("Figure 4: Impact of k_paths on AI-assisted performance")
    plt.legend()
    plt.tight_layout()

    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, "fig4_kpaths_effect.png"), dpi=300)
    plt.close()


def main():
    outdir = "figures"
    num_scenarios = 50

    # 1) 收集三方案 metrics（目前 K_PATHS 用你 env/paths.py 的設定）
    print("Running multi-scenario evaluation for 3 schemes...")
    arr = collect_metrics_all_schemes(num_scenarios=num_scenarios, base_seed=0)

    # 2) 繪製圖1、圖2
    print("Plotting Figure 1 (weighted_sum_utility)...")
    plot_fig1_weighted_utility(arr, outdir)

    print("Plotting Figure 2 (trade-off)...")
    plot_fig2_tradeoff(arr, outdir)

    # Removed Figure 3 plotting

    # 3) 圖4：針對不同 k_paths 重跑 AI + priority
    k_values = [1, 2, 4]
    print(f"Running AI+priority evaluation for k_paths in {k_values}...")
    results_k = collect_metrics_ai_vs_priority_for_kpaths(
        k_values=k_values,
        num_scenarios=num_scenarios,
        base_seed=100,
    )

    print("Plotting Figure 4 (k_paths effect)...")
    plot_fig4_kpaths(results_k, outdir)

    print(f"All figures saved under ./{outdir}/")


if __name__ == "__main__":
    main()
