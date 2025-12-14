# simulations/train_theta_rl.py
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from env.network import NetworkEnv, NetworkState
from env.paths import build_all_candidate_paths, Path
from env import qos
from baseline.path_selection import select_paths_baseline
from baseline.bandwidth_allocation import priority_weighted_backhaul_allocation
from simulations.run_baseline import compute_user_rate
from ai.features import extract_path_features, get_default_theta


# -------------------------
# Utility functions
# -------------------------

def softmax(x: np.ndarray, tau: float = 1.0) -> np.ndarray:
    """Numerically stable softmax with temperature tau."""
    x = x / max(tau, 1e-8)
    x_max = np.max(x)
    e = np.exp(x - x_max)
    s = e.sum()
    if s <= 0:
        return np.ones_like(x) / len(x)
    return e / s


def eval_priority_baseline(
    state: NetworkState,
    candidate_paths: dict[int, list[Path]],
) -> float:
    """
    在給定 scenario 上，用：
      - baseline routing
      - priority-weighted backhaul allocation
    計算 priority baseline 的 weighted_sum_utility。
    """
    num_gus = len(state.gus)
    num_bs = len(state.bs_nodes)

    chosen_paths = select_paths_baseline(state, candidate_paths)
    beta = priority_weighted_backhaul_allocation(state, chosen_paths)

    rates = np.zeros(num_gus, dtype=float)
    utilities = np.zeros(num_gus, dtype=float)
    weights = np.zeros(num_gus, dtype=float)

    for s in range(num_gus):
        user = state.gus[s]
        path = chosen_paths.get(s)
        r_s = compute_user_rate(state, beta, s, path)
        u_s = qos.user_utility(user, r_s)
        w_s = qos.user_weight(user)

        rates[s] = r_s
        utilities[s] = u_s
        weights[s] = w_s

    weighted_sum_utility = float((weights * utilities).sum())
    return weighted_sum_utility


def evaluate_with_theta(
    state: NetworkState,
    candidate_paths: dict[int, list[Path]],
    theta: np.ndarray,
    tau: float = 0.1,
) -> tuple[float, dict]:
    """
    給定 scenario / candidate_paths / theta：
      - 對每個 user 以 softmax policy 抽樣路徑
      - 用 priority-weighted allocation 求得 RL 方案的 QoS
      - 回傳：
          reward = WSU_RL - WSU_priority
          info: 各種 metrics + decisions（給 REINFORCE 用）
    """
    num_gus = len(state.gus)
    num_bs = len(state.bs_nodes)

    # baseline reward (priority-based heuristic)
    wsu_priority = eval_priority_baseline(state, candidate_paths)

    # RL routing (stochastic policy)
    chosen_paths: dict[int, Path | None] = {}
    decisions: list[tuple[np.ndarray, np.ndarray, int]] = []

    for s in range(num_gus):
        paths = candidate_paths.get(s, [])
        if not paths:
            chosen_paths[s] = None
            continue

        phi_list = []
        for p in paths:
            feats = extract_path_features(state, p)
            phi_list.append(feats.phi)
        phi_arr = np.stack(phi_list, axis=0)  # (k, d)

        scores = phi_arr.dot(theta)
        probs = softmax(scores, tau=tau)
        chosen_idx = int(np.random.choice(len(paths), p=probs))

        chosen_paths[s] = paths[chosen_idx]
        decisions.append((phi_arr, probs, chosen_idx))

    beta = priority_weighted_backhaul_allocation(state, chosen_paths)
    assert beta.shape == (num_gus, num_bs, num_bs)

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

    r_req_arr = np.array([u.r_req for u in state.gus])
    coverage_flags = rates >= r_req_arr

    wsu_rl = float((weights * utilities).sum())
    reward = wsu_rl - wsu_priority  # 相對 priority baseline 的提升量

    info = {
        "weighted_sum_utility_rl": wsu_rl,
        "weighted_sum_utility_priority": wsu_priority,
        "decisions": decisions,
    }

    return reward, info


# -------------------------
# Batch REINFORCE
# -------------------------

def train_theta_rl_batch(
    num_epochs: int = 40,
    num_scenarios: int = 50,
    base_seed: int = 3000,
    lr: float = 1e-4,
    tau: float = 0.1,
    lambda_reg: float = 1e-2,
) -> tuple[np.ndarray, list[float]]:
    """
    Batch 版本 REINFORCE（從 heuristic θ 開始）：

      - theta_0 = get_default_theta()
      - 每個 epoch：
          * 對固定的一批 seeds（num_scenarios 個）各跑一次
          * 累積 reward-weighted gradient
          * 加上 L2 regularization: -lambda_reg * (theta - theta_0)
          * 更新 theta

    回傳：
      theta           : 訓練後的參數
      reward_history  : 每個 epoch 的平均 reward（>0 表示優於 priority baseline）
    """
    theta0 = get_default_theta().astype(float)
    theta = theta0.copy()

    seeds = [base_seed + i for i in range(num_scenarios)]
    reward_history: list[float] = []

    for epoch in range(num_epochs):
        epoch_grad = np.zeros_like(theta)
        epoch_rewards: list[float] = []

        for seed in seeds:
            env = NetworkEnv(seed=seed)
            state = env.init_random_state()
            candidate_paths = build_all_candidate_paths(state)

            reward, info = evaluate_with_theta(state, candidate_paths, theta, tau=tau)
            epoch_rewards.append(reward)

            grad_s = np.zeros_like(theta)
            decisions = info["decisions"]

            for (phi_arr, probs, chosen_idx) in decisions:
                expected_phi = (probs[:, None] * phi_arr).sum(axis=0)
                grad_log_pi = phi_arr[chosen_idx] - expected_phi
                grad_s += grad_log_pi

            if len(decisions) > 0:
                grad_s /= len(decisions)

            epoch_grad += reward * grad_s

        epoch_grad /= len(seeds)

        # L2 regularization：鼓勵 theta 接近 theta0
        epoch_grad -= lambda_reg * (theta - theta0)

        theta += lr * epoch_grad
        theta = np.clip(theta, -10.0, 10.0)

        mean_reward = float(np.mean(epoch_rewards))
        reward_history.append(mean_reward)

        print(
            f"[Epoch {epoch+1}/{num_epochs}] "
            f"mean reward (WSU_RL - WSU_pri) = {mean_reward:.3f}, "
            f"theta = {theta}"
        )

    return theta, reward_history


def plot_reward_curve(reward_history: list[float]) -> None:
    rewards = np.array(reward_history)
    epochs = np.arange(1, len(rewards) + 1)

    plt.figure()
    plt.plot(epochs, rewards, marker="o")
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("Epoch")
    plt.ylabel("Mean reward (WSU_RL - WSU_priority)")
    plt.title("Batch REINFORCE (fine-tune from heuristic θ)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    num_epochs = 40
    num_scenarios = 50
    base_seed = 100
    lr = 1e-4
    tau = 0.1
    lambda_reg = 1e-2

    print("=== Start batch RL training from heuristic theta ===")
    theta_trained, reward_history = train_theta_rl_batch(
        num_epochs=num_epochs,
        num_scenarios=num_scenarios,
        base_seed=base_seed,
        lr=lr,
        tau=tau,
        lambda_reg=lambda_reg,
    )

    print("\n=== Training finished ===")
    print("Final theta:", theta_trained)

    try:
        np.save("ai/learned_theta_finetune.npy", theta_trained)
        print("Saved learned theta to ai/learned_theta_finetune.npy")
    except Exception as e:
        print("Warning: failed to save theta:", e)

    plot_reward_curve(reward_history)


if __name__ == "__main__":
    main()
