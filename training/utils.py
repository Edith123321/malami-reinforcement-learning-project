"""
Malami - Shared Training Utilities

"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from typing import List, Dict, Optional, Tuple
import csv


# ─── Colour scheme ────────────────────────────────────────────────────────────
ALGO_COLORS = {
    "DQN":       "#3B82F6",
    "REINFORCE": "#F59E0B",
    "PPO":       "#10B981",
    "A2C":       "#A78BFA",
}

plt.rcParams.update({
    "figure.facecolor": "#0A0E1A",
    "axes.facecolor":   "#12182C",
    "axes.edgecolor":   "#2D3A5E",
    "axes.labelcolor":  "#9CA3AF",
    "xtick.color":      "#6B7280",
    "ytick.color":      "#6B7280",
    "text.color":       "#E5E7EB",
    "grid.color":       "#1E2844",
    "grid.linestyle":   "--",
    "grid.linewidth":   0.6,
    "legend.facecolor": "#12182C",
    "legend.edgecolor": "#2D3A5E",
})


# ─── Reward Logger ─────────────────────────────────────────────────────────────

class RewardLogger:
    """Stores per-episode rewards for later plotting."""
    def __init__(self, algo_name: str, save_dir: str = "results"):
        self.algo_name    = algo_name
        self.save_dir     = save_dir
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int]   = []
        self.losses:          List[float] = []
        self.entropies:       List[float] = []
        os.makedirs(save_dir, exist_ok=True)

    def log(self, reward: float, length: int = 0):
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)

    def log_loss(self, loss: float):
        self.losses.append(loss)

    def log_entropy(self, entropy: float):
        self.entropies.append(entropy)

    def save_csv(self):
        path = os.path.join(self.save_dir, f"{self.algo_name}_rewards.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["episode", "reward", "length"])
            for i, (r, l) in enumerate(zip(self.episode_rewards, self.episode_lengths)):
                w.writerow([i, r, l])


def smooth(arr: List[float], window: int = 20) -> np.ndarray:
    a = np.array(arr, dtype=float)
    if len(a) < window:
        return a
    kernel = np.ones(window) / window
    return np.convolve(a, kernel, mode="valid")


# ─── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_model(model, env_fn, n_episodes: int = 30, deterministic: bool = True) -> Dict:
    """Run n_episodes with the model and collect metrics."""
    rewards, topics, lengths = [], [], []
    for _ in range(n_episodes):
        env  = env_fn()
        obs, _ = env.reset()
        done = False
        ep_r = 0.0
        ep_l = 0
        while not done:
            act, _ = model.predict(obs, deterministic=deterministic)
            obs, r, terminated, truncated, info = env.step(int(act))
            ep_r += r
            ep_l += 1
            done  = terminated or truncated
        rewards.append(ep_r)
        topics.append(info.get("topics_completed", 0))
        lengths.append(ep_l)
        env.close()
    return {
        "mean_reward":  float(np.mean(rewards)),
        "std_reward":   float(np.std(rewards)),
        "mean_topics":  float(np.mean(topics)),
        "mean_length":  float(np.mean(lengths)),
        "max_reward":   float(np.max(rewards)),
        "min_reward":   float(np.min(rewards)),
        "rewards":      rewards,
        "topics":       topics,
    }


# ─── Plotting Functions ─────────────────────────────────────────────────────────

def plot_cumulative_rewards(loggers: Dict[str, RewardLogger], save_path: str = "plots/cumulative_rewards.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Cumulative Reward Curves – Malami RL", fontsize=16, fontweight="bold", y=1.01)

    for ax, (name, logger) in zip(axes.flatten(), loggers.items()):
        color = ALGO_COLORS.get(name, "#ffffff")
        eps   = np.arange(len(logger.episode_rewards))
        raw   = np.array(logger.episode_rewards)
        sm    = smooth(raw, window=30)

        ax.plot(eps, raw, alpha=0.25, color=color, linewidth=0.8)
        if len(sm) > 0:
            sm_eps = eps[len(eps) - len(sm):]
            ax.plot(sm_eps, sm, color=color, linewidth=2.2, label="Smoothed (30ep)")
        ax.fill_between(eps, raw, alpha=0.08, color=color)
        ax.set_title(name, color=color, fontsize=13, fontweight="bold")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Episode Reward")
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Plot] Saved: {save_path}")


def plot_dqn_loss(losses: List[float], save_path: str = "plots/dqn_loss.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    color = ALGO_COLORS["DQN"]
    sm    = smooth(losses, window=50)
    ax.plot(losses, alpha=0.2, color=color, linewidth=0.8, label="Raw loss")
    if len(sm) > 0:
        ax.plot(range(len(losses) - len(sm), len(losses)), sm, color=color, linewidth=2, label="Smoothed")
    ax.set_title("DQN Objective (TD Loss)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Plot] Saved: {save_path}")


def plot_policy_entropy(entropy_data: Dict[str, List[float]], save_path: str = "plots/policy_entropy.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, entropies in entropy_data.items():
        color = ALGO_COLORS.get(name, "#aaa")
        sm    = smooth(entropies, window=20)
        ax.plot(entropies, alpha=0.2, color=color, linewidth=0.8)
        if len(sm) > 0:
            ax.plot(range(len(entropies) - len(sm), len(entropies)), sm,
                    color=color, linewidth=2, label=name)
    ax.set_title("Policy Entropy – Exploration Dynamics", fontsize=14, fontweight="bold")
    ax.set_xlabel("Update Step")
    ax.set_ylabel("Entropy")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Plot] Saved: {save_path}")


def plot_convergence(convergence_data: Dict[str, List[int]], threshold: float, save_path: str = "plots/convergence.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, episodes_to_converge in convergence_data.items():
        color = ALGO_COLORS.get(name, "#aaa")
        ax.plot(episodes_to_converge, color=color, linewidth=2, marker="o", markersize=4, label=name)
    ax.set_title(f"Convergence Speed (episodes to reach reward ≥ {threshold})", fontsize=13, fontweight="bold")
    ax.set_xlabel("Run Index")
    ax.set_ylabel("Episodes to Convergence")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Plot] Saved: {save_path}")


def plot_generalisation(results: Dict[str, Dict], save_path: str = "plots/generalisation.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    algos  = list(results.keys())
    means  = [results[a]["mean_reward"] for a in algos]
    stds   = [results[a]["std_reward"]  for a in algos]
    topics = [results[a]["mean_topics"] for a in algos]
    colors = [ALGO_COLORS.get(a, "#888") for a in algos]

    # Bar chart – mean reward
    bars = axes[0].bar(algos, means, yerr=stds, color=colors, alpha=0.8,
                       error_kw={"ecolor": "white", "capsize": 5})
    axes[0].set_title("Generalisation: Mean Reward\n(unseen student profiles)", fontweight="bold")
    axes[0].set_ylabel("Mean Episode Reward")
    axes[0].grid(True, axis="y")
    for bar, m in zip(bars, means):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f"{m:.1f}", ha="center", va="bottom", fontsize=11)

    # Box plot – topics completed
    topic_arrays = [results[a]["topics"] for a in algos]
    bp = axes[1].boxplot(topic_arrays, patch_artist=True, notch=True,
                         medianprops={"color": "white", "linewidth": 2})
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)
    axes[1].set_xticklabels(algos)
    axes[1].set_title("Generalisation: Topics Completed\n(distribution across episodes)", fontweight="bold")
    axes[1].set_ylabel("Topics Completed (out of 6)")
    axes[1].grid(True, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Plot] Saved: {save_path}")


def plot_hyperparameter_heatmap(hp_df_data: List[Dict], algo: str, metric: str = "mean_reward",
                                 save_path_prefix: str = "plots/hp_heatmap"):
    """Scatter-style hyperparameter sensitivity plot."""
    os.makedirs(os.path.dirname(save_path_prefix), exist_ok=True)
    if not hp_df_data:
        return
    metrics = [row[metric] for row in hp_df_data]
    lrs     = [row.get("learning_rate", 0) for row in hp_df_data]
    gammas  = [row.get("gamma", 0) for row in hp_df_data]

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(lrs, gammas, c=metrics, cmap="RdYlGn", s=200, alpha=0.85, edgecolors="white", linewidths=0.5)
    plt.colorbar(sc, ax=ax, label=metric)
    ax.set_title(f"{algo} Hyperparameter Sensitivity ({metric})", fontweight="bold")
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Gamma (Discount Factor)")
    ax.set_xscale("log")
    ax.grid(True)
    for i, row in enumerate(hp_df_data):
        ax.annotate(f"#{i+1}", (lrs[i], gammas[i]), fontsize=7, ha="center",
                    color="white", alpha=0.8)
    plt.tight_layout()
    path = f"{save_path_prefix}_{algo}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Plot] Saved: {path}")


def save_hyperparameter_table(rows: List[Dict], algo: str, save_dir: str = "results"):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{algo}_hyperparameter_table.csv")
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"[Table] Saved: {path}")


def print_hyperparameter_table(rows: List[Dict], algo: str):
    print(f"\n{'='*80}")
    print(f"  {algo} – Hyperparameter Tuning Results")
    print(f"{'='*80}")
    if not rows:
        return
    keys = list(rows[0].keys())
    widths = {k: max(len(k), max(len(str(r.get(k, ""))) for r in rows)) + 2 for k in keys}
    header = " | ".join(k.ljust(widths[k]) for k in keys)
    print(header)
    print("-" * len(header))
    for row in rows:
        print(" | ".join(str(row.get(k, "")).ljust(widths[k]) for k in keys))
    print()
