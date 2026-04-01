"""
Malami - Policy Gradient Training Script
==========================================
Trains REINFORCE (custom), PPO, and A2C using Stable-Baselines3.
Each algorithm undergoes 10 hyperparameter runs.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import explained_variance

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.custom_env import MalamiEnv
from training.utils import (
    RewardLogger, evaluate_model,
    save_hyperparameter_table, print_hyperparameter_table
)

MODELS_DIR  = "models/pg"
RESULTS_DIR = "results"
PLOTS_DIR   = "plots"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

TOTAL_TIMESTEPS = 150_000
EVAL_FREQ       = 10_000
N_EVAL_EPISODES = 20


def make_env(seed: int = 0):
    def _init():
        return Monitor(MalamiEnv(seed=seed))
    return _init


# ═══════════════════════════════════════════════════════════════════════════════
#  REINFORCE (Monte Carlo Policy Gradient)  – custom PyTorch implementation
# ═══════════════════════════════════════════════════════════════════════════════

class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, 128),     nn.ReLU(),
            nn.Linear(128, act_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(x), dim=-1)


class REINFORCEAgent:
    """
    Vanilla REINFORCE with optional entropy regularisation and baseline.
    """
    def __init__(self, obs_dim: int, act_dim: int, lr: float = 3e-4,
                 gamma: float = 0.99, entropy_coef: float = 0.01,
                 use_baseline: bool = True):
        self.gamma       = gamma
        self.entropy_coef = entropy_coef
        self.policy      = PolicyNet(obs_dim, act_dim)
        self.optimizer   = optim.Adam(self.policy.parameters(), lr=lr)
        self.use_baseline = use_baseline

    def select_action(self, obs: np.ndarray) -> tuple:
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        probs = self.policy(obs_t)
        dist  = Categorical(probs)
        action = dist.sample()
        return int(action.item()), dist.log_prob(action), dist.entropy()

    def update(self, episode_rewards, episode_log_probs, episode_entropies) -> dict:
        R = 0.0
        returns = []
        for r in reversed(episode_rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns_t = torch.FloatTensor(returns)
        if self.use_baseline:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        policy_loss = 0.0
        entropy_loss = 0.0
        for log_p, ret, ent in zip(episode_log_probs, returns_t, episode_entropies):
            policy_loss  -= log_p * ret
            entropy_loss -= ent

        total_loss = policy_loss + self.entropy_coef * entropy_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()
        return {
            "policy_loss": float(policy_loss.item()),
            "entropy":     float(entropy_loss.item() / max(len(episode_entropies), 1)),
        }

    def save(self, path: str):
        torch.save({"policy_state": self.policy.state_dict()}, path)

    @classmethod
    def load(cls, path: str, obs_dim: int, act_dim: int):
        agent = cls(obs_dim, act_dim)
        ckpt  = torch.load(path, map_location="cpu")
        agent.policy.load_state_dict(ckpt["policy_state"])
        return agent

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        probs = self.policy(obs_t)
        if deterministic:
            return int(torch.argmax(probs).item()), None
        dist = Categorical(probs)
        return int(dist.sample().item()), None


REINFORCE_HYPERPARAMS = [
    dict(lr=3e-4,  gamma=0.99,  entropy_coef=0.01,  episodes_per_update=1,  use_baseline=True),
    dict(lr=1e-3,  gamma=0.99,  entropy_coef=0.01,  episodes_per_update=1,  use_baseline=True),
    dict(lr=1e-4,  gamma=0.99,  entropy_coef=0.01,  episodes_per_update=1,  use_baseline=True),
    dict(lr=3e-4,  gamma=0.95,  entropy_coef=0.01,  episodes_per_update=1,  use_baseline=True),
    dict(lr=3e-4,  gamma=0.999, entropy_coef=0.01,  episodes_per_update=1,  use_baseline=True),
    dict(lr=3e-4,  gamma=0.99,  entropy_coef=0.05,  episodes_per_update=1,  use_baseline=True),
    dict(lr=3e-4,  gamma=0.99,  entropy_coef=0.001, episodes_per_update=1,  use_baseline=True),
    dict(lr=3e-4,  gamma=0.99,  entropy_coef=0.02,  episodes_per_update=5,  use_baseline=True),
    dict(lr=3e-4,  gamma=0.99,  entropy_coef=0.01,  episodes_per_update=1,  use_baseline=False),
    dict(lr=5e-4,  gamma=0.99,  entropy_coef=0.03,  episodes_per_update=3,  use_baseline=True),
]


def run_reinforce_hyperparameter_search() -> dict:
    from environment.custom_env import MalamiEnv, NUM_ACTIONS
    obs_dim  = MalamiEnv().observation_space.shape[0]
    act_dim  = NUM_ACTIONS
    results  = []
    best_score = -np.inf
    best_agent = None
    best_entropies = []

    for run_idx, params in enumerate(REINFORCE_HYPERPARAMS):
        print(f"\n[REINFORCE] Run {run_idx+1}/10 | lr={params['lr']} "
              f"gamma={params['gamma']} ent={params['entropy_coef']} "
              f"baseline={params['use_baseline']}")

        agent = REINFORCEAgent(
            obs_dim, act_dim,
            lr=params["lr"],
            gamma=params["gamma"],
            entropy_coef=params["entropy_coef"],
            use_baseline=params["use_baseline"],
        )
        env = MalamiEnv(seed=run_idx)
        episode_rewards_all = []
        entropies = []
        total_steps = 0
        N_STEPS_TARGET = 100_000

        while total_steps < N_STEPS_TARGET:
            obs, _ = env.reset()
            ep_rewards, ep_lp, ep_ent = [], [], []
            done = False
            while not done:
                act, lp, ent = agent.select_action(obs)
                obs, r, term, trunc, _ = env.step(act)
                ep_rewards.append(r)
                ep_lp.append(lp)
                ep_ent.append(ent)
                total_steps += 1
                done = term or trunc
            for _ in range(params["episodes_per_update"]):
                pass
            stats = agent.update(ep_rewards, ep_lp, ep_ent)
            episode_rewards_all.append(sum(ep_rewards))
            entropies.append(stats["entropy"])

        # Evaluate
        eval_rewards, eval_topics = [], []
        for _ in range(30):
            obs, _ = env.reset()
            done = False; ep_r = 0.0
            while not done:
                act, _ = agent.predict(obs, deterministic=True)
                obs, r, term, trunc, info = env.step(act)
                ep_r += r; done = term or trunc
            eval_rewards.append(ep_r)
            eval_topics.append(info.get("topics_completed", 0))

        mean_r = float(np.mean(eval_rewards))
        row = {
            "run":             run_idx + 1,
            "learning_rate":   params["lr"],
            "gamma":           params["gamma"],
            "entropy_coef":    params["entropy_coef"],
            "episodes/update": params["episodes_per_update"],
            "use_baseline":    params["use_baseline"],
            "mean_reward":     round(mean_r, 2),
            "std_reward":      round(float(np.std(eval_rewards)), 2),
            "mean_topics":     round(float(np.mean(eval_topics)), 2),
        }
        results.append(row)

        if mean_r > best_score:
            best_score     = mean_r
            best_agent     = agent
            best_entropies = list(entropies)
            print(f"  ✓ New best: {mean_r:.2f}")
        else:
            print(f"  → Mean reward: {mean_r:.2f}")
        env.close()

    if best_agent is not None:
        best_agent.save(os.path.join(MODELS_DIR, "reinforce_best.pt"))

    print_hyperparameter_table(results, "REINFORCE")
    save_hyperparameter_table(results, "REINFORCE", RESULTS_DIR)
    return {"results": results, "best_score": best_score, "entropies": best_entropies}


# ═══════════════════════════════════════════════════════════════════════════════
#  PPO
# ═══════════════════════════════════════════════════════════════════════════════

PPO_HYPERPARAMS = [
    dict(lr=3e-4, n_steps=2048, batch_size=64,  n_epochs=10, clip_range=0.2,  gae_lambda=0.95, ent_coef=0.01),
    dict(lr=1e-3, n_steps=2048, batch_size=64,  n_epochs=10, clip_range=0.2,  gae_lambda=0.95, ent_coef=0.01),
    dict(lr=1e-4, n_steps=2048, batch_size=64,  n_epochs=10, clip_range=0.2,  gae_lambda=0.95, ent_coef=0.01),
    dict(lr=3e-4, n_steps=1024, batch_size=64,  n_epochs=10, clip_range=0.2,  gae_lambda=0.95, ent_coef=0.01),
    dict(lr=3e-4, n_steps=512,  batch_size=64,  n_epochs=10, clip_range=0.2,  gae_lambda=0.95, ent_coef=0.01),
    dict(lr=3e-4, n_steps=2048, batch_size=128, n_epochs=10, clip_range=0.3,  gae_lambda=0.95, ent_coef=0.01),
    dict(lr=3e-4, n_steps=2048, batch_size=64,  n_epochs=20, clip_range=0.2,  gae_lambda=0.95, ent_coef=0.01),
    dict(lr=3e-4, n_steps=2048, batch_size=64,  n_epochs=10, clip_range=0.1,  gae_lambda=0.95, ent_coef=0.01),
    dict(lr=3e-4, n_steps=2048, batch_size=64,  n_epochs=10, clip_range=0.2,  gae_lambda=0.90, ent_coef=0.05),
    dict(lr=5e-4, n_steps=4096, batch_size=128, n_epochs=10, clip_range=0.25, gae_lambda=0.98, ent_coef=0.02),
]


def run_ppo_hyperparameter_search() -> dict:
    results    = []
    best_score = -np.inf
    best_path  = None
    best_entropies = []

    for run_idx, params in enumerate(PPO_HYPERPARAMS):
        print(f"\n[PPO] Run {run_idx+1}/10 | lr={params['lr']} "
              f"n_steps={params['n_steps']} clip={params['clip_range']} "
              f"batch={params['batch_size']}")

        env      = DummyVecEnv([make_env(seed=run_idx)])
        eval_env = DummyVecEnv([make_env(seed=run_idx + 200)])

        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=params["lr"],
            n_steps=params["n_steps"],
            batch_size=params["batch_size"],
            n_epochs=params["n_epochs"],
            clip_range=params["clip_range"],
            gae_lambda=params["gae_lambda"],
            ent_coef=params["ent_coef"],
            policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
            verbose=0,
        )

        entropies_this_run = []

        class EntropyCallback(EvalCallback):
            def _on_step(self) -> bool:
                if hasattr(self.model, "logger"):
                    lv = self.model.logger.name_to_value
                    if "train/entropy_loss" in lv:
                        entropies_this_run.append(abs(lv["train/entropy_loss"]))
                return super()._on_step()

        cb = EntropyCallback(
            eval_env, eval_freq=EVAL_FREQ,
            n_eval_episodes=N_EVAL_EPISODES, deterministic=True,
            render=False, verbose=0,
        )
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=cb, progress_bar=False)

        eval_result = evaluate_model(model, lambda: MalamiEnv(), n_episodes=30)
        mean_r = eval_result["mean_reward"]

        row = {
            "run":           run_idx + 1,
            "learning_rate": params["lr"],
            "n_steps":       params["n_steps"],
            "batch_size":    params["batch_size"],
            "n_epochs":      params["n_epochs"],
            "clip_range":    params["clip_range"],
            "gae_lambda":    params["gae_lambda"],
            "ent_coef":      params["ent_coef"],
            "mean_reward":   round(mean_r, 2),
            "std_reward":    round(eval_result["std_reward"], 2),
            "mean_topics":   round(eval_result["mean_topics"], 2),
        }
        results.append(row)

        if mean_r > best_score:
            best_score = mean_r
            best_path  = os.path.join(MODELS_DIR, "ppo_best")
            model.save(best_path)
            best_entropies = list(entropies_this_run)
            print(f"  ✓ New best: {mean_r:.2f} → saved")
        else:
            print(f"  → Mean reward: {mean_r:.2f}")
        env.close(); eval_env.close()

    print_hyperparameter_table(results, "PPO")
    save_hyperparameter_table(results, "PPO", RESULTS_DIR)
    return {"results": results, "best_score": best_score, "entropies": best_entropies,
            "best_path": (best_path + ".zip") if best_path else None}


# ═══════════════════════════════════════════════════════════════════════════════
#  A2C
# ═══════════════════════════════════════════════════════════════════════════════

A2C_HYPERPARAMS = [
    dict(lr=7e-4, n_steps=5,   gamma=0.99,  ent_coef=0.01,  vf_coef=0.5,  max_grad_norm=0.5),
    dict(lr=1e-3, n_steps=5,   gamma=0.99,  ent_coef=0.01,  vf_coef=0.5,  max_grad_norm=0.5),
    dict(lr=3e-4, n_steps=5,   gamma=0.99,  ent_coef=0.01,  vf_coef=0.5,  max_grad_norm=0.5),
    dict(lr=7e-4, n_steps=20,  gamma=0.99,  ent_coef=0.01,  vf_coef=0.5,  max_grad_norm=0.5),
    dict(lr=7e-4, n_steps=5,   gamma=0.95,  ent_coef=0.01,  vf_coef=0.5,  max_grad_norm=0.5),
    dict(lr=7e-4, n_steps=5,   gamma=0.99,  ent_coef=0.05,  vf_coef=0.5,  max_grad_norm=0.5),
    dict(lr=7e-4, n_steps=5,   gamma=0.99,  ent_coef=0.001, vf_coef=0.5,  max_grad_norm=0.5),
    dict(lr=7e-4, n_steps=5,   gamma=0.99,  ent_coef=0.01,  vf_coef=1.0,  max_grad_norm=0.5),
    dict(lr=7e-4, n_steps=5,   gamma=0.99,  ent_coef=0.01,  vf_coef=0.5,  max_grad_norm=1.0),
    dict(lr=5e-4, n_steps=10,  gamma=0.995, ent_coef=0.02,  vf_coef=0.6,  max_grad_norm=0.5),
]


def run_a2c_hyperparameter_search() -> dict:
    results    = []
    best_score = -np.inf
    best_path  = None
    best_entropies = []

    for run_idx, params in enumerate(A2C_HYPERPARAMS):
        print(f"\n[A2C] Run {run_idx+1}/10 | lr={params['lr']} "
              f"n_steps={params['n_steps']} gamma={params['gamma']} "
              f"ent={params['ent_coef']}")

        env      = DummyVecEnv([make_env(seed=run_idx + 300)])
        eval_env = DummyVecEnv([make_env(seed=run_idx + 400)])

        model = A2C(
            policy="MlpPolicy",
            env=env,
            learning_rate=params["lr"],
            n_steps=params["n_steps"],
            gamma=params["gamma"],
            ent_coef=params["ent_coef"],
            vf_coef=params["vf_coef"],
            max_grad_norm=params["max_grad_norm"],
            policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
            verbose=0,
        )

        entropies_this_run = []

        class A2CEntropyCallback(EvalCallback):
            def _on_step(self) -> bool:
                if hasattr(self.model, "logger"):
                    lv = self.model.logger.name_to_value
                    if "train/entropy_loss" in lv:
                        entropies_this_run.append(abs(lv["train/entropy_loss"]))
                return super()._on_step()

        cb = A2CEntropyCallback(
            eval_env, eval_freq=EVAL_FREQ,
            n_eval_episodes=N_EVAL_EPISODES, deterministic=True,
            render=False, verbose=0,
        )
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=cb, progress_bar=False)

        eval_result = evaluate_model(model, lambda: MalamiEnv(), n_episodes=30)
        mean_r = eval_result["mean_reward"]

        row = {
            "run":           run_idx + 1,
            "learning_rate": params["lr"],
            "n_steps":       params["n_steps"],
            "gamma":         params["gamma"],
            "ent_coef":      params["ent_coef"],
            "vf_coef":       params["vf_coef"],
            "max_grad_norm": params["max_grad_norm"],
            "mean_reward":   round(mean_r, 2),
            "std_reward":    round(eval_result["std_reward"], 2),
            "mean_topics":   round(eval_result["mean_topics"], 2),
        }
        results.append(row)

        if mean_r > best_score:
            best_score = mean_r
            best_path  = os.path.join(MODELS_DIR, "a2c_best")
            model.save(best_path)
            best_entropies = list(entropies_this_run)
            print(f"  ✓ New best: {mean_r:.2f} → saved")
        else:
            print(f"  → Mean reward: {mean_r:.2f}")
        env.close(); eval_env.close()

    print_hyperparameter_table(results, "A2C")
    save_hyperparameter_table(results, "A2C", RESULTS_DIR)
    return {"results": results, "best_score": best_score, "entropies": best_entropies,
            "best_path": (best_path + ".zip") if best_path else None}


if __name__ == "__main__":
    run_reinforce_hyperparameter_search()
    run_ppo_hyperparameter_search()
    run_a2c_hyperparameter_search()
