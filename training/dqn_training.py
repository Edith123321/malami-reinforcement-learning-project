"""
Malami - DQN Training Script
==============================
Value-based RL using Deep Q-Network from Stable-Baselines3.
Performs 10 hyperparameter runs and saves best model.
"""

import os
import sys
import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.custom_env import MalamiEnv
from training.utils import (
    RewardLogger, evaluate_model, plot_dqn_loss,
    save_hyperparameter_table, print_hyperparameter_table
)

MODELS_DIR  = "models/dqn"
RESULTS_DIR = "results"
PLOTS_DIR   = "plots"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

TOTAL_TIMESTEPS = 150_000
EVAL_FREQ       = 10_000
N_EVAL_EPISODES = 20


class LossCallback(BaseCallback):
    """Captures TD loss during DQN training."""
    def __init__(self, logger: RewardLogger):
        super().__init__()
        self._rl_logger = logger

    def _on_step(self) -> bool:
        if hasattr(self.model, "logger") and self.model.logger is not None:
            logs = self.model.logger.name_to_value
            if "train/loss" in logs:
                self._rl_logger.log_loss(logs["train/loss"])
        return True


# ─── Hyperparameter grid (10 runs) ─────────────────────────────────────────────
DQN_HYPERPARAMS = [
    # Run 1 – baseline
    dict(learning_rate=1e-3,  gamma=0.99, buffer_size=100_000, batch_size=64,
         exploration_fraction=0.3, exploration_final_eps=0.05, train_freq=4,
         target_update_interval=1000, gradient_steps=1, tau=1.0),
    # Run 2 – higher LR
    dict(learning_rate=5e-3,  gamma=0.99, buffer_size=100_000, batch_size=64,
         exploration_fraction=0.3, exploration_final_eps=0.05, train_freq=4,
         target_update_interval=1000, gradient_steps=1, tau=1.0),
    # Run 3 – low LR
    dict(learning_rate=1e-4,  gamma=0.99, buffer_size=100_000, batch_size=64,
         exploration_fraction=0.4, exploration_final_eps=0.02, train_freq=4,
         target_update_interval=1000, gradient_steps=1, tau=1.0),
    # Run 4 – low gamma
    dict(learning_rate=1e-3,  gamma=0.90, buffer_size=100_000, batch_size=64,
         exploration_fraction=0.3, exploration_final_eps=0.05, train_freq=4,
         target_update_interval=1000, gradient_steps=1, tau=1.0),
    # Run 5 – high gamma
    dict(learning_rate=1e-3,  gamma=0.999, buffer_size=100_000, batch_size=128,
         exploration_fraction=0.25, exploration_final_eps=0.05, train_freq=4,
         target_update_interval=500, gradient_steps=2, tau=0.9),
    # Run 6 – small buffer
    dict(learning_rate=1e-3,  gamma=0.99, buffer_size=20_000,  batch_size=64,
         exploration_fraction=0.35, exploration_final_eps=0.05, train_freq=4,
         target_update_interval=1000, gradient_steps=1, tau=1.0),
    # Run 7 – large batch
    dict(learning_rate=5e-4,  gamma=0.99, buffer_size=100_000, batch_size=256,
         exploration_fraction=0.3, exploration_final_eps=0.02, train_freq=8,
         target_update_interval=2000, gradient_steps=2, tau=1.0),
    # Run 8 – aggressive exploration
    dict(learning_rate=1e-3,  gamma=0.99, buffer_size=100_000, batch_size=64,
         exploration_fraction=0.6, exploration_final_eps=0.1,  train_freq=4,
         target_update_interval=1000, gradient_steps=1, tau=1.0),
    # Run 9 – soft update tau
    dict(learning_rate=1e-3,  gamma=0.99, buffer_size=100_000, batch_size=64,
         exploration_fraction=0.3, exploration_final_eps=0.05, train_freq=4,
         target_update_interval=500,  gradient_steps=1, tau=0.01),
    # Run 10 – more gradient steps
    dict(learning_rate=3e-4,  gamma=0.995, buffer_size=100_000, batch_size=128,
         exploration_fraction=0.2, exploration_final_eps=0.02, train_freq=4,
         target_update_interval=1000, gradient_steps=4, tau=1.0),
]


def make_env(seed: int = 0):
    def _init():
        env = MalamiEnv(seed=seed)
        return Monitor(env)
    return _init


def run_dqn_hyperparameter_search(verbose: bool = True) -> dict:
    results = []
    best_score = -np.inf
    best_model_path = None
    best_losses = []

    for run_idx, params in enumerate(DQN_HYPERPARAMS):
        print(f"\n[DQN] Run {run_idx+1}/10 | lr={params['learning_rate']} "
              f"gamma={params['gamma']} buf={params['buffer_size']} "
              f"batch={params['batch_size']} eps_frac={params['exploration_fraction']}")

        env   = DummyVecEnv([make_env(seed=run_idx)])
        eval_env = DummyVecEnv([make_env(seed=run_idx + 100)])
        logger   = RewardLogger("DQN", save_dir=RESULTS_DIR)

        model = DQN(
            policy="MlpPolicy",
            env=env,
            learning_rate=params["learning_rate"],
            gamma=params["gamma"],
            buffer_size=params["buffer_size"],
            batch_size=params["batch_size"],
            exploration_fraction=params["exploration_fraction"],
            exploration_final_eps=params["exploration_final_eps"],
            train_freq=params["train_freq"],
            target_update_interval=params["target_update_interval"],
            gradient_steps=params["gradient_steps"],
            tau=params["tau"],
            policy_kwargs=dict(net_arch=[256, 256, 128]),
            verbose=0,
            tensorboard_log=None,
        )

        loss_cb = LossCallback(logger)
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=None,
            log_path=None,
            eval_freq=EVAL_FREQ,
            n_eval_episodes=N_EVAL_EPISODES,
            deterministic=True,
            render=False,
            verbose=0,
        )

        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[loss_cb, eval_cb], progress_bar=False)

        # Evaluate
        eval_result = evaluate_model(model, lambda: MalamiEnv(), n_episodes=30)
        mean_r = eval_result["mean_reward"]

        row = {
            "run":                    run_idx + 1,
            "learning_rate":          params["learning_rate"],
            "gamma":                  params["gamma"],
            "buffer_size":            params["buffer_size"],
            "batch_size":             params["batch_size"],
            "exploration_fraction":   params["exploration_fraction"],
            "exploration_final_eps":  params["exploration_final_eps"],
            "gradient_steps":         params["gradient_steps"],
            "tau":                    params["tau"],
            "mean_reward":            round(mean_r, 2),
            "std_reward":             round(eval_result["std_reward"], 2),
            "mean_topics_completed":  round(eval_result["mean_topics"], 2),
        }
        results.append(row)

        if mean_r > best_score:
            best_score      = mean_r
            best_model_path = os.path.join(MODELS_DIR, "best_model")
            model.save(best_model_path)
            best_losses     = list(logger.losses)
            print(f"  ✓ New best: {mean_r:.2f} → saved to {best_model_path}")
        else:
            print(f"  → Mean reward: {mean_r:.2f}")

        env.close()
        eval_env.close()

    print_hyperparameter_table(results, "DQN")
    save_hyperparameter_table(results, "DQN", RESULTS_DIR)
    if best_losses:
        plot_dqn_loss(best_losses, save_path=os.path.join(PLOTS_DIR, "dqn_loss.png"))

    return {"results": results, "best_score": best_score, "best_model_path": best_model_path + ".zip"}


if __name__ == "__main__":
    run_dqn_hyperparameter_search()
