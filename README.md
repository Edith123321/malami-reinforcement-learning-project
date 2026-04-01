# Malami – Adaptive Learning Platform (RL Summative)

> **Malami** (Hausa for *teacher*) is an AI tutor that uses Reinforcement Learning to adaptively select the optimal learning activity for each student, maximising mastery progress while maintaining engagement.

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/malami_rl_summative
cd malami_rl_summative

# 2. Install dependencies (Python 3.10+)
pip install -r requirements.txt

# 3. Train all four algorithms (takes ~20–40 min on CPU)
python main.py train

# 4. Run the best agent with GUI
python main.py play --algo ppo

# 5. Random policy demo (no training required)
python main.py random

# 6. Generalisation benchmark
python main.py evaluate
```

---

## Project Structure

```
malami_rl_summative/
├── environment/
│   ├── custom_env.py       # Custom Gymnasium environment
│   └── rendering.py        # Pygame 2D GUI visualisation
├── training/
│   ├── dqn_training.py     # DQN with 10 hyperparameter runs
│   ├── pg_training.py      # REINFORCE, PPO, A2C – each 10 runs
│   └── utils.py            # Plotting, evaluation, logging helpers
├── models/
│   ├── dqn/best_model.zip
│   └── pg/{ppo,a2c,reinforce}_best.*
├── results/                # CSV tables, random demo screenshots
├── plots/                  # All report visualisation figures
├── main.py                 # CLI entry point
├── requirements.txt
└── README.md
```

---

## Environment Design

| Component | Description |
|-----------|-------------|
| **Agent** | AI tutor selecting instructional actions |
| **Actions** | 9 actions: Video Lesson, Practice Quiz, Problem Solving, Review Summary, Adaptive Hint, New Topic, Remediation, Peer Discussion, Worked Example |
| **Observations** | 21-dim vector: topic mastery, quiz scores, engagement, attempt counts, student profile |
| **Reward** | Δmastery × 10 + quiz_score × 3 - disengagement penalty + topic completion bonus |
| **Topics** | 6 topics: Biology → Chemistry → Physics → Maths → CS → AI & ML |
| **Termination** | All topics mastered, full disengagement, or 80 steps |

---

## Algorithms

| Algorithm | Category | Key Hyperparameters |
|-----------|----------|---------------------|
| **DQN** | Value-based | lr, gamma, buffer_size, batch_size, epsilon_decay |
| **REINFORCE** | Policy Gradient (MC) | lr, gamma, entropy_coef, baseline |
| **PPO** | Policy Gradient (modern) | lr, n_steps, clip_range, gae_lambda, n_epochs |
| **A2C** | Actor-Critic | lr, n_steps, gamma, ent_coef, vf_coef |

Each algorithm is trained for **150,000 timesteps** across **10 hyperparameter combinations**.

---

## Production Integration

The trained policy can be serialised to JSON and served as a REST API:

```python
import json
from stable_baselines3 import PPO

model = PPO.load("models/pg/ppo_best.zip")

# Serve via FastAPI
from fastapi import FastAPI
app = FastAPI()

@app.post("/recommend_action")
def recommend(obs: list[float]):
    action, _ = model.predict(obs, deterministic=True)
    return {"action": int(action), "action_name": ACTION_NAMES[int(action)]}
```

---

## Results

After training, all plots are saved to `plots/`:
- `cumulative_rewards.png` – reward curves for all 4 algorithms
- `dqn_loss.png` – TD loss curve
- `policy_entropy.png` – exploration dynamics (PPO, A2C, REINFORCE)
- `convergence.png` – episodes to convergence threshold
- `generalisation.png` – performance on unseen student profiles

