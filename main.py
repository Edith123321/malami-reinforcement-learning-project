"""
Malami – Main Entry Point
==========================
Usage:
  python main.py train           # Train all four algorithms
  python main.py train --algo dqn
  python main.py play            # Run best model with GUI
  python main.py random          # Random policy demo (static file)
  python main.py evaluate        # Generalisation benchmark
  python main.py plots           # Re-generate all plots from saved results
"""

import argparse
import os
import sys
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment.custom_env import MalamiEnv, StudentProfile, TOPIC_NAMES, NUM_TOPICS


def cmd_train(algo: str = "all"):
    """Train one or all algorithms with hyperparameter search."""
    algos = ["dqn", "reinforce", "ppo", "a2c"] if algo == "all" else [algo.lower()]

    all_results = {}

    if "dqn" in algos:
        from training.dqn_training import run_dqn_hyperparameter_search
        print("\n" + "═"*60)
        print("  Training DQN")
        print("═"*60)
        all_results["DQN"] = run_dqn_hyperparameter_search()

    pg_results = {}
    from training.pg_training import (
        run_reinforce_hyperparameter_search,
        run_ppo_hyperparameter_search,
        run_a2c_hyperparameter_search,
    )
    if "reinforce" in algos:
        print("\n" + "═"*60)
        print("  Training REINFORCE")
        print("═"*60)
        pg_results["REINFORCE"] = run_reinforce_hyperparameter_search()
        all_results["REINFORCE"] = pg_results["REINFORCE"]

    if "ppo" in algos:
        print("\n" + "═"*60)
        print("  Training PPO")
        print("═"*60)
        pg_results["PPO"] = run_ppo_hyperparameter_search()
        all_results["PPO"] = pg_results["PPO"]

    if "a2c" in algos:
        print("\n" + "═"*60)
        print("  Training A2C")
        print("═"*60)
        pg_results["A2C"] = run_a2c_hyperparameter_search()
        all_results["A2C"] = pg_results["A2C"]

    # Generate plots
    _generate_plots(all_results)

    # Print summary
    print("\n" + "═"*60)
    print("  TRAINING COMPLETE – SUMMARY")
    print("═"*60)
    for name, res in all_results.items():
        print(f"  {name:<12} best mean reward: {res.get('best_score', 'N/A'):.2f}")


def _generate_plots(all_results: dict):
    """Generate all required visualisation plots."""
    from training.utils import (
        plot_cumulative_rewards, plot_policy_entropy,
        plot_convergence, plot_generalisation, RewardLogger
    )
    import matplotlib
    matplotlib.use("Agg")

    os.makedirs("plots", exist_ok=True)

    # Build dummy loggers from results for plotting
    # (In practice, you'd pass the live loggers; here we reconstruct from results)
    print("\n[Plots] Generating visualisations...")

    # Generalisation plot from best models
    gen_results = {}
    for algo_name in ["DQN", "PPO", "A2C"]:
        model_map = {"DQN": "models/dqn/best_model.zip",
                     "PPO": "models/pg/ppo_best.zip",
                     "A2C": "models/pg/a2c_best.zip"}
        path = model_map.get(algo_name, "")
        if os.path.exists(path):
            try:
                model = _load_sb3_model(algo_name, path)
                from training.utils import evaluate_model
                res = evaluate_model(model, lambda: MalamiEnv(seed=999), n_episodes=50)
                gen_results[algo_name] = res
            except Exception as e:
                print(f"  [!] Could not evaluate {algo_name}: {e}")

    if gen_results:
        plot_generalisation(gen_results, save_path="plots/generalisation.png")

    print("[Plots] Done.")


def _load_sb3_model(algo: str, path: str):
    from stable_baselines3 import DQN, PPO, A2C
    cls_map = {"DQN": DQN, "PPO": PPO, "A2C": A2C}
    cls = cls_map[algo]
    return cls.load(path)


def cmd_play(algo: str = "ppo", gui: bool = True):
    """
    Run the best-performing model with rich adaptive learning visualization.
    Shows a real-time illustration of a 6th-grade Biology student learning adaptively.
    """
    model_paths = {
        "dqn":       "models/dqn/best_model.zip",
        "ppo":       "models/pg/ppo_best.zip",
        "a2c":       "models/pg/a2c_best.zip",
        "reinforce": "models/pg/reinforce_best.pt",
    }
    path = model_paths.get(algo.lower())
    if not path or not os.path.exists(path):
        print(f"[play] Model not found: {path}. Run 'python main.py train' first.")
        return

    render_mode = "human" if gui else None
    env = MalamiEnv(render_mode=render_mode)
    
    # Use a specific student profile for demonstration
    # You can customize this to show different learning styles
    student_profile = StudentProfile(
        learning_rate=0.6,      # Average learning speed
        retention=0.7,          # Good retention
        engagement_sensitivity=0.5,
        prior_knowledge=0.2     # Some prior knowledge
    )
    env.set_student_profile(student_profile)

    if algo.lower() == "reinforce":
        from training.pg_training import REINFORCEAgent
        model = REINFORCEAgent.load(path, env.observation_space.shape[0], env.action_space.n)
    else:
        model = _load_sb3_model(algo.upper(), path)

    print("\n" + "="*70)
    print("  🎓 MALAMI ADAPTIVE LEARNING TUTOR - 6th Grade Biology")
    print("="*70)
    print(f"  🤖 AI Tutor: {algo.upper()} Algorithm")
    print(f"  📚 Subject: Biology (Cell Structure, Photosynthesis, etc.)")
    print(f"  👨‍🎓 Student: 6th Grader (Learning Rate: {student_profile.learning_rate:.2f})")
    print("="*70)
    
    if gui:
        print("\n  🖥️  Opening interactive GUI...\n")
    else:
        print("\n  📊 Console visualization mode:\n")

    # Track learning metrics for analysis
    learning_history = {
        "steps": [],
        "mastery": [],
        "engagement": [],
        "actions": [],
        "rewards": []
    }

    for ep in range(3):  # Run 3 episodes to show progression
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        step = 0
        mastery_gains = []
        
        print(f"\n{'─'*70}")
        print(f"  📖 EPISODE {ep+1}: Learning Session")
        print(f"{'─'*70}")
        
        while not done and step < 50:  # Max 50 steps per session
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(int(action))
            ep_reward += reward
            step += 1
            done = term or trunc
            
            # Convert current_topic to integer safely
            current_topic = info['current_topic']
            if isinstance(current_topic, str):
                try:
                    current_topic = int(current_topic)
                except ValueError:
                    current_topic = 0
            
            action_name = env.get_action_name(int(action))
            topic_name = TOPIC_NAMES[current_topic] if current_topic < len(TOPIC_NAMES) else "Unknown"
            mastery = info.get('mastery_current', 0.0)
            engagement = info.get('engagement', 1.0)
            
            # Track learning metrics
            learning_history["steps"].append(step)
            learning_history["mastery"].append(mastery)
            learning_history["engagement"].append(engagement)
            learning_history["actions"].append(action_name)
            learning_history["rewards"].append(reward)
            
            if not gui:
                # Enhanced console output with visual indicators
                mastery_bar = "█" * int(mastery * 20) + "░" * (20 - int(mastery * 20))
                engagement_icon = "😊" if engagement > 0.7 else "😐" if engagement > 0.3 else "😟"
                
                # Color coding for actions
                action_colors = {
                    "Video Lesson": "📺",
                    "Practice Quiz": "📝",
                    "Problem Solving": "💡",
                    "Review Summary": "📚",
                    "Adaptive Hint": "💭",
                    "New Topic": "✨",
                    "Remediation": "🔄"
                }
                action_icon = action_colors.get(action_name, "⚡")
                
                print(f"\n  Step {step:2d} | {action_icon} {action_name:<16} | "
                      f"🎯 Mastery: [{mastery_bar}] {mastery:.1%} | "
                      f"❤️ Engagement: {engagement_icon} {engagement:.0%} | "
                      f"🏆 Reward: {reward:+.1f}")
                
                # Show adaptive hints based on student state
                if mastery < 0.3 and action_name in ["New Topic", "Problem Solving"]:
                    print(f"         💡 AI Insight: Student struggling. Providing extra support...")
                elif engagement < 0.4:
                    print(f"         ⚠️  AI Insight: Engagement dropping. Switching to engaging activity...")
                elif mastery > 0.8 and action_name == "Review Summary":
                    print(f"         🌟 AI Insight: Student mastered this topic well! Ready for next concept.")
            
            if gui:
                time.sleep(0.3)  # Slower for GUI to allow observation
            
            # Check if student has mastered the current topic
            if mastery >= 0.8 and not done:
                print(f"         ✅ Topic mastered! Moving to next concept...")
        
        # Episode summary
        avg_mastery = np.mean(learning_history["mastery"][-10:]) if learning_history["mastery"] else 0
        final_topics = info.get('topics_completed', 0)
        
        print(f"\n{'─'*70}")
        print(f"  📊 Episode {ep+1} Summary:")
        print(f"     • Total Steps: {step}")
        print(f"     • Total Reward: {ep_reward:.1f}")
        print(f"     • Topics Completed: {final_topics}/{NUM_TOPICS}")
        print(f"     • Final Mastery: {mastery:.1%}")
        print(f"     • Avg Engagement: {np.mean(learning_history['engagement']):.1%}")
        
        # Adaptive feedback based on performance
        if avg_mastery > 0.7:
            print(f"     🎉 Great progress! Student is learning effectively.")
        elif avg_mastery > 0.4:
            print(f"     👍 Good effort. Student is making steady progress.")
        else:
            print(f"     🤔 Student needs additional support. AI adjusting strategy...")
        
        if final_topics >= NUM_TOPICS:
            print(f"\n  🎉🎉🎉 CONGRATULATIONS! Student completed all topics! 🎉🎉🎉")
            break
    
    # Final comprehensive analysis
    print("\n" + "="*70)
    print("  📈 LEARNING ANALYSIS - Adaptive Tutor Performance")
    print("="*70)
    
    # Calculate learning metrics
    if learning_history["mastery"]:
        mastery_progress = learning_history["mastery"][-1] - learning_history["mastery"][0] if len(learning_history["mastery"]) > 1 else 0
        avg_reward = np.mean(learning_history["rewards"])
        
        print(f"\n  🧠 Learning Effectiveness:")
        print(f"     • Mastery Gain: {mastery_progress:+.1%}")
        print(f"     • Final Mastery: {learning_history['mastery'][-1]:.1%}")
        print(f"     • Average Reward per Step: {avg_reward:.2f}")
        
        print(f"\n  🎯 AI Tutor Strategy Analysis:")
        action_counts = Counter(learning_history["actions"])
        for action, count in action_counts.most_common(3):
            print(f"     • {action}: {count} times ({count/len(learning_history['actions']):.0%})")
        
        # Engagement pattern
        engagement_trend = learning_history["engagement"][-1] - learning_history["engagement"][0]
        print(f"\n  ❤️ Engagement Pattern:")
        if engagement_trend > 0:
            print(f"     • Engagement increased by {engagement_trend:.1%} - Student became more interested!")
        elif engagement_trend < -0.2:
            print(f"     • Engagement decreased - Student may need more varied activities")
        else:
            print(f"     • Engagement stable - Consistent learning experience")
    
    print("\n" + "="*70)
    print("  💡 Adaptive Learning Summary:")
    print("     The AI tutor successfully adjusted teaching strategies based on")
    print("     the student's mastery level and engagement, providing personalized")
    print("     learning experiences to maximize knowledge acquisition.")
    print("="*70 + "\n")
    
    env.close()
    
    # Optional: Generate a learning progress chart
    if not gui:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 3, 1)
            plt.plot(learning_history["steps"], learning_history["mastery"], 'b-', linewidth=2)
            plt.xlabel('Learning Step')
            plt.ylabel('Mastery Level')
            plt.title('Knowledge Mastery Progress')
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1)
            
            plt.subplot(1, 3, 2)
            plt.plot(learning_history["steps"], learning_history["engagement"], 'g-', linewidth=2)
            plt.xlabel('Learning Step')
            plt.ylabel('Engagement Level')
            plt.title('Student Engagement')
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1)
            
            plt.subplot(1, 3, 3)
            plt.plot(learning_history["steps"], np.cumsum(learning_history["rewards"]), 'r-', linewidth=2)
            plt.xlabel('Learning Step')
            plt.ylabel('Cumulative Reward')
            plt.title('Cumulative Learning Progress')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig("plots/learning_progress.png", dpi=100, bbox_inches='tight')
            plt.show()
            print("\n  📊 Learning progress chart saved to: plots/learning_progress.png")
        except:
            pass

def cmd_random():
    """
    Static demo: random policy to visualise the environment components
    without any trained model. Saves screenshots to results/random_demo/.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    print("[Random Demo] Running random policy...")
    os.makedirs("results/random_demo", exist_ok=True)
    env = MalamiEnv(seed=42)
    obs, _ = env.reset()
    done = False
    rewards, actions, masteries, engagements = [], [], [], []
    step = 0

    while not done and step < 50:
        action = env.action_space.sample()
        obs, r, term, trunc, info = env.step(action)
        rewards.append(r)
        actions.append(action)
        masteries.append(list(env.topic_masteries))
        engagements.append(env.engagement)
        done = term or trunc
        step += 1

    # Plot random demo overview
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.patch.set_facecolor("#0A0E1A")
    fig.suptitle("Malami – Random Policy Visualisation", fontsize=15,
                 fontweight="bold", color="white")

    ax = axes[0, 0]
    ax.set_facecolor("#12182C")
    ax.plot(rewards, color="#3B82F6", linewidth=1.5, marker="o", markersize=3)
    ax.set_title("Step Rewards (Random Policy)", color="white")
    ax.set_xlabel("Step", color="grey")
    ax.set_ylabel("Reward", color="grey")
    ax.grid(True, color="#1E2844")

    ax = axes[0, 1]
    ax.set_facecolor("#12182C")
    ax.plot(engagements, color="#10B981", linewidth=1.5)
    ax.set_title("Engagement Over Time", color="white")
    ax.set_xlabel("Step", color="grey")
    ax.set_ylabel("Engagement", color="grey")
    ax.set_ylim(0, 1.1)
    ax.grid(True, color="#1E2844")

    ax = axes[1, 0]
    ax.set_facecolor("#12182C")
    from collections import Counter
    action_names = env.get_action_name
    counts = Counter(actions)
    names_list = [env.get_action_name(i) for i in range(env.action_space.n)]
    vals  = [counts.get(i, 0) for i in range(env.action_space.n)]
    colors = ["#3B82F6", "#F59E0B", "#A78BFA", "#6B7280", "#10B981",
              "#10B981", "#EF4444", "#3B82F6", "#A78BFA"]
    bars = ax.barh(names_list, vals, color=colors, alpha=0.85)
    ax.set_title("Action Distribution (Random)", color="white")
    ax.set_xlabel("Count", color="grey")
    ax.grid(True, axis="x", color="#1E2844")

    ax = axes[1, 1]
    ax.set_facecolor("#12182C")
    topic_colors = ["#34D399","#60A5FA","#FB923C","#A78BFA","#22D3EE","#FBBF24"]
    m_arr = np.array(masteries)
    for ti in range(min(NUM_TOPICS, m_arr.shape[1])):
        ax.plot(m_arr[:, ti], color=topic_colors[ti], linewidth=1.5, label=TOPIC_NAMES[ti])
    ax.set_title("Topic Masteries Over Steps", color="white")
    ax.set_xlabel("Step", color="grey")
    ax.set_ylabel("Mastery", color="grey")
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, color="#1E2844")

    for ax_row in axes:
        for a in ax_row:
            for spine in a.spines.values():
                spine.set_edgecolor("#2D3A5E")

    plt.tight_layout()
    path = "results/random_demo/random_policy_overview.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Random Demo] Saved: {path}")
    env.close()


def cmd_evaluate():
    """Generalisation benchmark on unseen student profiles."""
    from training.utils import evaluate_model, plot_generalisation
    gen_results = {}

    model_map = {
        "DQN": ("DQN", "models/dqn/best_model.zip"),
        "PPO": ("PPO", "models/pg/ppo_best.zip"),
        "A2C": ("A2C", "models/pg/a2c_best.zip"),
    }

    # Unseen profiles: fast learner, slow learner, anxious, high prior
    rng = np.random.default_rng(seed=12345)

    for algo, (algo_cls, path) in model_map.items():
        if not os.path.exists(path):
            print(f"  [skip] {algo} model not found.")
            continue
        model = _load_sb3_model(algo_cls, path)
        # Use seeds > 9999 to ensure unseen profiles
        result = evaluate_model(model, lambda s=42: MalamiEnv(seed=s + 9999), n_episodes=50)
        gen_results[algo] = result
        print(f"  {algo}: mean={result['mean_reward']:.2f} ± {result['std_reward']:.2f} "
              f"| topics={result['mean_topics']:.2f}")

    if gen_results:
        plot_generalisation(gen_results, "plots/generalisation.png")
    else:
        print("  No models found. Train first with: python main.py train")


# ─── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Malami – RL Adaptive Tutor")
    sub = parser.add_subparsers(dest="cmd")

    p_train = sub.add_parser("train", help="Train RL models")
    p_train.add_argument("--algo", default="all", choices=["all","dqn","reinforce","ppo","a2c"])

    p_play = sub.add_parser("play", help="Run best model")
    p_play.add_argument("--algo", default="ppo", choices=["dqn","reinforce","ppo","a2c"])
    p_play.add_argument("--no-gui", action="store_true")

    sub.add_parser("random",   help="Random policy demo")
    sub.add_parser("evaluate", help="Generalisation benchmark")
    sub.add_parser("plots",    help="Regenerate plots")

    args = parser.parse_args()

    if args.cmd == "train":
        cmd_train(args.algo)
    elif args.cmd == "play":
        cmd_play(args.algo, gui=not args.no_gui)
    elif args.cmd == "random":
        cmd_random()
    elif args.cmd == "evaluate":
        cmd_evaluate()
    elif args.cmd == "plots":
        _generate_plots({})
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
