"""Basic environment sanity tests."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.custom_env import MalamiEnv, NUM_ACTIONS

def test_reset_shape():
    env = MalamiEnv(seed=0)
    obs, info = env.reset()
    assert obs.shape == (21,), f"Expected (21,), got {obs.shape}"
    print("  PASS: obs shape (21,)")

def test_step_valid():
    for action in range(NUM_ACTIONS):
        env = MalamiEnv(seed=1)
        env.reset()
        obs, r, term, trunc, info = env.step(action)
        assert obs.shape == (21,)
        assert isinstance(r, float)
    print("  PASS: all 9 actions valid")

def test_episode_terminates():
    env = MalamiEnv(seed=2)
    env.reset()
    done = False; steps = 0
    while not done and steps < 200:
        _, r, term, trunc, info = env.step(env.action_space.sample())
        done = term or trunc; steps += 1
    assert done
    print(f"  PASS: episode terminated in {steps} steps")

def test_mastery_bounds():
    env = MalamiEnv(seed=3)
    env.reset()
    for _ in range(50):
        env.step(env.action_space.sample())
        assert all(0 <= m <= 1 for m in env.topic_masteries)
    print("  PASS: mastery stays in [0,1]")

if __name__ == "__main__":
    print("Running environment tests...")
    test_reset_shape()
    test_step_valid()
    test_episode_terminates()
    test_mastery_bounds()
    print("\nAll tests passed!")
