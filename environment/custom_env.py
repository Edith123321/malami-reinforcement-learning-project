"""
Malami - Adaptive Learning Platform Environment
================================================
A custom Gymnasium environment simulating an AI tutor that selects
learning activities to maximise student mastery and engagement.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any


ACTION_VIDEO_LESSON     = 0
ACTION_PRACTICE_QUIZ    = 1
ACTION_PROBLEM_SOLVING  = 2
ACTION_REVIEW_SUMMARY   = 3
ACTION_ADAPTIVE_HINT    = 4
ACTION_NEW_TOPIC        = 5
ACTION_REMEDIATION      = 6
ACTION_PEER_DISCUSSION  = 7
ACTION_WORKED_EXAMPLE   = 8

NUM_ACTIONS = 9
NUM_TOPICS  = 6

TOPIC_NAMES = ["Cell Biology", "Genetics", "Evolution", "Ecology", "Human Body", "Plants"]
MASTERY_THRESHOLD = 0.85
MAX_STEPS = 80


class StudentProfile:
    """Realistic student profile with learning characteristics"""
    def __init__(self, rng=None, learning_rate=None, retention=None, 
                 engagement_sensitivity=None, prior_knowledge=None):
        if rng is None:
            rng = np.random.default_rng()
        
        # Learning characteristics with realistic ranges
        self.learning_rate = learning_rate if learning_rate is not None else rng.uniform(0.3, 0.9)
        self.retention = retention if retention is not None else rng.uniform(0.5, 0.9)
        self.engagement_sensitivity = engagement_sensitivity if engagement_sensitivity is not None else rng.uniform(0.3, 0.8)
        self.prior_knowledge = prior_knowledge if prior_knowledge is not None else rng.uniform(0.0, 0.25)
        
        # Additional profile attributes used in step logic
        self.learning_speed = self.learning_rate * 0.8
        self.preferred_modality = rng.choice([0, 1, 2])  # 0=visual, 1=kinesthetic, 2=auditory
        self.quiz_anxiety = rng.uniform(0.1, 0.4)
        self.curiosity = rng.uniform(0.2, 0.6)
        self.attention_span = rng.uniform(8, 15)
        
    def to_array(self):
        """Convert profile to numpy array for observation space"""
        return np.array([
            self.learning_rate,
            self.retention,
            self.engagement_sensitivity,
            self.prior_knowledge,
            self.learning_speed,
            self.attention_span / 20.0  # Normalize to 0-1 range
        ], dtype=np.float32)
    
    def __repr__(self):
        return (f"StudentProfile(learning_rate={self.learning_rate:.2f}, "
                f"retention={self.retention:.2f}, "
                f"engagement_sensitivity={self.engagement_sensitivity:.2f})")


class MalamiEnv(gym.Env):
    """
    Malami Adaptive Learning Environment.

    Observation (21 dims):
      [0]   current topic mastery
      [1]   avg recent quiz score
      [2]   time on activity (norm)
      [3]   attempt count (norm)
      [4]   engagement
      [5]   consecutive failures (norm)
      [6]   topics completed ratio
      [7]   global mastery mean
      [8]   hint requests (norm)
      [9-14] per-topic masteries (6 topics)
      [15-20] student profile features (6 features)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
        max_steps: int = MAX_STEPS,
        student_profile: Optional[StudentProfile] = None,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps
        self._fixed_profile = student_profile
        obs_dim = 9 + NUM_TOPICS + 6  # 9 base features + 6 topics + 6 profile features
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self._np_rng = np.random.default_rng(seed)
        
        # State variables
        self.profile = None
        self.topic_masteries = np.zeros(NUM_TOPICS, dtype=np.float32)
        self.current_topic = 0
        self.engagement = 1.0
        self.recent_scores = [0.0, 0.0, 0.0]
        self.time_on_activity = 0.0
        self.attempt_count = 0
        self.consecutive_fails = 0
        self.hint_requests = 0
        self.step_count = 0
        self._renderer = None

    def reset(self, *, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        if seed is not None:
            self._np_rng = np.random.default_rng(seed)
        
        # Initialize student profile
        self.profile = (
            self._fixed_profile if self._fixed_profile is not None
            else StudentProfile(rng=self._np_rng)
        )
        
        # Initialize topic masteries with prior knowledge
        self.topic_masteries = np.clip(
            self._np_rng.uniform(0.0, 0.15, NUM_TOPICS) + self.profile.prior_knowledge * 0.5,
            0.0, 1.0
        ).astype(np.float32)
        
        # Reset all state variables
        self.current_topic = 0
        self.engagement = float(self._np_rng.uniform(0.7, 1.0))
        self.recent_scores = [0.0, 0.0, 0.0]
        self.time_on_activity = 0.0
        self.attempt_count = 0
        self.consecutive_fails = 0
        self.hint_requests = 0
        self.step_count = 0
        
        # Return observation and info (current_topic as INTEGER)
        return self._get_obs(), {
            "topics_completed": 0,
            "current_topic": self.current_topic,  # Integer, not string
            "mastery_current": float(self.topic_masteries[self.current_topic]),
            "engagement": self.engagement
        }

    def step(self, action: int):
        """Execute one step in the environment."""
        assert self.action_space.contains(action)
        self.step_count += 1
        self.time_on_activity += 1.0
        
        reward, info = self._apply_action(action)
        self._update_engagement(action)
        
        # Check for topic advancement
        if self.topic_masteries[self.current_topic] >= MASTERY_THRESHOLD:
            if self.current_topic < NUM_TOPICS - 1:
                self.current_topic += 1
                reward += 5.0
                self.attempt_count = 0
                self.time_on_activity = 0.0
                info["event"] = f"Advanced to {TOPIC_NAMES[self.current_topic]}"
        
        # Check termination conditions
        topics_done = int(np.sum(self.topic_masteries >= MASTERY_THRESHOLD))
        terminated = (topics_done == NUM_TOPICS or self.engagement <= 0.05 or self.consecutive_fails >= 8)
        truncated = self.step_count >= self.max_steps
        
        # Bonus reward for completing all topics
        if terminated and topics_done == NUM_TOPICS:
            reward += 20.0
        
        # Build info dictionary
        info.update({
            "topics_completed": topics_done,
            "current_topic": self.current_topic,  # Integer
            "engagement": self.engagement,
            "step": self.step_count,
            "mastery_current": float(self.topic_masteries[self.current_topic]),
        })
        
        if self.render_mode == "human":
            self.render()
        
        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self):
        """Render the environment."""
        if self._renderer is None:
            from environment.rendering import MalamiRenderer
            self._renderer = MalamiRenderer()
        self._renderer.render(self._build_render_state())

    def close(self):
        """Close the environment."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _get_obs(self):
        """Construct observation vector."""
        avg_quiz = float(np.mean(self.recent_scores))
        obs = np.array([
            float(self.topic_masteries[self.current_topic]),
            avg_quiz,
            min(self.time_on_activity / 20.0, 1.0),
            min(self.attempt_count / 10.0, 1.0),
            self.engagement,
            min(self.consecutive_fails / 8.0, 1.0),
            float(np.sum(self.topic_masteries >= MASTERY_THRESHOLD)) / NUM_TOPICS,
            float(np.mean(self.topic_masteries)),
            min(self.hint_requests / 20.0, 1.0),
        ], dtype=np.float32)
        
        # Concatenate with topic masteries and student profile
        return np.concatenate([obs, self.topic_masteries, self.profile.to_array()]).astype(np.float32)

    def _apply_action(self, action: int):
        """Apply the action and return reward and info."""
        rng = self._np_rng
        p = self.profile
        m = float(self.topic_masteries[self.current_topic])
        reward = 0.0
        info = {"action": action}

        if action == ACTION_VIDEO_LESSON:
            mb = 0.15 if p.preferred_modality == 0 else 0.0
            delta = p.learning_speed * (0.4 + mb) * rng.uniform(0.8, 1.2)
            self._update_mastery(delta)
            reward = delta * 8.0 + 0.5
            self.consecutive_fails = 0

        elif action == ACTION_PRACTICE_QUIZ:
            raw_score = float(np.clip(m - p.quiz_anxiety * rng.uniform(0, 0.3) + rng.normal(0, 0.05), 0.0, 1.0))
            self.recent_scores = self.recent_scores[1:] + [raw_score]
            delta = p.learning_speed * raw_score * 0.6 * rng.uniform(0.9, 1.1)
            self._update_mastery(delta)
            reward = raw_score * 3.0 + delta * 6.0
            if raw_score < 0.4:
                self.consecutive_fails += 1
                reward -= 1.0
            else:
                self.consecutive_fails = 0

        elif action == ACTION_PROBLEM_SOLVING:
            if m > 0.5:
                delta = p.learning_speed * 0.8 * rng.uniform(0.7, 1.3)
                self._update_mastery(delta)
                reward = delta * 12.0 + m * 2.0
                self.consecutive_fails = 0
            else:
                reward = -1.5
                self.consecutive_fails += 1
                self.engagement = max(0.0, self.engagement - 0.05)

        elif action == ACTION_REVIEW_SUMMARY:
            delta = p.learning_speed * 0.2 * rng.uniform(0.9, 1.1)
            self._update_mastery(delta)
            reward = delta * 4.0 + 0.3
            self.consecutive_fails = max(0, self.consecutive_fails - 1)

        elif action == ACTION_ADAPTIVE_HINT:
            self.hint_requests += 1
            if self.consecutive_fails >= 2:
                delta = p.learning_speed * 0.35 * rng.uniform(0.8, 1.2)
                self._update_mastery(delta)
                reward = delta * 7.0 + 1.0
                self.consecutive_fails = max(0, self.consecutive_fails - 2)
            else:
                reward = -0.3

        elif action == ACTION_NEW_TOPIC:
            if m >= MASTERY_THRESHOLD and self.current_topic < NUM_TOPICS - 1:
                reward = 3.0 + p.curiosity
            elif m < 0.5:
                reward = -2.0
                self.engagement = max(0.0, self.engagement - 0.08)
            else:
                reward = m * 2.0

        elif action == ACTION_REMEDIATION:
            if self.consecutive_fails >= 3 and self.current_topic > 0:
                prev = self.current_topic - 1
                delta = p.learning_speed * 0.5
                self.topic_masteries[prev] = min(1.0, self.topic_masteries[prev] + delta)
                reward = 2.0
                self.consecutive_fails = max(0, self.consecutive_fails - 3)
            else:
                reward = -0.5

        elif action == ACTION_PEER_DISCUSSION:
            if 0.3 < m < 0.8:
                delta = p.learning_speed * 0.5 * rng.uniform(0.8, 1.2)
                self._update_mastery(delta)
                reward = delta * 9.0 + 0.8
                self.engagement = min(1.0, self.engagement + 0.03)
            else:
                delta = p.learning_speed * 0.15
                self._update_mastery(delta)
                reward = delta * 3.0

        elif action == ACTION_WORKED_EXAMPLE:
            mb = 0.2 if p.preferred_modality == 1 else 0.0
            delta = p.learning_speed * (0.45 + mb) * rng.uniform(0.8, 1.2)
            self._update_mastery(delta)
            reward = delta * 9.0 + 0.4
            self.consecutive_fails = max(0, self.consecutive_fails - 1)

        # Add engagement bonus
        if reward > 0:
            reward += self.engagement * 0.3
        
        reward = float(np.clip(reward, -5.0, 20.0))
        return reward, info

    def _update_mastery(self, delta: float):
        """Update mastery for current topic."""
        noise = float(self._np_rng.normal(0, 0.01))
        self.topic_masteries[self.current_topic] = float(
            np.clip(self.topic_masteries[self.current_topic] + delta + noise, 0.0, 1.0)
        )
        self.attempt_count += 1

    def _update_engagement(self, action: int):
        """Update engagement based on action."""
        decay = 0.01 / max(self.profile.attention_span, 0.1)
        self.engagement = max(0.0, self.engagement - decay)
        
        # Certain actions boost engagement
        if action in (ACTION_PEER_DISCUSSION, ACTION_PROBLEM_SOLVING):
            self.engagement = min(1.0, self.engagement + 0.04)
        if action == ACTION_NEW_TOPIC and self.topic_masteries[self.current_topic] >= MASTERY_THRESHOLD:
            self.engagement = min(1.0, self.engagement + 0.06)

    def _build_render_state(self):
        """Build state for rendering."""
        return {
            "topic_masteries": self.topic_masteries.tolist(),
            "current_topic": self.current_topic,
            "topic_names": TOPIC_NAMES,
            "engagement": self.engagement,
            "step": self.step_count,
            "recent_scores": self.recent_scores,
            "consecutive_fails": self.consecutive_fails,
        }

    def get_action_name(self, action: int) -> str:
        """Get human-readable action name."""
        names = [
            "Video Lesson", "Practice Quiz", "Problem Solving", "Review Summary",
            "Adaptive Hint", "New Topic", "Remediation", "Peer Discussion", "Worked Example"
        ]
        return names[action]

    def set_student_profile(self, profile: StudentProfile):
        """Set a custom student profile for the episode."""
        self._fixed_profile = profile
        if hasattr(self, 'profile'):
            self.profile = profile