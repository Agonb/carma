"""
agent.py — a tabular Q-learning agent, optionally augmented with an episodic replay memory.

The two experimental conditions differ by ONE switch (`use_memory`):
  - M0 (no memory): each environment step produces exactly one Q-update — the
                    transition the agent is currently standing on.
  - M1 (memory):    same online update, PLUS the agent stores every transition and
                    replays a small random batch of past transitions each step,
                    doing extra Q-updates on them.

Everything else — the Q-learning rule, reward, environment, autoencoder, exploration
schedule, and random seeds — is identical. So any difference in the learning curves is
attributable to the replay memory alone.

The autoencoder latent of each situation is stored with the transition (perception →
memory) and surfaced in the decision trace, the seed of the interpretability layer.
"""

import numpy as np
from episodic_memory import EpisodicMemory
from gridworld import ACTION_NAMES


class QAgent:
    def __init__(self, n_actions, encoder, use_memory=False, replay_batch=10,
                 backward_sweeps=2, replay_mode="uniform", alpha=0.5, gamma=0.98,
                 eps_start=1.0, eps_end=0.05, eps_decay=30, seed=0, **_ignore):
        self.n_actions = n_actions
        self.encoder = encoder
        self.use_memory = use_memory
        self.replay_batch = replay_batch
        self.backward_sweeps = backward_sweeps
        self.replay_mode = replay_mode      # uniform | prioritized | reverse | similarity
        self.alpha, self.gamma = alpha, gamma
        self.eps_start, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay
        self.rng = np.random.default_rng(seed)
        self.Q = {}
        self.memory = EpisodicMemory(seed=seed) if use_memory else None
        self._last_latent = None
        self._episode = []

    def _q(self, state):
        if state not in self.Q:
            self.Q[state] = np.zeros(self.n_actions, dtype=np.float32)
        return self.Q[state]

    def epsilon(self, episode):
        return self.eps_end + (self.eps_start - self.eps_end) * np.exp(-episode / self.eps_decay)

    def act(self, state, obs, episode, explain=False):
        # perception: encode the situation (stored with the transition; used for the
        # interpretability trace and Phase-2 similarity replay)
        self._last_latent = self.encoder.encode(obs)[0]
        q = self._q(state)
        if self.rng.random() < self.epsilon(episode):
            action, reason = int(self.rng.integers(self.n_actions)), "explore (epsilon)"
        else:
            action = int(np.argmax(q))
            reason = f"Q-table: {ACTION_NAMES[action]} has highest learned value ({q[action]:+.2f})"
        return (action, reason) if explain else action

    def _td_update(self, s, a, r, ns, done):
        q = self._q(s)
        target = r + (0.0 if done else self.gamma * np.max(self._q(ns)))
        q[a] += self.alpha * (target - q[a])

    def learn(self, state, action, reward, next_state, done):
        # TD-error of this transition (used as the priority for prioritized replay)
        q = self._q(state)
        target = reward + (0.0 if done else self.gamma * np.max(self._q(next_state)))
        priority = abs(target - q[action]) + 1e-3
        self._td_update(state, action, reward, next_state, done)   # online (all conditions)
        if self.use_memory:
            self.memory.add(state, action, reward, next_state, done,
                            latent=self._last_latent, priority=priority)
            self._episode.append((state, action, reward, next_state, done))
            for (s, a, r, ns, d) in self.memory.select(self.replay_mode, self.replay_batch,
                                                        query_latent=self._last_latent):
                self._td_update(s, a, r, ns, d)

    def end_episode(self, rewards=None):
        # BACKWARD REPLAY (hippocampal reverse replay): re-experience the episode in
        # reverse so the goal's value propagates along the whole path in one sweep.
        # Only the memory agent can do this — it requires having stored the episode.
        if self.use_memory and self._episode:
            for _ in range(self.backward_sweeps):
                for (s, a, r, ns, d) in reversed(self._episode):
                    self._td_update(s, a, r, ns, d)
        self._episode = []
