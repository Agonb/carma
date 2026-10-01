"""
episodic_memory.py — an episodic memory with several RETRIEVAL mechanisms.

The agent stores the episodes it experiences and replays them to learn from again
(experience replay; hippocampal-replay-inspired). The interesting research question
is not "does replay help?" (it does — textbook) but "given the SAME replay budget,
does it matter HOW we choose which experiences to replay?". This module therefore
offers four selection mechanisms over the same stored transitions:

  - uniform      : sample at random                         (DQN experience replay)
  - prioritized  : sample ∝ |TD-error|                       (Schaul et al. 2016, approx.)
  - reverse      : the most-recent transitions, in reverse   (reverse/backward replay)
  - similarity   : the k nearest past situations in the      (the autoencoder-latent key —
                   AUTOENCODER LATENT space (analogy)          makes the AE load-bearing)

`similarity` is the mechanism built from the architecture's own components (a learned
representation as the retrieval key, analogy-by-similarity). It is the only one that
uses the autoencoder latent, so comparing it against the others at EQUAL budget is
what tells us whether a perceptually-structured memory beats a flat one.
"""

import numpy as np


class EpisodicMemory:
    def __init__(self, capacity: int = 30000, seed: int = 0):
        self.cap = capacity
        self.rng = np.random.default_rng(seed)
        self.s, self.a, self.r, self.ns, self.done = [], [], [], [], []
        self.lat, self.prio = [], []

    def __len__(self):
        return len(self.s)

    def add(self, state, action, reward, next_state, done, latent=None, priority=1.0):
        self.s.append(state); self.a.append(action); self.r.append(reward)
        self.ns.append(next_state); self.done.append(done)
        self.lat.append(latent); self.prio.append(float(priority))
        if len(self.s) > int(self.cap * 1.1):            # lazy bulk trim
            cut = len(self.s) - self.cap
            for lst in (self.s, self.a, self.r, self.ns, self.done, self.lat, self.prio):
                del lst[:cut]

    def _gather(self, idx):
        return [(self.s[i], self.a[i], self.r[i], self.ns[i], self.done[i]) for i in idx]

    def sample(self, n):                                  # uniform
        if not self.s:
            return []
        return self._gather(self.rng.integers(0, len(self.s), size=min(n, len(self.s))))

    def sample_prioritized(self, n, alpha=0.6):
        if not self.s:
            return []
        p = np.asarray(self.prio, dtype=np.float64) ** alpha
        p /= p.sum()
        idx = self.rng.choice(len(self.s), size=min(n, len(self.s)), p=p, replace=True)
        return self._gather(idx)

    def sample_recent(self, n):                           # reverse / backward
        if not self.s:
            return []
        k = min(n, len(self.s))
        return self._gather(range(len(self.s) - 1, len(self.s) - 1 - k, -1))

    def sample_similar(self, n, query_latent):            # analogy by AE latent
        if not self.s or query_latent is None:
            return []
        L = np.asarray(self.lat, dtype=np.float32)
        L = L / (np.linalg.norm(L, axis=1, keepdims=True) + 1e-8)
        q = np.asarray(query_latent, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-8)
        sims = L @ q
        k = min(n, len(sims))
        idx = np.argpartition(sims, -k)[-k:]
        return self._gather(idx)

    def select(self, mode, n, query_latent=None):
        if mode == "uniform":
            return self.sample(n)
        if mode == "prioritized":
            return self.sample_prioritized(n)
        if mode == "reverse":
            return self.sample_recent(n)
        if mode == "similarity":
            return self.sample_similar(n, query_latent)
        raise ValueError(f"unknown replay mode {mode}")
