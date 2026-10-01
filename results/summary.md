# CARMA Phase 1 — results

**Question:** does adding episodic memory improve a reinforcement-learning agent?

15x15 maze, **random start each episode** (the agent
must learn to reach the goal from anywhere), 3 traps. Mean of
20 seeds, 120 episodes each. The two agents are identical
Q-learning agents with the same exploration and seeds — the ONLY difference is that M1 has
an episodic memory it replays (random replay each step + reverse-replay of each finished
episode). M0 has no memory and learns only from the single transition it is standing on.

| metric (last 25 episodes) | M0 — no memory | M1 — episodic memory |
|---|---|---|
| success rate (reaches goal) | 82% | 100% |
| episode return (higher better) | +0.152 | +0.727 |
| steps to reach goal (lower better) | 62.3 | 20.1 |
| success after only 20 episodes | 50% | 90% |
| **episodes to 90% success** | **47** | **16** |

**Headline: the memory agent reached 90% success by episode ~16; the no-memory agent took ~47.**

**Statistical reliability (paired across the 20 shared seeds):** the
episodes-to-90% gap is M0-M1 = 31.9 episodes (bootstrap 95% CI
[26.4, 37.4]), paired t(19) = 11.2,
p = 7.8e-10. The speedup is large and not attributable to seed noise.

The mechanism is exactly the point of memory: without it, value information propagates one
step per real visit, so learning the whole maze is slow. With it, the agent re-experiences
stored episodes — including a *reverse replay* of each finished episode, inspired by
hippocampal replay — so the goal's value propagates across the whole state space far faster.
Memory buys **sample efficiency**: more learning from the same real experience.

Honest caveat: M1 does more value-updates per real step (it has experiences to replay; M0
does not). That is not a confound — it is the mechanism. The fair axis is performance per
episode of *real* interaction, which is what these curves show.

See `learning_curves.png` and `decision_trace.txt`.
