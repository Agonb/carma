# CARMA — equal-budget memory-mechanism comparison

Same maze, same 15-update/step replay budget for every replay arm (M0 has no replay). Mean of 12 seeds, 100 episodes. The only thing that differs between replay arms is HOW the replayed experiences are chosen.

| mechanism | final success | final return | episodes to 90% success |
|---|---|---|---|
| M0 no replay | 77% | +0.10 | 74 |
| uniform | 100% | +0.69 | 29 |
| prioritized (PER) | 98% | +0.65 | 29 |
| reverse | 91% | +0.53 | 41 |
| similarity (CAE) | 86% | +0.43 | 50 |

**Fastest replay mechanism to 90% success: prioritized (PER).**

**The key test — does the autoencoder-latent ('analogy') key beat plain uniform replay at equal budget?**  Here, similarity-keyed replay is **slower than** uniform replay (similarity reaches 90% at ~50, uniform at ~29).

This difference is statistically reliable across the 12 shared seeds, not seed noise: similarity-uniform = +20.8 episodes-to-90%, paired t(11) = 3.8, p = 3.2e-03, bootstrap 95% CI [10.7, 31.3].

Honest reading: this maze is FULLY OBSERVABLE (the agent's true position is known), so the learned latent key mostly encodes position and has little to add over uniform/prioritized selection. The hypothesised advantage of a learned similarity key is under PARTIAL observability / perceptual aliasing (egocentric view + LSTM) — that is the Phase-2 test. What this run DID change: the autoencoder is now load-bearing in the `similarity` arm (its latent decides what gets replayed), not decorative.