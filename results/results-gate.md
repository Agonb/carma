# CARMA — minimal symbolic safety gate (PROMOTE / REJECT / ESCALATE)

A rule-based supervisor sits between the learned policy and the environment and enforces one
explicit safety constraint — *never enter a known catastrophic hazard tile* — logging a
human-readable reason for every decision. Its knowledge (the hazard set + the layout) is
symbolic and separate from the learned Q-values, so it enforces something the learner is never
guaranteed to respect. The environment differs from Experiments 1-2 in two ways chosen so a
safety gate is meaningful: hazards are **catastrophic** (entering one ends the episode), and the
maze is **braided** (loops / redundant routes), so safe detours around a hazard can exist. On a
veto the gate substitutes the safe action that best progresses toward the goal (ties broken by
learned value) and ESCALATEs only when no safe progress exists. We evaluate 12 trained
agents over 60 episodes each at eps=0.25 (residual exploration/uncertainty —
where a safety gate earns its place), gate OFF vs ON.

| metric | gate OFF | gate ON |
|---|---|---|
| hazard entries / episode | 0.172 | **0.067** |
| success rate | 81% | **88%** |
| episode return | +0.341 | **+0.457** |

**The gate provably changes outcomes:** hazard entries fall by **61%**
(paired across the 12 seeds, OFF-ON = 0.106 hazards/episode, paired
t(11) = 6.9, p = 2.7e-05). Because the hazards are catastrophic,
avoiding them is pure benefit: success rises from 81% to
88% and mean return from +0.341 to
+0.457. Of all supervisory decisions only 0.8%
were vetoes (REJECT) and 48 were ESCALATE (no safe progress —
a genuinely unavoidable hazard); the rest were PROMOTE.

Every intervention is auditable (see `gate_trace.txt`) — the same promote/reject/escalate +
cited-reason machinery CGX needs at its trading gate. This realises the *symbolic* half of the
hybrid architecture in its smallest honest form; the full Soar-style supervisor (goal/subgoal
stack, learned constraints, escalation-on-impasse) remains designed (Section 3.1).
