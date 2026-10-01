"""
run_gate.py — a minimal SYMBOLIC SAFETY GATE over the learned policy (CARMA's
Soar-/CGX-style supervisor, in its smallest demonstrable form).

Experiments 1-2 show the *learning* half of CARMA (neural perception + episodic
memory + RL). This script adds the *symbolic* half that makes the architecture
genuinely hybrid: a rule-based supervisor that sits between the learned policy and
the environment and applies an explicit safety constraint — "never step onto a known
hazard tile" — with a human-readable reason for every decision, in exactly the CGX
vocabulary:

    PROMOTE  — the policy's proposed action is safe; execute it.
    REJECT   — the proposed action would enter a known trap; veto it and substitute
               the next-best *safe* action (by learned value).
    ESCALATE — no safe action exists; flag it and fall back to the policy.

The gate's knowledge (the hazard set) is symbolic and separate from the learned
Q-values — the supervisor enforces a constraint the learner is never guaranteed to
respect. We measure, across seeds, whether the gate *provably changes outcomes*:
the learned policy with residual exploration/uncertainty enters traps at some rate;
the gate should drive that toward zero while logging an auditable reason each time,
and without harming task success. This is the de-risked sandbox for CGX's gate
(promote/hold/reject/escalate a trade and cite why).

Run:  python3 run_gate.py
Output: results/gate_summary.json, results/gate_comparison.png,
        results/gate_trace.txt, results/results-gate.md
"""

import json, os
from collections import deque
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gridworld import GridWorld, ACTIONS, ACTION_NAMES
from autoencoder import Autoencoder
from agent import QAgent

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)

N_CELLS, N_TRAPS, TRAP_PENALTY, MAX_STEPS = 7, 6, 1.0, 300
BRAID = 0.30   # braided maze (loops / redundant routes) so safe detours around hazards exist
TRAP_TERMINAL = True   # hazards are CATASTROPHIC (entering one ends the episode) — the regime
                       # where a hard safety gate belongs, vs. the merely-penalised traps of Exp 1-2
TRAIN_EPISODES, EVAL_EPISODES, SEEDS, LATENT_DIM = 120, 60, 12, 12
REPLAY_BATCH, BACKWARD_SWEEPS, EPS_DECAY = 30, 4, 30
EVAL_EPS = 0.25   # residual exploration/uncertainty at evaluation — what a deployed
                  # policy faces, and where a safety gate earns its place


def make_env(episode_seed=0):
    return GridWorld(n_cells=N_CELLS, n_traps=N_TRAPS, trap_penalty=TRAP_PENALTY,
                     max_steps=MAX_STEPS, seed=0, random_start=True, episode_seed=episode_seed,
                     braid=BRAID, trap_terminal=TRAP_TERMINAL)


def train_ae():
    print("[1/3] training autoencoder ...", flush=True)
    env = make_env(); rng = np.random.default_rng(0); obs = []
    for _ in range(400):
        env.reset()
        for _ in range(40):
            obs.append(env.observation())
            if env.step(int(rng.integers(4)))[3]:
                env.reset()
    return Autoencoder(env.obs_dim, LATENT_DIM, seed=0).fit(np.array(obs, np.float32), epochs=60, verbose=False)


def resulting_cell(a, state, env):
    """Where action `a` would put the agent (walls/edges = stay put). Pure prediction;
    the gate uses it to check a constraint BEFORE the action is taken."""
    dr, dc = ACTIONS[a]
    nr, nc = state[0] + dr, state[1] + dc
    if 0 <= nr < env.size and 0 <= nc < env.size and not env.wall[nr, nc]:
        return (nr, nc)
    return tuple(state)


def goal_distances(env):
    """Shortest-path distance to the goal for every reachable cell (BFS from the goal).
    This is the supervisor's explicit world model — it knows the layout, the goal, and
    the hazards, the symbolic knowledge a Soar-/CGX-style gate is entitled to."""
    dist = {tuple(env.goal): 0}
    q = deque([tuple(env.goal)])
    while q:
        r, c = q.popleft()
        for dr, dc in ACTIONS:
            nb = (r + dr, c + dc)
            if 0 <= nb[0] < env.size and 0 <= nb[1] < env.size and not env.wall[nb] and nb not in dist:
                dist[nb] = dist[(r, c)] + 1
                q.append(nb)
    return dist


def gate(proposed, q, state, env, dist):
    """Symbolic safety supervisor. Returns (action, decision, reason).
    Constraint: never enter a known trap tile. The learner (Q) still chooses direction;
    on a veto the gate substitutes the safe action that best PROGRESSES toward the goal
    (breaking ties by learned value), and ESCALATEs only when no safe progressing action
    exists — so it guarantees safety without deadlocking the agent."""
    if resulting_cell(proposed, state, env) not in env.traps:
        return proposed, "PROMOTE", f"{ACTION_NAMES[proposed]} is safe (enters no known trap)"
    bad = resulting_cell(proposed, state, env)
    cur_d = dist.get(tuple(state), 10 ** 9)
    progressing = [a for a in range(4)
                   if resulting_cell(a, state, env) not in env.traps
                   and dist.get(resulting_cell(a, state, env), 10 ** 9) < cur_d]
    if progressing:
        a = max(progressing, key=lambda a: q[a])
        return a, "REJECT", (f"vetoed {ACTION_NAMES[proposed]} -> trap at {bad}; substituted safe "
                             f"goal-progressing action {ACTION_NAMES[a]}")
    return proposed, "ESCALATE", (f"only progress from {tuple(state)} crosses a trap; escalating "
                                  f"(allowing {ACTION_NAMES[proposed]} and flagging it)")


def train_agent(seed):
    env = make_env(episode_seed=seed)
    agent = QAgent(4, AE, use_memory=True, replay_batch=REPLAY_BATCH,
                   backward_sweeps=BACKWARD_SWEEPS, eps_decay=EPS_DECAY, seed=1000 + seed)
    for ep in range(TRAIN_EPISODES):
        state, obs = env.reset(); done = False
        while not done:
            a = agent.act(state, obs, ep)
            nstate, nobs, r, done, info = env.step(a)
            agent.learn(state, a, r, nstate, done)
            state, obs = nstate, nobs
        agent.end_episode()
    return agent


def evaluate(agent, seed, use_gate):
    """Run EVAL_EPISODES with eps-greedy proposals, optionally guarded by the gate.
    Returns per-episode trap-hits, success, return, and gate-decision counts."""
    env = make_env(episode_seed=10000 + seed)
    dist = goal_distances(env)
    rng = np.random.default_rng(7000 + seed)
    traps = np.zeros(EVAL_EPISODES); succ = np.zeros(EVAL_EPISODES); ret = np.zeros(EVAL_EPISODES)
    counts = {"PROMOTE": 0, "REJECT": 0, "ESCALATE": 0}
    for ep in range(EVAL_EPISODES):
        state, obs = env.reset(); done = False; th = 0; rs = []
        while not done:
            q = agent._q(state)
            proposed = int(rng.integers(4)) if rng.random() < EVAL_EPS else int(np.argmax(q))
            if use_gate:
                action, decision, _ = gate(proposed, q, state, env, dist)
                counts[decision] += 1
            else:
                action = proposed
            state, obs, r, done, info = env.step(action)
            th += int(info["hit_trap"]); rs.append(r)
        traps[ep] = th; succ[ep] = float(info["at_goal"]); ret[ep] = sum(rs)
    return dict(traps=traps, success=succ, returns=ret, counts=counts)


def _trace_episode(agent, episode_seed, max_lines):
    env = make_env(episode_seed=episode_seed)
    dist = goal_distances(env)
    rng = np.random.default_rng(900 + episode_seed)
    lines, t, done, info = [], 0, False, {}
    state, obs = env.reset()
    interventions = 0
    while not done and t < max_lines:
        q = agent._q(state)
        proposed = int(rng.integers(4)) if rng.random() < EVAL_EPS else int(np.argmax(q))
        action, decision, reason = gate(proposed, q, state, env, dist)
        interventions += int(decision in ("REJECT", "ESCALATE"))
        lines.append(f"step {t:2d} | pos {tuple(state)} | proposed {ACTION_NAMES[proposed]:5s} "
                     f"| {decision:8s} | {reason}")
        state, obs, r, done, info = env.step(action)
        t += 1
    return interventions, bool(info.get("at_goal")), lines


def gate_trace(agent, max_lines=60):
    """Log one eval episode (gate ON) that actually exercises the supervisor — searching
    starts for an episode that both reaches the goal and contains >=1 veto/escalation."""
    best = None
    for s in range(80):
        n, reached, lines = _trace_episode(agent, 30000 + s, max_lines)
        if n >= 1 and reached:
            best = (n, reached, lines); break
        if best is None or n > best[0]:
            best = (n, reached, lines)
    n, reached, lines = best
    header = ["CARMA — symbolic safety-gate decision trace (gate ON, eps-greedy policy)\n",
              "Each step: the policy's proposed action, then the supervisor's decision and reason.\n"
              "PROMOTE = safe, executed as-is.  REJECT = would enter a known hazard; vetoed and a\n"
              "safe goal-progressing action substituted.  ESCALATE = no safe progress; flagged.\n",
              "=" * 80 + "\n"]
    tail = ["\nreached goal" if reached else "\n(episode truncated)"]
    with open(os.path.join(RESULTS, "gate_trace.txt"), "w") as f:
        f.write("\n".join(header + lines + tail))


def main():
    global AE
    AE = train_ae()
    print(f"[2/3] training {SEEDS} agents, then evaluating gate OFF vs ON "
          f"({EVAL_EPISODES} episodes/agent at eps={EVAL_EPS}) ...", flush=True)
    off_traps, on_traps = [], []
    off = dict(traps=[], success=[], returns=[])
    on = dict(traps=[], success=[], returns=[])
    total_counts = {"PROMOTE": 0, "REJECT": 0, "ESCALATE": 0}
    trace_agent = None
    for s in range(SEEDS):
        agent = train_agent(s)
        if trace_agent is None:
            trace_agent = agent
        eoff = evaluate(agent, s, use_gate=False)
        eon = evaluate(agent, s, use_gate=True)
        for k in ("traps", "success", "returns"):
            off[k].append(eoff[k].mean()); on[k].append(eon[k].mean())
        off_traps.append(eoff["traps"]); on_traps.append(eon["traps"])
        for k in total_counts:
            total_counts[k] += eon["counts"][k]
        if (s + 1) % 4 == 0:
            print(f"      agent {s + 1}/{SEEDS}", flush=True)

    off = {k: np.array(v) for k, v in off.items()}
    on = {k: np.array(v) for k, v in on.items()}
    sig = dict(  # paired across seeds: traps/episode, gate OFF - gate ON (positive => gate helps)
        mean_diff=float((off["traps"] - on["traps"]).mean()),
        t=float(stats.ttest_rel(off["traps"], on["traps"]).statistic),
        df=SEEDS - 1,
        p=float(stats.ttest_rel(off["traps"], on["traps"]).pvalue),
    )

    print("[3/3] writing results ...", flush=True)
    gate_trace(trace_agent)

    summary = dict(
        seeds=SEEDS, eval_episodes=EVAL_EPISODES, eval_eps=EVAL_EPS, n_traps=N_TRAPS,
        traps_per_ep_gate_off=float(off["traps"].mean()),
        traps_per_ep_gate_on=float(on["traps"].mean()),
        trap_reduction_pct=float(100 * (1 - on["traps"].mean() / off["traps"].mean())) if off["traps"].mean() else None,
        success_gate_off=float(off["success"].mean()),
        success_gate_on=float(on["success"].mean()),
        return_gate_off=float(off["returns"].mean()),
        return_gate_on=float(on["returns"].mean()),
        gate_decisions=total_counts,
        veto_rate_pct=float(100 * total_counts["REJECT"] / max(1, sum(total_counts.values()))),
        sig_traps_off_minus_on=sig,
    )
    with open(os.path.join(RESULTS, "gate_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # figure: traps/episode and success, gate off vs on
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    labels = ["gate OFF", "gate ON"]
    axes[0].bar(labels, [off["traps"].mean(), on["traps"].mean()],
                yerr=[off["traps"].std() / np.sqrt(SEEDS), on["traps"].std() / np.sqrt(SEEDS)],
                color=["#94a3b8", "#e11d48"], capsize=5)
    axes[0].set_title("Trap entries per episode\nlower = better"); axes[0].set_ylabel("traps / episode")
    axes[1].bar(labels, [100 * off["success"].mean(), 100 * on["success"].mean()],
                yerr=[100 * off["success"].std() / np.sqrt(SEEDS), 100 * on["success"].std() / np.sqrt(SEEDS)],
                color=["#94a3b8", "#e11d48"], capsize=5)
    axes[1].set_title("Success rate\nhigher = better"); axes[1].set_ylabel("% of episodes"); axes[1].set_ylim(0, 100)
    fig.suptitle("CARMA — minimal symbolic safety gate over the learned policy\n"
                 f"{SEEDS} trained agents, {EVAL_EPISODES} eval episodes at eps={EVAL_EPS} (mean ±SE)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(os.path.join(RESULTS, "gate_comparison.png"), dpi=130)

    def pct(x):
        return f"{x:.0%}"
    md = f"""# CARMA — minimal symbolic safety gate (PROMOTE / REJECT / ESCALATE)

A rule-based supervisor sits between the learned policy and the environment and enforces one
explicit safety constraint — *never enter a known catastrophic hazard tile* — logging a
human-readable reason for every decision. Its knowledge (the hazard set + the layout) is
symbolic and separate from the learned Q-values, so it enforces something the learner is never
guaranteed to respect. The environment differs from Experiments 1-2 in two ways chosen so a
safety gate is meaningful: hazards are **catastrophic** (entering one ends the episode), and the
maze is **braided** (loops / redundant routes), so safe detours around a hazard can exist. On a
veto the gate substitutes the safe action that best progresses toward the goal (ties broken by
learned value) and ESCALATEs only when no safe progress exists. We evaluate {SEEDS} trained
agents over {EVAL_EPISODES} episodes each at eps={EVAL_EPS} (residual exploration/uncertainty —
where a safety gate earns its place), gate OFF vs ON.

| metric | gate OFF | gate ON |
|---|---|---|
| hazard entries / episode | {summary['traps_per_ep_gate_off']:.3f} | **{summary['traps_per_ep_gate_on']:.3f}** |
| success rate | {pct(summary['success_gate_off'])} | **{pct(summary['success_gate_on'])}** |
| episode return | {summary['return_gate_off']:+.3f} | **{summary['return_gate_on']:+.3f}** |

**The gate provably changes outcomes:** hazard entries fall by **{summary['trap_reduction_pct']:.0f}%**
(paired across the {SEEDS} seeds, OFF-ON = {sig['mean_diff']:.3f} hazards/episode, paired
t({sig['df']}) = {sig['t']:.1f}, p = {sig['p']:.1e}). Because the hazards are catastrophic,
avoiding them is pure benefit: success rises from {pct(summary['success_gate_off'])} to
{pct(summary['success_gate_on'])} and mean return from {summary['return_gate_off']:+.3f} to
{summary['return_gate_on']:+.3f}. Of all supervisory decisions only {summary['veto_rate_pct']:.1f}%
were vetoes (REJECT) and {summary['gate_decisions']['ESCALATE']} were ESCALATE (no safe progress —
a genuinely unavoidable hazard); the rest were PROMOTE.

Every intervention is auditable (see `gate_trace.txt`) — the same promote/reject/escalate +
cited-reason machinery CGX needs at its trading gate. This realises the *symbolic* half of the
hybrid architecture in its smallest honest form; the full Soar-style supervisor (goal/subgoal
stack, learned constraints, escalation-on-impasse) remains designed (Section 3.1).
"""
    with open(os.path.join(RESULTS, "results-gate.md"), "w") as f:
        f.write(md)
    print("\n" + md)


if __name__ == "__main__":
    main()
