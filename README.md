# CARMA — a hybrid symbolic–neural cognitive architecture with episodic memory

Code and results for the seminar **"Memory-Augmented Cognitive Architectures for Reinforcement-Learning Agents"**,
Cognitive Robotics, doctoral programme in Computer Science and Engineering, Faculty of Computer Science and
Engineering (FCSE), Ss. Cyril and Methodius University in Skopje. Mentor: Prof. Dr. Andrea Kulakov.

Author: Agon Bajgora · agon.bajgora@umib.net

The seminar report is in [`docs/CARMA-seminar-report.pdf`](docs/CARMA-seminar-report.pdf).

## What it is

A small, CPU-only agent in a 15×15 gridworld, built from four parts:

| File | Part |
|---|---|
| `gridworld.py` | the environment (maze, start cells, hazards) |
| `autoencoder.py` | neural perception: compresses the agent's view into a latent code (NumPy) |
| `episodic_memory.py` | episodic memory with four replay mechanisms: uniform, prioritized, reverse, similarity-keyed (autoencoder latent) |
| `agent.py` | the reinforcement-learning core (tabular Q-learning) with memory replay |
| `run_gate.py` | a symbolic, Soar-inspired safety gate (PROMOTE / REJECT / ESCALATE) that logs a reason for every decision |

## Three experiments

| # | Question | Result | Script |
|---|---|---|---|
| 1 | Does episodic memory help? Memory on vs off, 20 seeds | **16 vs 47** episodes to 90% success, about 3× faster | `run_experiment.py` |
| 2 | Does a learned similarity key beat plain replay at equal budget? 12 seeds | No: uniform replay **29** vs similarity key **50** episodes (paired t(11) = 3.8, p = 0.003) | `run_mechanisms.py` |
| 3 | Does a symbolic safety gate change outcomes? 12 agents | Hazard entries **−61%** (p < 0.001), success **81% → 88%**, 0.8% of decisions vetoed | `run_gate.py` |

Experiment 2 is a deliberate negative result: the maze is fully observable, so a learned key has little to add.
The hypothesis it sharpens is that a learned, situation-keyed memory pays off under **partial observability**
(egocentric view + LSTM), which is the next step.

## Run it

```bash
pip install -r requirements.txt
python3 run_experiment.py     # Experiment 1: memory vs no memory
python3 run_mechanisms.py     # Experiment 2: replay mechanisms at equal budget
python3 run_gate.py           # Experiment 3: symbolic safety gate on vs off
```

Each script writes its figures, JSON summaries and a Markdown result note to `results/`. The committed
`results/` folder holds the outputs reported in the seminar. Everything runs on a laptop CPU; no GPU is needed.

## Licence

MIT, see `LICENSE`.
