# SafeClassroom

Reinforcement learning for **classroom occupancy control during epidemics**. RL
agents learn weekly admission decisions that trade off in-person attendance
against infection risk, trained on synthetic (sinusoidal) community-risk patterns
and evaluated on both sinusoidal (in-distribution) and a real COVID-19 risk
trace, against a Myopic heuristic, an analytical Critical-Capacity policy, and a
dynamic-programming upper bound.

## The environment

A single classroom of `N = 100` students is simulated over a 15-week horizon
(`campus_gym/`, a Gymnasium environment). Each week the controller admits
`u ∈ [0, N]` students (discrete `{0, 50, 100}` or continuous), and the infection
count evolves by

```
I(t+1) = min( α·I(t)·u(t) + β·c_risk(t)·u(t)²,  u(t) )
```

- **α** — within-classroom transmission risk (`0.005`)
- **β** — community-coupling coefficient (`0.01`)
- **c_risk(t)** — time-varying community risk in `[0, 1]` (sinusoidal in training, real CSV in evaluation)
- the `min(·, u)` caps new infections at the number admitted.

**State** `(c_risk, I)`, **action** `u`, **reward** `r = ω·u − (1−ω)·I`, where the
weight `ω ∈ {0.1,…,0.6}` sets the attendance-vs-safety trade-off. Episodes are
finite-horizon (no discounting). See `config.py` for all constants and
`threshold_behavior.py` for the R₀ / disease-free-vs-endemic analysis.

## Agents

Learned (one trainer script each):

| Agent | Script |
|---|---|
| Double DQN | `double_dqn.py` |
| PPO Discrete | `ppo_agent.py` |
| PPO Continuous (Beta policy + GAE) | `ppo_continuous_new.py` |

Baselines computed at evaluation time (no training):

- **Myopic** — greedy one-step-reward heuristic
- **Critical Capacity** — analytical policy that admits `u*(c_risk)` to keep R₀ < 1
- **DP Upper Bound** — clairvoyant dynamic-programming oracle (`optimal_dp_policy.py`)

## Setup

```bash
pip install -r requirements.txt
pip install -e campus_gym/        # register the Gymnasium environment
```

## Training

Each agent trains on sinusoidal risk; the `--eval-risk-type` flag only selects
which evaluation distribution its hyperparameters are tuned/saved against, and
sets the output directory.

```bash
# all three agents, both eval-risk-type variants, tune + train + evaluate
python run_pipeline.py
python run_pipeline.py --skip-tune          # reuse saved hyperparameters
python run_pipeline.py --eval-risk-type data  # one mode only

# all three agents (training only)
python train_all.py --eval-risk-type sinusoidal
python train_all.py --eval-risk-type data --skip-tune

# a single agent / single mode
python ppo_continuous_new.py --eval-risk-type data --skip-tune
```

Models, learning curves, and per-omega rollouts are written to
`<agent>_results_tuned_<sinusoidal|data>/` (git-ignored).

## Evaluation

```bash
python evaluate.py --eval-mode both          # sinusoidal (30 seeds) + data (1 trajectory)
python evaluate.py --eval-mode sinusoidal
python evaluate.py --eval-mode data
on regen_figures.py
```

Outputs go to `evaluation_results_<sinusoidal|data>/`:

- **`summary.csv`** — per (agent, ω): mean ± std reward, analytic & bootstrap 95% CIs,
  mean/std infected & attendance, `pct_of_upper_bound`, monotonicity (Spearman ρ),
  and the optimal-threshold safety scores (`optimal_x`, `optimal_y`, `safety_F`).
- **`safety_optimal_thresholds.csv`** — infection ceiling X\*, attendance floor Y\*, and F = ω·Y\* − (1−ω)·X\* (z = 90%).
- **`safety_optimal_thresholds_perturbation.csv`** — X\*/Y\*/F and reward under sensing-noise levels.
- **Figures** — reward vs ω, optimality gap, monotonicity, attendance–infection frontier,
  per-week trajectories, safety table/frontier, safety & reward robustness to noise,
  difference curves, tolerance intervals, and `policy_grids/`.

## Configuration

All shared constants live in [`config.py`](config.py): environment size, horizon,
ω values, training/eval seeds, the real-data file, and the safety percentage `z`.
Evaluation seeds never overlap the training seed (42), so there is no train/eval
leakage. Algorithm-specific hyperparameters stay inside each agent script.

Generated outputs (`*_results_tuned_*/`, `evaluation_results_*/`,
`model_threshold_figures/`) are git-ignored.
