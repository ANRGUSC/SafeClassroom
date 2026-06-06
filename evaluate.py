"""
Unified evaluation harness for SafeCampus RL agents.

Evaluates all agents on sinusoidal risk (in-distribution, 30 seeds) or
on the real-world CSV risk trajectory (10 seeds, env stochasticity only).

Agents (5-agent consolidated pipeline): Myopic, Double DQN,
        PPO Discrete, PPO Continuous, Upper Bound

Outputs (per eval mode, in evaluation_results_{sinusoidal,data}/):
  summary.csv                              — mean ± std + bootstrap CI + monotonicity + X*/Y*/F
  safety_optimal_thresholds.csv            — X*, Y*, F (optimal-threshold safety)
  safety_optimal_thresholds_perturbation.csv — X*/Y*/F under each perturbation level
  fig1_reward_vs_omega.png                 — mean reward with bootstrap 95% CI bands
  fig2_optimality_gap.png                  — % of upper bound (primary metric)
  fig3_monotonicity.png                    — Spearman rho_mean by agent and omega
  fig4_trajectory_omega*.png               — week-by-week trajectory per omega
  fig5_safety_optimal_thresholds.png       — optimal-threshold safety table (incl. F)
  fig_safety_frontier.png                  — X*, Y*, F vs omega
  fig_safety_perturbation.png              — F vs sensing-noise level (robustness)
  policy_grids/                            — per-agent and cross-agent policy plots
"""

import os
import json
import argparse
import colorsys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from scipy.stats import spearmanr, binom
import torch
import torch.nn as nn

from config import (
    TOTAL_STUDENTS, MAX_WEEKS, NUM_ACTION_LEVELS, OMEGA_VALUES,
    GLOBAL_SEED, EVAL_SEEDS_SINUSOIDAL, EVAL_SEEDS_DATA,
    REAL_DATA_FILE, FIXED_ALPHA, FIXED_BETA,
    SAFETY_Z, SAFETY_PERTURBATION_LEVELS, TRAIN_RISK_TYPE
)
from campus_gym.envs import ClassroomGymEnv
from optimal_dp_policy import UpperBoundSolver

# ============================================================
# 0. CONFIGURATION
# ============================================================
POLICY_GRID_POINTS = 50

# Default eval mode (overridable via CLI)
EVAL_RISK_TYPE = 'sinusoidal'   # primary evaluation; 'data' = real-world CSV


def _output_dir(risk_type):
    return f"evaluation_results_{risk_type}"


def _get_eval_seeds(risk_type: str):
    """Return the correct seed list for the given risk type."""
    if risk_type == 'sinusoidal':
        return EVAL_SEEDS_SINUSOIDAL
    elif risk_type == 'data':
        return EVAL_SEEDS_DATA
    else:
        raise ValueError(f"Unknown risk_type: {risk_type!r}")


# Artifact directories and filenames per agent. `dir_base` is suffixed with
# `_{eval_risk_type}` at load time so sinusoidal and data runs are isolated
# (and match what each training script writes when run with --eval-risk-type).
# 5-agent consolidated pipeline (REVISION R1). Retired agents' result folders are
# kept on disk but are no longer loaded or plotted. "PPO Continuous" is the Beta
# policy from ppo_continuous_new.py (display name only; dir base unchanged).
ARTIFACTS = {
    "Double DQN":            {"dir_base": "online_ddqn_results_tuned",        "fmt": "policy_ddqn_online_omega_{omega}.pth",        "type": "dqn"},
    "PPO Discrete":          {"dir_base": "ppo_discrete_results_tuned",       "fmt": "ppo_policy_optimal_omega_{omega}.pth",        "type": "ppo_discrete"},
    "PPO Continuous":        {"dir_base": "ppo_continuous_new_results_tuned", "fmt": "ppo_policy_optimal_omega_{omega}.pth",        "type": "ppo_continuous_beta"},
    "Myopic":                {"dir_base": None, "fmt": None, "type": "myopic"},
    "Critical Capacity":     {"dir_base": None, "fmt": None, "type": "critical_capacity"},
    "Upper Bound":           {"dir_base": None, "fmt": None, "type": "upper_bound"},
}


def _artifact_dir(agent_name, eval_risk_type):
    """Resolve the agent's per-risk-type artifact directory."""
    base = ARTIFACTS[agent_name]["dir_base"]
    if base is None:
        return None
    return f"{base}_{eval_risk_type}"

AGENT_ORDER = ["Myopic", "Double DQN", "PPO Discrete", "PPO Continuous",
               "Critical Capacity", "Upper Bound"]

AGENT_COLORS = {
    "Myopic":                "#E41A1C",
    "Double DQN":            "#F781BF",
    "PPO Discrete":          "#377EB8",
    "PPO Continuous":        "#4DAF4A",
    "Critical Capacity":     "#FF7F00",
    "Upper Bound":           "#000000",
}
AGENT_MARKERS = {
    "Myopic": "o", "Double DQN": "P",
    "PPO Discrete": "^", "PPO Continuous": "*",
    "Critical Capacity": "D", "Upper Bound": "h",
}
AGENT_LINESTYLES = {
    "Myopic": "--", "Double DQN": "-.",
    "PPO Discrete": "-", "PPO Continuous": "-",
    "Critical Capacity": "--", "Upper Bound": "-",
}

device = torch.device("cpu")

plt.rcParams.update({
    "font.size": 11,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "lines.linewidth": 2.0,
    "figure.dpi": 150,
})


# ============================================================
# 1. NETWORK DEFINITIONS
# ============================================================

class QNetwork(nn.Module):
    """Shared architecture for DQN, Online DQN, Double DQN."""
    def __init__(self, input_dim=2, output_dim=3, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.fc(x)


class ActorCriticDiscrete(nn.Module):
    """PPO-Discrete actor-critic (Softmax actor)."""
    def __init__(self, state_dim=2, action_dim=3, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, action_dim), nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )


class ActorCriticContinuousBeta(nn.Module):
    """PPO-Continuous actor-critic with Beta distribution (ppo_continuous_new.py)."""
    def __init__(self, state_dim=2, action_dim=1, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.conc_head = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def get_mean_action(self, state):
        h  = self.shared(state)
        mu = torch.clamp(torch.sigmoid(self.mean_head(h)), 1e-3, 1 - 1e-3)
        return mu


class ActorCriticContinuousNormal(nn.Module):
    """PPO-Continuous actor-critic with Normal distribution (ppo_continuous_actions.py).
    The actor's final layer is Tanh in [-1, 1]; the mean action is shifted to [0, 1]
    by `(mu_raw + 1) / 2` (matching the training-time `act` method)."""
    def __init__(self, state_dim=2, action_dim=1, hidden_dim=64, init_std=0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.log_std = nn.Parameter(torch.ones(1, action_dim) * float(np.log(init_std)))
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, action_dim), nn.Tanh(),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def get_mean_action(self, state):
        mu_raw = self.actor(state)
        return (mu_raw + 1.0) / 2.0


# ============================================================
# 2. AGENT LOADING
# ============================================================

def _load_hidden_dim(artifact_dir, omega, default=128):
    """Read tuned hidden_dim from saved JSON, else use default."""
    for fname in ("optimized_lrs.json", "optimized_lrs_online.json",
                  "optimized_lrs_ddqn_online.json"):
        path = os.path.join(artifact_dir, fname)
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            entry = data.get(str(omega)) or data.get(f"{omega:.1f}")
            if entry and isinstance(entry, dict):
                return int(entry.get("hidden_dim", default))
    return default


def load_agent_policy(agent_name, omega, eval_risk_type):
    """
    Load a saved agent artifact (from the eval-risk-type-specific dir) and
    return a unified policy callable:
        capacity = policy(infected: float, risk: float, week: int) -> float
    capacity is in [0, TOTAL_STUDENTS].

    Returns None if the artifact file is not found.
    Returns "UPPER_BOUND_SENTINEL" for the (clairvoyant) Upper Bound agent.
    """
    info = ARTIFACTS[agent_name]
    agent_type = info["type"]

    # ---- Myopic: greedy one-step lookahead ----
    if agent_type == "myopic":
        capacity_options = [0, 50, 100]

        def myopic_policy(infected, risk, week, omega=omega):
            best_cap, best_val = 0, -float("inf")
            for cap in capacity_options:
                pred = FIXED_ALPHA * infected * cap + FIXED_BETA * risk * cap ** 2
                pred = min(pred, TOTAL_STUDENTS)
                val = omega * cap - (1 - omega) * pred
                if val > best_val:
                    best_val, best_cap = val, cap
            return float(best_cap)

        return myopic_policy

    # ---- Critical Capacity: analytical DFE-enforcing threshold policy ----
    # Sets u(t)=clip(u*(t),0,N) with u*(t) the positive root of R0=1, observing
    # only c_risk(t). Independent of omega and of I(t); guarantees R0<1 every week.
    if agent_type == "critical_capacity":
        def cc_policy(infected, risk, week,
                      alpha=FIXED_ALPHA, beta=FIXED_BETA, N=TOTAL_STUDENTS):
            if risk <= 1e-9:
                return float(N)   # no community transmission -> full room is DFE
            disc = alpha ** 2 + 4.0 * beta * risk
            u_star = (-alpha + np.sqrt(disc)) / (2.0 * beta * risk)
            return float(np.clip(u_star, 0.0, N))
        return cc_policy

    # ---- Upper Bound: clairvoyant DP, built per (omega, seed) at eval time ----
    if agent_type == "upper_bound":
        return "UPPER_BOUND_SENTINEL"

    # ---- File-based agents ----
    artifact_dir  = _artifact_dir(agent_name, eval_risk_type)
    artifact_path = os.path.join(artifact_dir, info["fmt"].format(omega=omega))
    if not os.path.exists(artifact_path):
        warnings.warn(f"Artifact not found: {artifact_path}")
        return None

    # ---- Tabular Q ----
    if agent_type == "q_table":
        Q = np.load(artifact_path)
        num_levels = Q.shape[0]
        capacity_options = [0, 50, 100]

        def q_policy(infected, risk, week,
                     Q=Q, nl=num_levels, cap=capacity_options):
            i_idx = int(np.clip(infected / TOTAL_STUDENTS * (nl - 1), 0, nl - 1))
            r_idx = int(np.clip(risk * (nl - 1), 0, nl - 1))
            return float(cap[int(np.argmax(Q[i_idx, r_idx]))])

        return q_policy

    # ---- DQN variants ----
    if agent_type == "dqn":
        hidden_dim = _load_hidden_dim(artifact_dir, omega)
        net = QNetwork(input_dim=2, output_dim=NUM_ACTION_LEVELS, hidden_dim=hidden_dim)
        net.load_state_dict(torch.load(artifact_path, map_location=device))
        net.eval()
        cap = [0, 50, 100]
        s_buf = torch.zeros(1, 2)

        def dqn_policy(infected, risk, week, net=net, s_buf=s_buf, cap=cap):
            s_buf[0, 0] = infected / TOTAL_STUDENTS
            s_buf[0, 1] = risk
            with torch.no_grad():
                return float(cap[int(net(s_buf).argmax().item())])

        return dqn_policy

    # ---- PPO Discrete ----
    if agent_type == "ppo_discrete":
        hidden_dim = _load_hidden_dim(artifact_dir, omega, default=64)
        ac = ActorCriticDiscrete(state_dim=2, action_dim=NUM_ACTION_LEVELS, hidden_dim=hidden_dim)
        ac.load_state_dict(torch.load(artifact_path, map_location=device))
        ac.eval()
        cap = [0, 50, 100]
        s_buf = torch.zeros(1, 2)

        def ppo_d_policy(infected, risk, week, ac=ac, s_buf=s_buf, cap=cap):
            s_buf[0, 0] = infected / TOTAL_STUDENTS
            s_buf[0, 1] = risk
            with torch.no_grad():
                return float(cap[int(torch.argmax(ac.actor(s_buf)).item())])

        return ppo_d_policy

    # ---- PPO Continuous (Beta) — ppo_continuous_new.py architecture ----
    if agent_type == "ppo_continuous_beta":
        hidden_dim = _load_hidden_dim(artifact_dir, omega, default=64)
        ac = ActorCriticContinuousBeta(state_dim=2, action_dim=1, hidden_dim=hidden_dim)
        ac.load_state_dict(torch.load(artifact_path, map_location=device))
        ac.eval()
        s_buf = torch.zeros(1, 2)

        def ppo_c_beta_policy(infected, risk, week, ac=ac, s_buf=s_buf):
            s_buf[0, 0] = infected / TOTAL_STUDENTS
            s_buf[0, 1] = risk
            with torch.no_grad():
                mu = float(ac.get_mean_action(s_buf).item())
            return mu * TOTAL_STUDENTS

        return ppo_c_beta_policy

    # ---- PPO Continuous (Normal) — ppo_continuous_actions.py architecture ----
    if agent_type == "ppo_continuous_normal":
        hidden_dim = _load_hidden_dim(artifact_dir, omega, default=64)
        ac = ActorCriticContinuousNormal(state_dim=2, action_dim=1, hidden_dim=hidden_dim)
        ac.load_state_dict(torch.load(artifact_path, map_location=device))
        ac.eval()
        s_buf = torch.zeros(1, 2)

        def ppo_c_normal_policy(infected, risk, week, ac=ac, s_buf=s_buf):
            s_buf[0, 0] = infected / TOTAL_STUDENTS
            s_buf[0, 1] = risk
            with torch.no_grad():
                mu = float(ac.get_mean_action(s_buf).item())
            return mu * TOTAL_STUDENTS

        return ppo_c_normal_policy

    raise ValueError(f"Unknown agent type: {agent_type}")


# ============================================================
# 3. ENVIRONMENT FACTORY
# ============================================================

def make_eval_env(omega, seed, risk_type, perturbation_level=None):
    """Build a continuous-action eval env for the given risk type.

    perturbation_level: None or "none" -> deterministic eval; one of
    "low"/"moderate"/"high"/"severe" -> eval-perturbed mode, which injects
    Gaussian sensing noise into the observed infection count (the true SIR
    value is preserved in info["unperturbed_infected"])."""
    perturbed = perturbation_level not in (None, "none")
    mode = "eval-perturbed" if perturbed else "eval"
    common = dict(
        mode=mode,
        use_discrete_state=False,
        use_continuous_actions=True,
        num_action_levels=NUM_ACTION_LEVELS,
        total_students=TOTAL_STUDENTS,
        max_weeks=MAX_WEEKS,
        omega=omega,
        seed=seed,
    )
    if perturbed:
        common["perturbation_level"] = perturbation_level
    if risk_type == 'data':
        return ClassroomGymEnv(community_risk_data_file=REAL_DATA_FILE, **common)
    else:  # sinusoidal
        return ClassroomGymEnv(eval_risk_type="sinusoidal", **common)


# ============================================================
# 4. ROLLOUT
# ============================================================

def rollout(policy_fn, omega, seed, risk_type, upper_bound_solver=None,
            perturbation_level=None):
    """
    Run one episode and return a metrics dict.
    All policies return a capacity in [0, TOTAL_STUDENTS].

    Under perturbation, the agent observes a noisy infection count, but the
    metrics dict records BOTH the observed series ("infected") and the true
    SIR series ("infected_true"). Safety guarantees are measured on the true
    series; clean (perturbation=none) eval has the two series identical.
    """
    env = make_eval_env(omega, seed, risk_type, perturbation_level=perturbation_level)
    env.omega = omega
    raw_state, _ = env.reset(seed=seed)

    infected_list, infected_true_list = [], []
    allowed_list, risk_list, reward_list = [], [], []
    weeks = []

    done = False
    while not done:
        infected = float(raw_state[0])
        risk     = float(raw_state[1])
        week     = int(env.current_week)

        if upper_bound_solver is not None:
            capacity = float(upper_bound_solver.get_optimal_capacity(week, infected, risk))
        else:
            capacity = float(policy_fn(infected, risk, week))

        capacity = float(np.clip(capacity, 0.0, TOTAL_STUDENTS))
        action   = np.array([capacity / TOTAL_STUDENTS], dtype=np.float32)

        next_raw, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        weeks.append(week)
        infected_list.append(info["infected_students"])
        # True (unperturbed) outbreak size for safety; equals observed when clean.
        infected_true_list.append(info.get("unperturbed_infected", info["infected_students"]))
        allowed_list.append(info["allowed_students"])
        risk_list.append(info["community_risk"])
        reward_list.append(reward)

        raw_state = next_raw

    return {
        "total_reward":        sum(reward_list),
        "peak_infected":       max(infected_list),
        "cumulative_infected": sum(infected_list),
        "mean_infected":       float(np.mean(infected_list)),
        "mean_attendance":     float(np.mean(allowed_list)),
        "attendance_fraction": float(np.mean(allowed_list)) / TOTAL_STUDENTS,
        "weeks":               weeks,
        "infected":            infected_list,
        "infected_true":       infected_true_list,
        "allowed":             allowed_list,
        "risk":                risk_list,
        "rewards":             reward_list,
    }


# ============================================================
# 5. UPPER BOUND SOLVER
# ============================================================

# ── DP discretisation ────────────────────────────────────────
# Granularity of the (I, r) grid used by UpperBoundSolver. Defaults match the
# env's natural resolution: infected counts are integers in {0,...,N}, so
# n_infected_bins=N+1=101 makes every count its own bin (zero infected-axis
# discretisation error); risk is continuous in [0,1], 101 bins -> snap error
# <=0.005. _optimize_capacity is vectorised so the finer grid is cheap.
UB_INFECTED_BINS = 101
UB_RISK_BINS     = 101


def build_upper_bound_solver(omega, seed, risk_type,
                             n_infected_bins=UB_INFECTED_BINS,
                             n_risk_bins=UB_RISK_BINS):
    """Build and solve the DP upper bound for the given risk type / seed.

    n_infected_bins, n_risk_bins control the (I, r) discretisation. Bumping
    them yields a tighter ceiling at O(n_inf * n_risk * T) extra DP cells; the
    inner capacity search is vectorised over candidates so the marginal cost
    is small."""
    env = make_eval_env(omega, seed, risk_type)
    env.reset(seed=seed)
    solver = UpperBoundSolver(env,
                              n_infected_bins=n_infected_bins,
                              n_risk_bins=n_risk_bins)
    solver.solve(verbose=False)
    return solver


# ============================================================
# 6. MONOTONICITY METRIC
# ============================================================

def compute_monotonicity(policy_fn, n_grid=40):
    """
    Compute Spearman rank correlation between state variables and policy action.
    Negative rho = policy reduces attendance as risk/infections increase (monotonic).
    """
    infected_vals = np.linspace(0, TOTAL_STUDENTS, n_grid)
    risk_vals     = np.linspace(0, 1.0, n_grid)

    I_flat, R_flat, A_flat = [], [], []
    for inf in infected_vals:
        for crisk in risk_vals:
            action = policy_fn(inf, crisk)
            I_flat.append(inf)
            R_flat.append(crisk)
            A_flat.append(action)

    rho_I, _ = spearmanr(I_flat, A_flat)
    rho_r, _ = spearmanr(R_flat, A_flat)
    if np.isnan(rho_I):
        rho_I = 0.0
    if np.isnan(rho_r):
        rho_r = 0.0
    return {
        "rho_infected": round(float(rho_I), 4),
        "rho_risk":     round(float(rho_r), 4),
        "rho_mean":     round(float((rho_I + rho_r) / 2), 4),
    }


def _make_monotonicity_policy_fn(agent_name, policy, ub_solver_for_mono):
    """
    Adapt a (capacity = policy(infected, risk, week)) callable to a
    (action = fn(infected, risk)) callable suitable for monotonicity.
    For Upper Bound, use the representative seed's solver and evaluate at t=0.
    """
    if agent_name == "Upper Bound":
        if ub_solver_for_mono is None:
            return None

        def fn(infected, risk):
            return float(ub_solver_for_mono.get_optimal_capacity(0, infected, risk))
        return fn

    def fn(infected, risk):
        return float(policy(infected, risk, 0))
    return fn


# ============================================================
# 7. SAFETY EVALUATION (Optimal Threshold Search)
# ============================================================

def optimal_threshold_search(I, U, z):
    """
    Binary-search the optimal infection threshold X* and attendance threshold Y*
    for pooled per-week infection counts I and attendance counts U at safety
    percentage z (a probabilistic bound: each constraint must hold ≥ z% of weeks):

        X* = smallest integer x with  (% of weeks I_t > x)  <= (100 - z)
        Y* = largest  integer y with  (% of weeks u_t >= y) >= z

    X* is the tightest infection ceiling the policy respects; Y* the highest
    attendance floor it delivers. Equivalent to the (1 - z/100) upper / (z/100)
    lower empirical quantiles of the pooled trajectories.
    """
    I = np.asarray(I, dtype=np.int64)
    U = np.asarray(U, dtype=np.int64)
    if I.size == 0 or U.size == 0:
        return 0, 0

    x_l, x_h = 0, int(I.max())
    x_star = x_h
    while x_l <= x_h:
        x_m = (x_l + x_h) // 2
        if 100.0 * np.mean(I > x_m) <= (100 - z):
            x_star = x_m
            x_h = x_m - 1
        else:
            x_l = x_m + 1

    y_l, y_h = 0, int(U.max())
    y_star = y_l
    while y_l <= y_h:
        y_m = (y_l + y_h) // 2
        if 100.0 * np.mean(U >= y_m) >= z:
            y_star = y_m
            y_l = y_m + 1
        else:
            y_h = y_m - 1

    return int(x_star), int(y_star)


def safety_from_trajectories(infected_pool, allowed_pool, omega, z=SAFETY_Z):
    """Compute the optimal-threshold safety record from pooled per-week series.
    `infected_pool` should be the TRUE (unperturbed) infection counts."""
    x_star, y_star = optimal_threshold_search(infected_pool, allowed_pool, z)
    I_arr = np.asarray(infected_pool)
    U_arr = np.asarray(allowed_pool)
    pct_I_exceed  = 100.0 * float(np.mean(I_arr > x_star)) if I_arr.size else 0.0
    pct_A_atleast = 100.0 * float(np.mean(U_arr >= y_star)) if U_arr.size else 0.0
    return {
        "optimal_x":  int(x_star),
        "optimal_y":  int(y_star),
        "pct_I_gt_x": round(pct_I_exceed, 2),
        "safety_I":   bool(pct_I_exceed <= (100 - z)),
        "pct_A_ge_y": round(pct_A_atleast, 2),
        "safety_A":   bool(pct_A_atleast >= z),
        "F_score":    round(omega * y_star - (1.0 - omega) * x_star, 4),
    }


# ============================================================
# 8. SUMMARY HELPERS
# ============================================================

def _mean_std(vals, ndigits=4):
    """Mean and sample std (ddof=1) across seeds. std is NaN for n<2 (e.g. the
    single-trajectory data eval), matching the reward summary convention."""
    a = np.asarray(vals, dtype=float)
    mu = round(float(a.mean()), ndigits)
    sd = round(float(a.std(ddof=1)), ndigits) if a.size > 1 else float("nan")
    return mu, sd


def bootstrap_ci(rewards, n_boot=10_000, alpha=0.05):
    """
    Percentile bootstrap 95% CI for the mean. Returns (lower, upper).
    Recommended by Patterson et al. (2024) as the default CI method for RL.
    Deterministic (fixed rng seed) for reproducibility.
    """
    arr = np.asarray(rewards, dtype=float)
    if arr.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed=0)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot_means = arr[idx].mean(axis=1)
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return lo, hi


def summarise_rewards(rewards):
    """Return mean/std + analytic (Wald) and bootstrap 95% CIs from per-seed rewards.

    The analytic CI (mean ± 1.96·s/√n) goes in the summary CSV so values are
    deterministic and reproducible; the percentile bootstrap CI is reported in
    figures (R5a). A single trajectory (n < 2, e.g. the deterministic real-data
    eval) has no sampling distribution, so all spread metrics are NaN.
    """
    n = len(rewards)
    mu = float(np.mean(rewards))
    if n < 2:
        return {
            "n_seeds":     n,
            "mean_reward": round(mu, 4),
            "std_reward":  float("nan"),
            "ci95_lower":  float("nan"),
            "ci95_upper":  float("nan"),
            "ci95_bootstrap_lower": float("nan"),
            "ci95_bootstrap_upper": float("nan"),
        }
    sigma = float(np.std(rewards, ddof=1))
    se = sigma / np.sqrt(n)
    boot_lo, boot_hi = bootstrap_ci(rewards)
    return {
        "n_seeds":     n,
        "mean_reward": round(mu, 4),
        "std_reward":  round(sigma, 4),
        "ci95_lower":  round(mu - 1.96 * se, 4),
        "ci95_upper":  round(mu + 1.96 * se, 4),
        "ci95_bootstrap_lower": round(boot_lo, 4),
        "ci95_bootstrap_upper": round(boot_hi, 4),
    }


# ============================================================
# 9. MAIN EVALUATION LOOP
# ============================================================

def run_all(risk_type, output_dir):
    """
    Evaluate all agents for all omega values under the given risk type.

    Sinusoidal: multiple seeds (each draws an independent risk wave) -> mean ± std
                and a 95% CI across seeds.
    Real data:  a single deterministic trajectory (fixed CSV risk, fixed initial
                infected, deterministic dynamics). The seed is inert here, so we
                run exactly one rollout and report the reward without std/CI.
    """
    eval_seeds = _get_eval_seeds(risk_type)
    if risk_type == 'data':
        eval_seeds = eval_seeds[:1]   # single deterministic trajectory; seed is inert
        risk_label = f"real CSV ({REAL_DATA_FILE}) — single deterministic trajectory"
    else:
        risk_label = f"sinusoidal x{len(eval_seeds)} seeds"
    print(f"Evaluation mode : {risk_label}")

    # Build UB solvers — one per (omega, seed) combination
    print("Building Upper Bound solvers...")
    ub_solvers = {}
    for omega in OMEGA_VALUES:
        ub_solvers[omega] = {}
        for seed in eval_seeds:
            ub_solvers[omega][seed] = build_upper_bound_solver(omega, seed, risk_type)
        print(f"  UB solved: omega={omega} ({len(eval_seeds)} seed(s))")

    # Per-omega representative UB solver for monotonicity / policy grids
    rep_seed = eval_seeds[0]

    records = []
    safety_records = []
    # Paired per-seed rewards for difference curves / tolerance intervals (R5b/R5c).
    # Keyed per_seed_rewards[omega][agent] = {seed: total_reward} so agent–baseline
    # pairs share the same seed index when resampling.
    per_seed_rewards = {omega: {} for omega in OMEGA_VALUES}
    print("\nEvaluating agents...")
    for omega in OMEGA_VALUES:
        print(f"\n=== omega={omega} ===")

        for agent_name in AGENT_ORDER:
            policy = load_agent_policy(agent_name, omega, risk_type)
            if policy is None:
                print(f"  SKIP {agent_name} (artifact missing)")
                continue

            is_ub = (policy == "UPPER_BOUND_SENTINEL")
            seed_results = []
            seed_reward_map = {}

            for seed in eval_seeds:
                try:
                    m = rollout(
                        policy_fn=None if is_ub else policy,
                        omega=omega,
                        seed=seed,
                        risk_type=risk_type,
                        upper_bound_solver=ub_solvers[omega][seed] if is_ub else None,
                    )
                    seed_results.append(m)
                    seed_reward_map[seed] = m["total_reward"]
                except Exception as e:
                    warnings.warn(f"Rollout failed {agent_name}/omega={omega}/seed={seed}: {e}")

            if not seed_results:
                continue

            rewards_per_seed = [r["total_reward"] for r in seed_results]
            per_seed_rewards[omega][agent_name] = seed_reward_map
            stats = summarise_rewards(rewards_per_seed)

            # Optimal-threshold safety: pool TRUE per-week infection / attendance
            # across all seeds (clean eval => infected_true == observed).
            # Critical Capacity is EXCLUDED from the empirical safety table: it
            # enforces R0<1 by construction, so its thresholds are analytical, not
            # reward-derived. Its summary safety columns are left as NaN.
            is_cc = (ARTIFACTS[agent_name]["type"] == "critical_capacity")
            if is_cc:
                safety = {"optimal_x": float("nan"), "optimal_y": float("nan"),
                          "F_score": float("nan")}
            else:
                infected_pool = [v for r in seed_results for v in r["infected_true"]]
                allowed_pool  = [v for r in seed_results for v in r["allowed"]]
                safety = safety_from_trajectories(infected_pool, allowed_pool, omega)

            # Monotonicity: use representative UB solver for Upper Bound
            mono_policy = _make_monotonicity_policy_fn(
                agent_name, policy,
                ub_solver_for_mono=ub_solvers[omega][rep_seed] if is_ub else None,
            )
            mono = compute_monotonicity(mono_policy) if mono_policy is not None \
                   else {"rho_infected": 0.0, "rho_risk": 0.0, "rho_mean": 0.0}

            records.append({
                "agent":                    agent_name,
                "omega":                    omega,
                "n_seeds":                  stats["n_seeds"],
                "mean_reward":              stats["mean_reward"],
                "std_reward":               stats["std_reward"],
                "ci95_lower":               stats["ci95_lower"],
                "ci95_upper":               stats["ci95_upper"],
                "ci95_bootstrap_lower":     stats["ci95_bootstrap_lower"],
                "ci95_bootstrap_upper":     stats["ci95_bootstrap_upper"],
                "mean_peak_infected":       _mean_std([r["peak_infected"]       for r in seed_results])[0],
                "std_peak_infected":        _mean_std([r["peak_infected"]       for r in seed_results])[1],
                "mean_cumulative_infected": _mean_std([r["cumulative_infected"] for r in seed_results])[0],
                "std_cumulative_infected":  _mean_std([r["cumulative_infected"] for r in seed_results])[1],
                "mean_infected":            _mean_std([r["mean_infected"]        for r in seed_results])[0],
                "std_infected":             _mean_std([r["mean_infected"]        for r in seed_results])[1],
                "mean_attendance":          _mean_std([r["mean_attendance"]     for r in seed_results])[0],
                "std_attendance":           _mean_std([r["mean_attendance"]     for r in seed_results])[1],
                "attendance_fraction":      _mean_std([r["attendance_fraction"] for r in seed_results])[0],
                "std_attendance_fraction":  _mean_std([r["attendance_fraction"] for r in seed_results])[1],
                "rho_infected":             mono["rho_infected"],
                "rho_risk":                 mono["rho_risk"],
                "rho_mean":                 mono["rho_mean"],
                "optimal_x":                safety["optimal_x"],
                "optimal_y":                safety["optimal_y"],
                "safety_F":                 safety["F_score"],
            })

            if not is_cc:   # CC excluded from the empirical safety table
                safety_records.append({
                    "agent":      agent_name,
                    "omega":      omega,
                    "z":          SAFETY_Z,
                    "optimal_x":  safety["optimal_x"],
                    "optimal_y":  safety["optimal_y"],
                    "pct_I_gt_x": safety["pct_I_gt_x"],
                    "safety_I":   safety["safety_I"],
                    "pct_A_ge_y": safety["pct_A_ge_y"],
                    "safety_A":   safety["safety_A"],
                    "F_score":    safety["F_score"],
                })

            if np.isnan(stats["std_reward"]):
                reward_str = f"R={stats['mean_reward']:.2f} (single trajectory)"
            else:
                reward_str = (f"R={stats['mean_reward']:.2f}±{stats['std_reward']:.2f} "
                              f"[{stats['ci95_lower']:.2f},{stats['ci95_upper']:.2f}]")
            print(f"  {agent_name:22s} | {reward_str} | "
                  f"peak_I={records[-1]['mean_peak_infected']:.0f} | "
                  f"rho_mean={mono['rho_mean']:+.2f}")

    df = pd.DataFrame(records)

    # pct_of_upper_bound
    ub_row = df[df["agent"] == "Upper Bound"].set_index("omega")["mean_reward"]
    def pct_ub(row):
        ub = ub_row.get(row["omega"])
        if ub is not None and ub != 0:
            return round(100.0 * row["mean_reward"] / ub, 2)
        return np.nan
    df["pct_of_upper_bound"] = df.apply(pct_ub, axis=1)

    # Reorder summary CSV columns (R6 schema + optimal-threshold safety)
    columns = [
        "agent", "omega", "n_seeds",
        "mean_reward", "std_reward",
        "ci95_lower", "ci95_upper",
        "ci95_bootstrap_lower", "ci95_bootstrap_upper",
        "mean_peak_infected", "std_peak_infected",
        "mean_cumulative_infected", "std_cumulative_infected",
        "mean_infected", "std_infected",
        "mean_attendance", "std_attendance",
        "attendance_fraction", "std_attendance_fraction",
        "pct_of_upper_bound",
        "rho_infected", "rho_risk", "rho_mean",
        "optimal_x", "optimal_y", "safety_F",
    ]
    df = df[columns]

    df.to_csv(os.path.join(output_dir, "summary.csv"), index=False)
    print(f"\nSummary saved to {output_dir}/summary.csv")

    df_safety = pd.DataFrame(safety_records)
    df_safety.to_csv(os.path.join(output_dir, "safety_optimal_thresholds.csv"), index=False)
    print(f"Safety (optimal thresholds) saved to {output_dir}/safety_optimal_thresholds.csv")

    return df, df_safety, ub_solvers, per_seed_rewards


# ============================================================
# 10. FIGURE 2 — Optimality gap (primary metric)
# ============================================================

def plot_fig2_optimality_gap(df, output_dir):
    """Bar/line chart: % of Upper Bound per agent, per omega."""
    print("Generating Figure 2: optimality gap (pct_of_upper_bound)...")
    agents = [a for a in AGENT_ORDER if a in df["agent"].values and a != "Upper Bound"]
    omegas = sorted(df["omega"].unique())

    fig, ax = plt.subplots(figsize=(10, 6))
    for ag in agents:
        sub = df[df["agent"] == ag].sort_values("omega")
        if sub.empty:
            continue
        ax.plot(sub["omega"], sub["pct_of_upper_bound"],
                color=AGENT_COLORS.get(ag, "gray"),
                marker=AGENT_MARKERS.get(ag, "o"),
                linestyle=AGENT_LINESTYLES.get(ag, "-"),
                label=ag, linewidth=2, markersize=8)

    ax.axhline(100.0, color="black", linestyle="--", linewidth=1.5, alpha=0.7,
               label="Upper Bound (100%)")
    ax.set_xlabel("ω (attendance weight)")
    ax.set_ylabel("% of Upper Bound reward")
    ax.set_title("Optimality Gap — % of Upper Bound reward")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", fontsize=9, ncol=2, frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig2_optimality_gap.png"),
                dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# 11. FIGURE 1 — Reward vs omega with 95% CI bands
# ============================================================

def plot_fig1_reward_vs_omega(df, output_dir):
    """Mean reward per omega with shaded 95% CI bands per agent.
    CI bands are drawn only when available (sinusoidal multi-seed eval);
    the deterministic real-data eval has no CI, so only the reward line is shown."""
    print("Generating Figure 1: reward vs omega with 95% CI bands...")
    agents = [a for a in AGENT_ORDER if a in df["agent"].values]

    has_ci = False
    fig, ax = plt.subplots(figsize=(10, 6))
    for ag in agents:
        sub = df[df["agent"] == ag].sort_values("omega")
        if sub.empty:
            continue
        omegas = sub["omega"].values
        mean   = sub["mean_reward"].values
        ci_lo  = sub["ci95_bootstrap_lower"].values
        ci_hi  = sub["ci95_bootstrap_upper"].values

        color = AGENT_COLORS.get(ag, "gray")
        ax.plot(omegas, mean,
                color=color,
                marker=AGENT_MARKERS.get(ag, "o"),
                linestyle=AGENT_LINESTYLES.get(ag, "-"),
                label=ag, linewidth=2, markersize=7)
        if not np.all(np.isnan(ci_lo)):
            ax.fill_between(omegas, ci_lo, ci_hi, color=color, alpha=0.2)
            has_ci = True

    ax.set_xlabel("ω (attendance weight)")
    ax.set_ylabel("Mean total reward")
    ax.set_title("Mean Reward vs ω  (shaded = bootstrap 95% CI)" if has_ci
                 else "Reward vs ω  (single deterministic trajectory)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", fontsize=9, ncol=2, frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig1_reward_vs_omega.png"),
                dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# 12. FIGURE 3 — Monotonicity
# ============================================================

def plot_monotonicity(df, output_dir):
    """rho_mean vs omega per agent."""
    print("Generating Figure 3: monotonicity (rho_mean) vs omega...")
    agents = [a for a in AGENT_ORDER if a in df["agent"].values]

    fig, ax = plt.subplots(figsize=(10, 6))
    for ag in agents:
        sub = df[df["agent"] == ag].sort_values("omega")
        if sub.empty:
            continue
        ax.plot(sub["omega"], sub["rho_mean"],
                color=AGENT_COLORS.get(ag, "gray"),
                marker=AGENT_MARKERS.get(ag, "o"),
                linestyle=AGENT_LINESTYLES.get(ag, "-"),
                label=ag, linewidth=2, markersize=8)

    ax.axhline(0.0, color="black", linestyle=":", linewidth=1.0, alpha=0.5)
    ax.set_xlabel("ω (attendance weight)")
    ax.set_ylabel("rho_mean  (Spearman, avg of infected & risk)")
    ax.set_title("Policy Monotonicity — negative ρ = capacity ↓ as risk/infected ↑")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", fontsize=9, ncol=2, frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig3_monotonicity.png"),
                dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# 13. FIGURE 5 — Safety: optimal-threshold table + frontier
# ============================================================

def plot_safety_threshold_table(df_safety, output_dir, risk_type, z=SAFETY_Z):
    """Render the optimal-threshold safety table as a publication figure.
    Rows grouped by ω (alternating shade); best F per ω highlighted green
    (higher F = better infection/attendance trade-off at the policy's own ω)."""
    print("Generating Figure 5: optimal-threshold safety table...")
    omegas = sorted(df_safety["omega"].unique())
    yn = {True: "Yes", False: "No"}
    col_labels = ["Policy", "ω", "X*", "Y*", "I>X* (%)",
                  "Safe(I)", "u≥Y* (%)", "Safe(A)", "F"]

    cell_text, row_colors, best_F_rows = [], [], []
    shades = ["#eef3f8", "#ffffff"]
    ri = 0
    for gi, om in enumerate(omegas):
        sub = df_safety[df_safety["omega"] == om]
        ordered = [a for a in AGENT_ORDER if a in sub["agent"].values]
        f_vals = {a: float(sub[sub["agent"] == a].iloc[0]["F_score"]) for a in ordered}
        best = max(f_vals, key=f_vals.get) if f_vals else None
        for a in ordered:
            r = sub[sub["agent"] == a].iloc[0]
            cell_text.append([
                a, f"{om}", f"{int(r['optimal_x'])}", f"{int(r['optimal_y'])}",
                f"{r['pct_I_gt_x']:.1f}", yn[bool(r['safety_I'])],
                f"{r['pct_A_ge_y']:.1f}", yn[bool(r['safety_A'])],
                f"{r['F_score']:.2f}",
            ])
            row_colors.append(shades[gi % 2])
            if a == best:
                best_F_rows.append(ri)
            ri += 1

    n_rows, n_cols = len(cell_text), len(col_labels)
    if n_rows == 0:
        return
    cell_colours = [[row_colors[i]] * n_cols for i in range(n_rows)]

    fig, ax = plt.subplots(figsize=(13, max(4, 0.42 * n_rows + 1.4)))
    ax.axis("off")
    tbl = ax.table(cellText=cell_text, colLabels=col_labels,
                   cellColours=cell_colours, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.35)
    for j in range(n_cols):
        c = tbl[0, j]
        c.set_facecolor("#2c3e50")
        c.set_text_props(color="white", fontweight="bold")
    for i in range(1, n_rows + 1):
        tbl[i, 0].get_text().set_ha("left")
    for ri in best_F_rows:
        tbl[ri + 1, n_cols - 1].set_facecolor("#c6efce")
        tbl[ri + 1, n_cols - 1].set_text_props(fontweight="bold")

    title_risk = "Real CSV" if risk_type == "data" else "Sinusoidal"
    ax.set_title(f"Safety via Optimal Thresholds  (z={z}%, {title_risk})\n"
                 "X* = infection ceiling, Y* = attendance floor held ≥z% of weeks;  "
                 "F = ω·Y* − (1−ω)·X*  (green = best F per ω)\n"
                 "Critical Capacity excluded: it enforces $R_0<1$ by construction "
                 "(analytical, not empirical, thresholds)",
                 fontsize=10, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig5_safety_optimal_thresholds.png"),
                dpi=300, bbox_inches="tight")
    plt.close()


def plot_safety_frontier(df_safety, output_dir, z=SAFETY_Z):
    """X*, Y*, and F vs ω per agent — how each policy's guaranteed operating
    point shifts as attendance is weighted more heavily."""
    print("Generating safety frontier (X*, Y*, F vs ω)...")
    agents = [a for a in AGENT_ORDER if a in df_safety["agent"].values]
    omegas = sorted(df_safety["omega"].unique())

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = [("optimal_x", "X*  (infection ceiling)"),
               ("optimal_y", "Y*  (attendance floor)"),
               ("F_score",   "F = ω·Y* − (1−ω)·X*")]
    for ax, (col, label) in zip(axes, metrics):
        for ag in agents:
            sub = df_safety[df_safety["agent"] == ag].sort_values("omega")
            if sub.empty:
                continue
            ax.plot(sub["omega"], sub[col],
                    color=AGENT_COLORS.get(ag, "gray"),
                    marker=AGENT_MARKERS.get(ag, "o"),
                    linestyle=AGENT_LINESTYLES.get(ag, "-"),
                    label=ag, linewidth=2, markersize=6)
        ax.set_xlabel("ω (attendance weight)")
        ax.set_ylabel(label)
        ax.grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(loc="best", fontsize=8, frameon=True)
    fig.suptitle(f"Safety Frontier at z={z}%  —  guaranteed operating point vs ω",
                 fontweight="bold", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "fig_safety_frontier.png"),
                dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# 13b. PERTURBATION ROBUSTNESS SWEEP
# ============================================================

def run_perturbation_sweep(risk_type, output_dir,
                           levels=SAFETY_PERTURBATION_LEVELS, z=SAFETY_Z):
    """Recompute optimal-threshold safety under each perturbation level, for the
    DEPLOYABLE policies only (Myopic + the learned agents). The Upper Bound is an
    oracle ceiling — not a deployable controller and computed with the true risk
    trajectory — so it is excluded from the sensing-noise robustness sweep.

    Safety is measured on TRUE (unperturbed) infections; perturbation only
    corrupts the agent's observed inputs. Perturbation injects per-seed
    randomness, so the (otherwise deterministic) data mode also gets replication
    here — we use the sinusoidal seed list for replication in both modes."""
    sweep_agents = [a for a in AGENT_ORDER
                    if ARTIFACTS[a]["type"] not in ("upper_bound", "critical_capacity")]
    print(f"\nRunning perturbation robustness sweep ({risk_type}) over {levels} "
          f"for policies {sweep_agents} (Upper Bound, Critical Capacity excluded)...")
    sweep_seeds = EVAL_SEEDS_SINUSOIDAL   # replicate noise; ≥2 seeds for a distribution

    rows = []
    for omega in OMEGA_VALUES:
        for agent_name in sweep_agents:
            policy = load_agent_policy(agent_name, omega, risk_type)   # load once per (agent, ω)
            if policy is None:
                continue
            for level in levels:
                I_pool, U_pool, rewards = [], [], []
                for seed in sweep_seeds:
                    try:
                        m = rollout(
                            policy_fn=policy,
                            omega=omega, seed=seed, risk_type=risk_type,
                            perturbation_level=level,
                        )
                    except Exception as e:
                        warnings.warn(f"Perturb rollout failed {agent_name}/ω={omega}/"
                                      f"{level}/seed={seed}: {e}")
                        continue
                    I_pool.extend(m["infected_true"])
                    U_pool.extend(m["allowed"])
                    # TRUE achieved reward (uses true infections, like the safety
                    # metric): omega*allowed - (1-omega)*true_infected per week.
                    true_reward = sum(omega * a - (1.0 - omega) * i
                                      for a, i in zip(m["allowed"], m["infected_true"]))
                    rewards.append(true_reward)
                if not I_pool:
                    continue
                s = safety_from_trajectories(I_pool, U_pool, omega, z)
                r_mean, r_std = _mean_std(rewards)   # true reward under this noise level
                rows.append({"agent": agent_name, "omega": omega,
                             "perturbation": level, "z": z,
                             "mean_reward": r_mean, "std_reward": r_std, **s})
        print(f"  omega={omega} done")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "safety_optimal_thresholds_perturbation.csv"),
              index=False)
    print(f"Saved {output_dir}/safety_optimal_thresholds_perturbation.csv")
    plot_safety_perturbation(df, output_dir, levels, z)
    plot_reward_perturbation(df, output_dir, levels)
    return df


def plot_safety_perturbation(df, output_dir, levels, z=SAFETY_Z):
    """F-score vs perturbation level per agent, one subplot per ω — robustness
    of the safety/utility trade-off as sensing noise increases."""
    print("Generating safety perturbation robustness figure...")
    if df.empty:
        print("  (no perturbation rows — skipping)")
        return
    omegas = sorted(df["omega"].unique())
    agents = [a for a in AGENT_ORDER if a in df["agent"].values]
    x = np.arange(len(levels))

    ncols = 3
    nrows = int(np.ceil(len(omegas) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.2 * nrows), squeeze=False)
    axes = axes.flatten()
    for i, om in enumerate(omegas):
        ax = axes[i]
        for ag in agents:
            sub = df[(df["agent"] == ag) & (df["omega"] == om)]
            if sub.empty:
                continue
            sub = sub.set_index("perturbation").reindex(levels)
            ax.plot(x, sub["F_score"].values,
                    color=AGENT_COLORS.get(ag, "gray"),
                    marker=AGENT_MARKERS.get(ag, "o"),
                    linestyle=AGENT_LINESTYLES.get(ag, "-"),
                    label=ag, linewidth=2, markersize=6)
        ax.set_title(f"ω={om}", fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(levels, rotation=30, ha="right", fontsize=8)
        ax.set_xlabel("perturbation level"); ax.set_ylabel("F score")
        ax.grid(True, linestyle="--", alpha=0.4)
        if i == 0:
            ax.legend(loc="best", fontsize=8, frameon=True)
    for k in range(len(omegas), len(axes)):
        axes[k].set_visible(False)
    fig.suptitle(f"Safety Robustness to Sensing Noise (z={z}%)  —  "
                 "F vs perturbation level (true infections)",
                 fontweight="bold", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, "fig_safety_perturbation.png"),
                dpi=300, bbox_inches="tight")
    plt.close()


def plot_reward_perturbation(df, output_dir, levels):
    """Mean reward vs perturbation level per agent, one subplot per ω — analogous
    to the safety perturbation figure but on the reward objective. Shaded band is
    ±1 std across seeds (absent for the single-trajectory data eval)."""
    print("Generating reward perturbation robustness figure...")
    if df.empty or "mean_reward" not in df.columns:
        print("  (no perturbation reward rows — skipping)")
        return
    omegas = sorted(df["omega"].unique())
    agents = [a for a in AGENT_ORDER if a in df["agent"].values]
    x = np.arange(len(levels))

    ncols = 3
    nrows = int(np.ceil(len(omegas) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.2 * nrows), squeeze=False)
    axes = axes.flatten()
    for i, om in enumerate(omegas):
        ax = axes[i]
        for ag in agents:
            sub = df[(df["agent"] == ag) & (df["omega"] == om)]
            if sub.empty:
                continue
            sub = sub.set_index("perturbation").reindex(levels)
            y = sub["mean_reward"].values
            color = AGENT_COLORS.get(ag, "gray")
            ax.plot(x, y, color=color, marker=AGENT_MARKERS.get(ag, "o"),
                    linestyle=AGENT_LINESTYLES.get(ag, "-"),
                    label=ag, linewidth=2, markersize=6)
            sd = sub["std_reward"].values
            if not np.all(np.isnan(sd)):
                ax.fill_between(x, y - sd, y + sd, color=color, alpha=0.15)
        ax.set_title(f"ω={om}", fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(levels, rotation=30, ha="right", fontsize=8)
        ax.set_xlabel("perturbation level"); ax.set_ylabel("mean reward")
        ax.grid(True, linestyle="--", alpha=0.4)
        if i == 0:
            ax.legend(loc="best", fontsize=8, frameon=True)
    for k in range(len(omegas), len(axes)):
        axes[k].set_visible(False)
    fig.suptitle("Reward Robustness to Sensing Noise  —  "
                 "mean reward vs perturbation level (shaded = ±1 std)",
                 fontweight="bold", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, "fig_reward_perturbation.png"),
                dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# 14. FIGURE 4 — Week-by-week trajectory per omega
# ============================================================

def plot_real_trajectories(df_summary, ub_solvers, risk_type, output_dir):
    """One panel per agent showing infected/allowed/risk over MAX_WEEKS weeks, for each omega."""
    eval_seeds = _get_eval_seeds(risk_type)
    traj_seed = eval_seeds[0]
    risk_label = "Real CSV" if risk_type == 'data' else f"Sinusoidal (seed={traj_seed})"
    print(f"Generating Figure 4: week-by-week trajectories ({risk_label})...")

    for omega in OMEGA_VALUES:
        available = [a for a in AGENT_ORDER
                     if not df_summary[
                         (df_summary["agent"] == a) & (df_summary["omega"] == omega)
                     ].empty]

        n = len(available)
        if n == 0:
            continue

        fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=True)
        if n == 1:
            axes = [axes]

        for ax, agent_name in zip(axes, available):
            policy = load_agent_policy(agent_name, omega, risk_type)
            if policy is None:
                continue
            is_ub = (policy == "UPPER_BOUND_SENTINEL")

            m = rollout(
                policy_fn=None if is_ub else policy,
                omega=omega,
                seed=traj_seed,
                risk_type=risk_type,
                upper_bound_solver=ub_solvers[omega][traj_seed] if is_ub else None,
            )

            weeks = m["weeks"]
            ax2 = ax.twinx()
            ax2.plot(weeks, m["risk"], "g-", linewidth=1.5, alpha=0.6, label="Community Risk")
            ax2.fill_between(weeks, 0, m["risk"], alpha=0.1, color="green")
            ax2.set_ylim(0, 1.15)
            ax2.set_ylabel("Risk", color="green", fontsize=9)
            ax2.tick_params(axis="y", labelcolor="green")

            ax.bar([w - 0.2 for w in weeks], m["infected"], 0.4,
                   color=AGENT_COLORS[agent_name], alpha=0.7, label="Infected")
            ax.bar([w + 0.2 for w in weeks], m["allowed"], 0.4,
                   color="steelblue", alpha=0.5, label="Allowed")
            ax.set_ylabel("Students", fontsize=9)
            ax.set_ylim(0, TOTAL_STUDENTS * 1.15)
            ax.set_title(f"{agent_name}  |  R={m['total_reward']:.2f}", fontsize=10,
                         color=AGENT_COLORS[agent_name])
            ax.legend(fontsize=7, loc="upper left", frameon=False)

        axes[-1].set_xlabel("Week")
        fig.suptitle(f"Trajectories ({risk_label}) — ω={omega}", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"fig4_trajectory_omega{omega}.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()


# ============================================================
# 15. POLICY GRID HELPERS
# ============================================================

# Paper-friendly font sizes for policy grid figures (x-large, bold for print)
POLICY_TITLE_FS    = 24
POLICY_LABEL_FS    = 22
POLICY_TICK_FS     = 17
POLICY_LEGEND_FS   = 20
POLICY_SUPTITLE_FS = 28

# Bold/large defaults applied (via rc_context) while rendering policy grids.
POLICY_RC = {
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "xtick.labelsize": POLICY_TICK_FS,
    "ytick.labelsize": POLICY_TICK_FS,
}


def generate_distinct_colors(n):
    """Generate n visually distinct colors via an HSV ring.
    Same scheme used in the training scripts (deep_q_learning.py, double_dqn.py,
    online_dqn.py, train_tabular_q.py, ppo_agent.py) so the policy strips in
    evaluate.py colour-match the per-agent training visualisations."""
    HSV = [(x / n, 0.6, 0.95) for x in range(n)]
    RGB = [colorsys.hsv_to_rgb(*x) for x in HSV]
    return ['#{:02x}{:02x}{:02x}'.format(int(r * 255), int(g * 255), int(b * 255))
            for (r, g, b) in RGB]


# Discrete-action colours, one per action level — generated to match the
# training scripts' visualisations exactly.
DISCRETE_COLORS = generate_distinct_colors(NUM_ACTION_LEVELS)

# Continuous (PPO) colormap built from the same HSV ring used in the PPO training
# scripts, so the evaluate.py strips match the training visualisations.
PPO_CMAP = LinearSegmentedColormap.from_list(
    "ppo_policy", generate_distinct_colors(12), N=256)

# Agents shown in the policy-grid figures. Excluded: the Upper Bound (oracle) and
# Critical Capacity (analytical, risk-only policy — its grid is a degenerate
# horizontal gradient with no infection-axis structure).
POLICY_GRID_AGENTS = [a for a in AGENT_ORDER
                      if ARTIFACTS[a]["type"] not in ("upper_bound", "critical_capacity")]


def _is_continuous_agent(agent_name):
    return agent_name in ("PPO Continuous", "Critical Capacity", "Upper Bound")


def _extract_policy_grid(policy_fn, agent_name):
    """Return grid [n_inf × n_risk]. Discrete agents: action index. Continuous: capacity [0,100]."""
    infected_vals    = np.linspace(0, TOTAL_STUDENTS, POLICY_GRID_POINTS)
    risk_vals        = np.linspace(0, 1, POLICY_GRID_POINTS)
    capacity_options = [0.0, 50.0, 100.0]

    grid = np.zeros((POLICY_GRID_POINTS, POLICY_GRID_POINTS), dtype=np.float32)
    for i, inf in enumerate(infected_vals):
        for j, risk in enumerate(risk_vals):
            cap = float(policy_fn(inf, risk, 0))
            if _is_continuous_agent(agent_name):
                grid[i, j] = cap
            else:
                grid[i, j] = float(np.argmin([abs(cap - c) for c in capacity_options]))
    return grid


def _extract_ub_policy_grid(solver):
    """Trajectory-weighted UB policy grid, resized to display resolution."""
    agg = solver.get_aggregated_policy(method='trajectory')
    from scipy.ndimage import zoom
    zoom_i = POLICY_GRID_POINTS / agg.shape[0]
    zoom_r = POLICY_GRID_POINTS / agg.shape[1]
    grid = zoom(agg, (zoom_i, zoom_r), order=1).astype(np.float32)
    return np.clip(grid, 0, TOTAL_STUDENTS)


def _render_policy_ax(ax, grid, agent_name):
    if _is_continuous_agent(agent_name):
        im = ax.imshow(grid, extent=[0, 1, 0, TOTAL_STUDENTS],
                       origin="lower", aspect="auto",
                       cmap=PPO_CMAP, vmin=0, vmax=TOTAL_STUDENTS)
        return im
    else:
        cmap = ListedColormap(DISCRETE_COLORS)
        ax.imshow(grid, extent=[0, 1, 0, TOTAL_STUDENTS],
                  origin="lower", aspect="auto",
                  cmap=cmap, vmin=0, vmax=2)
        return None


def plot_per_agent_policy_strips(all_policies, ub_solvers_rep, output_dir):
    """One 1×6 figure per agent (Upper Bound excluded) showing the optimal policy
    for each omega. PPO uses the HSV colormap; legends/colorbars sit outside."""
    print("Generating per-agent policy strips...")
    capacity_options = [0, 50, 100]
    policy_grid_dir = os.path.join(output_dir, "policy_grids")
    os.makedirs(policy_grid_dir, exist_ok=True)

    with plt.rc_context(POLICY_RC):
        for agent_name in POLICY_GRID_AGENTS:
            fig, axes = plt.subplots(1, len(OMEGA_VALUES), figsize=(24, 4.8), sharey=True)

            for ax, omega in zip(axes, OMEGA_VALUES):
                policy = all_policies.get((agent_name, omega))
                if policy is None:
                    ax.set_visible(False)
                    continue
                grid = _extract_policy_grid(policy, agent_name)
                _render_policy_ax(ax, grid, agent_name)
                ax.set_title(f"ω={omega}", fontsize=POLICY_TITLE_FS, fontweight="bold")
                ax.set_xlabel("Risk", fontsize=POLICY_LABEL_FS, fontweight="bold")
                ax.tick_params(axis="both", labelsize=POLICY_TICK_FS)
                if omega == OMEGA_VALUES[0]:
                    ax.set_ylabel("Infected", fontsize=POLICY_LABEL_FS, fontweight="bold")

            if _is_continuous_agent(agent_name):
                sm = plt.cm.ScalarMappable(cmap=PPO_CMAP, norm=plt.Normalize(0, TOTAL_STUDENTS))
                sm.set_array([])
                # colorbar outside, to the right of the strip
                cbar = fig.colorbar(sm, ax=axes.tolist(), pad=0.05, shrink=0.9,
                                    fraction=0.035)
                cbar.set_label("Capacity (students)", fontsize=POLICY_LABEL_FS,
                               fontweight="bold")
                cbar.ax.tick_params(labelsize=POLICY_TICK_FS)
                rect = [0, 0, 0.93, 0.86]
            else:
                patches = [mpatches.Patch(color=DISCRETE_COLORS[i],
                                          label=f"Act {i} ({capacity_options[i]})")
                           for i in range(NUM_ACTION_LEVELS)]
                # legend outside, above the strip with room
                fig.legend(handles=patches, loc="upper center",
                           bbox_to_anchor=(0.5, 1.15), ncol=NUM_ACTION_LEVELS,
                           fontsize=POLICY_LEGEND_FS, frameon=True, borderaxespad=0.6)
                rect = [0, 0, 1, 0.84]

            fig.suptitle(f"{agent_name} — Optimal Policy π(infected, risk)",
                         fontweight="bold", fontsize=POLICY_SUPTITLE_FS, y=1.02)
            plt.tight_layout(rect=rect)
            safe_name = agent_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
            plt.savefig(os.path.join(policy_grid_dir, f"policy_strip_{safe_name}.png"),
                        dpi=300, bbox_inches="tight")
            plt.close()


def plot_cross_agent_comparison(all_policies, ub_solvers_rep, output_dir):
    """Rows = agents (Upper Bound excluded), columns = omegas. PPO uses the HSV
    colormap; the discrete legend and continuous colorbar are placed outside."""
    print("Generating cross-agent comparison grid...")
    capacity_options = [0, 50, 100]
    policy_grid_dir = os.path.join(output_dir, "policy_grids")
    os.makedirs(policy_grid_dir, exist_ok=True)

    agents = POLICY_GRID_AGENTS
    n_rows = len(agents)
    n_cols = len(OMEGA_VALUES)

    with plt.rc_context(POLICY_RC):
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(4.6 * n_cols, 4.0 * n_rows),
                                 sharey=True, sharex=True)

        cont_im = None
        for row, agent_name in enumerate(agents):
            for col, omega in enumerate(OMEGA_VALUES):
                ax = axes[row][col]
                policy = all_policies.get((agent_name, omega))

                if policy is None:
                    ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                            transform=ax.transAxes, fontsize=POLICY_TITLE_FS)
                    ax.set_facecolor("#f0f0f0")
                    if col == 0:
                        ax.set_ylabel(agent_name, fontsize=POLICY_LABEL_FS,
                                      fontweight="bold")
                    continue

                grid = _extract_policy_grid(policy, agent_name)
                im = _render_policy_ax(ax, grid, agent_name)
                if im is not None:
                    cont_im = im

                ax.tick_params(axis="both", labelsize=POLICY_TICK_FS)
                if row == 0:
                    ax.set_title(f"ω={omega}", fontsize=POLICY_TITLE_FS,
                                 fontweight="bold")
                if col == 0:
                    ax.set_ylabel(agent_name, fontsize=POLICY_LABEL_FS,
                                  fontweight="bold")
                if row == n_rows - 1:
                    ax.set_xlabel("Risk", fontsize=POLICY_LABEL_FS, fontweight="bold")

        # Leave room at the bottom for the legend + colorbar (outside the axes).
        fig.subplots_adjust(bottom=0.14, right=0.97, top=0.93, hspace=0.18, wspace=0.10)

        discrete_patches = [mpatches.Patch(color=DISCRETE_COLORS[i],
                                           label=f"Act {i} ({capacity_options[i]})")
                            for i in range(NUM_ACTION_LEVELS)]
        leg = fig.legend(handles=discrete_patches, loc="lower left",
                         bbox_to_anchor=(0.06, -0.02), ncol=NUM_ACTION_LEVELS,
                         fontsize=POLICY_LEGEND_FS, frameon=True,
                         title="Discrete agents")
        leg.get_title().set_fontsize(POLICY_LEGEND_FS)
        leg.get_title().set_fontweight("bold")

        if cont_im is not None:
            cbar_ax = fig.add_axes([0.60, -0.005, 0.32, 0.018])
            cbar = fig.colorbar(cont_im, cax=cbar_ax, orientation="horizontal")
            cbar.set_label("Capacity (students) — PPO Continuous",
                           fontsize=POLICY_LABEL_FS, fontweight="bold")
            cbar.ax.tick_params(labelsize=POLICY_TICK_FS)

        fig.suptitle("Cross-Agent Policy Comparison — π(infected, risk)",
                     fontweight="bold", fontsize=POLICY_SUPTITLE_FS, y=0.98)
        plt.savefig(os.path.join(policy_grid_dir, "policy_comparison_all_agents.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()


# ============================================================
# 15b. FIGURE 6 — Reward summary table (publication-ready)
# ============================================================

# Short agent labels for compact figures
AGENT_SHORT = {
    "Myopic":               "Myopic",
    "Tabular Q":            "Tab-Q",
    "DQN":                  "DQN",
    "Online DQN":           "On-DQN",
    "Double DQN":           "DDQN",
    "PPO Discrete":         "PPO-D",
    "PPO Continuous":       "PPO-C",
    "Critical Capacity":    "CC",
    "Upper Bound":          "DP-UB",
}


def plot_reward_table(df, output_dir, risk_type):
    """Publication-style reward table.

    Rows = agents, columns = ω values (cells = mean ± std across seeds),
    plus a final column with each agent's mean % of the DP Upper Bound.
    Best mean reward per ω column is highlighted green.
    """
    print("Generating Figure 6: reward summary table...")
    agents = [a for a in AGENT_ORDER if a in df["agent"].values]
    omegas = sorted(df["omega"].unique())
    col_labels = [f"ω={o}" for o in omegas] + ["% UB (mean)"]

    R = np.full((len(agents), len(omegas)), np.nan)
    cell_text = []
    for i, ag in enumerate(agents):
        row = []
        for j, om in enumerate(omegas):
            sub = df[(df["agent"] == ag) & (df["omega"] == om)]
            if sub.empty:
                row.append("—")
                continue
            mu = float(sub["mean_reward"].iloc[0])
            sd = float(sub["std_reward"].iloc[0])
            R[i, j] = mu
            row.append(f"{mu:.1f}" if np.isnan(sd) else f"{mu:.1f}\n±{sd:.1f}")
        pct = df[df["agent"] == ag]["pct_of_upper_bound"].dropna()
        row.append(f"{pct.mean():.1f}" if not pct.empty else "—")
        cell_text.append(row)

    n_rows, n_cols = len(agents), len(col_labels)
    row_colors = [["#f7f7f7" if i % 2 else "#ffffff"] * n_cols for i in range(n_rows)]

    fig, ax = plt.subplots(figsize=(1.5 * n_cols + 2, 0.62 * n_rows + 1.6))
    ax.axis("off")
    tbl = ax.table(cellText=cell_text, rowLabels=agents, colLabels=col_labels,
                   cellColours=row_colors, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.9)

    for j in range(n_cols):
        c = tbl[0, j]
        c.set_facecolor("#2c3e50")
        c.set_text_props(color="white", fontweight="bold")
    for i in range(1, n_rows + 1):
        c = tbl[i, -1]
        c.set_facecolor("#dce6f1")
        c.set_text_props(fontweight="bold", fontsize=8)

    # Highlight best mean reward per ω column
    for j in range(len(omegas)):
        col = R[:, j]
        if np.all(np.isnan(col)):
            continue
        best = np.nanmax(col)
        for i in range(n_rows):
            if not np.isnan(col[i]) and col[i] == best:
                tbl[i + 1, j].set_facecolor("#c6efce")
                tbl[i + 1, j].set_text_props(fontweight="bold")

    n_seeds = int(df["n_seeds"].iloc[0]) if "n_seeds" in df.columns and not df.empty else 0
    if risk_type == 'data':
        stat_label = "Total Reward (single deterministic trajectory) — Real CSV"
    else:
        stat_label = f"Total Reward (mean ± std across {n_seeds} seeds) — Sinusoidal"
    ax.set_title(f"{stat_label}\n"
                 "Green = best per ω column; final column = mean % of DP Upper Bound",
                 fontsize=11, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig6_reward_table.png"),
                dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# 15c. FIGURE 7 — Mean infected vs. mean allowed per agent
# ============================================================

def plot_infected_allowed(df, output_dir, risk_type):
    """Attendance–infection frontier: for each agent, mean weekly infected vs.
    mean weekly allowed traced across ω (0.1→0.6) as a scatter joined by a line.
    Lower-right (more attendance, fewer infections) is the efficient frontier."""
    print("Generating Figure 7: attendance–infection frontier...")
    agents = [a for a in AGENT_ORDER if a in df["agent"].values]
    risk_label = "Real COVID-19" if risk_type == "data" else "Sinusoidal"

    rc = {
        "font.size": 18, "font.weight": "bold",
        "axes.titlesize": 23, "axes.titleweight": "bold",
        "axes.labelsize": 21, "axes.labelweight": "bold",
        "xtick.labelsize": 16, "ytick.labelsize": 16,
        "legend.fontsize": 15, "lines.linewidth": 2.8,
    }
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(11, 8))
        for ag in agents:
            sub = df[df["agent"] == ag].sort_values("omega")
            if sub.empty:
                continue
            ax.plot(sub["mean_attendance"].values, sub["mean_infected"].values,
                    color=AGENT_COLORS.get(ag, "gray"),
                    marker=AGENT_MARKERS.get(ag, "o"),
                    linestyle=AGENT_LINESTYLES.get(ag, "-"),
                    linewidth=2.8, markersize=13, label=ag,
                    markeredgecolor="white", markeredgewidth=1.0)

        ax.set_xlabel("Mean Allowed (students / week)")
        ax.set_ylabel("Mean Infected (students / week)")
        ax.set_title(f"Attendance–Infection Frontier — {risk_label}\n"
                     "each line traces ω = 0.1 → 0.6", fontsize=20)
        ax.set_xlim(0, TOTAL_STUDENTS)
        ax.set_ylim(bottom=0)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.tick_params(width=1.6, length=6)
        # legend outside, to the right, with room
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                  frameon=True, title="Policy", borderaxespad=0.5)

        plt.tight_layout(rect=[0, 0, 0.82, 1])
        plt.savefig(os.path.join(output_dir, "fig7_infected_vs_allowed.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()


# ============================================================
# 15b. STATISTICAL-RIGOUR FIGURES (sinusoidal multi-seed only)
# ============================================================

def _paired_diff_ci(agent_map, base_map, n_boot=10_000, alpha=0.05):
    """Paired percentile bootstrap CI of the mean difference (agent − baseline).
    Pairs are matched by seed so shared-seed variance cancels. Returns
    (mean_diff, lo, hi) or None if fewer than 2 shared seeds."""
    seeds = sorted(set(agent_map) & set(base_map))
    if len(seeds) < 2:
        return None
    d = np.array([agent_map[s] for s in seeds]) - np.array([base_map[s] for s in seeds])
    rng = np.random.default_rng(seed=0)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    boot = d[idx].mean(axis=1)
    return (float(d.mean()),
            float(np.percentile(boot, 100 * alpha / 2)),
            float(np.percentile(boot, 100 * (1 - alpha / 2))))


def plot_difference_curves(per_seed_rewards, output_dir, baseline="Myopic"):
    """D = agent − baseline (mean reward) vs ω, with paired bootstrap 95% CI.
    A star marks omegas where the CI excludes 0 (statistically significant)."""
    print("Generating difference curves (paired bootstrap vs Myopic)...")
    omegas = sorted(per_seed_rewards.keys())
    rl_agents = [a for a in ("Double DQN", "PPO Discrete", "PPO Continuous")]

    fig, ax = plt.subplots(figsize=(10, 6))
    any_curve = False
    for ag in rl_agents:
        xs, ds, los, his, sig = [], [], [], [], []
        for om in omegas:
            amap = per_seed_rewards.get(om, {}).get(ag)
            bmap = per_seed_rewards.get(om, {}).get(baseline)
            if not amap or not bmap:
                continue
            res = _paired_diff_ci(amap, bmap)
            if res is None:
                continue
            dmean, lo, hi = res
            xs.append(om); ds.append(dmean); los.append(lo); his.append(hi)
            sig.append(lo > 0 or hi < 0)
        if not xs:
            continue
        any_curve = True
        color = AGENT_COLORS.get(ag, "gray")
        ax.plot(xs, ds, marker=AGENT_MARKERS.get(ag, "o"), color=color,
                label=ag, linewidth=2, markersize=7)
        ax.fill_between(xs, los, his, color=color, alpha=0.2)
        for x, d, s in zip(xs, ds, sig):
            if s:
                ax.annotate("*", (x, d), textcoords="offset points",
                            xytext=(0, 9), ha="center", fontsize=15, color=color)

    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.5)
    ax.set_xlabel("ω (attendance weight)")
    ax.set_ylabel(f"D = agent − {baseline}  (mean reward)")
    ax.set_title(f"Difference Curves vs {baseline}\n"
                 "shaded = paired bootstrap 95% CI;  * = CI excludes 0 (significant)")
    ax.grid(True, linestyle="--", alpha=0.4)
    if any_curve:
        ax.legend(loc="best", fontsize=9, frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig_difference_curves.png"),
                dpi=300, bbox_inches="tight")
    plt.close()


def plot_tolerance_intervals(per_seed_rewards, output_dir, alpha=0.05, beta=0.9):
    """Distribution-free (α, β) tolerance intervals per agent across ω.
    Shows the reward range a random deployment instance would fall in with
    confidence 1−α covering fraction β of the population."""
    print("Generating tolerance intervals (distribution-free)...")
    omegas = sorted(per_seed_rewards.keys())
    agents = [a for a in AGENT_ORDER
              if any(a in per_seed_rewards.get(om, {}) for om in omegas)]
    if not agents:
        print("  No per-seed rewards — skipping tolerance intervals.")
        return

    n_agents = len(agents)
    width = 0.8 / n_agents
    fig, ax = plt.subplots(figsize=(11, 6))
    for ai, ag in enumerate(agents):
        xs, means, los, his = [], [], [], []
        for oi, om in enumerate(omegas):
            amap = per_seed_rewards.get(om, {}).get(ag)
            if not amap:
                continue
            r = np.sort(np.array(list(amap.values()), dtype=float))
            n = r.size
            if n < 5:
                lo, hi = float(r.min()), float(r.max())
            else:
                nu = int(binom.ppf(alpha, n, 1 - beta))   # number excluded
                l_idx = min(nu // 2, n - 1)
                u_idx = max(n - 1 - nu // 2, 0)
                lo, hi = float(r[l_idx]), float(r[u_idx])
            xs.append(oi + ai * width); means.append(float(r.mean()))
            los.append(lo); his.append(hi)
        if not xs:
            continue
        yerr = np.array([np.array(means) - np.array(los),
                         np.array(his) - np.array(means)])
        ax.errorbar(xs, means, yerr=yerr, fmt="o", capsize=4,
                    color=AGENT_COLORS.get(ag, "gray"), label=ag, markersize=6)

    ax.set_xticks(np.arange(len(omegas)) + width * (n_agents - 1) / 2)
    ax.set_xticklabels([str(o) for o in omegas])
    ax.set_xlabel("ω (attendance weight)")
    ax.set_ylabel("Reward")
    ax.set_title(f"Distribution-free Tolerance Intervals (α={alpha}, β={beta})\n"
                 f"range covering ≥{int(beta*100)}% of deployments at "
                 f"{int((1-alpha)*100)}% confidence")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="best", fontsize=9, frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig_tolerance_intervals.png"),
                dpi=300, bbox_inches="tight")
    plt.close()


def plot_training_curves(output_dir, risk_type):
    """Per-agent learning curves: smoothed mean reward across runs + min/max band.
    Reads each agent's training_curves/training_rewards_omega_{w}_run*.csv."""
    print("Generating training learning curves...")
    import glob
    curves_root = os.path.join(output_dir, "training_curves")
    os.makedirs(curves_root, exist_ok=True)

    def smooth(y, w=20):
        if len(y) < w:
            return np.asarray(y, dtype=float)
        return np.convolve(y, np.ones(w) / w, mode="valid")

    any_plotted = False
    for agent_name in AGENT_ORDER:
        base = ARTIFACTS[agent_name]["dir_base"]
        if base is None:
            continue
        tc_dir = os.path.join(f"{base}_{risk_type}", "training_curves")
        if not os.path.isdir(tc_dir):
            continue

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()
        plotted = False
        n_runs = 0
        for i, om in enumerate(OMEGA_VALUES):
            ax = axes[i]
            files = sorted(glob.glob(os.path.join(
                tc_dir, f"training_rewards_omega_{om}_run*.csv")))
            if not files:
                ax.set_visible(False)
                continue
            runs = [pd.read_csv(fp)["total_reward"].values for fp in files]
            n_runs = max(n_runs, len(runs))
            L = min(len(r) for r in runs)
            mat = np.stack([r[:L] for r in runs])
            mean, mn, mx = mat.mean(0), mat.min(0), mat.max(0)
            color = AGENT_COLORS.get(agent_name, "gray")
            ep = np.arange(1, L + 1)
            ax.fill_between(ep, mn, mx, color=color, alpha=0.15)
            sm = smooth(mean)
            ax.plot(np.arange(1, len(sm) + 1), sm, color=color, linewidth=2)
            ax.set_title(f"ω={om}")
            ax.set_xlabel("Episode"); ax.set_ylabel("Reward")
            ax.grid(True, linestyle="--", alpha=0.4)
            plotted = True
        for k in range(len(OMEGA_VALUES), len(axes)):
            axes[k].set_visible(False)
        if plotted:
            fig.suptitle(f"Training Learning Curves — {agent_name} "
                         f"({n_runs} runs; min/max band, smoothed mean)",
                         fontweight="bold", fontsize=13)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            out = os.path.join(curves_root,
                               f"learning_curve_{agent_name.replace(' ', '_')}.png")
            plt.savefig(out, dpi=300, bbox_inches="tight")
            any_plotted = True
        plt.close()

    if any_plotted:
        print(f"  Learning curves saved to {curves_root}/")
    else:
        print("  No training_curves/ dirs found — skipping learning curve plots.")


# ============================================================
# 16. MAIN
# ============================================================

def run_eval_mode(risk_type):
    """Run a complete evaluation pipeline for a single risk type."""
    output_dir = _output_dir(risk_type)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "policy_grids"), exist_ok=True)

    print("=" * 65)
    print(f"SafeCampus Unified Evaluation Harness — risk_type={risk_type}")
    print(f"Agents : {AGENT_ORDER}")
    print(f"Omega  : {OMEGA_VALUES}")
    print("=" * 65)

    df, df_safety, ub_solvers, per_seed_rewards = run_all(risk_type, output_dir)

    # Pre-load policies for plotting
    print("\nLoading policies for policy grid figures...")
    all_policies = {}
    for agent_name in AGENT_ORDER:
        for omega in OMEGA_VALUES:
            all_policies[(agent_name, omega)] = load_agent_policy(agent_name, omega, risk_type)

    # Representative UB solver per omega for policy grids
    rep_seed = _get_eval_seeds(risk_type)[0]
    ub_solvers_rep = {omega: ub_solvers[omega][rep_seed] for omega in OMEGA_VALUES}

    # Primary figures
    plot_fig2_optimality_gap(df, output_dir)
    plot_fig1_reward_vs_omega(df, output_dir)
    plot_monotonicity(df, output_dir)
    plot_real_trajectories(df, ub_solvers, risk_type, output_dir)

    # Safety: optimal-threshold table + frontier (clean eval)
    plot_safety_threshold_table(df_safety, output_dir, risk_type)
    plot_safety_frontier(df_safety, output_dir)
    plot_reward_table(df, output_dir, risk_type)
    plot_infected_allowed(df, output_dir, risk_type)

    # Perturbation robustness sweep — deployable policies only (UB excluded)
    run_perturbation_sweep(risk_type, output_dir)

    # Statistical-rigour figures — sinusoidal multi-seed only (data is n=1)
    if risk_type == 'sinusoidal':
        plot_difference_curves(per_seed_rewards, output_dir)
        plot_tolerance_intervals(per_seed_rewards, output_dir)
        plot_training_curves(output_dir, risk_type)

    # Policy grid plots
    plot_per_agent_policy_strips(all_policies, ub_solvers_rep, output_dir)
    plot_cross_agent_comparison(all_policies, ub_solvers_rep, output_dir)

    print(f"\nAll outputs saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-mode", choices=["sinusoidal", "data", "both"],
                        default="both")
    args = parser.parse_args()

    if args.eval_mode in ("sinusoidal", "both"):
        run_eval_mode("sinusoidal")
    if args.eval_mode in ("data", "both"):
        run_eval_mode("data")

    print("\nDone.")


if __name__ == "__main__":
    main()
