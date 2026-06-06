import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.distributions import Categorical
import colorsys
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import os
import csv
import json
import time

from config import (
    TOTAL_STUDENTS, MAX_WEEKS, NUM_ACTION_LEVELS, OMEGA_VALUES,
    GLOBAL_SEED, TUNE_EVAL_SEEDS, REAL_DATA_FILE, FIXED_ALPHA, FIXED_BETA,
    TRAIN_RISK_TYPE, NUM_RUNS
)
from campus_gym.envs import ClassroomGymEnv


def _resolve_eval_risk_type():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--eval-risk-type", choices=["sinusoidal", "data"],
                        default="sinusoidal")
    args, _ = parser.parse_known_args()
    return args.eval_risk_type


EVAL_RISK_TYPE_RUN = _resolve_eval_risk_type()

# ============================================================
# 0. SETUP & CONFIGURATION
# ============================================================
ACTION_LEVELS = NUM_ACTION_LEVELS
GAMMA = 0.95
GAE_LAMBDA = 0.95
K_EPOCHS = 10
EPS_CLIP = 0.2
MAX_GRAD_NORM = 0.5
DEFAULT_LR = 0.005
DEFAULT_HIDDEN_DIM = 64

# Full Training Config
FULL_EPISODES = 2000
FULL_UPDATE_TIMESTEP = 2000  # Collect ~133 episodes per update (matches continuous PPO)

# Tuning Config — identical settings to full training for fair comparison
TUNE_EPISODES = 2000
TUNE_UPDATE_TIMESTEP = 2000
TUNE_K_EPOCHS = K_EPOCHS    # Same epochs as full training
LR_CANDIDATES = [0.0001, 0.001, 0.005, 0.01]
HIDDEN_DIM_CANDIDATES = [32, 64, 128]

# --- Tuning Evaluation Mode ---
# 'seeds': evaluate across multiple random seeds (train mode)
# 'data':  evaluate on real risk CSV file (eval mode)
TUNE_EVAL_MODE  = EVAL_RISK_TYPE_RUN  # follows --eval-risk-type CLI flag

# --- Policy Grid (for visualization only) ---
POLICY_GRID_POINTS = 20

# Multi-Omega Config
# NUM_RUNS imported from config.py (=5): independent training runs per omega.

# Output setup (per eval risk type, so sinusoidal/data runs are isolated)
OUTPUT_DIR = f"ppo_discrete_results_tuned_{EVAL_RISK_TYPE_RUN}"
LR_FILE = os.path.join(OUTPUT_DIR, "optimized_lrs.json")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Matplotlib formatting
plt.rcParams.update({
    'font.size': 12,
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'lines.linewidth': 2.0,
    'figure.titlesize': 14
})

# Device selection
# NOTE: For the small networks used here (2D input, 32-128 hidden, 3 output),
# CPU is faster than GPU (MPS/CUDA). The per-step tensor transfer overhead
# to GPU far exceeds any compute benefit for such tiny operations.
USE_GPU = False
if USE_GPU and torch.cuda.is_available():
    device = torch.device("cuda")
elif USE_GPU and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def set_seed(seed):
    """Sets seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def normalize_state(s):
    """Normalize: Infected (0-100) -> 0-1, Risk (0-1) -> 0-1"""
    return np.array([s[0] / 100.0, s[1]], dtype=np.float32)


def generate_distinct_colors(n):
    """Generates distinct colors for plotting."""
    HSV = [(x / n, 0.6, 0.95) for x in range(n)]
    RGB = [colorsys.hsv_to_rgb(*x) for x in HSV]
    return ['#{:02x}{:02x}{:02x}'.format(int(r * 255), int(g * 255), int(b * 255)) for (r, g, b) in RGB]


def moving_average(x, window=50):
    """Calculates moving average for smoothing plot data."""
    if len(x) < window: return x
    return np.convolve(x, np.ones(window) / window, mode='valid')


def extract_policy_grid(ppo_agent, grid_points=POLICY_GRID_POINTS):
    """
    Extracts a policy grid from the discrete PPO agent for visualization.
    Uses batched inference for performance.

    Returns:
        policy_grid: 2D array [infected_level, risk_level] -> action index
    """
    infected_vals = np.linspace(0, 100, grid_points)
    risk_vals = np.linspace(0, 1, grid_points)

    inf_grid, risk_grid = np.meshgrid(infected_vals, risk_vals, indexing='ij')
    inf_norm = inf_grid.flatten() / 100.0
    risk_flat = risk_grid.flatten()
    states = np.stack([inf_norm, risk_flat], axis=1).astype(np.float32)
    states_tensor = torch.FloatTensor(states).to(device)

    with torch.no_grad():
        probs = ppo_agent.policy.actor(states_tensor)
        actions = torch.argmax(probs, dim=1).cpu().numpy()

    return actions.reshape(grid_points, grid_points)


def extract_critic_grid(ppo_agent, grid_points=POLICY_GRID_POINTS):
    """
    Extracts critic value V(s) across the state grid using batched inference.

    Returns:
        value_grid: 2D array [infected_level, risk_level] -> V(s)
    """
    infected_vals = np.linspace(0, 100, grid_points)
    risk_vals = np.linspace(0, 1, grid_points)

    inf_grid, risk_grid = np.meshgrid(infected_vals, risk_vals, indexing='ij')
    states = np.stack([inf_grid.flatten() / 100.0, risk_grid.flatten()], axis=1).astype(np.float32)
    states_tensor = torch.as_tensor(states, device=device)

    with torch.no_grad():
        values = ppo_agent.policy.critic(states_tensor).squeeze(-1).cpu().numpy()

    return values.reshape(grid_points, grid_points)


def extract_value_gradient_grid(value_grid):
    """
    Computes the finite-difference gradient magnitude of V(s).

    Low magnitude  -> flat value landscape -> dynamically ambiguous region
                     -> policy has no strong preference -> expect non-monotonicity
    High magnitude -> steep value landscape -> clear signal -> policy should be monotone

    Returns:
        gradient_grid: 2D array of ||∇V(s)||
    """
    dV_dI = np.gradient(value_grid, axis=0)   # gradient along infected axis
    dV_dr = np.gradient(value_grid, axis=1)   # gradient along risk axis
    return np.sqrt(dV_dI ** 2 + dV_dr ** 2)


# ============================================================
# 1. PPO CLASSES (DISCRETE)
# ============================================================
class RolloutBuffer:
    def __init__(self, max_size, state_dim=2):
        self.max_size = max_size
        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros(max_size, dtype=np.int64)
        self.logprobs = np.zeros(max_size, dtype=np.float32)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.values = np.zeros(max_size, dtype=np.float32)
        self.is_terminals = np.zeros(max_size, dtype=np.bool_)
        self.pos = 0

    def add(self, state, action, logprob, reward, value, is_terminal):
        idx = self.pos
        self.states[idx] = state
        self.actions[idx] = action
        self.logprobs[idx] = logprob
        self.rewards[idx] = reward
        self.values[idx] = value
        self.is_terminals[idx] = is_terminal
        self.pos += 1

    def clear(self):
        self.pos = 0


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=DEFAULT_HIDDEN_DIM):
        super(ActorCritic, self).__init__()
        self.hidden_dim = hidden_dim

        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def act(self, state):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        return action_logprobs, state_values, dist_entropy

    @torch.no_grad()
    def get_value(self, state):
        return self.critic(state)


class PPO:
    def __init__(self, state_dim, action_dim, lr, gamma, gae_lambda,
                 K_epochs, eps_clip, hidden_dim=DEFAULT_HIDDEN_DIM):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.hidden_dim = hidden_dim
        self.policy = ActorCritic(state_dim, action_dim, hidden_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(state_dim, action_dim, hidden_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()
        # Pre-allocate inference tensor to avoid per-step allocation
        self.s_buffer = torch.zeros(1, state_dim, dtype=torch.float32, device=device)

    @torch.no_grad()
    def select_action(self, state):
        self.s_buffer[0] = torch.as_tensor(state)
        action, action_logprob = self.policy_old.act(self.s_buffer)
        value = self.policy_old.get_value(self.s_buffer).item()
        return action.item(), action_logprob.item(), value

    def _compute_gae(self, buffer):
        """GAE-λ advantage estimation using stored values."""
        n            = buffer.pos
        rewards      = buffer.rewards[:n]
        values       = buffer.values[:n]
        is_terminals = buffer.is_terminals[:n]

        advantages = np.zeros(n, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(n)):
            next_value = 0.0 if is_terminals[t] else (values[t + 1] if t + 1 < n else 0.0)
            delta = rewards[t] + self.gamma * next_value - values[t]
            gae   = delta + self.gamma * self.gae_lambda * (0.0 if is_terminals[t] else gae)
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    def update(self, buffer):
        advantages_np, returns_np = self._compute_gae(buffer)
        n = buffer.pos

        # Normalise advantages
        adv_mean, adv_std = advantages_np.mean(), advantages_np.std()
        advantages_np = (advantages_np - adv_mean) / (adv_std + 1e-8)

        old_states   = torch.as_tensor(buffer.states[:n],   device=device)
        old_actions  = torch.as_tensor(buffer.actions[:n],  device=device)
        old_logprobs = torch.as_tensor(buffer.logprobs[:n], device=device)
        advantages   = torch.as_tensor(advantages_np,        device=device)
        returns      = torch.as_tensor(returns_np,           device=device)

        if old_states.dim() == 1:
            old_states = old_states.unsqueeze(0)

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = state_values.squeeze(-1)

            ratios = torch.exp(logprobs - old_logprobs)
            surr1  = ratios * advantages
            surr2  = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss = (-torch.min(surr1, surr2)
                    + 0.5 * self.MseLoss(state_values, returns)
                    - 0.01 * dist_entropy)

            self.optimizer.zero_grad(set_to_none=True)
            loss.mean().backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), MAX_GRAD_NORM)
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        buffer.clear()


# ============================================================
# 2. CORE TRAINING FUNCTION
# ============================================================
def run_ppo_training(omega_val, run_seed, lr, episodes, update_timestep, hidden_dim=DEFAULT_HIDDEN_DIM, k_epochs=K_EPOCHS):
    """Runs a single PPO training session with specified hyperparameters."""
    set_seed(run_seed)

    env = ClassroomGymEnv(
        total_students=TOTAL_STUDENTS,
        max_weeks=MAX_WEEKS,
        use_discrete_state=False,
        num_action_levels=ACTION_LEVELS,
        num_discrete_levels=10,
        seed=run_seed,
        omega=omega_val,
        mode="train"
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    ppo_agent = PPO(state_dim, action_dim, lr, GAMMA, GAE_LAMBDA,
                    k_epochs, EPS_CLIP, hidden_dim)
    buffer = RolloutBuffer(max_size=update_timestep, state_dim=state_dim)

    episode_rewards = []
    time_step = 0
    visitation_grid = np.zeros((POLICY_GRID_POINTS, POLICY_GRID_POINTS), dtype=np.int32)

    for ep in range(episodes):
        raw_state, info = env.reset()
        state = normalize_state(raw_state)
        ep_reward = 0
        done = False

        while not done:
            time_step += 1

            # Track state visitation (state is already normalized to [0,1]^2)
            i_idx = min(int(state[0] * POLICY_GRID_POINTS), POLICY_GRID_POINTS - 1)
            j_idx = min(int(state[1] * POLICY_GRID_POINTS), POLICY_GRID_POINTS - 1)
            visitation_grid[i_idx, j_idx] += 1

            action, logprob, value = ppo_agent.select_action(state)
            next_raw_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            buffer.add(state, action, logprob, reward, value, done)

            state = normalize_state(next_raw_state)
            ep_reward += reward

            if time_step % update_timestep == 0:
                ppo_agent.update(buffer)

        episode_rewards.append(ep_reward)

    return ppo_agent, episode_rewards, env.capacity_options, visitation_grid


# ============================================================
# 3. GRID SEARCH TUNING
# ============================================================
def evaluate_tuning_agent(omega_val, ppo_agent):
    """
    Evaluates a trained PPO agent for hyperparameter selection.
    Mode is controlled by TUNE_EVAL_MODE:
      'data'       -> single episode on real CSV (out-of-distribution)
      'sinusoidal' -> average over TUNE_EVAL_SEEDS episodes (in-distribution)
    """
    s_buffer = torch.zeros(1, 2, dtype=torch.float32, device=device)

    def _run_episode(env):
        raw_state, _ = env.reset()
        state = normalize_state(raw_state)
        done = False
        ep_reward = 0
        while not done:
            with torch.no_grad():
                s_buffer[0] = torch.as_tensor(state)
                probs = ppo_agent.policy.actor(s_buffer)
                action = torch.argmax(probs).item()
            next_raw, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            state = normalize_state(next_raw)
        return ep_reward

    if TUNE_EVAL_MODE == 'data':
        env = ClassroomGymEnv(
            mode="eval", use_discrete_state=False,
            num_action_levels=ACTION_LEVELS, seed=999,
            omega=omega_val, community_risk_data_file=REAL_DATA_FILE
        )
        return _run_episode(env)
    else:  # sinusoidal
        rewards = []
        for seed in TUNE_EVAL_SEEDS:
            env = ClassroomGymEnv(
                mode="eval", use_discrete_state=False,
                num_action_levels=ACTION_LEVELS, seed=seed,
                omega=omega_val, eval_risk_type=TRAIN_RISK_TYPE
            )
            rewards.append(_run_episode(env))
        return float(np.mean(rewards))


def select_best_hyperparams(omega_results):
    """Selects the (lr, hidden_dim) combo with the highest average evaluation reward."""
    if not omega_results:
        raise ValueError("No results to select from")
    best = max(omega_results, key=lambda r: r['avg_eval_reward'])
    return best['lr'], best['hidden_dim'], best


def grid_search_tuning():
    """
    Performs grid search tuning for each omega value over LR and hidden_dim,
    selecting the combo that achieves the highest average evaluation reward.
    """
    optimized_hyperparams = {}
    all_tuning_results = {}

    total_trials = len(LR_CANDIDATES) * len(HIDDEN_DIM_CANDIDATES)
    print(f"\n--- Starting Grid Search Hyperparameter Tuning ---")
    print(f"Testing LRs: {LR_CANDIDATES}")
    print(f"Testing Hidden Dims: {HIDDEN_DIM_CANDIDATES}")
    print(f"Total trials per omega: {total_trials}")
    tune_label = f"real CSV ({REAL_DATA_FILE})" if TUNE_EVAL_MODE == 'data' else f"sinusoidal x{len(TUNE_EVAL_SEEDS)} seeds"
    print(f"Selection criterion: highest eval reward — {tune_label}")
    start_time = time.time()

    for omega in OMEGA_VALUES:
        print(f"\n{'=' * 60}")
        print(f"*** Tuning for Omega = {omega} ***")
        print(f"{'=' * 60}")

        omega_results = []
        trial_num = 0

        for lr in LR_CANDIDATES:
            for hidden_dim in HIDDEN_DIM_CANDIDATES:
                trial_num += 1
                print(f"\n  [{trial_num}/{total_trials}] LR={lr:.6f}, hidden_dim={hidden_dim}...", end=' ', flush=True)

                agent, _, _, _ = run_ppo_training(
                    omega_val=omega,
                    run_seed=GLOBAL_SEED,
                    lr=lr,
                    episodes=TUNE_EPISODES,
                    update_timestep=TUNE_UPDATE_TIMESTEP,
                    hidden_dim=hidden_dim,
                    k_epochs=TUNE_K_EPOCHS
                )

                avg_eval_reward = evaluate_tuning_agent(omega, agent)

                omega_results.append({'lr': lr, 'hidden_dim': hidden_dim, 'avg_eval_reward': avg_eval_reward})
                print(f"Avg Reward: {avg_eval_reward:.2f}")

        best_lr, best_hidden_dim, best_metrics = select_best_hyperparams(omega_results)
        optimized_hyperparams[str(omega)] = {'lr': best_lr, 'hidden_dim': best_hidden_dim}
        all_tuning_results[omega] = omega_results

        print(f"\n  --> Selected LR={best_lr:.6f}, hidden_dim={best_hidden_dim} "
              f"(Reward: {best_metrics['avg_eval_reward']:.2f})")

    # Save optimized hyperparams
    with open(LR_FILE, 'w') as f:
        json.dump(optimized_hyperparams, f, indent=4)

    # Save detailed tuning results
    tuning_results_file = os.path.join(OUTPUT_DIR, "tuning_results_detailed.json")
    with open(tuning_results_file, 'w') as f:
        json.dump(
            {str(omega): {'best_lr': optimized_hyperparams[str(omega)]['lr'],
                          'best_hidden_dim': optimized_hyperparams[str(omega)]['hidden_dim'],
                          'all_trials': trials}
             for omega, trials in all_tuning_results.items()},
            f, indent=4
        )

    end_time = time.time()
    print(f"\n--- Grid Search complete. Total time: {end_time - start_time:.2f}s ---")
    print(f"    Optimized hyperparams saved to {LR_FILE}")

    return {float(k): v for k, v in optimized_hyperparams.items()}


def load_optimized_hyperparams():
    """Loads optimized hyperparams from file, converting string keys to floats."""
    if os.path.exists(LR_FILE):
        with open(LR_FILE, 'r') as f:
            hyperparams = json.load(f)
            result = {}
            for k, v in hyperparams.items():
                omega = float(k)
                if isinstance(v, dict):
                    result[omega] = {'lr': float(v['lr']), 'hidden_dim': int(v['hidden_dim'])}
                else:
                    # Legacy format: just LR value
                    result[omega] = {'lr': float(v), 'hidden_dim': DEFAULT_HIDDEN_DIM}
            return result
    else:
        print(f"WARNING: Optimized hyperparams file not found at {LR_FILE}. Using defaults.")
        return {omega: {'lr': DEFAULT_LR, 'hidden_dim': DEFAULT_HIDDEN_DIM} for omega in OMEGA_VALUES}


# ============================================================
# 4. FINAL EVALUATION ROLLOUT FUNCTION
# ============================================================
def evaluate_final_rollout(omega_val, ppo_agent, seed=999):
    """Performs a single, detailed evaluation rollout for saving results."""
    set_seed(seed)
    eval_env = ClassroomGymEnv(
        mode="eval",
        use_discrete_state=False,
        num_action_levels=ACTION_LEVELS,
        seed=seed,
        omega=omega_val,
        community_risk_data_file=REAL_DATA_FILE
    )

    raw_state, info = eval_env.reset()
    state = normalize_state(raw_state)
    done = False
    total_reward = 0
    rollout_data = []
    s_buffer = torch.zeros(1, 2, dtype=torch.float32, device=device)

    while not done:
        with torch.no_grad():
            s_buffer[0] = torch.as_tensor(state)
            probs = ppo_agent.policy.actor(s_buffer)
            action = torch.argmax(probs).item()

        next_raw_state, reward, terminated, truncated, info = eval_env.step(action)
        done = terminated or truncated
        total_reward += reward

        rollout_data.append({
            "Week": eval_env.current_week,
            "Action_Index": action,
            "Allowed_Students": info['allowed_students'],
            "Infected_Students": info['infected_students'],
            "Community_Risk": round(info['community_risk'], 3),
            "Reward": round(reward, 4)
        })

        state = normalize_state(next_raw_state)

    return total_reward, rollout_data


# ============================================================
# 5. TRAINING AND EVALUATION WITH OPTIMIZED HYPERPARAMS
# ============================================================
def train_and_evaluate_optimal(optimized_hyperparams):
    """Runs full training and evaluation using the optimized hyperparameters."""
    all_rewards_matrix = {}
    representative_agents = {}
    visitation_grids = {}
    capacity_options = None

    print(f"\n--- Starting Full Training and Evaluation ---")
    print(f"Training {FULL_EPISODES} episodes per omega using optimized hyperparameters.")

    for omega in OMEGA_VALUES:
        hyperparams = optimized_hyperparams.get(omega, {'lr': DEFAULT_LR, 'hidden_dim': DEFAULT_HIDDEN_DIM})
        lr = hyperparams['lr']
        hidden_dim = hyperparams['hidden_dim']
        print(f"\n*** Training for Omega = {omega} (LR: {lr:.6f}, hidden_dim: {hidden_dim}) ***")

        omega_rewards_runs = []
        run_agents = []
        run_eval_scores = []
        run_visitations = []

        # Learning-curve directory (per-episode reward trajectories per run)
        curves_dir = os.path.join(OUTPUT_DIR, "training_curves")
        os.makedirs(curves_dir, exist_ok=True)

        for run_idx in range(NUM_RUNS):
            run_seed = GLOBAL_SEED + (run_idx * 100)
            print(f"  > Run {run_idx + 1}/{NUM_RUNS} (Seed {run_seed}) starting...")

            agent, rewards, caps, visitation = run_ppo_training(
                omega_val=omega,
                run_seed=run_seed,
                lr=lr,
                episodes=FULL_EPISODES,
                update_timestep=FULL_UPDATE_TIMESTEP,
                hidden_dim=hidden_dim
            )
            omega_rewards_runs.append(rewards)

            if capacity_options is None:
                capacity_options = caps

            # Save per-run learning curve
            curve_path = os.path.join(curves_dir, f"training_rewards_omega_{omega}_run{run_idx}.csv")
            with open(curve_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["episode", "total_reward"])
                for ep_idx, ep_reward in enumerate(rewards, start=1):
                    writer.writerow([ep_idx, ep_reward])

            # Score this run for best-model selection (mean over TUNE_EVAL_SEEDS)
            run_eval_score = evaluate_tuning_agent(omega, agent)
            run_agents.append(agent)
            run_eval_scores.append(run_eval_score)
            run_visitations.append(visitation)
            print(f"  > Run {run_idx + 1} finished. Tuning-eval score: {run_eval_score:.2f}")

        # --- Best-model selection: highest mean eval reward over TUNE_EVAL_SEEDS ---
        best_idx = max(range(NUM_RUNS), key=lambda i: run_eval_scores[i])
        best_agent = run_agents[best_idx]
        representative_agents[omega] = best_agent
        visitation_grids[omega] = run_visitations[best_idx]
        print(f"  >> Best run: #{best_idx + 1} (eval {run_eval_scores[best_idx]:.2f})")

        # Save best model
        torch.save(best_agent.policy.state_dict(),
                   os.path.join(OUTPUT_DIR, f"ppo_policy_optimal_omega_{omega}.pth"))

        # Evaluate best model (fixed seed 999 for the saved rollout)
        final_score, rollout = evaluate_final_rollout(omega, best_agent)

        # Save CSV
        csv_name = os.path.join(OUTPUT_DIR, f"eval_rollout_optimal_omega_{omega}.csv")
        with open(csv_name, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                "Week", "Action_Index", "Allowed_Students",
                "Infected_Students", "Community_Risk", "Reward"
            ])
            writer.writeheader()
            writer.writerows(rollout)

        # Save reward summary
        txt_name = os.path.join(OUTPUT_DIR, f"total_reward_optimal_omega_{omega}.txt")
        with open(txt_name, "w") as f:
            f.write(f"Optimal LR: {lr}\n")
            f.write(f"Optimal Hidden Dim: {hidden_dim}\n")
            f.write(f"Best Run: {best_idx + 1}/{NUM_RUNS} (tuning-eval {run_eval_scores[best_idx]:.4f})\n")
            f.write(f"Final Eval Score (Seed 999): {final_score:.4f}\n")

        print(f"  > Best-model eval score (seed 999): {final_score:.2f}")

        all_rewards_matrix[omega] = np.array(omega_rewards_runs)

    plot_combined_rewards(all_rewards_matrix, NUM_RUNS)
    if capacity_options:
        plot_policy_strips(representative_agents, capacity_options)
    plot_diagnostics(representative_agents, visitation_grids)


# ============================================================
# 6. PLOTTING FUNCTIONS
# ============================================================

def plot_diagnostics(representative_agents, visitation_grids):
    """
    4-row diagnostic strip plot (one column per omega) showing:

    Row 0 — Policy action π(s): discrete action index chosen at each state
    Row 1 — Critic value V(s): agent's learned estimate of state value
    Row 2 — Value gradient ||∇V(s)||: where the value landscape is flat (problematic)
             Low gradient = dynamically ambiguous region = expect non-monotonicity
    Row 3 — State visitation density: where the agent trained
             Low density = undertrained region = policy artifacts possible

    Reading the plot:
    - Non-monotone policy regions should co-locate with flat gradient AND/OR low visitation.
    - If non-monotone region has high visitation + flat gradient: the MDP is ambiguous there.
    - If non-monotone region has low visitation: it is a training coverage artifact.
    """
    print("\n--- Generating Diagnostic Strips Plot ---")

    n_omega = len(OMEGA_VALUES)
    row_labels = [
        'Policy $\\pi(s)$\n(action index)',
        'Critic $V(s)$',
        'Value Gradient $\\|\\nabla V\\|$\n(flat = ambiguous)',
        'State Visitation\n(sparse = undertrained)'
    ]

    # Collect grids for shared color scales
    all_policy   = {}
    all_critic   = {}
    all_gradient = {}
    all_visit    = {}

    for omega in OMEGA_VALUES:
        agent = representative_agents[omega]
        all_policy[omega]   = extract_policy_grid(agent).astype(np.float32)
        all_critic[omega]   = extract_critic_grid(agent)
        all_gradient[omega] = extract_value_gradient_grid(all_critic[omega])
        visit = visitation_grids[omega].astype(np.float32)
        all_visit[omega] = np.log1p(visit)   # log scale so rare states remain visible

    critic_vals   = np.concatenate([g.flatten() for g in all_critic.values()])
    gradient_vals = np.concatenate([g.flatten() for g in all_gradient.values()])
    visit_vals    = np.concatenate([g.flatten() for g in all_visit.values()])

    critic_lim   = (critic_vals.min(),   critic_vals.max())
    gradient_lim = (gradient_vals.min(), gradient_vals.max())
    visit_lim    = (visit_vals.min(),    visit_vals.max())

    fig, axes = plt.subplots(4, n_omega, figsize=(20, 13), sharey='row')

    # Discrete colormap for action indices (ACTION_LEVELS bins)
    action_cmap = plt.get_cmap('RdYlGn', ACTION_LEVELS)

    for idx, omega in enumerate(OMEGA_VALUES):
        grids = [
            all_policy[omega],
            all_critic[omega],
            all_gradient[omega],
            all_visit[omega]
        ]
        cmaps = [action_cmap, 'viridis', 'RdYlGn', 'Blues']
        vlims = [
            (-0.5, ACTION_LEVELS - 0.5),
            critic_lim,
            gradient_lim,
            visit_lim
        ]

        for row, (grid, cmap, (vmin, vmax)) in enumerate(zip(grids, cmaps, vlims)):
            ax = axes[row, idx]
            im = ax.imshow(
                grid,
                extent=[0, 1, 0, 100],
                origin='lower',
                aspect='auto',
                cmap=cmap,
                vmin=vmin,
                vmax=vmax
            )

            if row == 0:
                ax.set_title(f"$\\omega={omega}$", fontweight='bold', fontsize=12)
            if row == 3:
                ax.set_xlabel('Risk', fontsize=9)
            if idx == 0:
                ax.set_ylabel(row_labels[row], fontsize=9)
            else:
                ax.tick_params(axis='y', which='both', left=False, labelleft=False)

            if idx == n_omega - 1:
                cbar = fig.colorbar(im, ax=ax, fraction=0.08, pad=0.04)
                if row == 0:
                    cbar.set_ticks(range(ACTION_LEVELS))
                cbar.ax.tick_params(labelsize=7)

    fig.suptitle(
        'Policy Diagnostics: Action, Critic Value, Value Gradient, State Visitation\n'
        'Flat gradient + sparse visitation explain non-monotonic policy regions',
        fontsize=12, fontweight='bold', y=1.01
    )

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, 'diagnostics_strips.png'),
        dpi=300, bbox_inches='tight'
    )
    plt.close()
    print(f"    Saved: {os.path.join(OUTPUT_DIR, 'diagnostics_strips.png')}")


def plot_combined_rewards(all_rewards_matrix, num_runs):
    """Plots combined smoothed training rewards with confidence intervals."""
    print("\n--- Generating Combined Rewards Plot ---")

    plt.figure(figsize=(8, 5))

    cmap = plt.get_cmap('tab10')
    colors_lines = [cmap(i % 10) for i in range(len(OMEGA_VALUES))]

    def get_smoothed_data(data, window=100):
        kernel = np.ones(window) / window
        sliding_avg = np.convolve(data, kernel, mode='valid')
        growing_avg = np.cumsum(data[:window - 1]) / np.arange(1, window)
        return np.concatenate((growing_avg, sliding_avg))

    for idx, omega in enumerate(OMEGA_VALUES):
        data = all_rewards_matrix[omega]
        mean_rewards = np.mean(data, axis=0)
        std_rewards = np.std(data, axis=0)

        smoothed_mean = get_smoothed_data(mean_rewards, 100)
        smoothed_std = get_smoothed_data(std_rewards, 100)
        x_axis = np.arange(len(smoothed_mean))

        plt.plot(x_axis, smoothed_mean, label=f"$\\omega={omega}$", color=colors_lines[idx], linewidth=3)
        plt.fill_between(x_axis, smoothed_mean - smoothed_std, smoothed_mean + smoothed_std,
                         color=colors_lines[idx], alpha=0.2)

    plt.xlim(left=0)
    plt.margins(x=0)
    plt.xlabel("Episode", fontsize=14, fontweight='bold')
    plt.ylabel("Reward", fontsize=14, fontweight='bold')
    plt.title("Discrete PPO Training Reward", fontsize=14, fontweight='bold')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
               ncol=len(OMEGA_VALUES), borderaxespad=0., fontsize=12, frameon=False)
    plt.grid(True, linestyle='--', alpha=0.6, linewidth=1.5)
    plt.tick_params(axis='both', which='major', labelsize=12, width=2, length=6)

    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "combined_ppo_discrete_optimal_rewards_ci.png"),
                dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close()


def plot_policy_strips(representative_agents, capacity_options):
    """Plots the optimal policy as a heatmap strip for each omega value."""
    print("\n--- Generating Optimal Policy Strips Plot ---")

    action_colors = generate_distinct_colors(ACTION_LEVELS)
    cmap = ListedColormap(action_colors)

    legend_elems = [mpatches.Patch(color=action_colors[i], label=f"Act {i} ({capacity_options[i]})")
                    for i in range(ACTION_LEVELS)]

    fig, axes = plt.subplots(1, len(OMEGA_VALUES), figsize=(20, 3.5), sharey=True)

    if len(OMEGA_VALUES) == 1:
        axes = [axes]

    for idx, omega in enumerate(OMEGA_VALUES):
        ax = axes[idx]
        policy_grid = extract_policy_grid(representative_agents[omega])

        ax.imshow(
            policy_grid,
            extent=[0, 1, 0, 100],
            origin="lower",
            aspect="auto",
            cmap=cmap,
            vmin=0,
            vmax=ACTION_LEVELS - 1
        )

        ax.set_title(f"Optimal $\\omega={omega}$")
        ax.set_xlabel("Risk")
        if idx == 0:
            ax.set_ylabel("Infected Count")
        else:
            ax.tick_params(axis='y', which='both', left=False, labelleft=False)

    fig.legend(handles=legend_elems, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=ACTION_LEVELS, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "combined_ppo_discrete_optimal_policies.png"), dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================
# 7. MAIN CONTROL FUNCTION
# ============================================================
def main(mode='tune_and_train'):
    """
    Main function to control execution flow.

    Modes:
    - 'tune': Runs grid search to find optimal hyperparams (LR, hidden_dim) for each omega.
    - 'train': Loads optimal hyperparams and runs full training, evaluation, and plotting.
    - 'tune_and_train': Runs tuning, then immediately runs training.
    """
    optimized_hyperparams = {}

    if mode == 'tune' or mode == 'tune_and_train':
        optimized_hyperparams = grid_search_tuning()
    else:
        optimized_hyperparams = load_optimized_hyperparams()

    if mode == 'train' or mode == 'tune_and_train':
        train_and_evaluate_optimal(optimized_hyperparams)

    print(f"\nAll processing complete. Results in {OUTPUT_DIR}")


# --- Execution ---
if __name__ == '__main__':
    skip_tune = '--skip-tune' in sys.argv
    if skip_tune:
        main(mode='train')
    else:
        main(mode='tune_and_train')
