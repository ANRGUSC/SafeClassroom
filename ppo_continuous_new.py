import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta
import os
import csv
import json
import time
from matplotlib.colors import LinearSegmentedColormap
import colorsys

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
# 0. CONFIGURATION
# ============================================================

# Training
FULL_EPISODES      = 3000
UPDATE_TIMESTEP    = 2000
GAMMA              = 0.95
GAE_LAMBDA         = 0.95
K_EPOCHS           = 10
EPS_CLIP           = 0.2
MAX_GRAD_NORM      = 0.5
DEFAULT_LR         = 0.001
DEFAULT_HIDDEN_DIM = 64

# Tuning — same settings as full training
TUNE_EPISODES         = 3000
TUNE_UPDATE_TIMESTEP  = 2000
TUNE_K_EPOCHS         = K_EPOCHS
LR_CANDIDATES         = [0.003,0.005, 0.01, 0.03, 0.05]
HIDDEN_DIM_CANDIDATES = [32, 64, 128]

# Risk
CONSTANT_RISK_VALUE = 0.3
TUNE_EVAL_MODE      = EVAL_RISK_TYPE_RUN  # follows --eval-risk-type CLI flag
EVAL_RISK_TYPE      = EVAL_RISK_TYPE_RUN  # follows --eval-risk-type CLI flag

POLICY_GRID_POINTS = 20
# NUM_RUNS imported from config.py (=5): independent training runs per omega.

OUTPUT_DIR = f"ppo_continuous_new_results_tuned_{EVAL_RISK_TYPE_RUN}"
LR_FILE    = os.path.join(OUTPUT_DIR, "optimized_lrs.json")
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 12, 'font.weight': 'bold',
    'axes.labelweight': 'bold', 'axes.titleweight': 'bold',
    'lines.linewidth': 2.0, 'figure.titlesize': 14,
})

USE_GPU = False
if USE_GPU and torch.cuda.is_available():
    device = torch.device("cuda")
elif USE_GPU and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def normalize_state(s):
    s = s.astype(np.float32)
    return np.array([s[0] / 100.0, s[1]], dtype=np.float32)


def generate_distinct_colors(n):
    HSV = [(x / n, 0.6, 0.95) for x in range(n)]
    RGB = [colorsys.hsv_to_rgb(*x) for x in HSV]
    return ['#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255))
            for r, g, b in RGB]


# ============================================================
# 1. NETWORK
# ============================================================

def init_layer(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    """
    PPO continuous actor-critic with Beta distribution.

    Actor: outputs mean μ ∈ (0,1) via sigmoid and concentration κ > 1 via
           softplus+1. These give α = μκ, β = (1−μ)κ, so Beta(α,β) is
           always unimodal and naturally bounded in [0, 1]. No clipping
           needed — log_prob is exact at both collection and update time.
    Critic: outputs scalar V(s) for GAE advantage estimation.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=DEFAULT_HIDDEN_DIM):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.shared = nn.Sequential(
            init_layer(nn.Linear(state_dim, hidden_dim)), nn.Tanh(),
            init_layer(nn.Linear(hidden_dim, hidden_dim)), nn.Tanh(),
        )
        self.mean_head = init_layer(nn.Linear(hidden_dim, action_dim), std=0.01)
        self.conc_head = init_layer(nn.Linear(hidden_dim, action_dim), std=0.01)

        self.critic = nn.Sequential(
            init_layer(nn.Linear(state_dim, hidden_dim)), nn.Tanh(),
            init_layer(nn.Linear(hidden_dim, hidden_dim)), nn.Tanh(),
            init_layer(nn.Linear(hidden_dim, 1), std=1.0),
        )

    def _distribution(self, state):
        h   = self.shared(state)
        mu  = torch.clamp(torch.sigmoid(self.mean_head(h)), 1e-3, 1 - 1e-3)
        kap = F.softplus(self.conc_head(h)) + 1.0   # concentration > 1 → unimodal
        return Beta(mu * kap, (1.0 - mu) * kap)

    def act(self, state):
        dist     = self._distribution(state)
        action   = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def evaluate(self, state, action):
        dist         = self._distribution(state)
        action       = torch.clamp(action, 1e-6, 1 - 1e-6)   # keep inside Beta support
        log_prob     = dist.log_prob(action).sum(dim=-1)
        entropy      = dist.entropy().sum(dim=-1)
        state_values = self.critic(state)
        return log_prob, state_values, entropy

    @torch.no_grad()
    def get_value(self, state):
        return self.critic(state)

    def get_mean_action(self, state):
        """Deterministic mean action for evaluation."""
        h  = self.shared(state)
        mu = torch.clamp(torch.sigmoid(self.mean_head(h)), 1e-3, 1 - 1e-3)
        return mu


# ============================================================
# 2. ROLLOUT BUFFER  (stores values for GAE)
# ============================================================

class RolloutBuffer:
    def __init__(self, max_size, state_dim=2, action_dim=1):
        self.max_size = max_size
        self.states       = np.zeros((max_size, state_dim),  dtype=np.float32)
        self.actions      = np.zeros((max_size, action_dim), dtype=np.float32)
        self.logprobs     = np.zeros(max_size,               dtype=np.float32)
        self.rewards      = np.zeros(max_size,               dtype=np.float32)
        self.values       = np.zeros(max_size,               dtype=np.float32)
        self.is_terminals = np.zeros(max_size,               dtype=np.bool_)
        self.pos = 0

    def add(self, state, action, logprob, reward, value, is_terminal):
        i = self.pos
        self.states[i]       = state
        self.actions[i]      = action
        self.logprobs[i]     = logprob
        self.rewards[i]      = reward
        self.values[i]       = value
        self.is_terminals[i] = is_terminal
        self.pos += 1

    def clear(self):
        self.pos = 0


# ============================================================
# 3. PPO AGENT
# ============================================================

class PPO:
    def __init__(self, state_dim, action_dim, lr, gamma, gae_lambda,
                 K_epochs, eps_clip, hidden_dim=DEFAULT_HIDDEN_DIM):
        self.gamma      = gamma
        self.gae_lambda = gae_lambda
        self.eps_clip   = eps_clip
        self.K_epochs   = K_epochs

        self.policy     = ActorCritic(state_dim, action_dim, hidden_dim).to(device)
        self.optimizer  = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(state_dim, action_dim, hidden_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

        self._s_buf = torch.zeros(1, state_dim, dtype=torch.float32, device=device)

    @torch.no_grad()
    def select_action(self, state):
        self._s_buf[0] = torch.as_tensor(state)
        action, log_prob = self.policy_old.act(self._s_buf)
        value = self.policy_old.get_value(self._s_buf).item()
        return action.cpu().numpy().flatten(), log_prob.item(), value

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

        for _ in range(self.K_epochs):
            log_probs, state_values, entropy = self.policy.evaluate(old_states, old_actions)
            state_values = state_values.squeeze(-1)

            ratios = torch.exp(log_probs - old_logprobs)
            surr1  = ratios * advantages
            surr2  = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss = (-torch.min(surr1, surr2)
                    + 0.5 * self.MseLoss(state_values, returns)
                    - 0.01 * entropy)

            self.optimizer.zero_grad(set_to_none=True)
            loss.mean().backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), MAX_GRAD_NORM)
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        buffer.clear()


# ============================================================
# 4. TRAINING
# ============================================================

def run_ppo_training(omega_val, run_seed, lr, episodes, update_timestep,
                     hidden_dim=DEFAULT_HIDDEN_DIM, k_epochs=K_EPOCHS,
                     risk_type=TRAIN_RISK_TYPE, constant_risk_value=CONSTANT_RISK_VALUE):
    set_seed(run_seed)

    env = ClassroomGymEnv(
        total_students=TOTAL_STUDENTS, max_weeks=MAX_WEEKS,
        use_discrete_state=False, use_continuous_actions=True,
        seed=run_seed, omega=omega_val, mode="train",
        eval_risk_type=risk_type, constant_risk_value=constant_risk_value,
    )

    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent  = PPO(state_dim, action_dim, lr, GAMMA, GAE_LAMBDA,
                 k_epochs, EPS_CLIP, hidden_dim)
    buffer = RolloutBuffer(max_size=update_timestep,
                           state_dim=state_dim, action_dim=action_dim)

    episode_rewards = []
    time_step = 0
    visitation_grid = np.zeros((POLICY_GRID_POINTS, POLICY_GRID_POINTS), dtype=np.int32)

    for ep in range(episodes):
        raw_state, _ = env.reset()
        state     = normalize_state(raw_state)
        ep_reward = 0.0
        done      = False

        while not done:
            time_step += 1

            i_idx = min(int(state[0] * POLICY_GRID_POINTS), POLICY_GRID_POINTS - 1)
            j_idx = min(int(state[1] * POLICY_GRID_POINTS), POLICY_GRID_POINTS - 1)
            visitation_grid[i_idx, j_idx] += 1

            action, log_prob, value = agent.select_action(state)
            next_raw, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            buffer.add(state, action, log_prob, reward, value, done)
            state      = normalize_state(next_raw)
            ep_reward += reward

            if time_step % update_timestep == 0:
                agent.update(buffer)

        episode_rewards.append(ep_reward)

    return agent, episode_rewards, visitation_grid


# ============================================================
# 5. EVALUATION HELPERS
# ============================================================

def _make_eval_env(omega_val, seed, eval_risk_type=EVAL_RISK_TYPE):
    if eval_risk_type == 'data':
        return ClassroomGymEnv(
            mode="eval", use_discrete_state=False, use_continuous_actions=True,
            seed=seed, omega=omega_val, total_students=TOTAL_STUDENTS,
            community_risk_data_file=REAL_DATA_FILE,
        )
    return ClassroomGymEnv(
        mode="eval", use_discrete_state=False, use_continuous_actions=True,
        seed=seed, omega=omega_val, total_students=TOTAL_STUDENTS,
        eval_risk_type=eval_risk_type, constant_risk_value=CONSTANT_RISK_VALUE,
    )


def _run_eval_episode(eval_env, ppo_agent):
    """One deterministic episode using the actor mean."""
    s_buf = torch.zeros(1, 2, dtype=torch.float32, device=device)
    raw_state, _ = eval_env.reset()
    state = normalize_state(raw_state)
    done = False
    total_reward = 0.0
    while not done:
        with torch.no_grad():
            s_buf[0] = torch.as_tensor(state)
            action = ppo_agent.policy.get_mean_action(s_buf).cpu().numpy().flatten()
        next_raw, reward, terminated, truncated, _ = eval_env.step(action)
        done = terminated or truncated
        total_reward += reward
        state = normalize_state(next_raw)
    return total_reward


def evaluate_tuning_agent(omega_val, ppo_agent, eval_mode=TUNE_EVAL_MODE):
    if eval_mode == 'data':
        return _run_eval_episode(_make_eval_env(omega_val, seed=0, eval_risk_type='data'), ppo_agent)
    rewards = [
        _run_eval_episode(_make_eval_env(omega_val, seed=s, eval_risk_type=eval_mode), ppo_agent)
        for s in TUNE_EVAL_SEEDS
    ]
    return float(np.mean(rewards))


def evaluate_ppo(omega_val, ppo_agent, seed=999):
    """Detailed rollout for saving results."""
    eval_env = _make_eval_env(omega_val, seed)
    s_buf = torch.zeros(1, 2, dtype=torch.float32, device=device)
    raw_state, _ = eval_env.reset()
    state = normalize_state(raw_state)
    done = False
    total_reward = 0.0
    rollout_data = []

    while not done:
        with torch.no_grad():
            s_buf[0] = torch.as_tensor(state)
            action = ppo_agent.policy.get_mean_action(s_buf).cpu().numpy().flatten()
        next_raw, reward, terminated, truncated, info = eval_env.step(action)
        done = terminated or truncated
        total_reward += reward
        rollout_data.append({
            "Week":              eval_env.current_week,
            "Action_Value":      round(float(action[0]), 4),
            "Allowed_Students":  info['allowed_students'],
            "Infected_Students": info['infected_students'],
            "Community_Risk":    round(info['community_risk'], 3),
            "Reward":            round(reward, 4),
        })
        state = normalize_state(next_raw)

    return total_reward, rollout_data


# ============================================================
# 6. POLICY GRID HELPERS
# ============================================================

def extract_policy_grid(ppo_agent, grid_points=POLICY_GRID_POINTS):
    infected_vals = np.linspace(0, 100, grid_points)
    risk_vals     = np.linspace(0, 1,   grid_points)
    grid = np.zeros((grid_points, grid_points), dtype=np.float32)
    s_buf = torch.zeros(1, 2, dtype=torch.float32, device=device)

    with torch.no_grad():
        for i, inf in enumerate(infected_vals):
            for j, risk in enumerate(risk_vals):
                s_buf[0, 0] = inf / 100.0
                s_buf[0, 1] = risk
                grid[i, j] = ppo_agent.policy.get_mean_action(s_buf).item()

    return grid


def extract_critic_grid(ppo_agent, grid_points=POLICY_GRID_POINTS):
    infected_vals = np.linspace(0, 100, grid_points)
    risk_vals     = np.linspace(0, 1,   grid_points)
    inf_g, risk_g = np.meshgrid(infected_vals, risk_vals, indexing='ij')
    states = np.stack([inf_g.flatten() / 100.0, risk_g.flatten()], axis=1).astype(np.float32)

    with torch.no_grad():
        values = ppo_agent.policy.critic(
            torch.as_tensor(states, device=device)
        ).squeeze(-1).cpu().numpy()

    return values.reshape(grid_points, grid_points)


def extract_value_gradient_grid(value_grid):
    dI = np.gradient(value_grid, axis=0)
    dr = np.gradient(value_grid, axis=1)
    return np.sqrt(dI**2 + dr**2)


# ============================================================
# 7. HYPERPARAMETER TUNING
# ============================================================

def select_best_hyperparams(omega_results):
    if not omega_results:
        raise ValueError("No results to select from")
    best = max(omega_results, key=lambda r: r['avg_eval_reward'])
    return best['lr'], best['hidden_dim'], best


def grid_search_tuning():
    optimized = {}
    print(f"\n--- Grid Search Tuning (PPO Continuous + GAE) ---")
    print(f"LRs: {LR_CANDIDATES}  |  Hidden dims: {HIDDEN_DIM_CANDIDATES}")
    tune_label = (f"data ({REAL_DATA_FILE})" if TUNE_EVAL_MODE == 'data'
                  else f"{TUNE_EVAL_MODE} x{len(TUNE_EVAL_SEEDS)} seeds")
    print(f"Eval mode  : {tune_label}")
    t0 = time.time()

    for omega in OMEGA_VALUES:
        print(f"\n{'='*60}\n*** omega={omega} ***\n{'='*60}")
        results = []

        for lr in LR_CANDIDATES:
            for hidden_dim in HIDDEN_DIM_CANDIDATES:
                print(f"  LR={lr}, hidden_dim={hidden_dim} ...", end=' ', flush=True)
                agent, _, _ = run_ppo_training(
                    omega, GLOBAL_SEED, lr,
                    TUNE_EPISODES, TUNE_UPDATE_TIMESTEP,
                    hidden_dim=hidden_dim, k_epochs=TUNE_K_EPOCHS,
                )
                r = evaluate_tuning_agent(omega, agent)
                results.append({'lr': lr, 'hidden_dim': hidden_dim, 'avg_eval_reward': r})
                print(f"reward={r:.2f}")

        best_lr, best_hd, best = select_best_hyperparams(results)
        optimized[str(omega)] = {'lr': best_lr, 'hidden_dim': best_hd}
        print(f"--> Best: LR={best_lr}, hidden_dim={best_hd}, reward={best['avg_eval_reward']:.2f}")

    with open(LR_FILE, 'w') as f:
        json.dump(optimized, f, indent=4)

    print(f"\n--- Done in {time.time()-t0:.1f}s. Saved to {LR_FILE} ---")
    return {float(k): v for k, v in optimized.items()}


def load_optimized_hyperparams():
    if not os.path.exists(LR_FILE):
        print(f"WARNING: {LR_FILE} not found. Using defaults.")
        return {o: {'lr': DEFAULT_LR, 'hidden_dim': DEFAULT_HIDDEN_DIM} for o in OMEGA_VALUES}
    with open(LR_FILE) as f:
        data = json.load(f)
    return {float(k): {'lr': float(v['lr']), 'hidden_dim': int(v['hidden_dim'])}
            if isinstance(v, dict) else {'lr': float(v), 'hidden_dim': DEFAULT_HIDDEN_DIM}
            for k, v in data.items()}


# ============================================================
# 8. FULL TRAINING + EVALUATION
# ============================================================

def train_and_evaluate_optimal(optimized_hyperparams):
    all_rewards      = {}
    rep_agents       = {}
    visitation_grids = {}

    print(f"\n--- Full Training (PPO Continuous + GAE, {FULL_EPISODES} eps/omega) ---")
    print(f"Train risk: {TRAIN_RISK_TYPE}  |  Eval risk: {EVAL_RISK_TYPE}")

    for omega in OMEGA_VALUES:
        hp  = optimized_hyperparams.get(omega, {'lr': DEFAULT_LR, 'hidden_dim': DEFAULT_HIDDEN_DIM})
        lr, hd = hp['lr'], hp['hidden_dim']
        print(f"\n*** omega={omega}  LR={lr}  hidden_dim={hd} ***")

        omega_runs = []
        run_agents = []
        run_eval_scores = []
        run_visitations = []

        # Learning-curve directory (per-episode reward trajectories per run)
        curves_dir = os.path.join(OUTPUT_DIR, "training_curves")
        os.makedirs(curves_dir, exist_ok=True)

        for run_idx in range(NUM_RUNS):
            seed = GLOBAL_SEED + run_idx * 100
            print(f"  Run {run_idx+1}/{NUM_RUNS} (seed={seed}) ...", end=' ', flush=True)

            agent, rewards, visitation = run_ppo_training(
                omega, seed, lr, FULL_EPISODES, UPDATE_TIMESTEP, hidden_dim=hd,
            )
            omega_runs.append(rewards)
            print(f"done. mean_last_100={np.mean(rewards[-100:]):.2f}")

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

        # --- Best-model selection: highest mean eval reward over TUNE_EVAL_SEEDS ---
        best_idx = max(range(NUM_RUNS), key=lambda i: run_eval_scores[i])
        best_agent = run_agents[best_idx]
        rep_agents[omega]       = best_agent
        visitation_grids[omega] = run_visitations[best_idx]
        print(f"  >> Best run: #{best_idx + 1} (eval {run_eval_scores[best_idx]:.2f})")

        torch.save(best_agent.policy.state_dict(),
                   os.path.join(OUTPUT_DIR, f"ppo_policy_optimal_omega_{omega}.pth"))

        score, rollout = evaluate_ppo(omega, best_agent)

        with open(os.path.join(OUTPUT_DIR, f"eval_rollout_optimal_omega_{omega}.csv"),
                  'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                "Week","Action_Value","Allowed_Students",
                "Infected_Students","Community_Risk","Reward"])
            writer.writeheader()
            writer.writerows(rollout)

        with open(os.path.join(OUTPUT_DIR, f"total_reward_optimal_omega_{omega}.txt"), 'w') as f:
            f.write(str(score))

        print(f"  Eval score (best, seed 999): {score:.2f}")

        all_rewards[omega] = np.array(omega_runs)

    plot_combined_rewards(all_rewards)
    plot_policy_strips(rep_agents)
    plot_diagnostics(rep_agents, visitation_grids)


# ============================================================
# 9. PLOTTING
# ============================================================

def plot_combined_rewards(all_rewards_matrix):
    print("\n--- Rewards plot ---")
    plt.figure(figsize=(8, 5))
    cmap   = plt.get_cmap('tab10')
    colors = [cmap(i % 10) for i in range(len(OMEGA_VALUES))]

    def smooth(data, w=100):
        k = np.ones(w) / w
        return np.concatenate([
            np.cumsum(data[:w-1]) / np.arange(1, w),
            np.convolve(data, k, mode='valid'),
        ])

    for idx, omega in enumerate(OMEGA_VALUES):
        data = all_rewards_matrix[omega]
        m = smooth(np.mean(data, axis=0))
        s = smooth(np.std(data, axis=0))
        plt.plot(m, label=f"ω={omega}", color=colors[idx], linewidth=2)
        plt.fill_between(np.arange(len(m)), m - s, m + s, color=colors[idx], alpha=0.2)

    plt.xlabel("Episode"); plt.ylabel("Reward")
    plt.title("PPO Continuous (GAE) Training Reward")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
               ncol=len(OMEGA_VALUES), frameon=False)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "combined_rewards.png"), dpi=300, bbox_inches='tight')
    plt.close()


def plot_policy_strips(rep_agents):
    print("\n--- Policy strips ---")
    base   = generate_distinct_colors(12)
    cmap   = LinearSegmentedColormap.from_list("policy", base, N=256)
    fig, axes = plt.subplots(1, len(OMEGA_VALUES), figsize=(20, 3.5), sharey=True)

    for idx, omega in enumerate(OMEGA_VALUES):
        ax   = axes[idx]
        grid = extract_policy_grid(rep_agents[omega])
        im   = ax.imshow(grid, extent=[0,1,0,100], origin='lower',
                         aspect='auto', cmap=cmap, vmin=0, vmax=1)
        ax.set_title(f"ω={omega}"); ax.set_xlabel("Risk")
        if idx == 0:
            ax.set_ylabel("Infected Count")
        else:
            ax.tick_params(axis='y', left=False, labelleft=False)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
    fig.colorbar(im, cax=cbar_ax).set_label('Capacity Fraction', rotation=270, labelpad=15)
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(os.path.join(OUTPUT_DIR, "policy_strips.png"), dpi=300, bbox_inches='tight')
    plt.close()


def plot_diagnostics(rep_agents, visitation_grids):
    print("\n--- Diagnostics ---")
    base        = generate_distinct_colors(12)
    policy_cmap = LinearSegmentedColormap.from_list("policy", base, N=256)
    n_omega     = len(OMEGA_VALUES)

    all_critic   = {o: extract_critic_grid(rep_agents[o])           for o in OMEGA_VALUES}
    all_gradient = {o: extract_value_gradient_grid(all_critic[o])   for o in OMEGA_VALUES}
    all_visit    = {o: np.log1p(visitation_grids[o].astype(np.float32)) for o in OMEGA_VALUES}
    all_policy   = {o: extract_policy_grid(rep_agents[o])           for o in OMEGA_VALUES}

    critic_lim   = (min(v.min() for v in all_critic.values()),
                    max(v.max() for v in all_critic.values()))
    gradient_lim = (min(v.min() for v in all_gradient.values()),
                    max(v.max() for v in all_gradient.values()))
    visit_lim    = (min(v.min() for v in all_visit.values()),
                    max(v.max() for v in all_visit.values()))

    row_labels = ['Policy π(s)', 'Critic V(s)',
                  '||∇V|| (flat=ambiguous)', 'Visitation (sparse=undertrained)']
    fig, axes = plt.subplots(4, n_omega, figsize=(20, 13), sharey='row')

    for col, omega in enumerate(OMEGA_VALUES):
        grids = [all_policy[omega], all_critic[omega],
                 all_gradient[omega], all_visit[omega]]
        cmaps = [policy_cmap, 'viridis', 'RdYlGn', 'Blues']
        vlims = [(0, 1), critic_lim, gradient_lim, visit_lim]

        for row, (grid, cm, (vmin, vmax)) in enumerate(zip(grids, cmaps, vlims)):
            ax = axes[row, col]
            im = ax.imshow(grid, extent=[0,1,0,100], origin='lower',
                           aspect='auto', cmap=cm, vmin=vmin, vmax=vmax)
            if row == 0: ax.set_title(f"ω={omega}", fontweight='bold')
            if row == 3: ax.set_xlabel('Risk')
            if col == 0: ax.set_ylabel(row_labels[row], fontsize=9)
            else:        ax.tick_params(axis='y', left=False, labelleft=False)
            if col == n_omega - 1:
                fig.colorbar(im, ax=ax, fraction=0.08, pad=0.04).ax.tick_params(labelsize=7)

    fig.suptitle('PPO Continuous + GAE — Diagnostics', fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "diagnostics_strips.png"),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved diagnostics to {OUTPUT_DIR}/")


# ============================================================
# 10. MAIN
# ============================================================

def main(mode='tune_and_train'):
    """
    mode: 'tune' | 'train' | 'tune_and_train'
    """
    if mode in ('tune', 'tune_and_train'):
        hp = grid_search_tuning()
    else:
        hp = load_optimized_hyperparams()

    if mode in ('train', 'tune_and_train'):
        train_and_evaluate_optimal(hp)

    print(f"\nDone. Results in {OUTPUT_DIR}/")


if __name__ == '__main__':
    skip_tune = '--skip-tune' in sys.argv
    if skip_tune:
        main(mode='train')
    else:
        main(mode='tune_and_train')
