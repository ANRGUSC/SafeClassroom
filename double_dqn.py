import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import random
import colorsys
import matplotlib.patches as mpatches
import os
import csv
import json
import time
import sys
from matplotlib.colors import ListedColormap

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

# --- AGENT MODE ---
AGENT_MODE = 'DDQN_ONLINE'

# --- Hyperparameters ---
NUM_DISCRETE_LEVELS = 10
FULL_EPISODES = 3000
GAMMA = 0.95
EPS_START = 1.0
EPS_END = 0.001
TARGET_UPDATE = 10
UPDATE_EVERY = 4  # Gradient update every N env steps
DECAY_POWER = 2.0
DEFAULT_LR = 0.0001

# --- Tuning Config (Grid Search) ---
TUNE_EPISODES = 3000
LR_CANDIDATES = [0.00001, 0.0001, 0.0005, 0.001, 0.005]
HIDDEN_DIM_CANDIDATES = [32, 64, 128]  # Network hidden layer sizes to tune
DEFAULT_HIDDEN_DIM = 128  # Default hidden dim if tuning is skipped
TUNE_EVAL_MODE  = EVAL_RISK_TYPE_RUN  # follows --eval-risk-type CLI flag

# --- Policy Grid Config (visualization only) ---
POLICY_GRID_POINTS = 20

# --- Multi-Omega Config ---
# NUM_RUNS imported from config.py (=5): independent training runs per omega.

# --- Output setup (per eval risk type, so sinusoidal/data runs are isolated) ---
OUTPUT_DIR = f"online_ddqn_results_tuned_{EVAL_RISK_TYPE_RUN}"
LR_FILE = os.path.join(OUTPUT_DIR, f"optimized_lrs_{AGENT_MODE.lower()}.json")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Matplotlib Formatting ---
plt.rcParams.update({
    'font.size': 12,
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'lines.linewidth': 2.0,
    'figure.titlesize': 14
})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    """Sets seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# ============================================================
# 1. HELPERS & NETWORKS
# ============================================================

def normalize_state(s, total_students=100):
    """Normalize: Infected (0-100) -> 0-1, Risk (0-1) -> 0-1"""
    return np.array([s[0] / total_students, s[1]], dtype=np.float32)


def generate_distinct_colors(n):
    """Generates a list of visually distinct colors."""
    HSV = [(x / n, 0.6, 0.95) for x in range(n)]
    RGB = [colorsys.hsv_to_rgb(*x) for x in HSV]
    return ['#{:02x}{:02x}{:02x}'.format(int(r * 255), int(g * 255), int(b * 255)) for (r, g, b) in RGB]


def moving_average(x, window=50):
    """Calculates moving average for smoothing plot data."""
    if len(x) < window: return x
    return np.convolve(x, np.ones(window) / window, mode='valid')


class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.fc(x)


# ============================================================
# POLICY GRID EXTRACTION (visualization)
# ============================================================

def extract_policy_grid(policy_net, grid_points=POLICY_GRID_POINTS):
    """
    Extracts a policy grid from the DDQN agent for visualization.
    Uses batched inference for performance.

    Args:
        policy_net: Trained Q-network
        grid_points: Number of points along each dimension

    Returns:
        policy_grid: 2D array [infected_level, risk_level] -> action index
    """
    infected_vals = np.linspace(0, 1, grid_points)  # Normalized: 0-100 -> 0-1
    risk_vals = np.linspace(0, 1, grid_points)

    # Create meshgrid and flatten for batched inference
    inf_grid, risk_grid = np.meshgrid(infected_vals, risk_vals, indexing='ij')
    states = np.stack([inf_grid.flatten(), risk_grid.flatten()], axis=1).astype(np.float32)

    # Single batched forward pass
    states_tensor = torch.tensor(states, dtype=torch.float32).to(device)
    with torch.no_grad():
        q_values = policy_net(states_tensor)
        actions = q_values.argmax(dim=1).cpu().numpy()

    policy_grid = actions.reshape(grid_points, grid_points).astype(int)
    return policy_grid


# ============================================================
# 2. CORE DDQN ONLINE TRAINING FUNCTION
# ============================================================
def train_ddqn(omega_val, run_seed, lr, episodes, hidden_dim=DEFAULT_HIDDEN_DIM):
    """
    Trains a Double DQN agent using the online (single transition) update rule.
    """
    set_seed(run_seed)

    env = ClassroomGymEnv(
        total_students=TOTAL_STUDENTS,
        max_weeks=MAX_WEEKS,
        use_discrete_state=False,
        num_action_levels=NUM_ACTION_LEVELS,
        num_discrete_levels=NUM_DISCRETE_LEVELS,
        seed=run_seed,
        omega=omega_val,
        mode="train"
    )

    state_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n

    policy_net = QNetwork(state_dim, num_actions, hidden_dim=hidden_dim).to(device)
    target_net = QNetwork(state_dim, num_actions, hidden_dim=hidden_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    episode_rewards = []

    # Pre-allocate tensors for online update (avoids per-step allocation)
    s_buffer = torch.zeros(1, state_dim, dtype=torch.float32, device=device)
    ns_buffer = torch.zeros(1, state_dim, dtype=torch.float32, device=device)
    a_buffer = torch.zeros(1, 1, dtype=torch.int64, device=device)
    r_buffer = torch.zeros(1, dtype=torch.float32, device=device)
    d_buffer = torch.zeros(1, dtype=torch.float32, device=device)
    total_steps = 0

    for ep in range(episodes):
        raw_state, info = env.reset()
        state = normalize_state(raw_state, env.total_students)
        done = False
        ep_reward = 0

        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        ((1 - ep / episodes) ** DECAY_POWER)

        while not done:
            # Select Action
            if random.random() < eps_threshold:
                action = random.randint(0, num_actions - 1)
            else:
                with torch.no_grad():
                    s_buffer[0] = torch.as_tensor(state)
                    action = policy_net(s_buffer).argmax(dim=1).item()

            # Step in Environment
            next_raw_state, reward, terminated, truncated, _ = env.step(action)
            next_state = normalize_state(next_raw_state, env.total_students)
            done = terminated or truncated
            ep_reward += reward
            total_steps += 1

            # ONLINE UPDATE (DDQN Logic) - every UPDATE_EVERY steps
            if total_steps % UPDATE_EVERY == 0:
                s_buffer[0] = torch.as_tensor(state)
                a_buffer[0, 0] = action
                r_buffer[0] = reward
                ns_buffer[0] = torch.as_tensor(next_state)
                d_buffer[0] = float(done)

                # Q(s, a) from Policy Net
                q_values = policy_net(s_buffer).gather(1, a_buffer)

                # DDQN TARGET: Policy Net SELECTS action, Target Net EVALUATES
                with torch.no_grad():
                    next_action_index = policy_net(ns_buffer).argmax(1).unsqueeze(1)
                    next_q_values = target_net(ns_buffer).gather(1, next_action_index)
                    expected_q_values = r_buffer + (GAMMA * next_q_values.squeeze(1) * (1 - d_buffer))

                loss = loss_fn(q_values.squeeze(), expected_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            state = next_state

        episode_rewards.append(ep_reward)

        if ep % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    return policy_net, episode_rewards, env.capacity_options


# ============================================================
# 3. GRID SEARCH TUNING FUNCTIONS
# ============================================================

def select_best_hyperparams(omega_results):
    """Selects the (lr, hidden_dim) combo with the highest average evaluation reward."""
    if not omega_results:
        raise ValueError("No results to select from")
    best = max(omega_results, key=lambda r: r['avg_eval_reward'])
    return best['lr'], best['hidden_dim'], best


def evaluate_tuning_agent(omega_val, policy_net):
    """
    Evaluates a trained Q-network for hyperparameter selection.
    Mode is controlled by TUNE_EVAL_MODE:
      'data'       -> single episode on real CSV (out-of-distribution)
      'sinusoidal' -> average over TUNE_EVAL_SEEDS episodes (in-distribution)
    """
    s_buffer = torch.zeros(1, 2, dtype=torch.float32, device=device)

    def _run_episode(env):
        raw_state, _ = env.reset()
        state = normalize_state(raw_state, env.total_students)
        done = False
        ep_reward = 0
        while not done:
            with torch.no_grad():
                s_buffer[0] = torch.as_tensor(state)
                action = int(policy_net(s_buffer).argmax().item())
            next_raw, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            state = normalize_state(next_raw, env.total_students)
        return ep_reward

    if TUNE_EVAL_MODE == 'data':
        env = ClassroomGymEnv(
            mode="eval", use_discrete_state=False, max_weeks=MAX_WEEKS,
            num_action_levels=NUM_ACTION_LEVELS, seed=999,
            omega=omega_val, community_risk_data_file=REAL_DATA_FILE
        )
        return _run_episode(env)
    else:  # sinusoidal
        rewards = []
        for seed in TUNE_EVAL_SEEDS:
            env = ClassroomGymEnv(
                mode="eval", use_discrete_state=False, max_weeks=MAX_WEEKS,
                num_action_levels=NUM_ACTION_LEVELS, seed=seed,
                omega=omega_val, eval_risk_type=TRAIN_RISK_TYPE
            )
            rewards.append(_run_episode(env))
        return float(np.mean(rewards))


def grid_search_tuning():
    """
    Performs grid search tuning for each omega value over LR and hidden_dim.
    Selects the combo with the highest average evaluation reward on sinusoidal
    risk (training distribution). The real CSV is reserved for final evaluation.
    """
    optimized_hyperparams = {}
    tuning_results = {}

    total_trials = len(LR_CANDIDATES) * len(HIDDEN_DIM_CANDIDATES)
    print(f"\n--- Starting Grid Search Hyperparameter Tuning ({AGENT_MODE}) ---")
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

        for hidden_dim in HIDDEN_DIM_CANDIDATES:
            for lr in LR_CANDIDATES:
                trial_num += 1
                print(f"\n  [{trial_num}/{total_trials}] Testing LR={lr:.6f}, hidden_dim={hidden_dim}...", end=' ', flush=True)

                policy_net, _, _ = train_ddqn(
                    omega_val=omega,
                    run_seed=GLOBAL_SEED,
                    lr=lr,
                    episodes=TUNE_EPISODES,
                    hidden_dim=hidden_dim
                )

                avg_eval_reward = evaluate_tuning_agent(omega, policy_net)

                omega_results.append({
                    'lr': lr,
                    'hidden_dim': hidden_dim,
                    'avg_eval_reward': avg_eval_reward,
                })

                print(f"Avg Eval Reward: {avg_eval_reward:.2f}")

        best_lr, best_hidden_dim, best_metrics = select_best_hyperparams(omega_results)

        optimized_hyperparams[str(omega)] = {'lr': best_lr, 'hidden_dim': best_hidden_dim}
        tuning_results[omega] = {
            'best_lr': best_lr,
            'best_hidden_dim': best_hidden_dim,
            'best_metrics': best_metrics,
            'all_trials': omega_results
        }

        print(f"\n  --> Selected LR: {best_lr:.6f}, Hidden Dim: {best_hidden_dim} "
              f"(Reward: {best_metrics['avg_eval_reward']:.2f})")

    with open(LR_FILE, 'w') as f:
        json.dump(optimized_hyperparams, f, indent=4)

    tuning_results_file = os.path.join(OUTPUT_DIR, "tuning_results_detailed.json")
    with open(tuning_results_file, 'w') as f:
        json.dump(
            {str(omega): {'best_lr': d['best_lr'], 'best_hidden_dim': d['best_hidden_dim'],
                          'all_trials': d['all_trials']}
             for omega, d in tuning_results.items()},
            f, indent=4
        )

    end_time = time.time()
    print(f"\n--- Grid Search complete in {end_time - start_time:.2f}s. Saved to {LR_FILE} ---")
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
        print(f"WARNING: Optimized hyperparams file not found at {LR_FILE}. Using defaults for all omegas.")
        return {omega: {'lr': DEFAULT_LR, 'hidden_dim': DEFAULT_HIDDEN_DIM} for omega in OMEGA_VALUES}


# ============================================================
# 4. FULL TRAINING AND EVALUATION
# ============================================================
def evaluate_ddqn(omega_val, policy_net, seed=999):
    """Evaluates the final agent and returns reward and rollout data."""
    eval_env = ClassroomGymEnv(
        mode="eval",
        max_weeks=MAX_WEEKS,
        use_discrete_state=False,
        num_action_levels=NUM_ACTION_LEVELS,
        seed=seed,
        omega=omega_val,
        community_risk_data_file=REAL_DATA_FILE
    )

    raw_state, info = eval_env.reset()
    state = normalize_state(raw_state, eval_env.total_students)
    done = False
    total_reward = 0
    rollout_data = []
    s_buffer = torch.zeros(1, 2, dtype=torch.float32, device=device)

    while not done:
        with torch.no_grad():
            s_buffer[0] = torch.as_tensor(state)
            action = int(policy_net(s_buffer).argmax().item())

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

        state = normalize_state(next_raw_state, eval_env.total_students)

    return total_reward, rollout_data, eval_env.capacity_options


def train_and_evaluate_optimal(optimized_hyperparams):
    """Runs full training and evaluation using the optimized hyperparameters."""
    all_rewards_matrix = {}
    representative_nets = {}
    capacity_options = None

    print(f"\n--- Starting Full {AGENT_MODE} Training and Evaluation ---")
    print(f"Training {FULL_EPISODES} episodes per omega using optimized hyperparameters.")

    for omega in OMEGA_VALUES:
        hyperparams = optimized_hyperparams.get(omega, {'lr': DEFAULT_LR, 'hidden_dim': DEFAULT_HIDDEN_DIM})
        lr = hyperparams['lr']
        hidden_dim = hyperparams['hidden_dim']
        print(f"\n*** Training for Omega = {omega} (LR: {lr:.6f}, Hidden Dim: {hidden_dim}) ***")

        omega_rewards_runs = []
        run_nets = []
        run_eval_scores = []
        run_caps = []

        # Learning-curve directory (per-episode reward trajectories per run)
        curves_dir = os.path.join(OUTPUT_DIR, "training_curves")
        os.makedirs(curves_dir, exist_ok=True)

        for run_idx in range(NUM_RUNS):
            run_seed = GLOBAL_SEED + (run_idx * 100)
            print(f"  > Run {run_idx + 1}/{NUM_RUNS} (Seed {run_seed}) starting...")

            # Train using the optimal hyperparameters
            net, rewards, caps = train_ddqn(
                omega_val=omega,
                run_seed=run_seed,
                lr=lr,
                episodes=FULL_EPISODES,
                hidden_dim=hidden_dim
            )
            omega_rewards_runs.append(rewards)

            # Save per-run learning curve
            curve_path = os.path.join(curves_dir, f"training_rewards_omega_{omega}_run{run_idx}.csv")
            with open(curve_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["episode", "total_reward"])
                for ep_idx, ep_reward in enumerate(rewards, start=1):
                    writer.writerow([ep_idx, ep_reward])

            # Score this run for best-model selection (mean over TUNE_EVAL_SEEDS)
            run_eval_score = evaluate_tuning_agent(omega, net)
            run_nets.append(net)
            run_eval_scores.append(run_eval_score)
            run_caps.append(caps)
            print(f"  > Run {run_idx + 1} finished. Tuning-eval score: {run_eval_score:.2f}")

        # --- Best-model selection: highest mean eval reward over TUNE_EVAL_SEEDS ---
        best_idx = max(range(NUM_RUNS), key=lambda i: run_eval_scores[i])
        best_net = run_nets[best_idx]
        capacity_options = run_caps[best_idx]
        representative_nets[omega] = best_net
        print(f"  >> Best run: #{best_idx + 1} (eval {run_eval_scores[best_idx]:.2f})")

        # Save best Model
        model_path = os.path.join(OUTPUT_DIR, f"policy_{AGENT_MODE.lower()}_omega_{omega}.pth")
        torch.save(best_net.state_dict(), model_path)

        # Evaluate best model (fixed seed 999 for the saved rollout)
        final_score, rollout, _ = evaluate_ddqn(omega, best_net, seed=999)

        # Save Rollout CSV
        csv_name = os.path.join(OUTPUT_DIR, f"eval_rollout_{AGENT_MODE.lower()}_omega_{omega}.csv")
        with open(csv_name, mode='w', newline='') as f:
            writer = csv.DictWriter(f,
                                    fieldnames=["Week", "Action_Index", "Allowed_Students", "Infected_Students",
                                                "Community_Risk", "Reward"])
            writer.writeheader()
            writer.writerows(rollout)

        # Save Total Reward
        txt_name = os.path.join(OUTPUT_DIR, f"total_reward_{AGENT_MODE.lower()}_omega_{omega}.txt")
        with open(txt_name, "w") as f:
            f.write(f"Optimal LR: {lr}\n")
            f.write(f"Optimal Hidden Dim: {hidden_dim}\n")
            f.write(f"Best Run: {best_idx + 1}/{NUM_RUNS} (tuning-eval {run_eval_scores[best_idx]:.4f})\n")
            f.write(f"Final Eval Score (Seed 999): {final_score:.4f}\n")

        print(f"  > Best-model eval score (seed 999): {final_score:.2f}")

        all_rewards_matrix[omega] = np.array(omega_rewards_runs)

    plot_combined_rewards(all_rewards_matrix, NUM_RUNS)
    if capacity_options:
        plot_policy_strips(representative_nets, capacity_options)


# ============================================================
# 5. PLOTTING FUNCTIONS
# ============================================================
def plot_combined_rewards(all_rewards_matrix, num_runs):
    """Plots combined smoothed training rewards with confidence intervals."""
    print("\n--- Generating Combined Rewards Plot ---")

    # Figure size (Width, Height) in inches
    plt.figure(figsize=(8, 5))

    # Accessible Colors (Tab10)
    cmap = plt.get_cmap('tab10')
    colors_lines = [cmap(i % 10) for i in range(len(OMEGA_VALUES))]

    # --- Robust Smoothing Function ---
    def get_smoothed_data(data, window=100):
        # 1. Sliding window average (valid mode)
        kernel = np.ones(window) / window
        sliding_avg = np.convolve(data, kernel, mode='valid')
        # 2. Growing window average for the start
        growing_avg = np.cumsum(data[:window - 1]) / np.arange(1, window)
        # 3. Concatenate
        return np.concatenate((growing_avg, sliding_avg))

    # ---------------------------------

    for idx, omega in enumerate(OMEGA_VALUES):
        data = all_rewards_matrix[omega]
        mean_rewards = np.mean(data, axis=0)
        std_rewards = np.std(data, axis=0)

        smoothed_mean = get_smoothed_data(mean_rewards, 100)
        smoothed_std = get_smoothed_data(std_rewards, 100)
        x_axis = np.arange(len(smoothed_mean))

        plt.plot(x_axis, smoothed_mean, label=f"$\\omega={omega}$", color=colors_lines[idx], linewidth=3)
        plt.fill_between(x_axis, smoothed_mean - smoothed_std, smoothed_mean + smoothed_std, color=colors_lines[idx],
                         alpha=0.2)

    # 1. Eliminate internal whitespace gaps
    plt.xlim(left=0)
    plt.margins(x=0)

    # Styling
    plt.xlabel("Episode", fontsize=14, fontweight='bold')
    plt.ylabel("Reward", fontsize=14, fontweight='bold')
    plt.title(f"Double DQN Training Reward", fontsize=14,
              fontweight='bold')

    # Legend
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
               ncol=len(OMEGA_VALUES), borderaxespad=0., fontsize=12, frameon=False)

    plt.grid(True, linestyle='--', alpha=0.6, linewidth=1.5)
    plt.tick_params(axis='both', which='major', labelsize=12, width=2, length=6)

    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    # 2. Tight Layout to pack elements
    plt.tight_layout()

    # 3. Save with tight bounding box and minimal padding
    # 'pad_inches=0.05' leaves a tiny breathing room so text doesn't touch the edge.
    # Set 'pad_inches=0' if you want absolutely zero white border.
    plt.savefig(os.path.join(OUTPUT_DIR, f"combined_{AGENT_MODE.lower()}_rewards_ci.png"),
                dpi=300,
                bbox_inches='tight',
                pad_inches=0.05)
    plt.close()


def plot_policy_strips(representative_nets, capacity_options):
    """Plots the final policy as a heatmap strip for each omega value."""
    print("\n--- Generating Policy Strips Plot ---")

    action_colors = generate_distinct_colors(NUM_ACTION_LEVELS)
    cmap = ListedColormap(action_colors)

    legend_elems = [mpatches.Patch(color=action_colors[i], label=f"Act {i} ({capacity_options[i]} cap.)")
                    for i in range(NUM_ACTION_LEVELS)]

    fig, axes = plt.subplots(1, len(OMEGA_VALUES), figsize=(20, 3.5), sharey=True)
    if len(OMEGA_VALUES) == 1:
        axes = [axes]

    for idx, omega in enumerate(OMEGA_VALUES):
        ax = axes[idx]
        net = representative_nets[omega]

        # Use extract_policy_grid for consistency
        policy_grid = extract_policy_grid(net)

        ax.imshow(
            policy_grid,
            extent=[0, 1, 0, 100],
            origin="lower",
            aspect="auto",
            cmap=cmap,
            vmin=0,
            vmax=NUM_ACTION_LEVELS - 1
        )

        ax.set_title(f"{AGENT_MODE} $\\omega={omega}$")
        ax.set_xlabel("Risk")
        if idx == 0:
            ax.set_ylabel("Infected Count")
        else:
            ax.tick_params(axis='y', which='both', left=False, labelleft=False)

    fig.legend(handles=legend_elems, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=NUM_ACTION_LEVELS, fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(OUTPUT_DIR, f"combined_{AGENT_MODE.lower()}_policies.png"), dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================
# 6. MAIN CONTROL FUNCTION
# ============================================================
def main(mode='tune_and_train'):
    """
    Main function to control execution flow.

    Modes:
    - 'tune': Runs Grid Search to find optimal hyperparams (LR, hidden_dim) for each omega.
    - 'train': Loads optimal hyperparams and runs full training, evaluation, and plotting.
    - 'tune_and_train': Runs tuning, then immediately runs training.
    """
    optimized_hyperparams = {}

    if mode == 'tune' or mode == 'tune_and_train':
        optimized_hyperparams = grid_search_tuning()
    else:
        optimized_hyperparams = load_optimized_hyperparams()

    if mode == 'train' or mode == 'tune_and_train':
        if optimized_hyperparams:
            train_and_evaluate_optimal(optimized_hyperparams)
        else:
            print("Error: Could not determine optimized hyperparams for training.")
            sys.exit(1)

    print(f"\nAll processing complete. Results in {OUTPUT_DIR}")


# --- Execution ---
if __name__ == '__main__':
    skip_tune = '--skip-tune' in sys.argv
    if skip_tune:
        main(mode='train')
    else:
        main(mode='tune_and_train')