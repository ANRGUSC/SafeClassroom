"""
Upper Bound Policy Solver using (i, r, t) State Space

This solver computes the TRUE OPTIMAL policy π*(i, r, t) for a known risk trajectory.
Using the full (i, r, t) state allows proper visualization in (i, r) space.

State: (infected, risk, time)
Policy: π*(i, r, t) - optimal action given current infected, risk, and time

This is the UPPER BOUND because:
- At time t, we know the current risk r_t AND all future risks r_{t+1}, ..., r_T
- RL agents only observe (i, r) without knowing t or future risks

NOTE: BayesOptimalMDPSolver has been removed from the active pipeline.
Only UpperBoundSolver (backward induction with known risk trajectory) is used.

Author: SafeCampus Project
"""

import numpy as np
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import os
import time

from campus_gym.envs import ClassroomGymEnv


# =============================================================================
# UPPER BOUND SOLVER (i, r, t)
# =============================================================================

class UpperBoundSolver:
    """
    Compute optimal policy π*(i, r, t) for known risk trajectory.

    Uses backward induction over the full (i, r, t) state space:
        V_t(i, r) = max_c { R(i, r, c) + V_{t+1}(i', r_{t+1}) }

    where r_{t+1} is the KNOWN next risk from the trajectory.
    """

    def __init__(
        self,
        env: ClassroomGymEnv,
        n_infected_bins: int = 100,
        n_risk_bins: int = 100
    ):
        """Initialize solver from environment."""
        self.env = env

        # Extract parameters
        self.total_students = env.total_students
        self.omega = env.omega
        self.horizon = env.max_weeks
        self.alpha = env.current_alpha
        self.beta = env.current_beta
        self.risk_pattern = list(env.risk_pattern)

        # Action space
        self.use_continuous_actions = env.use_continuous_actions
        if self.use_continuous_actions:
            self.max_capacity = float(env.total_students)
        else:
            self.capacity_options = env.capacity_options

        # Discretization
        self.n_infected_bins = n_infected_bins
        self.n_risk_bins = n_risk_bins
        self.infected_vals = np.linspace(0, self.total_students, n_infected_bins)
        self.risk_vals = np.linspace(0, 1, n_risk_bins)

        # Results: V[t, i_idx, r_idx] and policy[t, i_idx, r_idx]
        self.V = None
        self.policy = None

    def _get_infected_index(self, infected: float) -> int:
        idx = int(round(infected * (self.n_infected_bins - 1) / self.total_students))
        return np.clip(idx, 0, self.n_infected_bins - 1)

    def _get_risk_index(self, risk: float) -> int:
        idx = int(round(risk * (self.n_risk_bins - 1)))
        return np.clip(idx, 0, self.n_risk_bins - 1)

    def _interpolate_value(self, t: int, infected: float, r_idx: int) -> float:
        """Interpolate V for continuous infected value."""
        idx_float = infected * (self.n_infected_bins - 1) / self.total_students
        idx_float = np.clip(idx_float, 0, self.n_infected_bins - 1)

        idx_low = int(np.floor(idx_float))
        idx_high = min(idx_low + 1, self.n_infected_bins - 1)

        if idx_low == idx_high:
            return self.V[t, idx_low, r_idx]

        frac = idx_float - idx_low
        return (1 - frac) * self.V[t, idx_low, r_idx] + frac * self.V[t, idx_high, r_idx]

    def _simulate_step(self, infected: float, risk: float, capacity: float) -> Tuple[float, float]:
        """Simulate one step using environment's dynamics."""
        next_infected = self.alpha * infected * capacity + self.beta * risk * (capacity ** 2)
        next_infected = min(int(round(next_infected)), self.total_students)
        reward = self.omega * capacity - (1 - self.omega) * next_infected
        return float(next_infected), reward

    def _compute_q_value(self, t: int, infected: float, r_idx: int, capacity: float) -> float:
        """
        Compute Q_t(i, r, c) = R(i, r, c) + V_{t+1}(i', r_{t+1})

        Note: r_{t+1} is the KNOWN next risk from the trajectory.
        """
        risk = self.risk_vals[r_idx]
        next_infected, reward = self._simulate_step(infected, risk, capacity)

        if t + 1 >= self.horizon:
            return reward

        # Get the KNOWN next risk from trajectory
        next_risk = self.risk_pattern[t + 1]
        next_r_idx = self._get_risk_index(next_risk)

        future_value = self._interpolate_value(t + 1, next_infected, next_r_idx)
        return reward + future_value

    def _optimize_capacity(self, t: int, infected: float, r_idx: int) -> Tuple[float, float]:
        """
        Find optimal capacity via integer grid search over [0, total_students],
        vectorised so finer (n_infected_bins, n_risk_bins) grids stay tractable.

        minimize_scalar (Brent's method) fails here because _simulate_step uses
        int(round(...)), making Q a staircase function with many local optima.
        Grid search over integers is exact: the env rounds capacity to the nearest
        integer anyway, so there are at most (total_students + 1) distinct Q values.
        """
        # capacities to test (numpy array)
        if self.use_continuous_actions:
            cs = np.arange(int(self.max_capacity) + 1, dtype=np.float64)
        else:
            cs = np.asarray(self.capacity_options, dtype=np.float64)

        # vectorised transition and reward at this (infected, risk, t)
        risk = float(self.risk_vals[r_idx])
        next_inf_real = self.alpha * infected * cs + self.beta * risk * (cs ** 2)
        next_inf = np.minimum(np.round(next_inf_real).astype(np.int64),
                              self.total_students)
        rewards = self.omega * cs - (1.0 - self.omega) * next_inf

        # terminal: future value is zero
        if t + 1 >= self.horizon:
            qs = rewards
        else:
            next_risk = self.risk_pattern[t + 1]
            next_r_idx = self._get_risk_index(next_risk)
            V_col = self.V[t + 1, :, next_r_idx]   # (n_infected_bins,)

            # linear interpolation in infected for every candidate capacity
            idx_float = next_inf.astype(np.float64) * (self.n_infected_bins - 1) \
                        / self.total_students
            idx_float = np.clip(idx_float, 0, self.n_infected_bins - 1)
            idx_low = np.floor(idx_float).astype(np.int64)
            idx_high = np.minimum(idx_low + 1, self.n_infected_bins - 1)
            frac = idx_float - idx_low
            future_vals = (1.0 - frac) * V_col[idx_low] + frac * V_col[idx_high]
            qs = rewards + future_vals

        best = int(np.argmax(qs))
        return float(cs[best]), float(qs[best])

    def solve(self, verbose: bool = True) -> Dict:
        """Solve via backward induction over (i, r, t)."""
        T = self.horizon
        n_inf = self.n_infected_bins
        n_risk = self.n_risk_bins

        if verbose:
            print("=" * 65)
            print("UPPER BOUND SOLVER (i, r, t)")
            print("=" * 65)
            print(f"Environment: total_students={self.total_students}, T={T}")
            print(f"Parameters: α={self.alpha}, β={self.beta}, ω={self.omega}")
            print(f"Risk pattern: {[f'{r:.2f}' for r in self.risk_pattern]}")
            print(f"State space: {n_inf} infected × {n_risk} risk × {T} time")
            print()

        # V[t, i_idx, r_idx] and policy[t, i_idx, r_idx]
        self.V = np.zeros((T + 1, n_inf, n_risk))
        self.policy = np.zeros((T, n_inf, n_risk))

        if verbose:
            print("Running backward induction...")

        start_time = time.time()

        for t in range(T - 1, -1, -1):
            current_risk = self.risk_pattern[t]
            current_r_idx = self._get_risk_index(current_risk)

            for i_idx, infected in enumerate(self.infected_vals):
                for r_idx in range(n_risk):
                    optimal_c, optimal_v = self._optimize_capacity(t, infected, r_idx)
                    self.V[t, i_idx, r_idx] = optimal_v
                    self.policy[t, i_idx, r_idx] = optimal_c

            if verbose:
                # Show value at actual risk for this time step
                avg_v = self.V[t, :, current_r_idx].mean()
                print(f"  t={t}: r_t={current_risk:.3f}, avg V(·, r_t, t)={avg_v:.2f}")

        elapsed = time.time() - start_time

        if verbose:
            print(f"\nSolve time: {elapsed:.2f}s")
            i20_idx = self._get_infected_index(20)
            r0_idx = self._get_risk_index(self.risk_pattern[0])
            print(f"Optimal V*(i=20, r=r_0, t=0) = {self.V[0, i20_idx, r0_idx]:.2f}")

        return {'policy': self.policy, 'V': self.V, 'time': elapsed}

    def get_optimal_capacity(self, t: int, infected: float, risk: float) -> float:
        """Get optimal capacity for state (i, r, t)."""
        i_idx = self._get_infected_index(infected)
        r_idx = self._get_risk_index(risk)
        return self.policy[t, i_idx, r_idx]

    def simulate_with_env(self, seed: int = None) -> Dict:
        """Simulate using the environment with optimal policy."""
        obs, _ = self.env.reset(seed=seed)

        weeks, risks, infected_list, capacities, rewards = [], [], [], [], []
        terminated = False

        while not terminated:
            t = self.env.current_week
            infected = float(self.env.current_infected)
            risk = self.env._get_risk()
            capacity = self.get_optimal_capacity(t, infected, risk)

            weeks.append(t)
            risks.append(risk)
            infected_list.append(infected)
            capacities.append(capacity)

            if self.use_continuous_actions:
                action = np.array([capacity / self.total_students], dtype=np.float32)
            else:
                action = np.argmin([abs(c - capacity) for c in self.capacity_options])

            obs, reward, terminated, _, info = self.env.step(action)
            rewards.append(reward)

        return {
            'weeks': weeks, 'risks': risks, 'infected': infected_list,
            'capacities': capacities, 'rewards': rewards,
            'total_reward': sum(rewards)
        }

    def get_policy_at_time(self, t: int) -> np.ndarray:
        """Get full (i, r) policy heatmap at time t."""
        return self.policy[t, :, :]

    def get_aggregated_policy(self, method: str = 'mean') -> np.ndarray:
        """
        Aggregate policy across time to get stationary-like π(i, r).

        Methods:
            'mean': Average across all time steps
            'first': Use t=0 policy
            'last': Use t=T-1 policy
            'trajectory': Weight by actual trajectory visits
        """
        if method == 'mean':
            return self.policy.mean(axis=0)
        elif method == 'first':
            return self.policy[0, :, :]
        elif method == 'last':
            return self.policy[-1, :, :]
        elif method == 'trajectory':
            # Weight by actual risk visits
            weighted = np.zeros((self.n_infected_bins, self.n_risk_bins))
            counts = np.zeros(self.n_risk_bins)
            for t in range(self.horizon):
                r_idx = self._get_risk_index(self.risk_pattern[t])
                weighted[:, r_idx] += self.policy[t, :, r_idx]
                counts[r_idx] += 1
            # Normalize
            for r_idx in range(self.n_risk_bins):
                if counts[r_idx] > 0:
                    weighted[:, r_idx] /= counts[r_idx]
                else:
                    # Interpolate from nearest visited
                    visited = np.where(counts > 0)[0]
                    if len(visited) > 0:
                        nearest = visited[np.argmin(np.abs(visited - r_idx))]
                        weighted[:, r_idx] = weighted[:, nearest]
            return weighted
        else:
            raise ValueError(f"Unknown method: {method}")


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_policy_at_times(solver: UpperBoundSolver, times: List[int], omega: float,
                         output_dir: str = "upper_bound_results"):
    """Plot (i, r) policy heatmaps at specific time steps."""
    os.makedirs(output_dir, exist_ok=True)

    n_times = len(times)
    fig, axes = plt.subplots(1, n_times, figsize=(5 * n_times, 5))
    if n_times == 1:
        axes = [axes]

    for idx, t in enumerate(times):
        ax = axes[idx]
        policy_ir = solver.get_policy_at_time(t)

        im = ax.imshow(policy_ir, extent=[0, 1, 0, solver.total_students],
                       origin='lower', aspect='auto', cmap='viridis', vmin=0, vmax=100)

        # Mark current risk at this time
        current_risk = solver.risk_pattern[t]
        ax.axvline(x=current_risk, color='red', linestyle='--', lw=2, label=f'r_{t}={current_risk:.2f}')

        ax.set_xlabel('Risk (r)', fontweight='bold')
        if idx == 0:
            ax.set_ylabel('Infected (i)', fontweight='bold')
        ax.set_title(f't={t}, r_t={current_risk:.2f}', fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        plt.colorbar(im, ax=ax, label='Capacity c*')

    plt.suptitle(f'Policy π*(i, r, t) at Different Time Steps (ω={omega})', fontweight='bold', y=1.02)
    plt.tight_layout()

    filepath = os.path.join(output_dir, f"policy_at_times_omega_{omega}.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")


def plot_all_omega(omega_values: List[float], output_dir: str = "upper_bound_results",
                   seed: int = 101):
    """Generate plots for all omega values."""
    os.makedirs(output_dir, exist_ok=True)

    # Figure 1: Aggregated (i, r) policies
    fig1, axes1 = plt.subplots(2, 3, figsize=(15, 10))
    axes1 = axes1.flatten()

    # Figure 2: Trajectories
    fig2, axes2 = plt.subplots(2, 3, figsize=(16, 10))
    axes2 = axes2.flatten()

    results = {}

    for idx, omega in enumerate(omega_values):
        print(f"\nSolving for ω = {omega}...")

        env = ClassroomGymEnv(
            total_students=100, max_weeks=15,
            use_discrete_state=False, use_continuous_actions=True,
            omega=omega, mode="eval", seed=seed
        )
        env.reset(seed=seed)

        solver = UpperBoundSolver(env, n_infected_bins=51, n_risk_bins=21)
        solver.solve(verbose=False)

        traj = solver.simulate_with_env(seed=seed)
        results[omega] = {'solver': solver, 'traj': traj}

        # Plot 1: Aggregated policy (trajectory-weighted)
        ax1 = axes1[idx]
        policy_agg = solver.get_aggregated_policy(method='trajectory')

        im1 = ax1.imshow(policy_agg, extent=[0, 1, 0, 100],
                         origin='lower', aspect='auto', cmap='viridis', vmin=0, vmax=100)

        # Contours
        X, Y = np.meshgrid(np.linspace(0, 1, solver.n_risk_bins),
                           np.linspace(0, 100, solver.n_infected_bins))
        ax1.contour(X, Y, policy_agg, levels=[25, 50, 75], colors='white', linewidths=1)

        # Mark visited risks
        for t in range(solver.horizon):
            ax1.axvline(x=solver.risk_pattern[t], color='red', alpha=0.3, lw=1)

        ax1.set_title(f'ω={omega}, R={traj["total_reward"]:.1f}', fontweight='bold')
        ax1.set_xlabel('Risk')
        if idx % 3 == 0:
            ax1.set_ylabel('Infected')

        # Plot 2: Trajectory
        ax2 = axes2[idx]
        weeks = np.array(traj['weeks'])
        width = 0.35

        ax2.bar(weeks - width/2, traj['infected'], width, label='Infected',
                color='#e74c3c', alpha=0.8)
        ax2.bar(weeks + width/2, traj['capacities'], width, label='Capacity',
                color='#3498db', alpha=0.8)

        ax2_twin = ax2.twinx()
        ax2_twin.plot(weeks, traj['risks'], 'g-o', lw=2, ms=5, label='Risk')
        ax2_twin.fill_between(weeks, 0, traj['risks'], alpha=0.15, color='green')
        ax2_twin.set_ylim(0, 1.15)
        ax2_twin.set_ylabel('Risk', color='green')

        ax2.set_xlabel('Week')
        ax2.set_ylabel('Count')
        ax2.set_title(f'ω={omega}, R={traj["total_reward"]:.1f}', fontweight='bold')
        ax2.set_ylim(0, 110)
        ax2.set_xticks(weeks)

        if idx == 0:
            lines1, labels1 = ax2.get_legend_handles_labels()
            lines2, labels2 = ax2_twin.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)

    # Finalize Figure 1
    fig1.subplots_adjust(right=0.9)
    cbar1 = fig1.colorbar(im1, cax=fig1.add_axes([0.92, 0.15, 0.02, 0.7]))
    cbar1.set_label('Capacity c*')
    fig1.suptitle('Upper Bound Policy π*(i, r) - Aggregated over Time\n(Red lines = visited risk values)',
                  fontweight='bold', y=1.02)
    filepath1 = os.path.join(output_dir, "upper_bound_policies_ir.png")
    fig1.savefig(filepath1, dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print(f"Saved: {filepath1}")

    # Finalize Figure 2
    fig2.suptitle('Trajectories under Upper Bound Policy (Perfect Foresight)',
                  fontweight='bold', y=1.02)
    fig2.tight_layout()
    filepath2 = os.path.join(output_dir, "upper_bound_trajectories.png")
    fig2.savefig(filepath2, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f"Saved: {filepath2}")

    # Summary
    print("\n" + "=" * 60)
    print("UPPER BOUND SUMMARY")
    print("=" * 60)
    print(f"{'Omega':<8} {'Total Reward':<15}")
    print("-" * 25)
    for omega in omega_values:
        print(f"{omega:<8} {results[omega]['traj']['total_reward']:<15.2f}")

    return results


def plot_policy_evolution(solver: UpperBoundSolver, omega: float,
                          output_dir: str = "upper_bound_results"):
    """Plot how policy evolves over time for a specific infected level."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Policy at i=20 as function of (r, t)
    ax1 = axes[0]
    i_idx = solver._get_infected_index(20)
    policy_rt = solver.policy[:, i_idx, :].T  # Shape: (n_risk, T)

    im1 = ax1.imshow(policy_rt, extent=[0, solver.horizon, 0, 1],
                     origin='lower', aspect='auto', cmap='viridis', vmin=0, vmax=100)

    # Overlay actual risk trajectory
    ax1.plot(np.arange(solver.horizon) + 0.5, solver.risk_pattern, 'r-o', lw=2, ms=4, label='Risk trajectory')

    ax1.set_xlabel('Time (t)', fontweight='bold')
    ax1.set_ylabel('Risk (r)', fontweight='bold')
    ax1.set_title(f'Policy π*(i=20, r, t)\n(Red = actual risk trajectory)', fontweight='bold')
    ax1.legend(loc='upper right')
    plt.colorbar(im1, ax=ax1, label='Capacity c*')

    # Right: Policy along actual trajectory
    ax2 = axes[1]

    infected_levels = [0, 20, 40, 60, 80]
    for i_level in infected_levels:
        i_idx = solver._get_infected_index(i_level)
        capacities = []
        for t in range(solver.horizon):
            r_idx = solver._get_risk_index(solver.risk_pattern[t])
            capacities.append(solver.policy[t, i_idx, r_idx])
        ax2.plot(range(solver.horizon), capacities, '-o', label=f'i={i_level}', ms=4)

    ax2.set_xlabel('Time (t)', fontweight='bold')
    ax2.set_ylabel('Optimal Capacity c*', fontweight='bold')
    ax2.set_title('Policy along Risk Trajectory\nπ*(i, r_t, t) for different i', fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 105)

    plt.suptitle(f'Policy Evolution (ω={omega})', fontweight='bold', y=1.02)
    plt.tight_layout()

    filepath = os.path.join(output_dir, f"policy_evolution_omega_{omega}.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")


# =============================================================================
# COMPARISON FUNCTION
# =============================================================================

def compare_with_rl(upper_bound_results: Dict, rl_rewards: Dict[float, float]):
    """Compare upper bound with RL agent rewards."""
    print("\n" + "=" * 70)
    print("COMPARISON: UPPER BOUND vs RL AGENT")
    print("=" * 70)
    print(f"{'Omega':<8} {'Upper Bound':<15} {'RL Agent':<15} {'Gap':<10} {'Gap %':<10}")
    print("-" * 60)

    for omega, ub_result in upper_bound_results.items():
        ub_reward = ub_result['traj']['total_reward']
        rl_reward = rl_rewards.get(omega, 0)
        gap = ub_reward - rl_reward
        gap_pct = 100 * gap / ub_reward if ub_reward != 0 else 0

        print(f"{omega:<8} {ub_reward:<15.2f} {rl_reward:<15.2f} {gap:<10.2f} {gap_pct:<10.1f}%")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("UPPER BOUND SOLVER (i, r, t)")
    print("=" * 70)

    # Single example
    env = ClassroomGymEnv(
        total_students=100, max_weeks=15,
        use_discrete_state=False, use_continuous_actions=True,
        omega=0.4, mode="eval", seed=101
    )
    env.reset(seed=101)

    solver = UpperBoundSolver(env, n_infected_bins=51, n_risk_bins=21)
    solver.solve(verbose=True)

    traj = solver.simulate_with_env(seed=101)
    print(f"\nTrajectory total reward: {traj['total_reward']:.2f}")

    # Plot policy at specific times
    plot_policy_at_times(solver, times=[0, 7, 14], omega=0.4)

    # Plot policy evolution
    plot_policy_evolution(solver, omega=0.4)

    # All omega values
    print("\n" + "=" * 70)
    print("Generating plots for all omega values...")
    print("=" * 70)

    omega_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    results = plot_all_omega(omega_values, seed=101)

    # Example comparison
    print("\n--- Example Comparison (placeholder RL values) ---")
    rl_rewards = {0.1: 15.0, 0.2: 55.0, 0.3: 120.0, 0.4: 220.0, 0.5: 350.0, 0.6: 510.0}
    compare_with_rl(results, rl_rewards)

    print("\nDone!")