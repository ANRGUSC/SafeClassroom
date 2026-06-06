import gymnasium as gym
from gymnasium.spaces import Discrete, MultiDiscrete, Box
import numpy as np
import random
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def get_discrete_value(x, max_val, num_levels):
    """Stable discretization without float jitter."""
    if max_val <= 0 or num_levels <= 1:
        return 0
    x = np.clip(float(x), 0, max_val)
    level = int(np.floor((x / max_val) * (num_levels - 1)))
    return np.clip(level, 0, num_levels - 1)


def estimate_infected_students(current_infected, allowed_students, community_risk, total_students, alpha, beta):
    """
    Calculates the number of newly infected students for the coming week.

    Recovery is implicit: the model assumes all individuals infected in week t
    recover fully before week t+1 (SIS-like with a full weekly reset). The
    returned value replaces current_infected entirely -- it is NOT added to it.
    Cumulative infection burden must therefore be computed by summing
    infected_students across all steps in an evaluation rollout, not by reading
    any single state observation.

    Formula (internal spread + community spread):
        I_{t+1} = alpha * I_t * A_t + beta * r_t * A_t^2
    where I_t = current infected, A_t = allowed students, r_t = community risk.
    """
    # Infection Formula: Internal Spread + Community Spread
    val = (alpha * current_infected * allowed_students) + \
          (beta * community_risk * (allowed_students ** 2))

    return min(int(round(val)), allowed_students)


# --------------------------------------------------
# Classroom Environment
# --------------------------------------------------

class ClassroomGymEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    # Perturbation level presets: maps level name to noise standard deviation (as fraction of infected count)
    PERTURBATION_PRESETS = {
        "none": 0.0,      # No noise
        "low": 0.25,      # 25% noise std dev
        "moderate": 0.5,  # 50% noise std dev
        "high": 0.75,     # 75% noise std dev
        "severe": 1.0,    # 100% noise std dev
    }

    def __init__(
            self,
            total_students=100,
            max_weeks=15,
            use_discrete_state=True,
            use_continuous_actions=False,  # <--- NEW FLAG
            num_action_levels=3,
            num_discrete_levels=10,
            omega=0.4,
            # Fixed values (used for Eval)
            fixed_alpha=0.005,
            fixed_beta=0.01,
            # Random ranges (used for Training)
            # alpha_range=(0.000, 0.01),
            # beta_range=(0.000, 0.03),
            alpha_range=(0.002, 0.008),
            beta_range=(0.005, 0.015),
            seed=123,
            community_risk_data_file=None,
            mode="train",
            perturbation_level="low",  # <--- NEW: "low", "medium", "high", or a float value
            eval_risk_type="sinusoidal",  # "sinusoidal", "random" (i.i.d. uniform), or "constant"
            constant_risk_value=0.5       # used when eval_risk_type="constant"
    ):
        # Basic settings
        self.total_students = total_students
        self.max_weeks = max_weeks
        self.num_discrete_levels = num_discrete_levels
        self.use_discrete_state = use_discrete_state
        self.use_continuous_actions = use_continuous_actions
        self.omega = omega
        self.mode = mode
        self.seed_value = seed

        # Risk type: "sinusoidal", "random" (i.i.d. uniform), or "constant"
        self.eval_risk_type     = eval_risk_type
        self.constant_risk_value = float(np.clip(constant_risk_value, 0.0, 1.0))

        # Perturbation settings for eval-perturbed mode
        if isinstance(perturbation_level, str):
            self.perturbation_std = self.PERTURBATION_PRESETS.get(perturbation_level, 0.1)
        else:
            # Allow numeric value directly
            self.perturbation_std = float(perturbation_level)

        # Infection Dynamics Parameters
        self.fixed_alpha = fixed_alpha
        self.fixed_beta = fixed_beta
        self.alpha_range = alpha_range
        self.beta_range = beta_range

        # Current Episode Dynamics (will be set in reset)
        self.current_alpha = fixed_alpha
        self.current_beta = fixed_beta

        # Seed all RNGs ONCE for reproducibility
        np.random.seed(seed)
        random.seed(seed)
        self.rng = np.random.default_rng(seed)

        # Episode index to generate deterministic-but-varied randomness
        self.episode_index = 0

        # --- Action Space ---
        if self.use_continuous_actions:
            # Continuous Action: Normalized [0.0, 1.0] representing fraction of total capacity
            self.action_space = Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
            # We still keep capacity_options for logging/compatibility if needed,
            # but won't use them for the action map.
            self.capacity_options = []
        else:
            # Discrete Action
            step = total_students / (num_action_levels - 1)
            actions = [min(total_students, int(round(i * step))) for i in range(num_action_levels)]
            self.capacity_options = sorted(list(set(actions)))
            self.action_space = Discrete(len(self.capacity_options))

        # --- Observation Space ---
        if use_discrete_state:
            # [Infected Level, Risk Level]
            self.observation_space = MultiDiscrete([num_discrete_levels, num_discrete_levels])
        else:
            # [Infected Count, Risk Value]
            self.observation_space = Box(
                low=np.array([0, 0.0]),
                high=np.array([total_students, 1.0]),
                dtype=np.float32,
            )

        # --- Load CSV Risk (Eval Mode) ---
        self.csv_risk_data = None
        if community_risk_data_file and os.path.exists(community_risk_data_file):
            try:
                self.csv_risk_data = np.genfromtxt(
                    community_risk_data_file, delimiter=",", skip_header=1, usecols=1
                )
                self.max_weeks = len(self.csv_risk_data)
            except Exception as e:
                logging.error(f"Error reading CSV: {e}")

        # Initialize first episode
        self.reset()

    # --------------------------------------------------
    # Risk pattern generator (varies across episodes)
    # --------------------------------------------------
    def _generate_risk_pattern(self, episode_seed):
        # Local RNG for risk pattern generation ensures consistency per episode index
        rng = np.random.default_rng(episode_seed)

        t = np.linspace(0, 2 * np.pi, self.max_weeks)
        risk = np.zeros(self.max_weeks)

        num_components = rng.integers(1, 4)
        for _ in range(num_components):
            amp = rng.uniform(0.2, 0.4)
            freq = rng.uniform(0.5, 2.0)
            phase = rng.uniform(0, 2 * np.pi)
            risk += amp * np.sin(freq * t + phase)

        # normalize to exactly [0.0, 1.0]
        r_min, r_max = risk.min(), risk.max()
        if r_max - r_min > 1e-6:
            risk = (risk - r_min) / (r_max - r_min)
        else:
            risk = np.zeros_like(risk)

        return risk.tolist()

    def _generate_random_risk_pattern(self, episode_seed):
        """Generates i.i.d. uniform random risk values -- no temporal correlation."""
        rng = np.random.default_rng(episode_seed)
        return [float(rng.uniform(0.0, 1.0)) for _ in range(self.max_weeks)]

    def _generate_constant_risk_pattern(self):
        """Generates a flat risk pattern at constant_risk_value for every week."""
        return [self.constant_risk_value] * self.max_weeks

    # --------------------------------------------------
    # Perturbation helper for eval-perturbed mode
    # --------------------------------------------------
    def _apply_perturbation(self, infected_count):
        """
        Apply Gaussian noise to the infection count for eval-perturbed mode.
        Noise std dev is proportional to the infected count (or a minimum baseline).
        """
        # Use a minimum baseline to ensure some noise even at low infection counts
        baseline = max(infected_count, 5)
        noise_std = self.perturbation_std * baseline

        # Sample noise and add to infected count
        noise = self.rng.normal(0, noise_std)
        perturbed = infected_count + noise

        # Clip to valid range [0, total_students] and round to integer
        perturbed = int(round(np.clip(perturbed, 0, self.total_students)))

        return perturbed

    # --------------------------------------------------
    # Reset
    # --------------------------------------------------
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed_value = seed
            self.rng = np.random.default_rng(seed)

        self.current_week = 0
        self.episode_index += 1

        # --- DOMAIN RANDOMIZATION LOGIC ---
        if self.mode == "train":
            # 1. Randomize Infection Dynamics
            # self.current_alpha = self.rng.uniform(self.alpha_range[0], self.alpha_range[1])
            # self.current_beta = self.rng.uniform(self.beta_range[0], self.beta_range[1])
            self.current_alpha = self.fixed_alpha
            self.current_beta = self.fixed_beta

            # 2. Randomize Initial State
            self.current_infected = int(self.rng.integers(1, max(2, self.total_students // 3)))

            # 3. Generate Risk Pattern (respects eval_risk_type for distribution matching)
            episode_seed = (self.seed_value or 0) + self.episode_index
            if self.eval_risk_type == "random":
                self.risk_pattern = self._generate_random_risk_pattern(episode_seed)
            elif self.eval_risk_type == "constant":
                self.risk_pattern = self._generate_constant_risk_pattern()
            else:  # "sinusoidal" (default)
                self.risk_pattern = self._generate_risk_pattern(episode_seed)

        elif self.mode == "eval" or self.mode == "eval-perturbed":
            # 1. Fixed Infection Dynamics
            self.current_alpha = self.fixed_alpha
            self.current_beta = self.fixed_beta

            # 2. Fixed Initial State
            self.current_infected = 20

            # 3. Risk Pattern
            if self.csv_risk_data is not None:
                self.risk_pattern = self.csv_risk_data
            else:
                eval_seed = seed if seed else 999
                if self.eval_risk_type == "random":
                    self.risk_pattern = self._generate_random_risk_pattern(eval_seed)
                elif self.eval_risk_type == "constant":
                    self.risk_pattern = self._generate_constant_risk_pattern()
                else:  # "sinusoidal" (default)
                    self.risk_pattern = self._generate_risk_pattern(eval_seed)

        risk = self._get_risk()

        # Create observation
        if self.use_discrete_state:
            obs = np.array([
                get_discrete_value(self.current_infected, self.total_students, self.num_discrete_levels),
                get_discrete_value(risk, 1.0, self.num_discrete_levels)
            ], dtype=np.int32)
        else:
            obs = np.array([float(self.current_infected), float(risk)], dtype=np.float32)

        return obs, {}

    # --------------------------------------------------
    # Risk lookup
    # --------------------------------------------------
    def _get_risk(self):
        idx = min(self.current_week, self.max_weeks - 1)
        if idx < len(self.risk_pattern):
            return float(self.risk_pattern[idx])
        return float(self.risk_pattern[-1])

    # --------------------------------------------------
    # Step
    # --------------------------------------------------
    def step(self, action):
        if self.use_continuous_actions:
            # Continuous Action: action is in [0.0, 1.0]
            # Clip to be safe
            val = np.clip(action, 0.0, 1.0)
            # If it's an array (Box), extract item
            if isinstance(val, np.ndarray):
                val = float(val.item())

            # Map [0,1] -> [0, total_students]
            allowed = int(round(val * self.total_students))
        else:
            # Discrete Action: index lookup
            allowed = self.capacity_options[int(action)]

        risk = self._get_risk()

        # Infection update using the episode-specific Alpha/Beta
        infected_estimate = estimate_infected_students(
            self.current_infected,
            allowed,
            risk,
            self.total_students,
            self.current_alpha,  # <-- Passed here
            self.current_beta  # <-- Passed here
        )

        # Apply perturbation if in eval-perturbed mode
        if self.mode == "eval-perturbed":
            self.current_infected = self._apply_perturbation(infected_estimate)
        else:
            self.current_infected = infected_estimate

        reward = self.omega * allowed - (1 - self.omega) * self.current_infected

        # Advance time
        self.current_week += 1
        terminated = self.current_week >= self.max_weeks

        next_risk = self._get_risk()

        # Build next observation
        if self.use_discrete_state:
            obs = np.array([
                get_discrete_value(self.current_infected, self.total_students, self.num_discrete_levels),
                get_discrete_value(next_risk, 1.0, self.num_discrete_levels)
            ], dtype=np.int32)
        else:
            obs = np.array([self.current_infected, next_risk], dtype=np.float32)

        info = {
            "infected_students": self.current_infected,
            "allowed_students": allowed,
            "community_risk": next_risk,
            "current_alpha": self.current_alpha,
            "current_beta": self.current_beta
        }

        # Include perturbation info when in eval-perturbed mode
        if self.mode == "eval-perturbed":
            info["perturbation_std"] = self.perturbation_std
            info["unperturbed_infected"] = infected_estimate

        return obs, reward, terminated, False, info

    def render(self):
        print(
            f"Week {self.current_week}: I={self.current_infected} | Risk={self._get_risk():.2f} | α={self.current_alpha:.4f}, β={self.current_beta:.4f}")