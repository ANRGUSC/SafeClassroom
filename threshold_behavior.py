#!/usr/bin/env python3
"""
threshold_behavior.py — Basic-reproduction-number (R0) threshold plots.

Reproduces Figure 1 of the IEEE Access paper. From the infection law
(campus_gym_env), for a single index case (I = 1) the basic reproduction number
at a CONSTANT community risk c is

    R0(u) = alpha * u + beta * c * u^2                                   (Eq. 2)

R0 is a reproduction RATE (new infections per index case); the threshold R0 = 1
separates the two equilibria:

    * Disease-Free Equilibrium (DFE), R0 < 1  -> infection dies out
    * Endemic Equilibrium      (EE),  R0 >= 1 -> infection persists

Because R0 is monotone in u, the threshold occurs at a single critical capacity
u* (R0(u*) = 1): u < u* is DFE, u >= u* is EE. The plane is shaded into these two
regions (blue DFE / pink EE), with the R0 curve and the R0 = 1 line overlaid.
Two panels contrast a lower community risk (c = 0.1) with a higher one (c = 0.3);
higher risk steepens the curve and moves u* left, shrinking the DFE region.

Parameters are the experiment values from config.py (alpha = 0.005, beta = 0.01),
i.e. the same dynamics the RL agents were trained/evaluated on. (The published
Figure 1 used beta = 0.001 for a compressed 0-4 range; with the true beta = 0.01
R0 reaches ~30 at full occupancy and the DFE region is correspondingly smaller.)

Outputs (model_threshold_figures/):
    threshold_behavior_R0_medium.png   (community risk 0.1, panel (a))
    threshold_behavior_R0_high.png     (community risk 0.3, panel (b))

Usage:
    python threshold_behavior.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from config import FIXED_ALPHA, FIXED_BETA, TOTAL_STUDENTS, MAX_WEEKS, REAL_DATA_FILE
from campus_gym.envs import ClassroomGymEnv

OUTPUT_DIR = "model_threshold_figures"
GRID = 600

# ALPHA = FIXED_ALPHA          # transmission risk within the classroom (config: 0.005)
# BETA  = FIXED_BETA           # community scaling coefficient         (config: 0.01)

ALPHA = 0.005
BETA  = 0.01
# (filename label, community risk c) — matches the paper's panels (a) and (b)
SCENARIOS = [
    ("medium", 0.1),
    ("high",   0.3),
]

DFE_COLOR = "#9ecae1"   # blue  — Disease-Free Equilibrium region
EE_COLOR  = "#fcae91"   # pink  — Endemic Equilibrium region

plt.rcParams.update({"font.size": 11})


def R0(u, c_risk, alpha=ALPHA, beta=BETA):
    """Basic reproduction number for a single index case at constant risk (Eq. 2)."""
    return alpha * u + beta * c_risk * (u ** 2)


def critical_capacity(c_risk, alpha=ALPHA, beta=BETA):
    """Critical capacity u* with R0 = 1: positive root of beta*c*u^2 + alpha*u - 1 = 0."""
    bc = beta * c_risk
    if bc <= 0:
        return (1.0 / alpha) if alpha > 0 else np.inf
    return (-alpha + np.sqrt(alpha ** 2 + 4.0 * bc)) / (2.0 * bc)


def plot_panel(label, c_risk):
    u = np.linspace(0, TOTAL_STUDENTS, GRID)
    r0 = R0(u, c_risk)
    u_star = critical_capacity(c_risk)
    ymax = max(r0.max() * 1.05, 1.2)

    fig, ax = plt.subplots(figsize=(6.2, 4.6))

    # --- DFE / EE regions: shade ONLY the area under the R0 curve (nothing above
    # it), coloured by the equilibrium of that occupancy — DFE where R0<1 (u<u*),
    # EE where R0>=1 (u>=u*). R0 is single-valued in u, so points above the curve
    # are not attainable and are left unshaded. ---
    ax.fill_between(u, 0, r0, where=(u <= u_star), color=DFE_COLOR, alpha=0.55,
                    zorder=0, interpolate=True)
    ax.fill_between(u, 0, r0, where=(u >= u_star), color=EE_COLOR, alpha=0.55,
                    zorder=0, interpolate=True)

    # --- R0 curve and the R0 = 1 threshold ---
    ax.plot(u, r0, color="#08306b", linewidth=2.4, zorder=3)
    ax.axhline(1.0, color="#1a9850", linestyle="--", linewidth=1.8, zorder=2)

    ax.set_xlim(0, TOTAL_STUDENTS)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("Allowed Population $u$")
    ax.set_ylabel("$R_0$")
    ax.set_title(f"Threshold Behavior of $R_0$\n"
                 f"(Community Risk: {c_risk}, $\\alpha_m$ = {ALPHA:g}, $\\beta$ = {BETA:g})",
                 fontsize=11)

    legend_handles = [
        Line2D([0], [0], color="#08306b", lw=2.4, label="$R_0$"),
        Patch(facecolor=DFE_COLOR, alpha=0.55, label="DFE region ($R_0 < 1$)"),
        Patch(facecolor=EE_COLOR,  alpha=0.55, label="EE region ($R_0 \\geq 1$)"),
        Line2D([0], [0], color="#1a9850", lw=1.8, ls="--", label="$R_0 = 1$"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=8.5, framealpha=0.95)
    ax.grid(True, linestyle=":", alpha=0.35, zorder=1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"threshold_behavior_R0_{label}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path, u_star


# ============================================================
# Threshold behavior for a TIME-VARYING risk pattern
# ============================================================

def get_risk_pattern(risk_type, seed=101):
    """Return the weekly community-risk sequence for the given pattern, taken from
    the simulator (so it matches what the agents see)."""
    kwargs = dict(mode="eval", use_discrete_state=False, use_continuous_actions=True,
                  total_students=TOTAL_STUDENTS, max_weeks=MAX_WEEKS, omega=0.4, seed=seed)
    if risk_type == "data":
        env = ClassroomGymEnv(community_risk_data_file=REAL_DATA_FILE, **kwargs)
    else:
        env = ClassroomGymEnv(eval_risk_type="sinusoidal", **kwargs)
    env.reset(seed=seed)
    return np.asarray(env.risk_pattern, dtype=float)[:MAX_WEEKS]


def plot_pattern(risk_type, seed=101):
    """Threshold behavior over a time-varying risk pattern: the critical capacity
    u*(t) (R0=1 boundary) tracks the risk each week, splitting the admissible
    occupancy into a DFE region (u < u*) and an EE region (u >= u*)."""
    c = get_risk_pattern(risk_type, seed)
    weeks = np.arange(1, len(c) + 1)
    u_star = np.clip([critical_capacity(ci) for ci in c], 0, TOTAL_STUDENTS)

    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(8.2, 6.4), sharex=True,
        gridspec_kw={"height_ratios": [1, 2.2]})

    # Top: the community-risk pattern
    ax0.plot(weeks, c, color="#6a3d9a", marker="o", linewidth=2, markersize=4)
    ax0.set_ylabel("Community\nrisk $c_{\\mathrm{risk}}$")
    ax0.set_ylim(0, 1)
    ax0.grid(True, linestyle=":", alpha=0.4)
    risk_label = "Real COVID-19" if risk_type == "data" else "Sinusoidal"
    ax0.set_title(f"Threshold Behavior over a {risk_label} Risk Pattern\n"
                  f"($\\alpha_m$ = {ALPHA:g}, $\\beta$ = {BETA:g})", fontsize=11)

    # Bottom: DFE / EE regions vs week, bounded by the critical capacity u*(t)
    ax1.fill_between(weeks, 0, u_star, color=DFE_COLOR, alpha=0.6,
                     step="mid", zorder=0)
    ax1.fill_between(weeks, u_star, TOTAL_STUDENTS, color=EE_COLOR, alpha=0.6,
                     step="mid", zorder=0)
    ax1.step(weeks, u_star, where="mid", color="#08306b", linewidth=2.2,
             zorder=3, label="critical capacity $u^*(t)$")
    ax1.set_xlim(1, len(c))
    ax1.set_ylim(0, TOTAL_STUDENTS)
    ax1.set_xlabel("Week $t$")
    ax1.set_ylabel("Allowed students $u$")
    ax1.grid(True, linestyle=":", alpha=0.4)

    legend_handles = [
        Line2D([0], [0], color="#08306b", lw=2.2, label="critical capacity $u^*(t)$"),
        Patch(facecolor=DFE_COLOR, alpha=0.6, label="DFE region ($u < u^*$, $R_0<1$)"),
        Patch(facecolor=EE_COLOR,  alpha=0.6, label="EE region ($u \\geq u^*$, $R_0\\geq 1$)"),
    ]
    ax1.legend(handles=legend_handles, loc="upper right", fontsize=8.5, framealpha=0.95)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"threshold_behavior_pattern_{risk_type}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path, u_star


# Large, bold styling for print-quality combined figure.
PAPER_RC = {
    "font.size": 18, "font.weight": "bold",
    "axes.titlesize": 21, "axes.titleweight": "bold",
    "axes.labelsize": 20, "axes.labelweight": "bold",
    "xtick.labelsize": 16, "ytick.labelsize": 16,
    "legend.fontsize": 15, "lines.linewidth": 3.0,
    "axes.linewidth": 1.6,
}


def plot_patterns_combined(seed=101):
    """Single side-by-side figure: sinusoidal (left) and real COVID-19 (right)
    risk patterns, each with its weekly risk (top) and the DFE/EE safe-occupancy
    boundary u*(t) (bottom)."""
    patterns = [("sinusoidal", "Sinusoidal"), ("data", "Real COVID-19")]

    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(
            2, 2, figsize=(16, 8.5), sharex="col",
            gridspec_kw={"height_ratios": [1, 2.3]})

        for col, (risk_type, title) in enumerate(patterns):
            c = get_risk_pattern(risk_type, seed)
            weeks = np.arange(1, len(c) + 1)
            u_star = np.clip([critical_capacity(ci) for ci in c], 0, TOTAL_STUDENTS)
            ax_top, ax_bot = axes[0, col], axes[1, col]

            # Top: community-risk pattern
            ax_top.plot(weeks, c, color="#6a3d9a", marker="o", linewidth=3, markersize=7)
            ax_top.set_ylim(0, 1.0)
            ax_top.set_title(f"{title} Risk Pattern", pad=10)
            ax_top.grid(True, linestyle=":", alpha=0.5)
            ax_top.tick_params(width=1.6, length=5)

            # Bottom: DFE / EE regions bounded by u*(t)
            ax_bot.fill_between(weeks, 0, u_star, step="mid",
                                color=DFE_COLOR, alpha=0.6, zorder=0)
            ax_bot.fill_between(weeks, u_star, TOTAL_STUDENTS, step="mid",
                                color=EE_COLOR, alpha=0.6, zorder=0)
            ax_bot.step(weeks, u_star, where="mid", color="#08306b",
                        linewidth=3.2, zorder=3)
            ax_bot.set_xlim(1, len(c))
            ax_bot.set_ylim(0, TOTAL_STUDENTS)
            ax_bot.set_xlabel("Week $t$")
            ax_bot.grid(True, linestyle=":", alpha=0.5)
            ax_bot.tick_params(width=1.6, length=5)

            if col == 0:
                ax_top.set_ylabel("Community\nrisk $c_{\\mathrm{risk}}$")
                ax_bot.set_ylabel("Allowed students $u$")

        # Single legend (regions + boundary) on the right bottom panel
        legend_handles = [
            Line2D([0], [0], color="#08306b", lw=3.2, label="critical capacity $u^*(t)$"),
            Patch(facecolor=DFE_COLOR, alpha=0.6, label="DFE region ($u<u^*$, $R_0<1$)"),
            Patch(facecolor=EE_COLOR,  alpha=0.6, label="EE region ($u\\geq u^*$, $R_0\\geq 1$)"),
        ]
        axes[1, 1].legend(handles=legend_handles, loc="upper right",
                          framealpha=0.95, borderpad=0.4)

        fig.suptitle("Threshold Behavior of the Safe-Occupancy Boundary "
                     f"($\\alpha_m$ = {ALPHA:g}, $\\beta$ = {BETA:g})",
                     fontsize=22, fontweight="bold", y=1.0)

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        path = os.path.join(OUTPUT_DIR, "threshold_behavior_patterns.png")
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
    return path


def main():
    print(f"Threshold behavior: R0(u) = alpha*u + beta*c*u^2  "
          f"(alpha={ALPHA:g}, beta={BETA:g})\n")
    print("-- constant-risk panels --")
    for label, c_risk in SCENARIOS:
        path, u_star = plot_panel(label, c_risk)
        print(f"[{label:6s}] community risk c={c_risk}:  critical capacity u*={u_star:.1f}  "
              f"(DFE for u<u*, EE for u>=u*)  ->  {path}")

    print("\n-- combined time-varying risk patterns (side by side) --")
    path = plot_patterns_combined()
    print(f"  -> {path}")

    print(f"\nDone. Figures in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
