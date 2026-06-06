#!/usr/bin/env python3
"""
train_all.py — Train all SafeCampus RL agents sequentially.

Usage:
    python train_all.py                              # tune + train for sinusoidal eval
    python train_all.py --eval-risk-type data        # tune + train for real-CSV eval
    python train_all.py --skip-tune                  # train only (existing tuning results)

Training itself always uses TRAIN_RISK_TYPE = sinusoidal. The --eval-risk-type
flag controls (a) which risk distribution is used during hyperparameter tuning
evaluation, and (b) the per-agent OUTPUT_DIR suffix so sinusoidal/data runs do
not overwrite each other.

Each agent script is run as a subprocess so its internal __name__=='__main__'
block executes.
"""
import subprocess
import sys
import argparse
import time

AGENTS = [
    ("Double DQN",            "double_dqn.py"),
    ("PPO Discrete",          "ppo_agent.py"),
    ("PPO Continuous",  "ppo_continuous_new.py"),
]

# AGENTS = [
#     ("Tabular Q-Learning",    "train_tabular_q.py"),
#     ("DQN",                   "deep_q_learning.py"),
#     ("Online DQN",            "online_dqn.py"),
#     ("Double DQN",            "double_dqn.py"),
#     ("PPO Discrete",          "ppo_agent.py"),
#     ("PPO Continuous",        "ppo_continuous_actions.py"),
#     ("PPO Continuous (New)",  "ppo_continuous_new.py"),
# ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-tune", action="store_true",
                        help="Skip hyperparameter tuning; use existing JSON files")
    parser.add_argument("--eval-risk-type", choices=["sinusoidal", "data"],
                        default="sinusoidal",
                        help="Risk distribution for tuning eval + OUTPUT_DIR tag")
    args = parser.parse_args()

    extra_args = ["--eval-risk-type", args.eval_risk_type]
    if args.skip_tune:
        extra_args.append("--skip-tune")

    results = {}
    for name, script in AGENTS:
        print(f"\n{'='*70}")
        print(f"  TRAINING: {name}  ({script})  [eval={args.eval_risk_type}]")
        print(f"{'='*70}")
        t0 = time.time()
        try:
            subprocess.run(
                [sys.executable, script] + extra_args,
                check=True
            )
            elapsed = time.time() - t0
            results[name] = f"OK ({elapsed/60:.1f} min)"
        except subprocess.CalledProcessError as e:
            results[name] = f"FAILED (exit code {e.returncode})"
            print(f"  ERROR: {name} failed. Continuing with next agent.")

    print(f"\n{'='*70}")
    print(f"  TRAINING SUMMARY  (eval risk type: {args.eval_risk_type})")
    print(f"{'='*70}")
    for name, status in results.items():
        print(f"  {name:<30} {status}")


if __name__ == "__main__":
    main()
