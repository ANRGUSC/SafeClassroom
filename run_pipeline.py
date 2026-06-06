#!/usr/bin/env python3
"""
run_pipeline.py — Full SafeCampus experiment pipeline.

Training always uses TRAIN_RISK_TYPE = sinusoidal. The pipeline runs the
tune + train + evaluate loop once per --eval-risk-type so each evaluation
mode gets hyperparameters tuned against its own risk distribution.

By default both modes are run:
  Step 1a. Train (eval-risk-type = sinusoidal)  -> *_results_tuned_sinusoidal/
  Step 1b. Train (eval-risk-type = data)        -> *_results_tuned_data/
  Step 2a. Evaluate sinusoidal                  -> evaluation_results_sinusoidal/
  Step 2b. Evaluate data                        -> evaluation_results_data/

Usage:
    python run_pipeline.py                              # full: both modes, tune+train+eval
    python run_pipeline.py --skip-tune                  # both modes, train only
    python run_pipeline.py --eval-only                  # both modes, evaluate only
    python run_pipeline.py --eval-risk-type sinusoidal  # only sinusoidal mode
    python run_pipeline.py --eval-risk-type data        # only data mode
"""
import subprocess, sys, argparse, time


def run(cmd, label):
    print(f"\n{'='*70}\n  {label}\n{'='*70}")
    t0 = time.time()
    subprocess.run(cmd, check=True)
    print(f"  Done in {(time.time()-t0)/60:.1f} min")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-tune", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-risk-type", choices=["sinusoidal", "data", "both"],
                        default="both",
                        help="Which eval mode(s) to run end-to-end")
    args = parser.parse_args()

    if args.eval_risk_type == "both":
        risk_types = ["sinusoidal", "data"]
    else:
        risk_types = [args.eval_risk_type]

    for rt in risk_types:
        if not args.eval_only:
            train_cmd = [sys.executable, "train_all.py", "--eval-risk-type", rt]
            if args.skip_tune:
                train_cmd.append("--skip-tune")
            run(train_cmd, f"STEP: Train all agents  (eval risk type = {rt})")

        run([sys.executable, "evaluate.py", "--eval-mode", rt],
            f"STEP: Evaluate ({rt})")

    print("\n Pipeline complete.")
    out_dirs = [f"evaluation_results_{rt}" for rt in risk_types]
    print("  Results: " + " and ".join(out_dirs))


if __name__ == "__main__":
    main()
