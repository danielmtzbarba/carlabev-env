import os
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Root directory where all experiments live
ROOT = "/Users/danielmtz/Data/results/TUDresden/results/runs_final"

# Scalar tags we care about (adapt to your logging)
SCALAR_TAGS = [
    "stats/success_rate",
    "stats/collision_rate",
    "stats/episodic_length",
    "stats/episodic_return",
    "stats/mean_reward",
]

def parse_experiment_name(exp_name: str) -> dict:
    """
    Parse experiment folder name of the form:
      exp-1_cnn-ppo_traffic-off_input-rgb_rwd-shaping_curr-off

    Returns a dict with:
      exp_id, algo, traffic, input, reward, curriculum, experiment
    """
    meta = {
        "experiment": exp_name,
        "exp_id": None,
        "algo": None,
        "traffic": None,
        "input": None,
        "reward": None,
        "curriculum": None,
    }

    parts = exp_name.split("_")
    for p in parts:
        if p.startswith("exp-"):
            try:
                meta["exp_id"] = int(p.split("-")[1])
            except ValueError:
                meta["exp_id"] = None
        elif p.startswith("cnn") or p.startswith("ppo") or "cnn-ppo" in p:
            meta["algo"] = p
        elif p.startswith("traffic-"):
            meta["traffic"] = p.split("traffic-")[1]
        elif p.startswith("input-"):
            meta["input"] = p.split("input-")[1]
        elif p.startswith("rwd-"):
            meta["reward"] = p.split("rwd-")[1]
        elif p.startswith("curr-"):
            meta["curriculum"] = p.split("curr-")[1]

    return meta


def load_scalar_series(event_path: str, tag: str):
    """
    Load full scalar series (step, value) from a TB event file for a given tag.
    Returns list of (step, value) or None.
    """
    try:
        ea = EventAccumulator(event_path)
        ea.Reload()

        scalar_tags = ea.Tags().get("scalars", [])
        if tag not in scalar_tags:
            return None

        events = ea.Scalars(tag)
        return [(e.step, e.value) for e in events]

    except Exception as e:
        print(f"⚠️ Could not read TensorBoard file {event_path}: {e}")
        return None


def load_eval_results(eval_path: str):
    """
    Load the dict inside eval-results-last.npy.
    Returns a dict or None.
    """
    if not os.path.exists(eval_path):
        return None

    try:
        data = np.load(eval_path, allow_pickle=True).item()
        return data
    except Exception as e:
        print(f"⚠️ Could not read {eval_path}: {e}")
        return None

def load_all_results(
    root: str = ROOT,
    scalar_tags = None,
):
    """
    Walk over all experiments in `root`, load:
      - training curves for given scalar_tags
      - eval-results-last.npy

    Returns:
      df_train: columns = [exp_id, experiment, algo, traffic, input, reward,
                           curriculum, step, tag, value]
      df_eval:  columns = [exp_id, experiment, algo, traffic, input, reward,
                           curriculum, mean_return, success_rate, ...]
    """
    if scalar_tags is None:
        scalar_tags = SCALAR_TAGS

    all_train_rows = []
    all_eval_rows = []

    for exp in sorted(os.listdir(root)):
        exp_dir = os.path.join(root, exp)
        if not os.path.isdir(exp_dir):
            continue

        meta = parse_experiment_name(exp)
        print(f"\n📁 Experiment: {exp}  ({meta})")

        # -------------------------
        # 1) Find TensorBoard event file
        # -------------------------
        event_file = None
        for r, dirs, files in os.walk(exp_dir):
            for f in files:
                if f.startswith("events.out"):
                    event_file = os.path.join(r, f)
                    break
            if event_file:
                break

        if event_file is None:
            print("  ⚠️ No TensorBoard event file found")
        else:
            # For each tag we care about, extract series
            try:
                ea = EventAccumulator(event_file)
                ea.Reload()
                available = ea.Tags().get("scalars", [])
            except Exception as e:
                print(f"  ⚠️ Could not load EventAccumulator: {e}")
                available = []

            for tag in scalar_tags:
                if tag not in available:
                    # silently skip or print
                    # print(f"  ⚠️ Tag '{tag}' not in scalars for {exp}")
                    continue

                series = load_scalar_series(event_file, tag)
                if series is None:
                    continue

                for step, value in series:
                    row = {
                        **meta,
                        "step": step,
                        "tag": tag,
                        "value": value,
                    }
                    all_train_rows.append(row)
                print(f"  ✓ Loaded {len(series)} points for tag '{tag}'")

        # -------------------------
        # 2) Load eval-results-last.npy
        # -------------------------
        eval_file = os.path.join(exp_dir, "eval-results-last.npy")
        eval_data = load_eval_results(eval_file)

        if eval_data:
            eval_row = {**meta, **eval_data}
            all_eval_rows.append(eval_row)

            print(
                f"  ✓ Eval: "
                f"success={eval_data.get('success_rate'):.3f}, "
                f"collision={eval_data.get('collision_rate'):.3f}, "
                f"return={eval_data.get('mean_return'):.2f}"
            )
        else:
            print("  ⚠️ No eval results found")

    df_train = pd.DataFrame(all_train_rows)
    df_eval = pd.DataFrame(all_eval_rows)

    return df_train, df_eval