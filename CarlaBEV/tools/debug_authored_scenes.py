import json
import os
import random
from pathlib import Path

import numpy as np
import pygame
import tyro

from CarlaBEV.envs import make_env
from CarlaBEV.tools.debug.controls import (
    get_action_from_keys,
    init_key_tracking,
    process_events,
)
from CarlaBEV.tools.debug.cfg import ArgsCarlaBEV
from CarlaBEV.src.deeprl.logger import create_loggers


cfg = tyro.cli(ArgsCarlaBEV)
AUTHORED_SCENE_DIR = Path(
    os.environ.get("CARLABEV_AUTHORED_SCENE_DIR", "CarlaBEV/assets/scenes")
)
AUTHORED_MODE = os.environ.get("CARLABEV_AUTHORED_MODE", "train").strip().lower()
AUTHORED_SCENE_FILTER = os.environ.get("CARLABEV_AUTHORED_SCENE_FILTER", "").strip()
AUTHORED_SCENARIO_FILTER = os.environ.get("CARLABEV_AUTHORED_SCENARIO_ID", "").strip()
VARIATION_ENABLED = os.environ.get("CARLABEV_VARIATION_ENABLED", "1").lower() not in {"0", "false", "no"}
VARIATION_SEED_MIN = int(os.environ.get("CARLABEV_VARIATION_SEED_MIN", "0"))
VARIATION_SEED_MAX = int(os.environ.get("CARLABEV_VARIATION_SEED_MAX", "2147483647"))

sim_logger = create_loggers(cfg)


def list_authored_scene_files(scene_dir: Path):
    if not scene_dir.exists():
        raise FileNotFoundError(f"Authored scene directory not found: {scene_dir}")

    scene_files = []
    for path in sorted(scene_dir.glob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            continue
        if "actors" not in data:
            continue
        scene_id = data.get("scene_id", path.stem)
        scenario_id = data.get("scenario_id", "")
        if AUTHORED_SCENE_FILTER and AUTHORED_SCENE_FILTER not in scene_id:
            continue
        if AUTHORED_SCENARIO_FILTER and AUTHORED_SCENARIO_FILTER != scenario_id:
            continue
        scene_files.append((path, data))

    if not scene_files:
        raise RuntimeError(
            f"No authored scenes found in {scene_dir} "
            f"(scene_filter={AUTHORED_SCENE_FILTER!r}, scenario_filter={AUTHORED_SCENARIO_FILTER!r})."
        )
    return scene_files


def resolve_mode_variation_enabled():
    if AUTHORED_MODE == "eval":
        return False
    return VARIATION_ENABLED


def build_train_reset_options(reset_mask, scene_pool, rng):
    scene_path, scene_data = rng.choice(scene_pool)
    variation_enabled = resolve_mode_variation_enabled()
    options = {
        "config_file": str(scene_path),
        "reset_mask": reset_mask,
        "variation_enabled": variation_enabled,
    }
    variation_seed = None
    if variation_enabled:
        variation_seed = rng.randint(VARIATION_SEED_MIN, VARIATION_SEED_MAX)
        options["variation_seed"] = variation_seed
    meta = {
        "mode": "train",
        "scene_path": str(scene_path),
        "scene_id": scene_data.get("scene_id", scene_path.stem),
        "scenario_id": scene_data.get("scenario_id"),
        "variation_enabled": variation_enabled,
        "variation_seed": variation_seed,
    }
    return options, meta


def build_eval_reset_options(reset_mask, scene_pool, scene_index):
    scene_path, scene_data = scene_pool[scene_index % len(scene_pool)]
    options = {
        "config_file": str(scene_path),
        "reset_mask": reset_mask,
        "variation_enabled": False,
    }
    meta = {
        "mode": "eval",
        "scene_path": str(scene_path),
        "scene_id": scene_data.get("scene_id", scene_path.stem),
        "scenario_id": scene_data.get("scenario_id"),
        "variation_enabled": False,
        "variation_seed": None,
        "scene_index": scene_index % len(scene_pool),
    }
    return options, meta


def build_authored_reset_options(reset_mask, scene_pool, rng, scene_index):
    if AUTHORED_MODE == "eval":
        return build_eval_reset_options(reset_mask, scene_pool, scene_index)
    return build_train_reset_options(reset_mask, scene_pool, rng)


def debug_print_reset(meta):
    print(
        "[authored-debug] reset",
        {
            "mode": meta["mode"],
            "scene_id": meta["scene_id"],
            "scenario_id": meta["scenario_id"],
            "variation_enabled": meta["variation_enabled"],
            "variation_seed": meta["variation_seed"],
            "scene_path": meta["scene_path"],
        },
    )


def main():
    if AUTHORED_MODE not in {"train", "eval"}:
        raise ValueError(
            f"Unsupported CARLABEV_AUTHORED_MODE={AUTHORED_MODE!r}. Use 'train' or 'eval'."
        )
    pygame.init()
    keys_held = init_key_tracking()
    envs = make_env(cfg)
    rng = random.Random(cfg.seed)
    scene_pool = list_authored_scene_files(AUTHORED_SCENE_DIR)
    scene_index = 0

    print("Observation space:", envs.observation_space)
    print(
        "[authored-debug] scene-pool",
        {
            "mode": AUTHORED_MODE,
            "count": len(scene_pool),
            "dir": str(AUTHORED_SCENE_DIR),
            "scene_filter": AUTHORED_SCENE_FILTER,
            "scenario_filter": AUTHORED_SCENARIO_FILTER,
            "variation_enabled": resolve_mode_variation_enabled(),
        },
    )

    options, meta = build_authored_reset_options(
        np.array([True], dtype=bool),
        scene_pool,
        rng,
        scene_index,
    )
    debug_print_reset(meta)
    observation, info = envs.reset(options=options)
    spawn_info = info.get("spawn_validation", {})
    if spawn_info and not spawn_info.get("valid", False):
        raise RuntimeError(f"Invalid ego spawn at reset: {spawn_info}")
    running = True

    while running:
        running = process_events(keys_held)
        action_idx = get_action_from_keys(keys_held)

        if cfg.env.action_space == "continuous":
            steer = 0.0
            gas = 0.0
            brake = 0.0

            if action_idx == 1:
                gas = 1.0
            elif action_idx == 2:
                brake = 1.0
            elif action_idx == 3:
                gas = 1.0
                steer = 1.0
            elif action_idx == 4:
                gas = 1.0
                steer = -1.0
            elif action_idx == 5:
                steer = 1.0
            elif action_idx == 6:
                steer = -1.0
            elif action_idx == 7:
                brake = 1.0
                steer = 1.0
            elif action_idx == 8:
                brake = 1.0
                steer = -1.0

            action = np.array([gas, steer, brake], dtype=np.float32)
        else:
            action = action_idx

        observation, reward, terminated, trunks, info = envs.step([action])
        for i, ended in enumerate(terminated):
            if ended:
                sim_logger.log_episode(info["episode_info"], i)
                if AUTHORED_MODE == "eval":
                    scene_index += 1
                options, meta = build_authored_reset_options(
                    np.logical_or(terminated, trunks),
                    scene_pool,
                    rng,
                    scene_index,
                )
                debug_print_reset(meta)
                observation, info = envs.reset(options=options)
                spawn_info = info.get("spawn_validation", {})
                if spawn_info and not spawn_info.get("valid", False):
                    raise RuntimeError(f"Invalid ego spawn during debug reset: {spawn_info}")

    envs.close()


if __name__ == "__main__":
    main()
