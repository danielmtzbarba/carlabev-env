"""Child process for cross-process replay characterization tests.

Runs one seeded random-navigation episode fragment (scene library disabled)
and writes a JSON trajectory fingerprint to the path given as argv[1].

Executed by ``test_replay_determinism.py`` in a fresh interpreter
(with a per-run ``PYTHONHASHSEED``) so that any process-dependent randomness
— e.g. seeding an RNG from ``hash(...)`` — becomes observable. Not a test
module itself (leading underscore keeps pytest from collecting it).
"""

from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


def npc_states(env) -> list[list[float]]:
    states = []
    for kind in ("vehicle", "pedestrian"):
        for actor in env.map.actor_manager.actors.get(kind, []):
            ctrl = actor._controller
            states.append(
                [
                    round(float(ctrl.x_m), 9),
                    round(float(ctrl.y_m), 9),
                    round(float(ctrl.yaw), 9),
                    round(float(ctrl.v_mps), 9),
                    int(getattr(actor, "recycle_count", 0)),
                ]
            )
    return states


def main() -> int:
    out_path = sys.argv[1]
    num_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 170

    from CarlaBEV.src.diagnostics.rollouts import (
        RANDOM_NAV_OPTIONS,
        brake_only_action,
        make_env,
    )

    # Ego holds still (brake) so the episode cannot terminate early while the
    # background traffic runs long enough for the lifecycle to recycle a
    # finished vehicle (first event is at step 63 under generation policy v3 for this seed/options).
    env = make_env(max_episode_duration_s=120.0)
    try:
        env.reset(seed=3, options={**RANDOM_NAV_OPTIONS, "num_vehicles": 6})
        steps = []
        for k in range(num_steps):
            _, reward, terminated, truncated, _ = env.step(brake_only_action(k))
            hero = env.map.hero
            steps.append(
                {
                    "step": k,
                    "hero": [round(float(hero.x_m), 9), round(float(hero.y_m), 9)],
                    "reward": round(float(reward), 9),
                    "npcs": npc_states(env),
                }
            )
            if terminated or truncated:
                break
        lifecycle = env.map.traffic_lifecycle
        payload = {
            "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
            "recycled": 0 if lifecycle is None else int(lifecycle.stats.recycled),
            "steps": steps,
        }
    finally:
        env.close()

    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
