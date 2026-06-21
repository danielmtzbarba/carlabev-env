from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from loguru import logger

from CarlaBEV.config import EnvConfig, RandomNavigationReset, build_random_navigation_options
from CarlaBEV.envs.carlabev import CarlaBEV


logger.remove()
logger.add(sys.stdout, format="{message}", colorize=True)


def _derive_structured_seed(
    *,
    study_id: str,
    backbone_id: str,
    study_seed: int,
    episode_index: int,
    kind: str,
) -> int:
    token = f"{study_id}:{backbone_id}:{study_seed}:{episode_index}:{kind}"
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**31 - 1)


def build_scene_library(
    *,
    episodes: int,
    difficulty_preset_id: str = "medium",
    route_extent: str | None = None,
    traffic_density: str | None = None,
    speed_profile: str | None = None,
    seed_offset: int = 0,
    study_seed: int | None = None,
    study_id: str = "manual",
    backbone_id: str = "default",
    episode_index_start: int = 0,
    scene_library_path: str = "CarlaBEV/assets/scenes/scene_library.db",
    route_dist_range: tuple[int, int] | None = None,
    num_vehicles: int | None = None,
    num_vehicles_near_ego: int | None = None,
    traffic_role_profile: str | None = None,
    guaranteed_candidate_role: str | None = None,
    ego_route_graph: str = "canonical",
    render_mode: str = "rgb_array",
) -> dict:
    env = CarlaBEV(
        EnvConfig(
            map_name="Town01",
            render_mode=render_mode,
            obs_mode="bev_rgb",
            action_mode="discrete",
            size=128,
            scene_library_enabled=True,
            scene_library_path=scene_library_path,
        )
    )

    hits = 0
    misses = 0
    scene_keys: list[str] = []
    samples: list[dict] = []
    try:
        for episode in range(int(episodes)):
            episode_index = int(episode_index_start) + episode
            if study_seed is not None:
                scene_seed = _derive_structured_seed(
                    study_id=study_id,
                    backbone_id=backbone_id,
                    study_seed=int(study_seed),
                    episode_index=episode_index,
                    kind="scene",
                )
                route_seed = _derive_structured_seed(
                    study_id=study_id,
                    backbone_id=backbone_id,
                    study_seed=int(study_seed),
                    episode_index=episode_index,
                    kind="route",
                )
                traffic_seed = _derive_structured_seed(
                    study_id=study_id,
                    backbone_id=backbone_id,
                    study_seed=int(study_seed),
                    episode_index=episode_index,
                    kind="traffic",
                )
                scenario_seed = _derive_structured_seed(
                    study_id=study_id,
                    backbone_id=backbone_id,
                    study_seed=int(study_seed),
                    episode_index=episode_index,
                    kind="scenario",
                )
            else:
                seed_base = int(seed_offset) + episode * 1000
                scene_seed = seed_base
                route_seed = seed_base + 1
                traffic_seed = seed_base + 2
                scenario_seed = seed_base + 3
            request = RandomNavigationReset(
                difficulty_preset_id=difficulty_preset_id,
                route_extent=route_extent,
                traffic_density=traffic_density,
                speed_profile=speed_profile,
                route_dist_range=route_dist_range,
                num_vehicles=num_vehicles,
                num_vehicles_near_ego=num_vehicles_near_ego,
                traffic_role_profile=traffic_role_profile,
                guaranteed_candidate_role=guaranteed_candidate_role,
                ego_route_graph=ego_route_graph,
                scene_seed=scene_seed,
                route_seed=route_seed,
                traffic_seed=traffic_seed,
                scenario_seed=scenario_seed,
            )
            options = build_random_navigation_options(request)
            _, info = env.reset(options=options)
            cache_info = info.get("scene_cache", {})
            descriptor = info.get("scene_descriptor", {})
            cache_hit = bool(cache_info.get("hit", False))
            if cache_hit:
                hits += 1
            else:
                misses += 1
            scene_key = str(cache_info.get("scene_key", ""))
            scene_keys.append(scene_key)
            sample = {
                "episode": episode,
                "episode_index": episode_index,
                "scene_key": scene_key,
                "cache_hit": cache_hit,
                "study_seed": study_seed,
                "scene_seed": descriptor.get("scene_seed"),
                "route_seed": descriptor.get("route_seed"),
                "traffic_seed": descriptor.get("traffic_seed"),
                "route_profile": descriptor.get("route_profile"),
                "main_actor_role": descriptor.get("main_actor_role"),
            }
            samples.append(sample)
            logger.opt(colors=True).info(
                "<cyan>Build</cyan> <yellow>{episode}</yellow> | "
                "<cyan>cache</cyan> <yellow>{cache}</yellow> | "
                "<cyan>scene</cyan> <yellow>{scene_seed}</yellow> | "
                "<cyan>route</cyan> <yellow>{route_profile}</yellow> | "
                "<cyan>role</cyan> <yellow>{main_actor_role}</yellow>",
                episode=episode,
                cache="hit" if cache_hit else "miss",
                scene_seed=sample["scene_seed"],
                route_profile=sample["route_profile"] or "unknown",
                main_actor_role=sample["main_actor_role"] or "n/a",
            )
    finally:
        env.close()

    summary = {
        "episodes": int(episodes),
        "difficulty_preset_id": difficulty_preset_id,
        "route_extent": route_extent,
        "traffic_density": traffic_density,
        "speed_profile": speed_profile,
        "study_seed": study_seed,
        "study_id": study_id,
        "backbone_id": backbone_id,
        "scene_library_path": scene_library_path,
        "hits": hits,
        "misses": misses,
        "unique_scene_keys": len(set(scene_keys)),
        "samples": samples,
    }
    logger.opt(colors=True).info(
        "<cyan>SceneLibrary</cyan> <yellow>{path}</yellow> | "
        "<cyan>episodes</cyan> <yellow>{episodes}</yellow> | "
        "<cyan>misses</cyan> <yellow>{misses}</yellow> | "
        "<cyan>hits</cyan> <yellow>{hits}</yellow> | "
        "<cyan>unique</cyan> <yellow>{unique}</yellow>",
        path=Path(scene_library_path).resolve(),
        episodes=summary["episodes"],
        misses=summary["misses"],
        hits=summary["hits"],
        unique=summary["unique_scene_keys"],
    )
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the SQLite scene library without stepping episodes.")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--difficulty-preset-id", default="medium")
    parser.add_argument("--route-extent", default=None)
    parser.add_argument("--traffic-density", default=None)
    parser.add_argument("--speed-profile", default=None)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--study-seed", type=int, default=None)
    parser.add_argument("--study-id", default="manual")
    parser.add_argument("--backbone-id", default="default")
    parser.add_argument("--episode-index-start", type=int, default=0)
    parser.add_argument(
        "--scene-library-path",
        default="CarlaBEV/assets/scenes/scene_library.db",
    )
    parser.add_argument("--min-distance", type=int, default=None)
    parser.add_argument("--max-distance", type=int, default=None)
    parser.add_argument("--num-vehicles", type=int, default=None)
    parser.add_argument("--num-vehicles-near-ego", type=int, default=None)
    parser.add_argument("--traffic-role-profile", default=None)
    parser.add_argument("--guaranteed-candidate-role", default=None)
    parser.add_argument("--ego-route-graph", default="canonical")
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    route_dist_range = None
    if args.min_distance is not None or args.max_distance is not None:
        if args.min_distance is None or args.max_distance is None:
            raise SystemExit("Both --min-distance and --max-distance are required when overriding route distance.")
        route_dist_range = (int(args.min_distance), int(args.max_distance))

    summary = build_scene_library(
        episodes=args.episodes,
        difficulty_preset_id=args.difficulty_preset_id,
        route_extent=args.route_extent,
        traffic_density=args.traffic_density,
        speed_profile=args.speed_profile,
        seed_offset=args.seed_offset,
        study_seed=args.study_seed,
        study_id=args.study_id,
        backbone_id=args.backbone_id,
        episode_index_start=args.episode_index_start,
        scene_library_path=args.scene_library_path,
        route_dist_range=route_dist_range,
        num_vehicles=args.num_vehicles,
        num_vehicles_near_ego=args.num_vehicles_near_ego,
        traffic_role_profile=args.traffic_role_profile,
        guaranteed_candidate_role=args.guaranteed_candidate_role,
        ego_route_graph=args.ego_route_graph,
    )
    if args.as_json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
