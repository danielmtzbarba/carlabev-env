from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random

import numpy as np


_SEED_MODULUS = 2**31 - 1


def derive_seed(base_seed: int, *parts: object) -> int:
    token = ":".join([str(int(base_seed)), *(str(part) for part in parts)])
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % _SEED_MODULUS


@dataclass
class RNGBundle:
    scene_seed: int
    route_seed: int
    traffic_seed: int
    scenario_seed: int
    scene_rng: random.Random
    route_rng: random.Random
    traffic_rng: random.Random
    scenario_rng: random.Random
    scene_np_rng: np.random.Generator
    route_np_rng: np.random.Generator
    traffic_np_rng: np.random.Generator
    scenario_np_rng: np.random.Generator


def build_rng_bundle(
    *,
    scene_seed: int,
    route_seed: int | None = None,
    traffic_seed: int | None = None,
    scenario_seed: int | None = None,
) -> RNGBundle:
    scene_seed = int(scene_seed)
    route_seed = derive_seed(scene_seed, "route") if route_seed is None else int(route_seed)
    traffic_seed = (
        derive_seed(scene_seed, "traffic") if traffic_seed is None else int(traffic_seed)
    )
    scenario_seed = (
        derive_seed(scene_seed, "scenario")
        if scenario_seed is None
        else int(scenario_seed)
    )
    return RNGBundle(
        scene_seed=scene_seed,
        route_seed=route_seed,
        traffic_seed=traffic_seed,
        scenario_seed=scenario_seed,
        scene_rng=random.Random(scene_seed),
        route_rng=random.Random(route_seed),
        traffic_rng=random.Random(traffic_seed),
        scenario_rng=random.Random(scenario_seed),
        scene_np_rng=np.random.default_rng(scene_seed),
        route_np_rng=np.random.default_rng(route_seed),
        traffic_np_rng=np.random.default_rng(traffic_seed),
        scenario_np_rng=np.random.default_rng(scenario_seed),
    )
