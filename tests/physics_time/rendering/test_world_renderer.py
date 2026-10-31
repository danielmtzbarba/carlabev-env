"""RENDERING / layered world renderer (Phase 3): class-id raster, dirty rects, decoupled domain.

Expected: the world is an 8-bit class-id surface (static map layer + dirty-rect
dynamic layer) that equals a full redraw every frame; the observation RGB is the
palette lookup of the class ids; the reset observation already contains actors
and targets; traffic-light strips are rendered at their true position; the ego
overlay uses the EGO class; actors/control never import pygame at module level.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from CarlaBEV.config import EnvConfig
from CarlaBEV.envs.carlabev import CarlaBEV
from CarlaBEV.envs.world_renderer import WorldRenderer, surface_ids
from CarlaBEV.semantics import (
    PALETTE_LUT,
    VOID_ID,
    SemanticClass,
    semantic_class_from_rgb,
)
from CarlaBEV.src.diagnostics.rollouts import JAYWALK_OPTIONS, RANDOM_NAV_OPTIONS

from physics_time_helpers import make_env

REPO = Path(__file__).resolve().parents[3]
NO_PYGAME_ROOTS = (
    "CarlaBEV/src/actors",
    "CarlaBEV/src/control",
    "CarlaBEV/src/geometry",
    "CarlaBEV/src/scenes/target.py",
)


def _full_redraw_ids(env) -> np.ndarray:
    """Reference: static layer + every drawable, drawn on a fresh renderer."""
    wr = env.map.world_renderer
    ref = WorldRenderer(
        env.map.static_ids, padding=wr.padding, metric=wr.metric, frame=wr.frame
    )
    for drawable in env.map.scene.drawables():
        ref.draw(drawable)
    return ref.ids()


def test_dirty_rect_world_equals_full_redraw_every_frame():
    env = make_env(geometry_profile="realistic")
    try:
        env.reset(seed=3, options={**RANDOM_NAV_OPTIONS, "num_vehicles": 6})
        assert np.array_equal(env.map.world_renderer.ids(), _full_redraw_ids(env))
        for k in range(25):
            env.step(
                np.array([0.6, 0.3 if k % 10 < 5 else -0.3, 0.0], dtype=np.float32)
            )
            # Targets collected by the collision check after drawing disappear on the
            # next frame (as before); redraw at the current state and compare.
            env.map.draw_world()
            assert np.array_equal(
                env.map.world_renderer.ids(), _full_redraw_ids(env)
            ), f"step {k}"
        assert (
            0 < env.map.world_renderer.last_frame_rects < 60
        )  # a handful of rects, not the whole map
    finally:
        env.close()


def test_observation_rgb_is_palette_lookup_of_class_ids():
    env = make_env()
    try:
        obs, _ = env.reset(seed=3, options=RANDOM_NAV_OPTIONS)
        ids = env.semantic_ids
        assert ids.shape == (128, 128) and ids.dtype == np.uint8
        assert np.array_equal(PALETTE_LUT[ids], obs)
        # every non-void id round-trips through the colour palette
        for class_id in np.unique(ids):
            if class_id != VOID_ID:
                assert semantic_class_from_rgb(PALETTE_LUT[class_id]) == SemanticClass(
                    int(class_id)
                )
        assert set(np.unique(ids)) >= {
            int(SemanticClass.DRIVABLE),
            int(SemanticClass.EGO),
        }
    finally:
        env.close()


def test_bev_class_observation_mode():
    env = CarlaBEV(
        EnvConfig(
            render_mode="rgb_array",
            obs_mode="bev_class",
            action_mode="continuous",
            scene_library_enabled=False,
        )
    )
    try:
        obs, _ = env.reset(seed=3, options=RANDOM_NAV_OPTIONS)
        assert (
            env.observation_space.contains(obs)
            and obs.shape == (128, 128)
            and obs.dtype == np.uint8
        )
        assert np.array_equal(obs, env.semantic_ids)
    finally:
        env.close()


def test_reset_observation_contains_actors_and_route_targets():
    """R-04: the pre-Phase-3 reset frame showed the bare map (actors were drawn only in step())."""
    env = make_env()
    try:
        env.reset(seed=3, options={**RANDOM_NAV_OPTIONS, "num_vehicles": 6})
        world = env.map.world_renderer.ids()
        assert (world == int(SemanticClass.VEHICLE)).any()
        assert (world == int(SemanticClass.ROUTE)).any()
        ids = env.semantic_ids
        assert ids[
            env.map.fov_renderer.anchor_px[1], env.map.fov_renderer.anchor_px[0]
        ] == int(SemanticClass.EGO)
    finally:
        env.close()


def test_traffic_light_strips_are_rendered_at_their_surface_position():
    """R-05: strips used to be drawn without the render-frame padding (misplaced by the crop size)."""
    env = make_env()
    try:
        env.reset(seed=3, options={"scene": "red_light_runner", "level": 1})
        world = env.map.world_renderer.ids()
        frame = env.map.render_frame
        for light in env.map.actor_manager.actors["traffic_light"]:
            left, top, w, h = light.rect_px()
            origin = frame.world_to_surface((left, top))
            patch = world[
                int(origin.y) : int(origin.y + h) + 1,
                int(origin.x) : int(origin.x + w) + 1,
            ]
            assert (patch == int(light.semantic_class)).any(), light.signal_state
        classes = {
            int(light.semantic_class)
            for light in env.map.actor_manager.actors["traffic_light"]
        }
        assert classes == {
            int(SemanticClass.TRAFFIC_LIGHT_RED),
            int(SemanticClass.TRAFFIC_LIGHT_GREEN),
        }
    finally:
        env.close()


def test_masked_fov_corners_are_void():
    env = make_env(fov_masked=True)
    try:
        env.reset(seed=3, options=JAYWALK_OPTIONS)
        ids = env.semantic_ids
        assert (
            ids[0, 0] == VOID_ID
            and ids[0, -1] == VOID_ID
            and ids[-1, 0] == VOID_ID
            and ids[-1, -1] == VOID_ID
        )
        assert (PALETTE_LUT[VOID_ID] == 0).all()
    finally:
        env.close()


def test_world_surface_ids_match_static_map_outside_actors():
    env = make_env()
    try:
        env.reset(seed=3, options=JAYWALK_OPTIONS)
        wr = env.map.world_renderer
        world, static = wr.ids(), surface_ids(wr.static)
        dynamic = world != static
        assert set(np.unique(world[dynamic])) <= {int(c) for c in SemanticClass}
        assert dynamic.sum() < 0.01 * dynamic.size
    finally:
        env.close()


def test_domain_modules_do_not_import_pygame():
    offenders = []
    for root in NO_PYGAME_ROOTS:
        path = REPO / root
        files = [path] if path.is_file() else list(path.rglob("*.py"))
        for file in files:
            tree = ast.parse(file.read_text(encoding="utf-8"))
            for node in (
                tree.body
            ):  # module level only; lazy GUI imports inside functions are allowed
                if isinstance(node, ast.Import) and any(
                    a.name.split(".")[0] == "pygame" for a in node.names
                ):
                    offenders.append(str(file.relative_to(REPO)))
                if (
                    isinstance(node, ast.ImportFrom)
                    and (node.module or "").split(".")[0] == "pygame"
                ):
                    offenders.append(str(file.relative_to(REPO)))
    assert not offenders, offenders
