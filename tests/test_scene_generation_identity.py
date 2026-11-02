"""Cache identity must cover the inputs that determine a frozen scene."""

import os
import sqlite3
import json
import hashlib
import shutil

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pytest

from CarlaBEV.config import (
    EnvConfig,
    RandomNavigationReset,
    build_random_navigation_options,
)
from CarlaBEV.envs.carlabev import CarlaBEV
from CarlaBEV.src.managers.scene_library import SceneLibraryMissError
from CarlaBEV.src.scenes.scene_identity import (
    SceneGenerationConfig,
    SceneGenerationIdentity,
)
from CarlaBEV.tools.verify_scene_library import verify_scene_library


def options():
    return build_random_navigation_options(
        RandomNavigationReset(
            difficulty_preset_id="hard",
            scene_seed=1,
            num_vehicles=12,
            route_dist_range=(30, 60),
        )
    )


def make_env(**overrides):
    return CarlaBEV(
        EnvConfig(
            **{
                "render_mode": "rgb_array",
                "obs_mode": "bev_rgb",
                "scene_library_enabled": False,
                **overrides,
            }
        )
    )


def test_geometry_profiles_cannot_alias_different_generated_routes():
    legacy, realistic = make_env(geometry_profile="legacy"), make_env()
    try:
        _, a = legacy.reset(seed=1, options=options())
        _, b = realistic.reset(seed=1, options=options())
        assert [n.route_x for n in legacy.scene_spec.npcs] != [
            n.route_x for n in realistic.scene_spec.npcs
        ]
        assert a["scene_descriptor"]["scene_key"] != b["scene_descriptor"]["scene_key"]
    finally:
        legacy.close()
        realistic.close()


def test_strict_realistic_replay_rejects_a_legacy_geometry_corpus(tmp_path):
    db = str(tmp_path / "corpus.db")
    writer = make_env(
        scene_library_enabled=True, scene_library_path=db, geometry_profile="legacy"
    )
    try:
        writer.reset(seed=1, options=options())
    finally:
        writer.close()
    before = (tmp_path / "corpus.db").read_bytes()
    reader = make_env(
        scene_library_enabled=True,
        scene_library_path=db,
        scene_library_read_only=True,
        scene_library_require_hit=True,
    )
    try:
        with pytest.raises(SceneLibraryMissError):
            reader.reset(seed=1, options=options())
    finally:
        reader.close()
    assert (tmp_path / "corpus.db").read_bytes() == before


@pytest.mark.parametrize(
    "changed",
    [
        {"traffic_enabled": False},
        {"map_collision_mode": "center"},
        {"map_contact_inset_m": 0.1},
        {"npc_route_geometry": "legacy"},
        {"npc_fillet_target_radius_m": 7.0},
        {"physics_dt": 0.05},
    ],
)
def test_generation_and_spawn_settings_change_identity(changed):
    a, b = make_env(), make_env(**changed)
    try:
        assert a._scene_identity_key(options()) != b._scene_identity_key(options())
    finally:
        a.close()
        b.close()


def test_presentation_and_variation_preserve_cache_hits(tmp_path):
    db = str(tmp_path / "corpus.db")
    writer = make_env(scene_library_enabled=True, scene_library_path=db)
    try:
        _, original = writer.reset(seed=1, options=options())
        frozen = writer.scene_spec.to_payload()
    finally:
        writer.close()
    reader = make_env(
        scene_library_enabled=True,
        scene_library_path=db,
        scene_library_read_only=True,
        scene_library_require_hit=True,
        fps=15,
        obs_size=(64, 64),
        ego_anchor_y_frac=0.2,
    )
    try:
        _, replay = reader.reset(
            seed=1,
            options={
                **options(),
                "variation_seed": 42,
                "variation_profile": "spawn_v1",
            },
        )
        assert replay["scene_cache"]["hit"]
        assert (
            original["scene_descriptor"]["scene_key"]
            == replay["scene_descriptor"]["scene_key"]
        )
        assert reader.scene_spec.to_payload() == frozen
    finally:
        reader.close()


@pytest.mark.parametrize(
    "mutation", ["missing", "missing_version", "old_version", "wrong_config"]
)
def test_required_hit_checks_stored_identity_even_under_a_matching_key(
    tmp_path, mutation
):
    db = str(tmp_path / "corpus.db")
    writer = make_env(scene_library_enabled=True, scene_library_path=db)
    try:
        writer.reset(seed=1, options=options())
    finally:
        writer.close()
    with sqlite3.connect(db) as conn:
        payload = json.loads(
            conn.execute("SELECT actors_json FROM scenes").fetchone()[0]
        )
        if mutation == "missing":
            del payload["identity"]["generation"]
        elif mutation == "missing_version":
            del payload["identity"]["generation"]["identity_version"]
        elif mutation == "old_version":
            payload["identity"]["generation"]["identity_version"] = 1
        else:
            payload["identity"]["generation"]["config"]["environment"][
                "geometry_profile"
            ] = "legacy"
        conn.execute("UPDATE scenes SET actors_json = ?", (json.dumps(payload),))
    before = (tmp_path / "corpus.db").read_bytes()
    reader = make_env(
        scene_library_enabled=True,
        scene_library_path=db,
        scene_library_read_only=True,
        scene_library_require_hit=True,
    )
    try:
        with pytest.raises(SceneLibraryMissError, match="not replayable"):
            reader.reset(seed=1, options=options())
    finally:
        reader.close()
    assert (tmp_path / "corpus.db").read_bytes() == before


def test_map_asset_contents_participate_in_identity(tmp_path):
    from CarlaBEV.envs.utils import asset_path

    env = make_env()
    try:
        config = env._generation_config
        for name in config.asset_sha256:
            target = tmp_path / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(os.path.join(asset_path, name), target)
        same = SceneGenerationConfig.from_environment(
            env.cfg, env.units, env.scene_generator, assets_root=tmp_path
        )
        assert same == config  # absolute checkout location is not identity
        name = "Town01/town01-canonical-vehicle.pkl"
        with (tmp_path / name).open("ab") as output:
            output.write(b"changed asset")
        changed = SceneGenerationConfig.from_environment(
            env.cfg, env.units, env.scene_generator, assets_root=tmp_path
        )
        original = env._scene_generation_identity(options())
        altered = SceneGenerationIdentity.from_request(
            generator_version=original.generator_version,
            config=changed,
            options=original.options,
            seeds=original.seeds,
        )
        assert altered.key() != original.key()
    finally:
        env.close()


def test_old_key_namespace_is_not_reused(tmp_path):
    env = make_env(
        scene_library_enabled=True, scene_library_path=str(tmp_path / "corpus.db")
    )
    try:
        env.reset(seed=1, options=options())
        identity = SceneGenerationIdentity.model_validate(
            env.scene_spec.identity["generation"]
        )
        old_payload = {
            "generator_version": identity.generator_version,
            "surface_size": env.size,
            "options": identity.options,
            "seeds": identity.seeds,
        }
        old_key = hashlib.sha256(
            json.dumps(old_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        assert old_key != identity.key()
        with sqlite3.connect(tmp_path / "corpus.db") as conn:
            conn.execute("UPDATE scenes SET scene_key = ?", (old_key,))
        _, info = env.reset(seed=1, options=options())
        assert not info["scene_cache"]["hit"]
    finally:
        env.close()


def test_verifier_reconstructs_nondefault_generation_settings(tmp_path):
    db = str(tmp_path / "corpus.db")
    env = make_env(
        scene_library_enabled=True,
        scene_library_path=db,
        map_collision_mode="center",
        npc_route_geometry="legacy",
        traffic_enabled=False,
    )
    try:
        env.reset(seed=1, options=options())
    finally:
        env.close()
    report = verify_scene_library(db)
    assert report["passed"], report
