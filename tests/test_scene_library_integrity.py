"""A required cache hit must contain the exact recorded NPC population."""

import json
import os
import sqlite3

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pytest

from CarlaBEV.config import (
    EnvConfig,
    RandomNavigationReset,
    build_random_navigation_options,
)
from CarlaBEV.envs.carlabev import CarlaBEV
from CarlaBEV.src.managers.scene_library import (
    SceneLibraryInvalidEntryError,
    SceneLibraryMissError,
)
from CarlaBEV.tools.verify_scene_library import verify_scene_library


def make_env(db, **overrides):
    return CarlaBEV(
        EnvConfig(
            **{
                "render_mode": "rgb_array",
                "obs_mode": "bev_rgb",
                "scene_library_path": str(db),
                **overrides,
            }
        )
    )


@pytest.fixture
def corpus(tmp_path):
    db = tmp_path / "corpus.db"
    options = build_random_navigation_options(
        RandomNavigationReset(
            difficulty_preset_id="easy",
            scene_seed=5,
            num_vehicles=4,
            route_dist_range=(30, 60),
        )
    )
    env = make_env(db)
    try:
        env.reset(seed=5, options=options)
        payload = env.scene_spec.to_payload()
        assert len(payload["npcs"]) == 4
    finally:
        env.close()
    return db, options, payload


CORRUPTIONS = (
    "missing_all",
    "missing_one",
    "foreign_scene",
    "extra_row",
    "missing_references",
    "missing_reference",
    "duplicate_references",
    "substituted_reference",
    "reordered_references",
    "reordered_ordinals",
    "duplicate_ordinal",
    "spec_uid",
    "row_id",
    "spec_kind",
    "malformed_spec",
    "malformed_scene",
    "inline_bypass",
    "invalid_route",
)


def corrupt(db, payload, mode):
    with sqlite3.connect(db) as conn:
        refs = [npc["npc_uid"] for npc in payload["npcs"]]
        stored = json.loads(
            conn.execute("SELECT actors_json FROM scenes").fetchone()[0]
        )
        if mode == "missing_all":
            conn.execute("DELETE FROM npcs")
        elif mode == "missing_one":
            conn.execute("DELETE FROM npcs WHERE npc_uid = ?", (refs[0],))
        elif mode == "foreign_scene":
            conn.execute(
                "UPDATE npcs SET scene_key = 'another-scene' WHERE npc_uid = ?",
                (refs[0],),
            )
        elif mode == "extra_row":
            conn.row_factory = sqlite3.Row
            row = dict(conn.execute("SELECT * FROM npcs LIMIT 1").fetchone())
            row.update(npc_uid="unexpected", ordinal=99)
            conn.execute(
                f"INSERT INTO npcs ({', '.join(row)}) VALUES ({', '.join('?' for _ in row)})",
                tuple(row.values()),
            )
        elif mode == "missing_references":
            del stored["npc_uids"]
        elif mode == "missing_reference":
            stored["npc_uids"].pop()
        elif mode == "duplicate_references":
            stored["npc_uids"][1] = refs[0]
        elif mode == "substituted_reference":
            stored["npc_uids"][0] = "another-npc"
        elif mode == "reordered_references":
            stored["npc_uids"].reverse()
        elif mode == "reordered_ordinals":
            conn.execute(
                "UPDATE npcs SET ordinal = CASE ordinal WHEN 0 THEN 1 WHEN 1 THEN 0 ELSE ordinal END"
            )
        elif mode == "duplicate_ordinal":
            conn.execute("UPDATE npcs SET ordinal = 1 WHERE ordinal = 0")
        elif mode == "row_id":
            conn.execute(
                "UPDATE npcs SET npc_id = 'vehicle:999' WHERE npc_uid = ?", (refs[0],)
            )
        elif mode == "malformed_spec":
            conn.execute(
                "UPDATE npcs SET spec_json = '{broken' WHERE npc_uid = ?", (refs[0],)
            )
        elif mode == "malformed_scene":
            conn.execute("UPDATE scenes SET actors_json = '{broken'")
            return
        elif mode == "inline_bypass":
            stored["npcs"] = payload["npcs"]
            conn.execute("DELETE FROM npcs")
        else:
            npc = dict(payload["npcs"][0])
            if mode == "spec_uid":
                npc["npc_uid"] = refs[1]
            elif mode == "spec_kind":
                npc["kind"] = "pedestrian"
            elif mode == "invalid_route":
                npc["route_y"] = []
            else:
                raise AssertionError(mode)
            conn.execute(
                "UPDATE npcs SET spec_json = ? WHERE npc_uid = ?",
                (json.dumps(npc), refs[0]),
            )
        conn.execute("UPDATE scenes SET actors_json = ?", (json.dumps(stored),))


@pytest.mark.parametrize("mode", CORRUPTIONS)
@pytest.mark.parametrize("read_only", [True, False])
def test_required_hit_rejects_corruption_without_writing_or_generating(
    corpus, monkeypatch, mode, read_only
):
    db, options, payload = corpus
    corrupt(db, payload, mode)
    before = db.read_bytes()
    env = make_env(
        db, scene_library_read_only=read_only, scene_library_require_hit=True
    )

    def forbidden(*args, **kwargs):
        pytest.fail("strict replay must not generate, evict or touch an invalid scene")

    monkeypatch.setattr(env.scene_generator, "build_scene", forbidden)
    monkeypatch.setattr(env.scene_library, "delete_scene", forbidden)
    monkeypatch.setattr(env.scene_library, "touch_scene", forbidden)
    try:
        with pytest.raises(SceneLibraryMissError, match="not replayable") as exc:
            env.reset(seed=5, options=options)
        assert isinstance(exc.value.__cause__, SceneLibraryInvalidEntryError)
        assert exc.value.scene_key == payload["scene_key"]
    finally:
        env.close()
    assert db.read_bytes() == before


@pytest.mark.parametrize("read_only", [True, False])
def test_optional_cache_regenerates_corruption_and_only_repairs_writable_corpora(
    corpus, read_only
):
    db, options, payload = corpus
    corrupt(db, payload, "missing_one")
    before = db.read_bytes()
    env = make_env(db, scene_library_read_only=read_only)
    try:
        _, info = env.reset(seed=5, options=options)
        assert not info["scene_cache"]["hit"]
        assert env.scene_spec.to_payload() == payload
        assert env.num_vehicles == 4
        if not read_only:
            _, replay = env.reset(seed=5, options=options)
            assert replay["scene_cache"]["hit"]
            assert env.num_vehicles == 4
    finally:
        env.close()
    if read_only:
        assert db.read_bytes() == before
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM scenes").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM npcs").fetchone()[0] == (
            3 if read_only else 4
        )


@pytest.mark.parametrize("mode", CORRUPTIONS)
def test_verifier_reports_corrupt_entries_without_mutating_the_corpus(corpus, mode):
    db, _, payload = corpus
    corrupt(db, payload, mode)
    before = db.read_bytes()
    report = verify_scene_library(str(db))
    assert not report["passed"]
    assert report["counts"] == {"unreadable": 1}
    assert report["results"][0]["scene_key"] == payload["scene_key"]
    assert db.read_bytes() == before


def test_reference_order_is_independent_of_physical_row_order(corpus):
    db, options, payload = corpus
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        rows = [
            dict(row)
            for row in conn.execute("SELECT * FROM npcs ORDER BY ordinal DESC")
        ]
        conn.execute("DELETE FROM npcs")
        columns = list(rows[0])
        conn.executemany(
            f"INSERT INTO npcs ({', '.join(columns)}) VALUES ({', '.join('?' for _ in columns)})",
            [tuple(row.values()) for row in rows],
        )
    before = db.read_bytes()
    env = make_env(db, scene_library_read_only=True, scene_library_require_hit=True)
    try:
        _, info = env.reset(seed=5, options=options)
        assert info["scene_cache"]["hit"]
        assert env.scene_spec.to_payload() == payload
    finally:
        env.close()
    assert db.read_bytes() == before


def test_diagnostic_scene_reads_also_reject_corrupt_references(corpus):
    db, _, payload = corpus
    corrupt(db, payload, "missing_one")
    env = make_env(db, scene_library_read_only=True)
    try:
        with pytest.raises(SceneLibraryInvalidEntryError):
            env.scene_library.get_scene_by_key(payload["scene_key"])
        with pytest.raises(SceneLibraryInvalidEntryError):
            env.scene_library.get_scene_by_filters()
    finally:
        env.close()


def test_duplicate_rows_in_a_damaged_schema_cannot_be_hidden_by_uid_lookup(corpus):
    db, options, _ = corpus
    with sqlite3.connect(db) as conn:
        # Simulate an externally rebuilt table that lost its primary key.
        conn.execute("CREATE TABLE damaged_npcs AS SELECT * FROM npcs")
        conn.execute("INSERT INTO damaged_npcs SELECT * FROM npcs LIMIT 1")
        conn.execute("DROP TABLE npcs")
        conn.execute("ALTER TABLE damaged_npcs RENAME TO npcs")
    before = db.read_bytes()
    env = make_env(db, scene_library_read_only=True, scene_library_require_hit=True)
    try:
        with pytest.raises(SceneLibraryMissError, match="expected 4, found 5"):
            env.reset(seed=5, options=options)
    finally:
        env.close()
    assert db.read_bytes() == before


def test_scene_and_npcs_are_read_from_the_same_snapshot(corpus, monkeypatch):
    db, _, payload = corpus
    with sqlite3.connect(db) as writer:
        writer.execute("PRAGMA journal_mode=WAL")
    env = make_env(db, scene_library_read_only=True)
    original = env.scene_library._row_to_record

    def concurrent_delete(row, *, conn):
        # A writer commits after the scene SELECT, before NPC reference resolution.
        with sqlite3.connect(db) as writer:
            writer.execute("DELETE FROM npcs")
        return original(row, conn=conn)

    monkeypatch.setattr(env.scene_library, "_row_to_record", concurrent_delete)
    try:
        record = env.scene_library.get_scene_by_key(payload["scene_key"])
        assert record["actors_payload"]["npcs"] == payload["npcs"]
        # The next read sees the committed deletion and must fail.
        monkeypatch.setattr(env.scene_library, "_row_to_record", original)
        with pytest.raises(SceneLibraryInvalidEntryError):
            env.scene_library.get_scene_by_key(payload["scene_key"])
    finally:
        env.close()


def test_explicit_empty_reference_list_is_a_valid_scene(tmp_path):
    db = tmp_path / "empty.db"
    options = build_random_navigation_options(
        RandomNavigationReset(
            difficulty_preset_id="no_traffic",
            scene_seed=5,
            num_vehicles=0,
        )
    )
    env = make_env(db)
    try:
        env.reset(seed=5, options=options)
        assert not env.scene_spec.npcs
    finally:
        env.close()
    reader = make_env(db, scene_library_read_only=True, scene_library_require_hit=True)
    try:
        _, info = reader.reset(seed=5, options=options)
        assert info["scene_cache"]["hit"]
        assert reader.num_vehicles == 0
    finally:
        reader.close()
