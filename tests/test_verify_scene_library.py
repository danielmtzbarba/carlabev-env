"""``carlabev scene verify-library`` and the Stage 5 generation bookkeeping changes."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from CarlaBEV.cli.main import main as cli_main  # noqa: E402
from CarlaBEV.src.actors.actor import Node  # noqa: E402
from CarlaBEV.src.actors.vehicle import Vehicle  # noqa: E402
from CarlaBEV.src.managers.scene_generator import PlannerManager  # noqa: E402
from CarlaBEV.src.units import DEFAULT_UNIT_CONTRACT  # noqa: E402
from CarlaBEV.tools.build_scene_library import build_scene_library  # noqa: E402
from CarlaBEV.tools.verify_scene_library import verify_scene_library  # noqa: E402


def _build(db_path: str, episodes: int = 2) -> None:
    build_scene_library(
        episodes=episodes,
        difficulty_preset_id="easy",
        num_vehicles=3,
        route_dist_range=(30, 60),
        scene_library_path=db_path,
    )


class VerifySceneLibraryTests(unittest.TestCase):
    def test_fresh_library_verifies_identically_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = f"{tmpdir}/lib.db"
            _build(db)
            report = verify_scene_library(db)
            self.assertTrue(report["passed"], report)
            self.assertEqual(report["counts"], {"identical": 2})
            self.assertEqual(report["checked"], 2)

            with sqlite3.connect(db) as conn:
                key = conn.execute("SELECT scene_key FROM scenes ORDER BY scene_id LIMIT 1").fetchone()[0]
                payload = json.loads(conn.execute("SELECT actors_json FROM scenes WHERE scene_key = ?", (key,)).fetchone()[0])
                payload["agent"]["target_speed_mps"] = 99.0
                conn.execute("UPDATE scenes SET actors_json = ? WHERE scene_key = ?", (json.dumps(payload), key))
            report = verify_scene_library(db)
            self.assertFalse(report["passed"])
            self.assertEqual(report["counts"], {"identical": 1, "mismatch": 1})
            mismatch = next(r for r in report["results"] if r["status"] == "mismatch")
            self.assertEqual(mismatch["scene_key"], key)
            self.assertIn("spec.agent.target_speed_mps", mismatch["detail"])

            with sqlite3.connect(db) as conn:
                conn.execute("UPDATE scenes SET actors_json = '{\"schema_version\": 1}' WHERE scene_key = ?", (key,))
            report = verify_scene_library(db)
            self.assertEqual(report["counts"], {"identical": 1, "unreadable": 1})

    def test_cli_sample_and_exit_codes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = f"{tmpdir}/lib.db"
            _build(db)
            report_path = f"{tmpdir}/verify.json"
            self.assertEqual(
                cli_main(["scene", "verify-library", "--scene-library-path", db, "--sample", "1", "--quiet", "--report", report_path]),
                0,
            )
            report = json.loads(open(report_path, encoding="utf-8").read())
            self.assertEqual(report["checked"], 1)
            with sqlite3.connect(db) as conn:
                conn.execute("UPDATE scenes SET actors_json = '{\"schema_version\": 1}'")
            self.assertEqual(cli_main(["scene", "verify-library", "--scene-library-path", db, "--quiet"]), 1)
        with self.assertRaises(SystemExit):
            cli_main(["scene", "verify-library", "--scene-library-path", "/nonexistent/lib.db"])


class GenerationBookkeepingTests(unittest.TestCase):
    def test_actor_path_nodes_are_built_lazily_and_settable(self):
        vehicle = Vehicle(map_size=128, routeX=[], routeY=[], target_speed=5.0)
        vehicle.set_route_wp("n1", 10, 20)
        vehicle.set_route_wp("n2", 30, 40)
        self.assertIsNone(vehicle._path_nodes)  # nothing materialised yet
        path = vehicle.path
        self.assertEqual([n.id for n in path], ["n1", "n2"])
        self.assertEqual(path[1].scaled_pos, [30, 40])
        self.assertIs(vehicle.path, path)  # cached
        vehicle.path = []
        self.assertEqual(vehicle.path, [])
        vehicle.path = [Node("z", (1.0, 2.0))]
        self.assertEqual([(n.id, n.scaled_pos) for n in vehicle.path], [("z", [1, 2])])
        self.assertEqual(vehicle._path_waypoints, [("z", 1, 2)])

    def test_shortest_paths_are_memoised_and_returned_as_copies(self):
        planner = PlannerManager("Town01", metric=DEFAULT_UNIT_CONTRACT.metric).get("vehicle-canonical")
        nodes = list(planner.nodes["vehicle"])
        start, end = nodes[0], nodes[len(nodes) // 2]
        first = planner.shortest_path_nodes(start, end)
        self.assertEqual(list(planner._shortest_path_cache), [(start, end)])
        second = planner.shortest_path_nodes(start, end)
        self.assertEqual(first, second)
        self.assertIsNot(first, second)
        second.append("mutated")
        self.assertEqual(planner.shortest_path_nodes(start, end), first)
        # Node positions are cached read-only arrays with identical values.
        pos = planner.get_node_pos_surface(start)
        self.assertIs(pos, planner.get_node_pos_surface(start))
        self.assertFalse(pos.flags.writeable)
        self.assertTrue(np.array_equal(pos, DEFAULT_UNIT_CONTRACT.metric.raw_to_surface(planner.get_node_pos(start))))


if __name__ == "__main__":
    unittest.main()
