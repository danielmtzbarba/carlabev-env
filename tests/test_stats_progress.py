import math

from CarlaBEV.src.deeprl.stats import EpisodeStats


def test_episode_stats_derives_progress_from_scene_when_reward_omits_it():
    episode = EpisodeStats()

    info = {
        "reward": {"reward": 1.0, "cause": None},
        "hero": {"state_m": [0.0, 0.0, 0.0, 8.0], "speed_mps": 8.0},
        "scene": {"route_length": 100.0, "dist2goal_m": 25.0},
    }

    episode.step(info)

    assert math.isclose(episode.final_progress, 0.75, rel_tol=1e-9)


def test_episode_stats_prefers_explicit_reward_progress():
    episode = EpisodeStats()

    info = {
        "reward": {"reward": 1.0, "cause": None, "progress": 0.2},
        "hero": {"state_m": [0.0, 0.0, 0.0, 8.0], "speed_mps": 8.0},
        "scene": {"route_length": 100.0, "dist2goal_m": 25.0},
    }

    episode.step(info)

    assert math.isclose(episode.final_progress, 0.2, rel_tol=1e-9)


def test_episode_stats_ignores_progress_when_route_length_is_missing():
    episode = EpisodeStats()

    info = {
        "reward": {"reward": 1.0, "cause": None},
        "hero": {"state_m": [0.0, 0.0, 0.0, 8.0], "speed_mps": 8.0},
        "scene": {"route_length": 0.0, "dist2goal_m": 25.0},
    }

    episode.step(info)

    assert episode.progress == []
