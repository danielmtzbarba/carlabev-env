<div align="center">

# CarlaBEV

**A bird's-eye-view simulator for vision-based autonomous-driving reinforcement learning**

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Gymnasium](https://img.shields.io/badge/gymnasium-API-brightgreen)](https://gymnasium.farama.org/)
[![PyGame](https://img.shields.io/badge/pygame-2.1.3-yellow)](https://www.pygame.org/)

</div>

## Overview

`CarlaBEV` is a lightweight BEV simulator for autonomous-driving research. It is designed for rapid iteration on:

- vision-based DRL training
- route-following and avoidance behavior
- procedurally generated traffic scenes
- edge-case scenario design and evaluation

The simulator uses a Gymnasium-compatible API, a 2D raster BEV pipeline, explicit route and actor management, and a growing set of scenario generators for structured safety-critical evaluation.

It is not a full dynamic clone of CARLA. The project is best understood as a fast research simulator for BEV decision-making and control experiments.

<div align="center">
  <img src="CarlaBEV/assets/images/carlabev_hero_banner.png" alt="CarlaBEV Hero Banner" width="800">
</div>

## Current Status

The project currently supports:

- Gymnasium vectorized environment usage
- RGB or semantic-mask BEV observations
- discrete or continuous ego actions
- route-following ego dynamics with a kinematic bicycle update
- explicit camera/FOV centering on the true ego start state
- synchronized rendering, collision, semantic tile lookup, and checkpoint detection
- random scene generation
- predefined edge-case scenarios including:
  - `lead_brake`
  - `jaywalk`
  - `red_light_runner`

Recent simulator hardening work also introduced:

- explicit geometry conversion utilities across raw asset, runtime, and metric frames
- reset-time spawn validation for valid ego initialization
- stronger steering authority for meaningful turns and avoidance
- executable simulator validity checks

## Key Features

- **Gymnasium-compatible API** for training with standard RL tooling.
- **BEV observation pipeline** with optional semantic masking and frame stacking.
- **Scenario generation** for both random traffic scenes and structured edge cases.
- **Configurable ego control** with discrete and continuous actions.
- **Semantic map awareness** for drivable-space checks, collision logic, and route progress.
- **Debug tooling** for manual inspection of environment behavior.
- **Technical documentation** covering kinematics, geometry, control, and coordinate conventions.

## Documentation

Project documentation lives in `CarlaBEV/docs/`:

- [Ego Kinematics](CarlaBEV/docs/ego_kinematics.md)
- [Geometry And Metric Frame](CarlaBEV/docs/geometry_and_metric_frame.md)
- [Control And Actions](CarlaBEV/docs/control_and_actions.md)
- [Coordinate Conventions](CarlaBEV/docs/coordinate_conventions.md)
- [Scenario Specifications](CarlaBEV/docs/scenario_specifications.md)

These docs describe the simulator as it exists in the repository today.

## Repository Layout

`CarlaBEV/` is organized into a few main layers:

- `CarlaBEV/envs/`
  Core environment, rendering, camera, transforms, spaces, and wrappers entrypoint.

- `CarlaBEV/src/actors/`
  Ego, vehicle, pedestrian, traffic-light, and actor behavior logic.

- `CarlaBEV/src/control/`
  Vehicle state integration, Stanley tracking, and control utilities.

- `CarlaBEV/src/scenes/`
  Scene orchestration, targets, utilities, and predefined scenarios.

- `CarlaBEV/src/managers/`
  Actor management, scene generation, and serialization.

- `CarlaBEV/src/deeprl/`
  Reward functions, reward signals, stats, and logging helpers.

- `CarlaBEV/tools/`
  Debug scripts, scene tools, and validation utilities.

## Installation

Using [`uv`](https://github.com/astral-sh/uv) is recommended.

```bash
git clone https://github.com/danielmtzbarba/carlabev-env.git
cd carlabev-env
uv sync
```

## Running The Simulator

### Debug Viewer

The fastest way to inspect the simulator manually is:

```bash
uv run CarlaBEV/tools/debug_env.py
```

This launches a `pygame` window and steps the environment with keyboard control.

### Validity Checks

The repository includes a simulator validation script:

```bash
uv run python CarlaBEV/tools/validate_simulator_semantics.py
```

This checks core simulator contracts such as:

- bicycle yaw update consistency
- geometry conversion consistency
- observation-shape correctness
- valid scenario spawns

## Configuration

Primary debug/training configuration lives in [cfg.py](CarlaBEV/tools/debug/cfg.py).

Important environment options include:

- `obs_space`: `bev` or `vector`
- `masked`: semantic-mask observations vs grayscale pipeline
- `action_space`: `discrete` or `continuous`
- `reward_type`: `carl` or default reward
- `size`: base BEV size
- `obs_size`: final resized observation size
- `traffic_enabled`
- `max_vehicles`

## Example Reset

```python
import numpy as np

from CarlaBEV.envs import make_env
from CarlaBEV.tools.debug.cfg import ArgsCarlaBEV, EnvConfig, LoggerConfig

cfg = ArgsCarlaBEV(
    env=EnvConfig(render_mode="rgb_array"),
    logging=LoggerConfig(),
)

envs = make_env(cfg)

obs, info = envs.reset(
    options={
        "scene": "red_light_runner",
        "num_vehicles": 10,
        "route_dist_range": [30, 100],
        "reset_mask": np.array([True], dtype=bool),
    }
)
```

## Research Scope

CarlaBEV is a strong fit for:

- BEV policy learning
- scenario-driven DRL benchmarking
- failure analysis in edge-case traffic situations
- rapid experimentation on observation design and reward shaping

It is a weaker fit for:

- high-fidelity vehicle dynamics
- sensor-realism studies
- direct simulation-to-real transfer claims without additional validation

## Known Modeling Boundaries

- Ego dynamics use a simplified kinematic bicycle model.
- Collision footprints are coarse relative to real vehicle geometry.
- Some scenario anchors remain map-specific even though the geometry pipeline now exposes explicit metric conversions.
- The simulator is optimized for research iteration speed, not photorealism or detailed physics.
