# Documentation

Start with the repository [README](../../README.md) for installation and a
minimal DRL loop. These guides describe the current implementation.

## Using CarlaBEV

- [Public integration API](downstream_integration_api.md): construction, observations, resets, telemetry and corpus compatibility.
- [Configuration reference](config_reference.md): accepted settings and defaults.
- [Scenario specifications](scenario_specifications.md): random, predefined and authored scenes.
- [Diagnostics and validation](debugging_and_diagnostics.md): CLI tools, replay checks and scene libraries.

## Simulator contracts

- [Simulation clock and units](simulation_clock_and_units.md), [multi-rate execution](multi_rate_architecture.md).
- [Ego kinematics](ego_kinematics.md), [control and actions](control_and_actions.md).
- [Geometry and metric frame](geometry_and_metric_frame.md), [coordinate conventions](coordinate_conventions.md).
- [Actor geometry and collision](actor_geometry.md).
- [Traffic rules and actor behaviors](traffic_rules_and_behaviors.md).
- [Route generation and planning](route_generation_and_planning.md).
- [Rendering architecture](rendering_architecture.md).
- [Numerical determinism](numerical_determinism.md), [benchmarking](benchmarking.md).

## Historical material

Completed plans, audit ledgers and their diagrams are retained under
[history](history/README.md). They explain earlier decisions and measurements;
use the current guides above for defaults, import paths and runtime behavior.
