# Stage 1: Simulator Semantics Hardening

This document defines the proposed Stage 1 fixes for CarlaBEV before we refactor the simulator core. The goal is to make the simulator scientifically legible, numerically consistent, and testable enough for later publication-grade experiments.

## Scope

Stage 1 covers four foundations:

1. Ego and non-ego kinematics
2. Coordinate conventions
3. Units and scaling
4. Reward semantics

The intent is not to make the simulator fully dynamic or high-fidelity yet. The intent is to eliminate silent inconsistencies that would otherwise invalidate downstream reinforcement-learning claims.

## Current Diagnosis

The current codebase has a good modular decomposition, but several semantics are mixed together:

- World motion, path-tracking, and screen rendering are partially coupled.
- Pixels are often used as if they were meters.
- Speed is sometimes treated as a simulator unit, sometimes as km/h, and sometimes as if it were m/s.
- Reward logic mixes route progress, semantic tiles, and actor collision heuristics without a fully explicit contract.

This is acceptable for early prototyping, but it is not strong enough for a production-level simulator or a paper claiming robust autonomous-driving evaluation.

## Proposed Canonical State

All simulator logic should use a canonical world state in SI units:

- Position: `(x_w, y_w)` in meters
- Heading: `psi` in radians
- Speed: `v` in meters per second
- Longitudinal acceleration: `a` in meters per second squared
- Steering angle: `delta` in radians
- Time step: `dt` in seconds

Rendering should become a pure projection layer:

- Pixels are only for visualization
- Semantic rasters are generated from world geometry and actor states
- Reward uses world quantities, not rendered colors

## Coordinate Convention

Use one world frame and one renderer frame.

### World Frame

- Right-handed planar vehicle/world frame
- `+x_w`: east / right
- `+y_w`: north / up
- `psi = 0`: facing `+x_w`
- Positive yaw is counter-clockwise

### Screen Frame

- `+x_s`: right
- `+y_s`: down

Projection should be explicit:

`x_s = x_origin + s * x_w`

`y_s = y_origin - s * y_w`

where `s` is pixels-per-meter.

This prevents the current ambiguity where route generation, control, and rendering all partially inherit image-space conventions.

## Ego and Actor Kinematics

Stage 1 should use a single kinematic bicycle model for the ego and a compatible reduced model for scripted actors. This is a standard approximation for path tracking and low-to-moderate-speed autonomy studies [1].

### Ego Kinematics

For wheelbase `L`, the discrete-time update should be:

`x_{k+1} = x_k + v_k cos(psi_k) dt`

`y_{k+1} = y_k + v_k sin(psi_k) dt`

`psi_{k+1} = psi_k + (v_k / L) tan(delta_k) dt`

`v_{k+1} = clip(v_k + a_k dt, v_min, v_max)`

Important implementation notes:

- Do not add steering angle directly to yaw.
- Heading update must come from the bicycle relation, not from a rendering shortcut.
- Steering and acceleration limits should be explicit parameters.
- Apply the same integration order consistently for ego and scripted vehicles.

### Stanley Tracking Law

If Stanley remains the path-tracking controller, the steering command should follow the standard form [2]:

`delta = theta_e + arctan(k e_f / v)`

where:

- `theta_e` is heading error
- `e_f` is signed cross-track error at the front axle
- `k` is the controller gain

This controller assumes a smooth path and a consistent vehicle frame. If the route is in image coordinates while yaw is interpreted as a vehicle-frame quantity, the controller becomes hard to tune and hard to trust.

### Scripted Actor Kinematics

Background vehicles and pedestrians should not bypass the semantics used by the ego.

Vehicles:

- Same state variables as ego, possibly with simpler controls
- Longitudinal behavior may come from a target-speed policy or event script
- Steering should be derived from the route geometry or the same tracker family

Pedestrians:

- Position and heading in world coordinates
- Constant-velocity or finite-state behavior is acceptable in Stage 1
- Use meters per second, not pixel-per-step rules

Traffic-light or event scripts should change policy targets, not mutate physical state directly.

## Units and Scaling

The simulator needs a strict split between geometry scale and rendering scale.

### Canonical Conversions

- `meters_per_pixel = 1 / pixels_per_meter`
- Route length for reward and evaluation: meters
- Speed limit: meters per second internally
- Logging may additionally export km/h for readability

Recommended helpers:

- `world_to_screen()`
- `screen_to_world()`
- `meters_to_pixels()`
- `pixels_to_meters()`

These should live in one module and be the only allowed conversion path.

### Unit Rules

- Dynamics: SI units only
- Reward: SI units only
- Rendering: pixels only
- Dataset/scenario specs: meters for positions and lengths, seconds for event timing

If legacy assets are pixel-anchored, convert them once at load time and keep the runtime state in SI units.

## Reward Semantics

Reward should describe the task, not compensate for simulator inconsistencies.

### Hard Events

Termination events should be semantic and explicit:

- Collision with dynamic actor
- Collision with non-drivable geometry
- Goal reached
- Max horizon
- Invalid state if the simulator itself becomes numerically inconsistent

### Shaping Signals

Non-terminal shaping should be defined in world units:

- Route progress `Delta s` in meters along the reference path
- Lateral error `e_lat` in meters
- Heading error `e_psi` in radians
- Speed error relative to a target or speed limit in meters per second
- TTC in seconds
- Control smoothness from `a`, steering rate, and jerk

One reasonable Stage 1 reward family is:

`r_t = w_p Delta s - w_lat |e_lat| - w_psi |e_psi| - w_v max(0, v - v_lim) - w_c C_ttc - w_u C_smooth`

with terminal overrides:

- success: large positive terminal bonus
- collision: large negative terminal penalty

This is not the only valid design, but it has two advantages:

- every term has a unit and an interpretation
- every term can be validated independently

### TTC Semantics

Time-to-collision is defined as the time until collision under continued motion assumptions [3]. For Stage 1:

- TTC should be computed from relative world motion
- TTC is only valid when actors are on a collision course under the chosen approximation
- If no collision is predicted, treat TTC as `+inf`
- Penalties should be monotone as TTC decreases

Do not infer safety purely from raster overlap or center-to-center pixel distance.

## Semantic Observation Contract

The rendered RGB frame may remain available, but the authoritative semantic representation should be generated from known geometry/state, not reverse-engineered from the RGB surface.

Recommended semantic layers:

- drivable area
- non-drivable area
- lane boundaries / markings
- ego footprint
- vehicles
- pedestrians
- traffic controls
- route / goal

This avoids color-fragility and makes reward and observation semantics align.

## Stage 1 Validity Checks

The simulator should fail validation if any of the following happen:

1. Yaw update is inconsistent with the bicycle model.
2. Straight-line motion is inconsistent with heading.
3. Speed or route units are mixed.
4. Reward speed penalties are non-monotone or use inverted conversions.
5. Observation-space declarations disagree with returned tensors.
6. Coordinate transforms are not invertible up to tolerance.
7. Scenario sampling can fail silently.

## Immediate Refactor Targets

These are the concrete code-level changes Stage 1 should drive:

1. Replace direct-yaw updates in the vehicle state integrator with the documented bicycle-model heading update.
2. Introduce an explicit world-to-screen transform layer.
3. Convert scenario geometry and route lengths to meters at load time.
4. Standardize speed, acceleration, and speed-limit units to SI.
5. Rework reward computation to depend on world semantics rather than tile colors where possible.
6. Add deterministic, executable validity checks to gate future changes.

## Implemented So Far

The current codebase now includes the following Stage 1 foundations:

- Ego yaw integration follows the kinematic bicycle update.
- Camera following and render rectangles go through an explicit transform layer instead of using draw rectangles as state.
- A geometry module defines explicit conversions among:
  - raw asset coordinates
  - surface/world coordinates used by the current simulator runtime
  - metric coordinates in meters
- Map-graph accessors now expose node positions in surface and metric frames.
- Scenario route lengths are now computed in meters.
- Scenario parameters for gaps and offsets have begun migrating from pixel semantics to meter semantics.
- A validity-check script now tests both simulator semantics and geometry roundtrips.

This is still an intermediate state. The simulator runtime is structurally separated from screen space, and geometry conversions are explicit, but not every scenario or reward path is fully metric-native yet.

## References

[1] J. Snider, "Automatic Steering Methods for Autonomous Automobile Path Tracking," Carnegie Mellon University Robotics Institute, 2009.  
URL: https://www.ri.cmu.edu/pub_files/2009/2/Automatic_Steering_Methods_for_Autonomous_Automobile_Path_Tracking.pdf

[2] S. Thrun et al., "Stanley: The Robot That Won the DARPA Grand Challenge," Journal of Field Robotics, 2006.  
URL: https://robots.stanford.edu/papers/thrun.stanley05.pdf

[3] C. Schwarz, "On Computing Time-to-Collision for Automation Scenarios," Transportation Research Part F, 2014.  
URL: https://doi.org/10.1016/j.trf.2014.06.015
