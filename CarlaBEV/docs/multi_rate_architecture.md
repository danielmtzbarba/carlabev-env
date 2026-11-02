# Multi-Rate Execution

Physics, agent decisions, observations and display use separate rates with one
simulation clock. The original design and trade-offs are retained in
[history](history/multi_rate_architecture_design.md).

| Rate | Setting / owner | Current behavior |
| --- | --- | --- |
| Physics | `physics_dt`, `SimClock` | Ego and NPC integration, collision and reward evaluation per substep |
| Control | `action_repeat`, `CarlaBEV.step()` | One action held over up to `action_repeat` physics substeps |
| Observation | `BaseMap.render_world()`, environment wrappers | One boundary observation per decision; wrappers stack decision frames |
| Display | `fps` or `display_fps` | Wall-clock pacing or snapshot interpolation, independent of dynamics |

The default is `physics_dt=0.1`, `action_repeat=1`: one decision advances 0.1
simulated seconds unless an episode boundary is reached first. For example,
`physics_dt=0.05`, `action_repeat=2` keeps the same decision interval with finer
physics integration. It changes numerical sampling, so it does not promise
bit-identical trajectories or returns.

## One environment step

1. Decode the action once.
2. For each physics substep, advance `Scene` through `BaseMap.advance()`; update
   actors, traffic rules and lifecycle; compute collision information and reward.
3. Accumulate rewards and stop on the first termination or horizon truncation.
4. Render the final state once through `BaseMap.render_world()` and obtain the
   observation. Record actual simulated duration and substep counts.
5. In interpolated human mode, present captured `WorldSnapshot` poses through
   `Presenter`; this does not modify simulation state or observations.

`info["reward"]["control_reward"]` is the reward returned by `step()`;
`substep_rewards` records its components. Episode summaries distinguish physics
steps from decisions. Detailed clock, reward and horizon contracts are in
[Simulation Clock And Units](simulation_clock_and_units.md).

## Running

```bash
# Finer integration at the default 0.1-second decision interval
uv run carlabev run sim --physics-dt 0.05 --action-repeat 2

# Smooth real-time presentation without changing integration
uv run carlabev run sim --display-fps 60
```

`--fps` is a wall-clock cap, not a physics rate. It cannot be combined with
`--display-fps`. Observation videos use the control rate; interpolated display
frames are not a separate recorded video stream.

## Implementation and checks

- `src/sim_time.py`: `TimeConfig` and `SimClock`.
- `envs/carlabev.py`: substep loop, reward accumulation and episode boundary.
- `envs/world.py`: simulation advance separated from world rendering.
- `envs/presenter.py`: immutable drawable snapshots and interpolation.

```bash
uv run pytest tests/physics_time/clock/test_action_repeat.py
uv run pytest tests/physics_time/rendering/test_presenter.py
```
