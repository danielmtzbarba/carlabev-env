# Traffic rules and actor behaviours (Phase 4)

Background traffic is driven by two layers, both pure functions of the
simulated state and the scene clock (no accumulated timers, no wall-clock,
no rendering dependency):

1. **Behaviours** — per-actor finite-state machines (`src/actors/behavior/framework.py`)
   that set the *nominal* target speed (cruise, timed brake, wait-then-cross …).
2. **Traffic rules** — a reactive longitudinal policy (`src/actors/traffic_policy.py`)
   that may only *reduce* the behaviour's acceleration: car following (IDM),
   stop lines, junction right-of-way.

The ego is never constrained by either layer; it is perceived by them (B-01) and
its signal compliance is only *observed* (`info["signals"]`).

## Behaviour framework

```
Behavior(states, initial)          State(name, on_enter, each_step, transitions, speed_mps)
  reset(actor)   → initial state, on_enter actions (defines spawn state, e.g. rest at the curb)
  apply(actor, t, dt, other_actors, ego, signals)
      each_step actions → evaluate transitions in order → first firing trigger enters the next
      state (one transition per step; on_enter actions run immediately)
```

| Triggers | Actions |
| --- | --- |
| `AtTimeS(s)` (since behaviour start), `AfterS(s)` (in state), `RouteIndexAtLeast(f)`, `RouteComplete`, `NearPoint`, `EgoWithinM(m)`, `GapAccepted(min_ttc_s)`, `AllOf`, `AnyOf`, `Always` | `SetTargetSpeed`, `RampSpeed(rate)`, `Stop`, `Retreat`, `SetInitialSpeed`, `SetFlag(name, value)` |

Timers use `time_reached()` / `timer_elapsed()` on the end-of-step time `t`
(C-03 convention): `AfterS(1.0)` fires in the step whose end time is 1.0 s, at
any `physics_dt`. Specs round-trip (`to_spec()` / `Behavior.from_spec()`), and
scenes may author behaviours inline:

```json
{"type": "fsm", "initial": "cruise",
 "states": {"cruise":  {"transitions": [{"trigger": {"type": "at_time_s", "seconds": 2.5}, "to": "braking"}]},
            "braking": {"each_step": [{"type": "ramp_speed", "rate_mps2": -4.0}]}}}
```

Built-ins (`registry.py`): `timed_brake` (`LeadBrakeBehavior`: cruise → braking),
`cross` / `stop_mid` / `yield_return` (jaywalk family: waiting → entering →
yielding|stalled → crossing|retreating → cleared|retreated). The pedestrian
behaviours accept `gap_accept_s`: with it the pedestrian leaves the curb only
when no vehicle (incl. the ego) heading towards it would arrive within that
TTC (`GapAccepted`); `0`/unset keeps the adversarial "jaywalk regardless".

`actor.behavior_state` exposes the state name (goldens: the lead-brake vehicle
now reports `cruise`/`braking` instead of the old placeholder `idle`).

## Traffic rules (`UnitContract.rules`, `EnvConfig`)

| Switch | Default | Effect |
| --- | --- | --- |
| `car_following` | `"idm"` | IDM on bumper-to-bumper gaps; `"legacy"` = old TTC lookup table (attribution only) |
| `traffic_signals_enabled` | `True` | vehicles stop at red/yellow stop lines controlling their heading |
| `junction_right_of_way_enabled` | `True` | vehicles yield at junction zones |
| `junction_radius_m` | `7.0` | zone radius around each junction centre |

Per-vehicle opt-outs for adversaries: `obeys_signals=False`,
`yields_at_junctions=False` (scene JSON fields of the same name; the
`red_light_runner` adversary sets both).

### Car following — IDM

`a = a_max [1 − (v/v0)^4 − (s*/s)^2]`, `s* = s0 + max(0, vT + v·Δv / (2√(a_max b)))`
(`src/actors/car_following.py`). `s` is the **shape gap** (`geometry.gap_between`
on the oriented bodies; legacy squares or realistic boxes/circles), `v0` the
behaviour's target speed, the lead the nearest actor ahead in the lane corridor
(20 m, ±2.25 m), including pedestrians and the ego. Parameters per generated
speed class:

| class | a_max | b | s0 | T |
| --- | --- | --- | --- | --- |
| slow | 1.5 | 2.0 | 2.5 m | 1.8 s |
| medium (default) | 2.0 | 2.5 | 2.0 m | 1.5 s |
| fast | 2.5 | 3.0 | 1.5 m | 1.2 s |

Wiring: the rule engine returns `min(a_behaviour, a_lead, a_signal, a_yield)`.
With no active constraint the actor uses the unchanged speed loop
`Kp (v_t − v) + a_ff` (free driving is bit-identical to the legacy path); with a
constraint the controller target is set to the current speed and the IDM
acceleration is commanded as feed-forward, clipped to the actor's limits
(2.5 / 3.5 m/s²) and floored so the speed never integrates below zero. Once
nearly at rest within `s0 + 1 m` of a stopped obstacle/stop point the actor
commands a full stop instead of IDM's asymptotic creep.

### Signals

`TrafficLight(controls_heading="north"|"south"|"east"|"west", cycle=SignalCycle)`.
A light controls the vehicles travelling in that direction (within 45°) whose
lateral position is within the strip span; the stop line is a virtual lead at
rest `stop_line_margin_m` (0.5 m) before the strip. Red → stop; yellow → stop
only if `gap ≥ v²/(2b)`; green → ignore. Lights without `controls_heading` are
decorative (legacy authored scenes). `SignalCycle(green_s, yellow_s, red_s,
offset_s)` derives the phase from `t` (`state_at(t)`), so two lights with
complementary offsets form a fixed-time plan; static lights keep their authored
state. Scenario JSON: `"controls_heading": "east"`, `"cycle": {"green_s": 10, "yellow_s": 3, "red_s": 13, "offset_s": 0}`.

### Junction right-of-way

`src/actors/junctions.py` builds circular zones from the canonical vehicle graph:
centre = crossing of the road centrelines (mean x of north–south lane nodes,
mean y of east–west nodes — the plain node centroid is biased ≈ 4 m), only
junctions with ≥ 3 arms (Town01: 12 T-junctions; bends/straights excluded),
cached per town. A vehicle approaching a zone (heading ray enters it within
`yield_lookahead_m` = 20 m) yields — zone edge as a virtual stop — to **crossing**
traffic (heading difference 45°–135°; incl. the ego) that is inside the zone
(unless already leaving it) or that will reach the zone earlier
(`distance_to_edge / v`, deterministic tie-break). Vehicles inside a zone never
yield (no deadlock inside), and after `yield_max_wait_s` = 6 s at the edge a
vehicle only yields to traffic already inside. Simplifications (documented, not
modelled): turns across oncoming traffic, lane-level conflict matrices.

### Diagnostics

`actor.last_traffic_policy` (also merged into `controller_info`): `constraint`
(`lead`/`signal`/`yield`/None), `lead_gap_m`, `lead_speed_mps`, `lead_ttc_s`,
`signal_state`, `signal_gap_m`, `junction_id`, `yield_gap_m`, `yield_to`,
`accel_ff_mps2`, `settle`. Ego: `info["signals"] = {state, gap_m,
red_light_violations}` and `episode_info["red_light_violations"]` (count of red
stop lines the ego crossed; never enforced).

## Executable checks

`tests/physics_time/behavior/test_behavior_framework.py` (8) and
`test_traffic_rules.py` (17): spec round-trip, one-transition-per-step, timers
at three `dt`, gap acceptance, IDM fixed points/monotonicity, following without
overlap (both geometry profiles), standstill gap, legacy attribution, free
driving bit-identical, signal cycle = f(t) at two `dt`, stop at red / runner,
proceed on green, yellow rule, decorative lights, junction map geometry,
yield-then-proceed, disabled right-of-way, bounded wait, ego violation count.
