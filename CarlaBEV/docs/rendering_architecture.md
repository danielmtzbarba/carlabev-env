# Rendering Architecture

Phase 3 of the actor/rendering/behaviour plan. Rendering is a **pure observer**
of the simulation: it reads actor bodies and poses, never mutates state, and
cannot influence a trajectory (enforced by `tests/physics_time/rendering`).

## Layers and data flow

```
static semantic raster (map PNG labels → class ids, uint8)      ─┐
                                                                  ├─► WorldRenderer (8-bit palettized surface of class ids)
Scene.drawables(): targets · traffic-light strips · actor bodies ─┘     static layer + dirty-rect dynamic layer
                                                                          │
                                                                          ▼ crop (subsurface) → rotate (nearest, VOID fill) → compose → mask → ego overlay
                                                                  FovRenderer  (8-bit class ids, preallocated output)
                                                                          │
                                              ┌───────────────────────────┼─────────────────────────────┐
                                              ▼                           ▼                             ▼
                                   observation_ids()              observation_rgb()                 human window
                                   (size, size) uint8             palette LUT (single gather)        blit (palette → display)
                                   obs_mode="bev_class"           obs_mode="bev_rgb"/"bev_semantic"
```

* **Class ids are the authority.** `BaseMap.semantic_class_at()` reads the
  static raster directly; `SemanticClass` values are the pixel values of the
  world and FOV surfaces; `VOID_ID = 255` marks "nothing" (outside the rotated
  crop, masked corners) and renders black. RGB is only ever a palette lookup,
  so colour matching cannot drift from the semantics.
* **Dirty rects.** `WorldRenderer.begin_frame()` restores only last frame's
  dynamic rects from the static layer (≈ 10 rects/frame) instead of blitting
  the whole padded map (the former `scene.redraw_world`, ≈ 2 ms per step).
* **Bodies, not sprites.** Vehicles/pedestrians are drawn from
  `Body.corner_tuples()` (oriented polygon), circles (pedestrians) or the legacy
  axis-aligned square, pixel-identical to the previous renderer for the legacy
  profile. Route targets and traffic-light strips are drawn from their surface
  rects. No sprite cache is needed: drawing all actors costs ≈ 0.1 ms.
* **Ego overlay.** `FovRenderer.draw_ego_overlay()` places the ego footprint
  with its *reference point* on the FOV anchor and the body extending forward,
  using the `EGO` class (previously the ego was drawn black — indistinguishable
  from void; finding R-03).

## Decoupling

| Before | After |
| --- | --- |
| `Actor(pygame.sprite.Sprite)` owning `rect`, `draw`, `isCollided` | `Actor` is a plain object exposing `body()`, `semantic_class`, `surface_rect()`; no `pygame` import in `src/actors`, `src/control`, `src/geometry` (guarded by `test_domain_modules_do_not_import_pygame`) |
| `BaseAgent(Controller, Hero)` (Hero = pygame sprite) | `BaseAgent(Controller)`; overlay drawn by the FOV renderer |
| `BaseMap(Scene)` — map, scene and rendering in one object | `BaseMap` composes `Scene`, `WorldRenderer`, `FovRenderer`; attribute access not found on the map is delegated to the scene (`map.hero`, `map.actor_manager`, `map.clock` …) |
| `Scene._scene_step` blitted the map and drew sprites | `Scene` only advances actors and exposes `drawables()`; `BaseMap.step()` draws after the physics step |
| `Target.isCollided(hero)` on pygame rects; `TrafficLight.draw` with pygame | `Target.trigger(hero_rect)` / `TrafficLight.rect_px()` on plain `IntRect`s (`geometry/rects.py`, same half-open overlap semantics as `Rect.colliderect`) |

## Pixel equivalence and intended changes

Frame-by-frame comparison against the previous renderer over three seeded
scenes (128 px, incl. a masked/anchored variant) shows identical pixels except
for four **intended** fixes:

| ID | Fix |
| --- | --- |
| R-03 | The ego is drawn in the `EGO` class colour (cyan (0, 200, 255) since Phase 4; yellow before) instead of the hard-coded black passed by `spawn_hero`, which made it indistinguishable from the void outside the crop. |
| R-04 | The **reset observation** now contains vehicles and route targets; previously actors were only drawn inside `step()`, so the first frame of every episode showed the bare map. |
| R-05 | Traffic-light stop-line strips are drawn at their true surface position; `TrafficLight.draw` ignored the render-frame padding and painted them ≈ 181 px away. |
| S-02 | Yellow and green lights have their own classes (`TRAFFIC_LIGHT_YELLOW` (255, 200, 0), `TRAFFIC_LIGHT_GREEN` (0, 200, 0)); previously a yellow light used the ego colour and a green light the route colour (and therefore counted as route in the semantic masks). |

Semantic masks (`rgb_to_semantic_mask`) also accept an `(H, W)` class-id
raster directly (`ids_to_semantic_mask`).

## Performance (standard benchmark, this sandbox)

| config | before (ms/step) | after | speed-up |
| --- | --- | --- | --- |
| 64 px rgb_array | 1.95 | 1.64 | 1.2× |
| 128 px rgb_array | 4.22 | **1.49** | **2.8×** |
| 128 px random traffic | 4.67 | 2.69 | 1.7× |
| 128 px human | 4.25 | 2.09 | 2.0× |
| 256 px rgb_array | 10.48 | 2.35 | 4.5× |

Phase breakdown at 128 px: `scene.draw_actors` 0.10 ms (was 2.0 ms redraw +
0.2 ms sprites), `map.draw_fov` 0.22 ms, `env.render` 0.13 ms,
`scene.collision_check` 0.24 ms, `scene.npc_step` 0.23 ms. The palette list is
cached (building it cost ~1 ms per frame in a first version) and the FOV
output surface is preallocated — both found with `carlabev benchmark`.

## Validation

`tests/physics_time/rendering/test_world_renderer.py`: dirty-rect world equals
a full redraw every frame; observation RGB equals the palette lookup of the
ids; `bev_class` mode; reset frame contains actors (R-04); traffic-light strips
at their position (R-05); masked corners are VOID; pygame-free domain modules.
FPS-independence digests and all goldens unchanged.

## Multi-rate presentation (Phase 5)

Drawing is no longer part of the physics step: `BaseMap.advance()` integrates
and updates the semantic tile under the ego, `BaseMap.render_world()` draws the
dynamic layer and the FOV once per control step (boundary state). In `human`
mode with `EnvConfig.display_fps`, a `Presenter` (`envs/presenter.py`) shows
`display_fps × control_dt` frames per control step, interpolating
`WorldSnapshot`s captured after every physics sub-step (poses linear, headings
along the short arc) into its own FOV output, paced at `display_fps` in real
time. It reads snapshots only — the observation surface, actors and clock are
untouched (`tests/physics_time/rendering/test_presenter.py`).
