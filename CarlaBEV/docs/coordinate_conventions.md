# Coordinate Conventions

This document describes the coordinate conventions used by CarlaBEV.

## Core Principle

Rendering, collision, camera, and semantic lookup should all agree on the same ego reference point.

The simulator now enforces that the configured FOV ego anchor corresponds to the true ego scene position at reset and during stepping.

## Frames

### 1. Raw Asset Frame

Used by:

- graph assets
- map-specific annotations where applicable

This frame is not used directly by dynamics or rendering.

### 2. Surface/World Runtime Frame

Used by:

- legacy ego and actor state views (`state`, `x`, `y`, `v`)
- camera following
- route geometry
- current collision rectangles
- semantic tile queries

This is the rendering and compatibility frame. Dynamics integration uses the metric frame below.

### 3. Metric Dynamics Frame

Canonical actor dynamics use `x_m` and `y_m` in meters, `v_mps` in meters/second,
and `yaw` in radians. `state_m` exposes this representation. Route points are
converted from surface coordinates before controller calculations.

### 4. FOV/Image Frame

Used by:

- cropped and rotated BEV observations
- the DRL input image
- the ego marker drawn at the configured ego anchor of the FOV

The ego is rendered at the configured FOV anchor, and the cropped map content is aligned to the same world pose.

## Ego Anchor Rule

The camera follows the ego world position through [camera.py](../envs/camera.py).

The world-aligned crop remains centered on the ego pose before rotation. The final rotated image is then composed so the ego lands at the configured anchor in image space.

That is important:

- final FOV size: `128 x 128`
- crop size before rotation: derived from the configured anchor and rotation support

If the crop center and final image anchor are mixed together, the ego overlay may look correct while the underlying map content drifts with yaw.

## Semantic Tile Convention

The world and the FOV are rendered as **class-id rasters** (8-bit palettized
surfaces whose pixel values are `SemanticClass` ids; RGB is a palette lookup).
The semantic tile under the ego is taken from the authoritative static
class raster using the ego world position, not from the center pixel of the
rotated FOV. See [Rendering Architecture](rendering_architecture.md).

That sampled tile is also normalized into a shared semantic class id
(`collision.tile_class`) before reward/off-road logic consumes it.

This avoids false obstacle/off-road detections caused by:

- ego overlay pixels
- crop rotation
- masked corners

## Collision Convention

Collision checks now use the actual ego rectangle and actor rectangles in the same projected frame.

The previous offset-based collision heuristic was removed because it became incorrect once the FOV centering was fixed.

## Target / Waypoint Convention

Checkpoint and goal targets are handled as scene actors:

- excluded from invalid-spawn checks
- included in runtime collision/trigger checks

This lets the ego start on its route without spawn rejection while still allowing checkpoint and goal detection during rollout.

## Notes

- Canonical dynamics state is metric; surface/world state properties are compatibility views used at rendering and geometry boundaries.
- Actor footprints are rectangle approximations.
- Some scenario anchors are map-specific rather than lane-semantic.

## Implementation References

- [camera.py](../envs/camera.py)
- [world.py](../envs/world.py)
- [transforms.py](../envs/transforms.py)
- [scene.py](../src/scenes/scene.py)
