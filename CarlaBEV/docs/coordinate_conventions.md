# Coordinate Conventions

This document describes the coordinate conventions used by CarlaBEV.

## Core Principle

Rendering, collision, camera, and semantic lookup should all agree on the same ego reference point.

The simulator now enforces that the FOV center corresponds to the true ego scene position at reset and during stepping.

## Frames

### 1. Raw Asset Frame

Used by:

- graph assets
- map-specific annotations where applicable

This frame is not used directly by dynamics or rendering.

### 2. Surface/World Runtime Frame

Used by:

- ego and actor state
- camera following
- route geometry
- current collision rectangles
- semantic tile queries

At the current stage of the simulator, this is the main operational frame.

### 3. FOV/Image Frame

Used by:

- cropped and rotated BEV observations
- the DRL input image
- the ego marker drawn at the center of the FOV

The ego is rendered at the FOV center, and the cropped map content is now centered on the same world pose.

## Ego Centering Rule

The camera follows the ego world position through [camera.py](/Users/danielmtz/Data/projects/driverless/carlabev-env/CarlaBEV/envs/camera.py).

Because the crop region is larger than the final FOV, the camera centers on the crop resolution, not on the final display resolution.

That is important:

- final FOV size: `128 x 128`
- crop size before rotation: `192 x 192`

If the camera centers on the wrong size, the ego overlay may appear centered while the underlying map content is shifted.

## Semantic Tile Convention

The semantic tile under the ego is taken from the authoritative semantic map using ego world position, not from the center pixel of the rotated FOV.

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

- Runtime actor state is stored in the surface/world frame used by the simulator runtime.
- Actor footprints are rectangle approximations.
- Some scenario anchors are map-specific rather than lane-semantic.

## Implementation References

- [camera.py](/Users/danielmtz/Data/projects/driverless/carlabev-env/CarlaBEV/envs/camera.py)
- [world.py](/Users/danielmtz/Data/projects/driverless/carlabev-env/CarlaBEV/envs/world.py)
- [transforms.py](/Users/danielmtz/Data/projects/driverless/carlabev-env/CarlaBEV/envs/transforms.py)
- [scene.py](/Users/danielmtz/Data/projects/driverless/carlabev-env/CarlaBEV/src/scenes/scene.py)
