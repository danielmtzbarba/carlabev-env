from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

from CarlaBEV.config import (
    AuthoredSceneReset,
    EnvConfig,
    RandomNavigationReset,
    build_authored_scene_options,
    build_random_navigation_options,
)
from CarlaBEV.envs.carlabev import CarlaBEV
from CarlaBEV.wrappers.rgb_to_semantic import (
    fuse_vehicle_temporal_channels,
    fuse_weighted_vehicle_history,
    rgb_to_semantic_mask,
    stacked_semantic_channel_labels,
    vehicle_temporal_channel_labels,
    weighted_vehicle_history_channel_labels,
)


DEFAULT_AUTHORED_SCENE = "CarlaBEV/assets/scenes/redlightrunner-01.01.json"


@dataclass
class Args:
    output: str | None = None
    size: int = 128
    map_name: str = "Town01"
    semantic_mask_ch: str = "6-class"
    scene: str = "rdm"
    authored_scene: str | None = DEFAULT_AUTHORED_SCENE
    variation_enabled: bool = False
    variation_seed: int | None = None
    warmup_steps: int = 3
    history_frames: int = 3
    num_vehicles: int = 10
    route_min: int = 30
    route_max: int = 100
    figure_dpi: int = 180
    interpolation: str = "nearest"
    weights: tuple[float, float, float] = (1.0, 0.5, 0.25)
    max_cols: int = 4


def _build_reset_options(args: Args) -> dict[str, object]:
    if args.authored_scene:
        return build_authored_scene_options(
            AuthoredSceneReset(
                config_file=args.authored_scene,
                variation_enabled=args.variation_enabled,
                variation_seed=args.variation_seed,
            )
        )
    if args.scene == "rdm":
        return build_random_navigation_options(
            RandomNavigationReset(
                num_vehicles=args.num_vehicles,
                route_dist_range=(args.route_min, args.route_max),
            )
        )
    return {"scene": args.scene}


def _capture_rgb_history(args: Args) -> np.ndarray:
    cfg = EnvConfig(
        size=args.size,
        map_name=args.map_name,
        obs_mode="bev_rgb",
        render_mode="rgb_array",
        action_mode="discrete",
    )
    env = CarlaBEV(cfg)
    frames = []
    try:
        obs, _ = env.reset(options=_build_reset_options(args))
        for _ in range(args.warmup_steps):
            obs, _, terminated, truncated, _ = env.step(0)
            if terminated or truncated:
                obs, _ = env.reset(options=_build_reset_options(args))
                break

        frames.append(np.asarray(obs))
        while len(frames) < args.history_frames:
            obs, _, terminated, truncated, _ = env.step(0)
            if terminated or truncated:
                obs, _ = env.reset(options=_build_reset_options(args))
            frames.append(np.asarray(obs))
        return np.stack(frames, axis=0)
    finally:
        env.close()


def _build_variants(rgb_history: np.ndarray, args: Args):
    stacked_semantics = np.stack(
        [rgb_to_semantic_mask(frame, mode=args.semantic_mask_ch) for frame in rgb_history],
        axis=0,
    ).astype(np.float32)

    current_stack = stacked_semantics.reshape(-1, *stacked_semantics.shape[2:])
    temporal = fuse_vehicle_temporal_channels(
        stacked_semantics,
        mode=args.semantic_mask_ch,
        history_frames=args.history_frames,
    )
    weighted = fuse_weighted_vehicle_history(
        stacked_semantics,
        mode=args.semantic_mask_ch,
        weights=args.weights,
    )

    return [
        ("rgb_history", rgb_history, tuple(f"rgb_t-{args.history_frames - 1 - idx}" if idx < args.history_frames - 1 else "rgb_t" for idx in range(args.history_frames))),
        (
            "current_stack",
            current_stack,
            stacked_semantic_channel_labels(args.semantic_mask_ch, args.history_frames),
        ),
        (
            "vehicle_temporal",
            temporal,
            vehicle_temporal_channel_labels(args.semantic_mask_ch, args.history_frames),
        ),
        (
            "vehicle_weighted",
            weighted,
            weighted_vehicle_history_channel_labels(args.semantic_mask_ch),
        ),
    ]


def _short_label(label: str) -> str:
    replacements = {
        "non_drivable": "non_drv",
        "drivable": "drv",
        "sidewalk": "sidewalk",
        "pedestrian": "ped",
        "route": "route",
        "traffic_light_red": "red_light",
        "vehicle_history_weighted": "veh_hist_w",
        "vehicle_t": "veh@t",
        "vehicle_t-1": "veh@t-1",
        "vehicle_t-2": "veh@t-2",
        "rgb_t": "rgb@t",
        "rgb_t-1": "rgb@t-1",
        "rgb_t-2": "rgb@t-2",
    }
    return replacements.get(label, label)


def _draw_variant_grid(subfig, row_name: str, images: np.ndarray, labels: tuple[str, ...], args: Args) -> None:
    cols = min(args.max_cols, images.shape[0])
    rows = int(np.ceil(images.shape[0] / cols))
    axes = subfig.subplots(rows, cols, squeeze=False)
    subfig.suptitle(row_name, fontsize=13, fontweight="bold", x=0.02, ha="left", y=0.98)

    for idx in range(rows * cols):
        row = idx // cols
        col = idx % cols
        ax = axes[row][col]
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)

        if idx >= images.shape[0]:
            ax.axis("off")
            continue

        image = images[idx]
        if row_name == "rgb_history":
            ax.imshow(image, interpolation=args.interpolation)
        else:
            ax.imshow(
                image,
                cmap="magma",
                vmin=0.0,
                vmax=1.0,
                interpolation=args.interpolation,
            )
        ax.set_title(_short_label(labels[idx]), fontsize=9, pad=4)


def _render_variant_figure(plt, row_name: str, images: np.ndarray, labels: tuple[str, ...], args: Args):
    cols = min(args.max_cols, images.shape[0])
    rows = int(np.ceil(images.shape[0] / cols))
    fig = plt.figure(
        figsize=(cols * 2.8, rows * 2.8 + 0.8),
        dpi=args.figure_dpi,
        constrained_layout=True,
    )
    fig.patch.set_facecolor("white")
    fig.suptitle(
        f"{row_name} | semantic_mask_ch={args.semantic_mask_ch}",
        fontsize=13,
    )
    _draw_variant_grid(fig, row_name, images, labels, args)
    return fig


def main(args: Args) -> None:
    if args.history_frames != 3:
        raise ValueError("This visualization currently expects history_frames=3 to match the implemented ablations.")
    if args.semantic_mask_ch not in {"4-class", "5-class", "6-class", "7-class"}:
        raise ValueError("semantic_mask_ch must expose a vehicle channel for this visualization.")

    rgb_history = _capture_rgb_history(args)
    variants = _build_variants(rgb_history, args)

    try:
        if args.output:
            import matplotlib

            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required to display the vehicle temporal fusion montage."
        ) from exc

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        for row_name, images, labels in variants:
            fig = _render_variant_figure(plt, row_name, images, labels, args)
            method_path = output_path.with_name(f"{output_path.stem}-{row_name}{output_path.suffix}")
            fig.savefig(method_path, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {row_name} visualization to {method_path.resolve()}")
    else:
        figures = []
        for row_name, images, labels in variants:
            fig = _render_variant_figure(plt, row_name, images, labels, args)
            try:
                fig.canvas.manager.set_window_title(f"CarlaBEV {row_name}")
            except Exception:
                pass
            figures.append(fig)
        plt.show()


if __name__ == "__main__":
    main(tyro.cli(Args))
