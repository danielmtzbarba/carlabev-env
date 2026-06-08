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
    rgb_to_semantic_mask,
    semantic_mask_channels,
)


MODE_ORDER = ("binary", "2-class", "4-class", "5-class", "7-class")
DEFAULT_AUTHORED_SCENE = "CarlaBEV/assets/scenes/redlightrunner-01.01.json"


@dataclass
class Args:
    output: str | None = None
    size: int = 128
    map_name: str = "Town01"
    scene: str = "rdm"
    authored_scene: str | None = DEFAULT_AUTHORED_SCENE
    variation_enabled: bool = False
    variation_seed: int | None = None
    warmup_steps: int = 3
    num_vehicles: int = 10
    route_min: int = 30
    route_max: int = 100
    figure_dpi: int = 180
    interpolation: str = "bilinear"


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


def _capture_rgb_frame(args: Args) -> np.ndarray:
    cfg = EnvConfig(
        size=args.size,
        map_name=args.map_name,
        obs_mode="bev_rgb",
        render_mode="rgb_array",
    )
    env = CarlaBEV(cfg)
    try:
        obs, _ = env.reset(options=_build_reset_options(args))
        for _ in range(args.warmup_steps):
            action = 0 if cfg.action_mode == "discrete" else np.zeros(3, dtype=np.float32)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset(options=_build_reset_options(args))
                break
        return obs
    finally:
        env.close()


def _semantic_rows(rgb_frame: np.ndarray):
    rows = [("raw_rgb", [(rgb_frame, "raw_rgb")])]
    for mode in MODE_ORDER:
        semantic = rgb_to_semantic_mask(rgb_frame, mode=mode)
        labels = semantic_mask_channels(mode)
        rows.append((mode, list(zip(semantic, labels, strict=True))))
    return rows


def main(args: Args) -> None:
    rgb_frame = _capture_rgb_frame(args)
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required to display the semantic channel montage. "
            "Install project dependencies and rerun, or pass --output to save instead."
        ) from exc

    rows = _semantic_rows(rgb_frame)
    max_cols = max(len(images) for _, images in rows)
    fig, axes = plt.subplots(
        len(rows),
        max_cols,
        figsize=(max_cols * 3.2, len(rows) * 3.0),
        dpi=args.figure_dpi,
        squeeze=False,
        constrained_layout=True,
    )
    fig.patch.set_facecolor("white")

    for row_idx, (row_name, images) in enumerate(rows):
        for col_idx in range(max_cols):
            ax = axes[row_idx][col_idx]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)
            if col_idx >= len(images):
                ax.axis("off")
                continue

            image, label = images[col_idx]
            if row_name == "raw_rgb":
                ax.imshow(image, interpolation=args.interpolation)
            else:
                ax.imshow(
                    image,
                    cmap="magma",
                    vmin=0.0,
                    vmax=1.0,
                    interpolation=args.interpolation,
                )
            ax.set_title(label, fontsize=10, pad=8)

        axes[row_idx][0].set_ylabel(
            row_name,
            rotation=0,
            labelpad=36,
            va="center",
            fontsize=11,
        )

    if args.output:
        output_path = Path(args.output)
        fig.savefig(output_path, bbox_inches="tight")
        print(f"Saved semantic channel visualization to {output_path.resolve()}")

    plt.show()


if __name__ == "__main__":
    main(tyro.cli(Args))
