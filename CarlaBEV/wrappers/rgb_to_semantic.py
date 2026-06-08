import gymnasium as gym
import numpy as np

from CarlaBEV.semantics import SemanticClass, semantic_color_array

SEMANTIC_MASK_CHANNELS = {
    "binary": ("drivable",),
    "2-class": (
        "drivable",
        "route",
    ),
    "4-class": (
        "drivable",
        "vehicle",
        "pedestrian",
        "route",
    ),
    "5-class": (
        "drivable",
        "sidewalk",
        "vehicle",
        "pedestrian",
        "route",
    ),
    "6-class": (
        "non_drivable",
        "drivable",
        "sidewalk",
        "vehicle",
        "pedestrian",
        "route",
    ),
    "7-class": (
        "non_drivable",
        "drivable",
        "sidewalk",
        "vehicle",
        "pedestrian",
        "route",
        "traffic_light_red",
    ),
}


def semantic_mask_channels(mode: str) -> tuple[str, ...]:
    try:
        return SEMANTIC_MASK_CHANNELS[mode]
    except KeyError as exc:
        choices = ", ".join(sorted(SEMANTIC_MASK_CHANNELS))
        raise ValueError(
            f"Unsupported semantic_mask_ch={mode!r}. Expected one of: {choices}"
        ) from exc


def rgb_to_semantic_mask(rgb_image, mode="6-class"):
    """
    Convert an RGB frame into one of the supported semantic channel layouts.

    Args:
        rgb_image (np.ndarray): (H, W, 3) RGB image.
        mode (str): semantic channel layout to emit.

    Returns:
        np.ndarray: (C, H, W) semantic mask (binary channels).
    """
    semantic_mask_channels(mode)

    rgb = rgb_image.astype(np.uint8)
    h, w, _ = rgb.shape

    white = semantic_color_array(SemanticClass.DRIVABLE)
    red = semantic_color_array(SemanticClass.PEDESTRIAN)
    red_light = semantic_color_array(SemanticClass.TRAFFIC_LIGHT_RED)
    blue = semantic_color_array(SemanticClass.VEHICLE)
    green = semantic_color_array(SemanticClass.ROUTE)
    gray = semantic_color_array(SemanticClass.NON_DRIVABLE)
    grays = semantic_color_array(SemanticClass.SIDEWALK)

    is_white = np.all(rgb == white, axis=-1)
    is_red = np.all(rgb == red, axis=-1)
    is_red_light = np.all(rgb == red_light, axis=-1)
    is_blue = np.all(rgb == blue, axis=-1)
    is_green = np.all(rgb == green, axis=-1)
    is_gray = np.all(rgb == gray, axis=-1)
    is_grays = np.all(rgb == grays, axis=-1)

    if mode == "binary":
        mask = np.zeros((1, h, w), dtype=np.float32)
        mask[0, np.logical_or(is_white, is_green)] = 1.0
        return mask

    if mode == "2-class":
        mask = np.zeros((2, h, w), dtype=np.float32)
        mask[0, np.logical_or(is_white, is_green)] = 1.0
        mask[1, is_green] = 1.0
        return mask

    if mode == "4-class":
        mask = np.zeros((4, h, w), dtype=np.float32)
        mask[0, np.logical_or(is_white, is_green)] = 1.0
        mask[1, is_blue] = 1.0
        mask[2, is_red] = 1.0
        mask[3, is_green] = 1.0
        return mask

    if mode == "5-class":
        mask = np.zeros((5, h, w), dtype=np.float32)
        mask[0, np.logical_or(is_white, is_green)] = 1.0
        mask[1, is_grays] = 1.0
        mask[2, is_blue] = 1.0
        mask[3, is_red] = 1.0
        mask[4, is_green] = 1.0
        return mask

    mask = np.zeros((6, h, w), dtype=np.float32)
    mask[0, is_gray] = 1.0
    mask[1, np.logical_or(is_white, is_green)] = 1.0
    mask[2, is_grays] = 1.0
    mask[3, is_blue] = 1.0
    mask[4, is_red] = 1.0
    mask[5, is_green] = 1.0
    if mode == "7-class":
        mask = np.zeros((7, h, w), dtype=np.float32)
        mask[0, is_gray] = 1.0
        mask[1, np.logical_or(is_white, is_green)] = 1.0
        mask[2, is_grays] = 1.0
        mask[3, is_blue] = 1.0
        mask[4, is_red] = 1.0
        mask[5, is_green] = 1.0
        mask[6, is_red_light] = 1.0
        return mask
    return mask

class SemanticMaskWrapper(gym.ObservationWrapper):
    """
    A Gym wrapper to convert RGB observations into semantic masks.

    This wrapper assumes the environment's observation is an RGB image
    and converts it into the configured semantic mask layout.
    """

    def __init__(self, env, mode="6-class"):
        super(SemanticMaskWrapper, self).__init__(env)
        self.mode = mode
        self.channel_names = semantic_mask_channels(mode)
        obs_shape = self.observation_space.shape
        h, w = obs_shape[:2]
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(len(self.channel_names), h, w),
            dtype=np.float32,
        )

    def observation(self, observation):
        """
        Convert the RGB observation into a semantic mask.

        Args:
            observation (np.ndarray): (H, W, 3) RGB image.

        Returns:
            np.ndarray: (C, H, W) semantic mask.
        """
        return rgb_to_semantic_mask(observation, mode=self.mode)


class FlattenStackedFrames(gym.ObservationWrapper):
    """
    Merge frame and channel dimensions: (num_frames, num_channels, H, W) → (num_frames * num_channels, H, W)
    """
    def __init__(self, env):
        super().__init__(env)
        obs_shape = env.observation_space.shape  # e.g. (4, 6, 96, 96)
        assert len(obs_shape) == 4, f"Unexpected shape: {obs_shape}"
        num_frames, num_channels, h, w = obs_shape
        merged_shape = (num_frames * num_channels, h, w)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=merged_shape, dtype=np.float32
        )

    def observation(self, obs):
        # merge the first two dims: (frames, channels, H, W) → (frames*channels, H, W)
        return np.asarray(obs).reshape(-1, *obs.shape[2:])
