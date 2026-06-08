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


def vehicle_channel_index(mode: str) -> int:
    channels = semantic_mask_channels(mode)
    try:
        return channels.index("vehicle")
    except ValueError as exc:
        raise ValueError(
            f"semantic_mask_ch={mode!r} does not expose a vehicle channel, so vehicle history fusion is unsupported."
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


def flatten_stacked_frames(obs: np.ndarray) -> np.ndarray:
    stacked = np.asarray(obs, dtype=np.float32)
    if stacked.ndim != 4:
        raise ValueError(f"Expected stacked observation with shape (frames, channels, H, W), got {stacked.shape}")
    return stacked.reshape(-1, *stacked.shape[2:])


def fuse_vehicle_temporal_channels(obs: np.ndarray, mode: str, history_frames: int = 3) -> np.ndarray:
    stacked = np.asarray(obs, dtype=np.float32)
    if stacked.ndim != 4:
        raise ValueError(f"Expected stacked observation with shape (frames, channels, H, W), got {stacked.shape}")
    if stacked.shape[0] < history_frames:
        raise ValueError(
            f"Vehicle temporal fusion requires at least {history_frames} stacked frames, got {stacked.shape[0]}"
        )

    vehicle_idx = vehicle_channel_index(mode)
    history = stacked[-history_frames:]
    current = history[-1]

    static_without_vehicle = np.delete(current, vehicle_idx, axis=0)
    vehicle_history = history[::-1, vehicle_idx, :, :]
    return np.concatenate([static_without_vehicle, vehicle_history], axis=0).astype(np.float32)


def fuse_weighted_vehicle_history(
    obs: np.ndarray,
    mode: str,
    weights: tuple[float, float, float] = (1.0, 0.5, 0.25),
) -> np.ndarray:
    stacked = np.asarray(obs, dtype=np.float32)
    if stacked.ndim != 4:
        raise ValueError(f"Expected stacked observation with shape (frames, channels, H, W), got {stacked.shape}")
    history_frames = len(weights)
    if stacked.shape[0] < history_frames:
        raise ValueError(
            f"Weighted vehicle history requires at least {history_frames} stacked frames, got {stacked.shape[0]}"
        )

    vehicle_idx = vehicle_channel_index(mode)
    history = stacked[-history_frames:][::-1]
    current = history[0]

    static_without_vehicle = np.delete(current, vehicle_idx, axis=0)
    weighted_vehicle = np.zeros_like(current[vehicle_idx], dtype=np.float32)
    for frame, weight in zip(history, weights, strict=True):
        weighted_vehicle += weight * frame[vehicle_idx]
    weighted_vehicle = np.clip(weighted_vehicle, 0.0, 1.0)
    return np.concatenate([static_without_vehicle, weighted_vehicle[None, ...]], axis=0).astype(np.float32)


def stacked_semantic_channel_labels(mode: str, num_frames: int) -> tuple[str, ...]:
    labels = []
    channels = semantic_mask_channels(mode)
    for frame_idx in range(num_frames):
        age = num_frames - 1 - frame_idx
        suffix = "t" if age == 0 else f"t-{age}"
        labels.extend(f"{channel}_{suffix}" for channel in channels)
    return tuple(labels)


def vehicle_temporal_channel_labels(mode: str, history_frames: int = 3) -> tuple[str, ...]:
    channels = list(semantic_mask_channels(mode))
    channels.remove("vehicle")
    labels = list(channels)
    for age in range(history_frames):
        suffix = "t" if age == 0 else f"t-{age}"
        labels.append(f"vehicle_{suffix}")
    return tuple(labels)


def weighted_vehicle_history_channel_labels(mode: str) -> tuple[str, ...]:
    channels = list(semantic_mask_channels(mode))
    channels.remove("vehicle")
    channels.append("vehicle_history_weighted")
    return tuple(channels)

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
        return flatten_stacked_frames(obs)


class VehicleTemporalFusionWrapper(gym.ObservationWrapper):
    """
    Replace the single current-frame vehicle channel with a three-frame vehicle history.
    Output: current static channels without vehicle + vehicle_t/t-1/t-2.
    """

    def __init__(self, env, mode="6-class", history_frames: int = 3):
        super().__init__(env)
        self.mode = mode
        self.history_frames = history_frames
        obs_shape = env.observation_space.shape
        assert len(obs_shape) == 4, f"Unexpected shape: {obs_shape}"
        num_frames, num_channels, h, w = obs_shape
        if num_frames < history_frames:
            raise ValueError(
                f"VehicleTemporalFusionWrapper requires frame_stack >= {history_frames}, got {num_frames}"
            )
        vehicle_channel_index(mode)
        merged_shape = (num_channels - 1 + history_frames, h, w)
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=merged_shape,
            dtype=np.float32,
        )

    def observation(self, obs):
        return fuse_vehicle_temporal_channels(obs, mode=self.mode, history_frames=self.history_frames)


class WeightedVehicleHistoryWrapper(gym.ObservationWrapper):
    """
    Replace the vehicle channel with a single decayed vehicle history channel.
    Output: current static channels without vehicle + weighted vehicle history.
    """

    def __init__(self, env, mode="6-class", weights: tuple[float, float, float] = (1.0, 0.5, 0.25)):
        super().__init__(env)
        self.mode = mode
        self.weights = weights
        obs_shape = env.observation_space.shape
        assert len(obs_shape) == 4, f"Unexpected shape: {obs_shape}"
        num_frames, num_channels, h, w = obs_shape
        if num_frames < len(weights):
            raise ValueError(
                f"WeightedVehicleHistoryWrapper requires frame_stack >= {len(weights)}, got {num_frames}"
            )
        vehicle_channel_index(mode)
        merged_shape = (num_channels, h, w)
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=merged_shape,
            dtype=np.float32,
        )

    def observation(self, obs):
        return fuse_weighted_vehicle_history(obs, mode=self.mode, weights=self.weights)
