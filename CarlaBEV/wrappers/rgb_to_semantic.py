import gymnasium as gym
import numpy as np

def rgb_to_semantic_mask(rgb_image):
    """
    Convert RGB frame from simulator into a 5-channel semantic mask.

    Args:
        rgb_image (np.ndarray): (H, W, 3) RGB image.

    Returns:
        np.ndarray: (5, H, W) semantic mask (binary channels).
    """
    # Normalize input if needed (assume 0-255)
    rgb = rgb_image.astype(np.uint8)
    h, w, _ = rgb.shape
    mask = np.zeros((6, h, w), dtype=np.float32)

    # Define color thresholds (loose matching in case of slight rendering variation)
    white = np.array([255, 255, 255])
    red = np.array([255, 0, 0])
    blue = np.array([0, 7, 175])
    green = np.array([0, 255, 0])
    gray = np.array([150, 150, 150])  # Adjusted if your gray is different
    grays = np.array([220, 220, 220])  # Adjusted if your gray is different

    # Create masks
    is_white = np.all(rgb == white, axis=-1)
    is_red = np.all(rgb == red, axis=-1)
    is_blue = np.all(rgb == blue, axis=-1)
    is_green = np.all(rgb == green, axis=-1)
    is_gray = np.all(rgb == gray, axis=-1)
    is_grays = np.all(rgb == grays, axis=-1)

    # Fill semantic channels
    mask[0, is_gray] = 1.0  # Non-drivable
    mask[1, is_white] = 1.0  # Drivable
    mask[2, is_grays] = 1.0  # Non-drivable
    mask[3, is_blue] = 1.0  # Vehicle
    mask[4, is_red] = 1.0  # Pedestrian
    mask[5, is_green] = 1.0  # Checkpoint

    return mask

class SemanticMaskWrapper(gym.ObservationWrapper):
    """
    A Gym wrapper to convert RGB observations into semantic masks.

    This wrapper assumes the environment's observation is an RGB image
    and converts it into a 6-channel semantic mask using the rgb_to_semantic_mask function.
    """

    def __init__(self, env):
        super(SemanticMaskWrapper, self).__init__(env)
        # Update the observation space to reflect the semantic mask shape
        obs_shape = self.observation_space.shape
        h, w = obs_shape[:2]
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(6, h, w), dtype=np.float32
        )

    def observation(self, observation):
        """
        Convert the RGB observation into a semantic mask.

        Args:
            observation (np.ndarray): (H, W, 3) RGB image.

        Returns:
            np.ndarray: (6, H, W) semantic mask.
        """
        return rgb_to_semantic_mask(observation)
