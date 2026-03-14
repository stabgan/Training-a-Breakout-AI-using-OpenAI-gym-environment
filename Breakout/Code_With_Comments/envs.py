# Improvement of the Gym environment with universe

import cv2
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box


def create_atari_env(env_id, video=False):
    env = gym.make(env_id, render_mode="rgb_array")
    if video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder='test',
            episode_trigger=lambda ep: True,
            disable_logger=True,
        )
    env = MyAtariRescale42x42(env)
    env = MyNormalizedEnv(env)
    return env


def _process_frame42(frame):
    frame = frame[34:34 + 160, :160]
    # Resize by half, then down to 42x42 (essentially mipmapping). If
    # we resize directly we lose pixels that, when mapped to 42x42,
    # aren't close enough to the pixel boundary.
    frame = cv2.resize(frame, (80, 80))
    frame = cv2.resize(frame, (42, 42))
    frame = frame.mean(2)
    frame = frame.astype(np.float32)
    frame *= (1.0 / 255.0)
    return frame


class MyAtariRescale42x42(gym.ObservationWrapper):

    def __init__(self, env=None):
        super(MyAtariRescale42x42, self).__init__(env)
        self.observation_space = Box(0.0, 1.0, shape=(1, 42, 42), dtype=np.float32)

    def observation(self, observation):
        return _process_frame42(observation)


class MyNormalizedEnv(gym.ObservationWrapper):

    def __init__(self, env=None):
        super(MyNormalizedEnv, self).__init__(env)
        self.state_mean = 0
        self.state_std = 0
        self.alpha = 0.9999
        self.num_steps = 0

    def observation(self, observation):
        self.num_steps += 1
        self.state_mean = self.state_mean * self.alpha + \
            observation.mean() * (1 - self.alpha)
        self.state_std = self.state_std * self.alpha + \
            observation.std() * (1 - self.alpha)

        unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.num_steps))
        unbiased_std = self.state_std / (1 - pow(self.alpha, self.num_steps))

        ret = (observation - unbiased_mean) / (unbiased_std + 1e-8)
        return np.expand_dims(ret, axis=0)
