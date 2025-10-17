import gym
import torch
import random, datetime, numpy as np
import cv2

from gym.spaces import Box

class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        resize_obs = cv2.resize(observation, self.shape[::-1], interpolation=cv2.INTER_AREA)
        return resize_obs


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class RewardShaping(gym.Wrapper):
    """
    Reward shaping to make learning faster:
    - Reward for moving right (progress)
    - Penalty for dying
    - Bonus for reaching flag
    """
    def __init__(self, env):
        super().__init__(env)
        self._current_x = 0
        self._max_x = 0

    def reset(self, **kwargs):
        self._current_x = 0
        self._max_x = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # Track Mario's x position
        new_x = info.get('x_pos', 0)

        # Reward for moving right (exploration bonus)
        if new_x > self._max_x:
            reward += (new_x - self._max_x) / 10.0  # Small bonus for progress
            self._max_x = new_x

        # Penalty for moving backward
        if new_x < self._current_x - 5:
            reward -= 1

        self._current_x = new_x

        # Big penalty for dying
        if done and not info.get('flag_get', False):
            reward -= 50

        # Big bonus for reaching flag
        if info.get('flag_get', False):
            reward += 500

        return obs, reward, done, info
