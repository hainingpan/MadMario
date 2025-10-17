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
    Optimal reward shaping for faster DQN learning in Super Mario Bros.

    Based on research and best practices from:
    - PyTorch Mario RL Tutorial (https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html)
    - "Internal Model from Observations for Reward Shaping" (arXiv:1806.01267)
    - Community implementations and empirical findings

    Reward components:
    1. Base reward: x-position change (from environment)
    2. Exploration bonus: Reward for reaching new max x-position
    3. Time efficiency: Small bonus per step alive
    4. Enemy kills: Bonus for defeating enemies (detected via score change)
    5. Coin collection: Bonus for collecting coins
    6. Power-ups: Bonus for getting mushrooms/flowers
    7. Death penalty: Scaled by progress (dying early is worse)
    8. Flag completion: Large bonus for reaching the goal
    9. Stuck penalty: Discourage getting stuck in place
    """
    def __init__(self, env):
        super().__init__(env)
        self._current_x = 0
        self._max_x = 0
        self._previous_score = 0
        self._previous_coins = 0
        self._previous_status = 'small'
        self._stuck_count = 0
        self._stuck_threshold = 20  # Frames

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._current_x = 40  # Starting x position
        self._max_x = 40
        self._previous_score = 0
        self._previous_coins = 0
        self._previous_status = 'small'
        self._stuck_count = 0
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # Extract info
        new_x = info.get('x_pos', 0)
        new_score = info.get('score', 0)
        new_coins = info.get('coins', 0)
        new_status = info.get('status', 'small')
        flag_get = info.get('flag_get', False)

        # 1. Base reward (already included from environment)
        # reward = Δx_pos (from environment)

        # 2. Exploration bonus: Reward for reaching new areas
        if new_x > self._max_x:
            exploration_bonus = (new_x - self._max_x) * 0.1
            reward += exploration_bonus
            self._max_x = new_x
            self._stuck_count = 0  # Reset stuck counter

        # 3. Time efficiency: Small bonus for staying alive
        if not done:
            reward += 0.1

        # 4. Enemy kill bonus (detected via score increase)
        # Score increases by 100-1000 per enemy kill
        score_delta = new_score - self._previous_score
        if score_delta >= 100:  # Enemy killed
            # Estimate number of enemies killed (100, 200, 400, 800 points)
            if score_delta >= 800:
                reward += 50  # Multiple enemies
            elif score_delta >= 400:
                reward += 30  # Chain kill
            elif score_delta >= 200:
                reward += 20  # Double kill
            else:
                reward += 10  # Single enemy

        # 5. Coin collection bonus
        coins_collected = new_coins - self._previous_coins
        if coins_collected > 0:
            reward += coins_collected * 5  # +5 per coin

        # 6. Power-up bonus (status change: small -> big -> fire)
        if new_status != self._previous_status:
            if new_status == 'tall':  # Got mushroom
                reward += 20
            elif new_status == 'fireball':  # Got fire flower
                reward += 30

        # 7. Death penalty (scaled by progress)
        if done and not flag_get:
            # Calculate progress through level (flag at x ~3161)
            progress_ratio = min(self._max_x / 3161.0, 1.0)
            # Dying early is penalized more heavily
            death_penalty = -100 * (1 - progress_ratio * 0.5)  # Range: -50 to -100
            reward += death_penalty

        # 8. Flag completion bonus (HUGE reward for success!)
        if flag_get:
            time_remaining = info.get('time', 0)
            time_bonus = time_remaining * 2  # Reward speed
            flag_bonus = 1000 + time_bonus
            reward += flag_bonus

        # 9. Stuck penalty: Punish standing still
        if new_x == self._current_x:
            self._stuck_count += 1
            if self._stuck_count >= self._stuck_threshold:
                reward -= 1  # Small penalty for being stuck
        else:
            self._stuck_count = 0

        # Update tracking variables
        self._current_x = new_x
        self._previous_score = new_score
        self._previous_coins = new_coins
        self._previous_status = new_status

        return obs, reward, done, info
