"""
Mario RL Training Script - Deep Q-Network (DQN)

This script demonstrates the complete Reinforcement Learning training loop.
The learning happens through the DQN algorithm implemented in agent.py.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import random, datetime
from pathlib import Path

import gym
import gym_super_mario_bros
# Use old gym wrappers since gym_super_mario_bros uses old gym API
from gym.wrappers import FrameStack, GrayScaleObservation
from nes_py.wrappers import JoypadSpace

from metrics import MetricLogger
from agent import Mario
from wrappers import ResizeObservation, SkipFrame, RewardShaping


# ============================================================================
# STEP 1: ENVIRONMENT SETUP
# ============================================================================
# Initialize Super Mario Bros environment
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')

# Limit the action space to just 2 actions:
#   0. walk right
#   1. jump right
env = JoypadSpace(
    env,
    [['right'],
    ['right', 'A']]
)

# Apply preprocessing wrappers to make learning more efficient
env = SkipFrame(env, skip=4)              # Process every 4th frame
env = RewardShaping(env)                  # Add reward shaping for faster learning
env = GrayScaleObservation(env, keep_dim=False)  # Convert to grayscale
env = ResizeObservation(env, shape=84)     # Resize to 84x84
env = FrameStack(env, num_stack=4)         # Stack 4 frames together

env.reset()

# ============================================================================
# STEP 2: AGENT INITIALIZATION
# ============================================================================
# Look for existing checkpoint to resume from
checkpoints_base = Path('checkpoints')
checkpoint = None
save_dir = None

# Find the latest checkpoint directory
if checkpoints_base.exists():
    checkpoint_dirs = sorted([d for d in checkpoints_base.iterdir() if d.is_dir()], reverse=True)
    for dir in checkpoint_dirs:
        latest_checkpoint = dir / 'mario_net_latest.chkpt'
        if latest_checkpoint.exists():
            checkpoint = latest_checkpoint
            save_dir = dir
            print(f"Found existing checkpoint: {checkpoint}")
            print("Resuming training from this checkpoint...\n")
            break

# If no checkpoint found, create new directory
if save_dir is None:
    save_dir = checkpoints_base / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    save_dir.mkdir(parents=True)
    print("No existing checkpoint found. Starting fresh training...\n")

# Initialize Mario agent
# - state_dim: (4, 84, 84) = 4 stacked grayscale 84x84 frames
# - action_dim: 2 actions (right, jump right)
mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir, checkpoint=checkpoint)

# Initialize metrics logger
logger = MetricLogger(save_dir)

# ============================================================================
# STEP 3: TRAINING LOOP
# ============================================================================
episodes = 40000  # Total number of episodes to train

print("Starting training...")
print(f"Checkpoints will be saved to: {save_dir}")
print(f"Training for {episodes} episodes\n")

for e in range(episodes):
    # Reset environment for new episode (old gym API returns just state)
    state = env.reset()

    # ========================================================================
    # EPISODE LOOP - Play one complete game
    # ========================================================================
    while True:
        # ---------------------------------------------------------------------
        # 1. AGENT ACTS (Exploration vs Exploitation)
        # ---------------------------------------------------------------------
        # Mario chooses an action based on epsilon-greedy strategy
        # - With probability epsilon: random action (EXPLORE)
        # - With probability 1-epsilon: best action from Q-network (EXPLOIT)
        action = mario.act(state)

        # ---------------------------------------------------------------------
        # 2. ENVIRONMENT RESPONDS
        # ---------------------------------------------------------------------
        # Execute the action in the environment (old gym API returns 4 values)
        next_state, reward, done, info = env.step(action)

        # ---------------------------------------------------------------------
        # 3. STORE EXPERIENCE (Experience Replay)
        # ---------------------------------------------------------------------
        # Save this experience in the replay buffer
        # Experience = (state, next_state, action, reward, done)
        mario.cache(state, next_state, action, reward, done)

        # ---------------------------------------------------------------------
        # 4. LEARN FROM EXPERIENCE (This is where learning happens!)
        # ---------------------------------------------------------------------
        # The agent samples a batch of past experiences and learns from them
        # Learning happens in agent.py's learn() method:
        #   - Sample random batch from replay buffer
        #   - Compute TD estimate: Q(s, a) using online network
        #   - Compute TD target: r + γ * max Q(s', a') using target network
        #   - Compute loss and backpropagate to update online network
        #   - Periodically sync target network with online network
        q, loss = mario.learn()

        # ---------------------------------------------------------------------
        # 5. LOG METRICS
        # ---------------------------------------------------------------------
        logger.log_step(reward, loss, q)

        # ---------------------------------------------------------------------
        # 6. UPDATE STATE
        # ---------------------------------------------------------------------
        state = next_state

        # ---------------------------------------------------------------------
        # 7. CHECK IF EPISODE IS DONE
        # ---------------------------------------------------------------------
        if done or info['flag_get']:
            break

    # Log end of episode
    logger.log_episode()

    # Print progress every 20 episodes
    if e % 20 == 0:
        logger.record(
            episode=e,
            epsilon=mario.exploration_rate,
            step=mario.curr_step
        )

print("\nTraining complete!")
print(f"Checkpoints saved to: {save_dir}")
