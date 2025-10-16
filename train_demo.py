"""
Quick Training Demo for Mario

This is a shorter training script for demonstration purposes.
For full training (80 hours), use train.py instead.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import datetime
from pathlib import Path

import gym
import gym_super_mario_bros
from gym.wrappers import FrameStack, GrayScaleObservation
from nes_py.wrappers import JoypadSpace

from metrics import MetricLogger
from agent import Mario
from wrappers import ResizeObservation, SkipFrame


print("=" * 70)
print("🍄 Mario RL Training Demo 🍄")
print("=" * 70)
print("\nThis is a QUICK DEMO training run.")
print("For serious training, use train.py with 40,000 episodes.")
print("=" * 70 + "\n")

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================
# Create Mario environment (old gym API)
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, [['right'], ['right', 'A']])

# Apply preprocessing wrappers
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env, keep_dim=False)
env = ResizeObservation(env, shape=84)
env = FrameStack(env, num_stack=4)

env.reset()

# ============================================================================
# AGENT INITIALIZATION
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

# Initialize Mario
mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir, checkpoint=checkpoint)

logger = MetricLogger(save_dir)

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
episodes = 500  # Quick demo - just 500 episodes (vs 40,000 for full training)

print(f"Training configuration:")
print(f"  Episodes: {episodes}")
print(f"  Checkpoint directory: {save_dir}")
print(f"  Burnin period: {int(mario.burnin)} steps (learning starts after this)")
print(f"  Learn every: {mario.learn_every} steps")
print(f"  Batch size: {mario.batch_size}")
print(f"  Exploration rate: {mario.exploration_rate} → {mario.exploration_rate_min}")
print(f"  GPU available: {mario.use_cuda}")
print("\nStarting training...\n")

# ============================================================================
# TRAINING LOOP
# ============================================================================
for e in range(episodes):
    state = env.reset()  # Old gym returns just state
    episode_reward = 0
    episode_steps = 0

    # Episode loop - play one complete game
    while True:
        # 1. Agent chooses action (epsilon-greedy)
        action = mario.act(state)

        # 2. Environment responds
        next_state, reward, done, info = env.step(action)  # Old gym returns 4 values

        # 3. Store experience in replay buffer
        mario.cache(state, next_state, action, reward, done)

        # 4. LEARN - This is where the neural network is updated!
        q, loss = mario.learn()

        # 5. Log metrics
        logger.log_step(reward, loss, q)

        episode_reward += reward
        episode_steps += 1
        state = next_state

        # Check if episode is done
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

print("\n" + "=" * 70)
print("🎉 Training Complete!")
print("=" * 70)

# ============================================================================
# SAVE FINAL MODEL
# ============================================================================
final_save_path = Path('mario_trained.chkpt')
import torch
torch.save(
    dict(
        model=mario.net.state_dict(),
        exploration_rate=mario.exploration_rate,
        exploration_rate_decay=mario.exploration_rate_decay,
        optimizer=mario.optimizer.state_dict(),
        curr_step=mario.curr_step,
        learning_rate=mario.learning_rate
    ),
    final_save_path
)

print(f"\n✅ Final model saved to: {final_save_path}")
print(f"   Total steps trained: {mario.curr_step}")
print(f"   Final exploration rate: {mario.exploration_rate:.4f}")
print(f"   Device used: {mario.device}")
print(f"\nTo watch Mario play with this trained model, run:")
print(f"   python watch.py")
print("\n" + "=" * 70)
