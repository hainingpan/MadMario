"""
Watch Pre-trained Mario Play

This script loads a trained Mario agent and watches it play the game.
No learning happens here - this is purely for evaluation/visualization.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import time
from pathlib import Path
import gym
import gym_super_mario_bros
from gym.wrappers import FrameStack, GrayScaleObservation
from nes_py.wrappers import JoypadSpace
from agent import Mario
from wrappers import ResizeObservation, SkipFrame
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT

print("🍄 Watch Pre-trained Mario Play 🍄\n")

# ============================================================================
# PLAYBACK SPEED CONTROL
# ============================================================================
FRAME_DELAY = 0.02  # Seconds to pause between frames
                    # 0.01 = very fast
                    # 0.03 = comfortable viewing (RECOMMENDED)
                    # 0.05 = slow, easy to see details
                    # 0.10 = very slow, good for analysis

# ============================================================================
# ENVIRONMENT SETUP (Same as training)
# ============================================================================
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, [['right'], ['right', 'A']])
# env = JoypadSpace(env, SIMPLE_MOVEMENT)
# env = JoypadSpace(env, COMPLEX_MOVEMENT)
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env, keep_dim=False)
env = ResizeObservation(env, shape=84)
env = FrameStack(env, num_stack=4)

# ============================================================================
# LOAD PRE-TRAINED MODEL
# ============================================================================
# checkpoint = Path('mario_trained.chkpt')
checkpoint = Path('mario_net_latest-1-1.chkpt')
# checkpoint = Path('mario_net_latest_optimal.chkpt')
# checkpoint = Path('mario_net_latest_full.chkpt')
# checkpoint = Path('mario_net_latest_all.chkpt')
save_dir = Path('checkpoints') / 'watch'
save_dir.mkdir(parents=True, exist_ok=True)

if not checkpoint.exists():
    print(f"❌ Error: {checkpoint} not found!")
    print("Train a model first using:")
    print("  python train.py        # Full training (40k episodes)")
    print("  python train_demo.py   # Quick demo (500 episodes)")
    exit(1)

mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir, checkpoint=checkpoint)
mario.exploration_rate=0.0
print(f"✓ Loaded pre-trained model (exploration_rate={mario.exploration_rate:.4f})")
print(f"✓ Frame delay: {FRAME_DELAY} seconds")
print(f"\nA game window should open showing Mario playing!")
print(f"Playing episodes... Close window or press Ctrl+C to stop.\n")

# ============================================================================
# PLAY EPISODES
# ============================================================================
num_episodes = 20
successful_episodes = 0

for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    steps = 0

    print(f"Episode {episode + 1}/{num_episodes}...", end=" ", flush=True)

    while True:
        # Render the game window
        env.render()

        # Let Mario decide the action (using trained Q-network)
        action = mario.act(state)

        # Perform the action
        next_state, reward, done, info = env.step(action)

        # Slow down visualization for comfortable viewing
        time.sleep(FRAME_DELAY)

        total_reward += reward
        steps += 1
        state = next_state

        # Check if Mario finished or died
        if done or info.get('flag_get', False):
            if info.get('flag_get', False):
                print(f"🎉 FLAG! x: {info['x_pos']:4d}, reward: {total_reward:7.1f}, steps: {steps:3d}")
                successful_episodes += 1
            else:
                print(f"☠️  died at x: {info.get('x_pos', 0):4d}, reward: {total_reward:7.1f}, steps: {steps:3d}")
            break

env.close()

print(f"\n✅ Finished!")
print(f"Success rate: {successful_episodes}/{num_episodes} ({successful_episodes/num_episodes*100:.0f}%)")
