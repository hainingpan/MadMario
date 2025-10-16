# MadMario - Deep Q-Learning for Super Mario Bros

Train an AI agent to play Super Mario Bros using Deep Q-Learning (DQN).

**Note:** This repository is adapted from the [PyTorch official MadMario tutorial](https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html) for learning purposes only.

## Quick Start

**Requirements:** Python 3.8-3.11 (nes_py not compatible with Python 3.12+)

If using Python 3.12+, create a new environment:
```bash
mamba create -n mario python=3.11
mamba activate mario
```

```bash
# Install dependencies
pip install -r requirements.txt

# Train (40,000 episodes, ~20 hours on GPU)
python train.py

# Quick demo (500 episodes, ~1 hour)
python train_demo.py

# Watch trained model play
python watch.py
```

## Key Features

**Automatic Resume** - Training automatically resumes from the latest checkpoint. Just run `python train.py` again after interrupting.

**Device Support** - Auto-detects best device: CUDA > MPS > CPU

**Full State Saving** - Checkpoints include optimizer state, exploration rate, and step counter

## Project Structure

```
├── agent.py        # DQN agent (learning algorithm)
├── neural.py       # CNN Q-network architecture
├── wrappers.py     # Environment preprocessing
├── metrics.py      # Training metrics & logging
├── train.py        # Main training script
└── watch.py        # Watch trained agent play
```

## Training Details

**Hyperparameters:**
- Episodes: 40,000
- Replay buffer: 100,000
- Batch size: 32
- Learning rate: 0.00025
- Discount factor γ: 0.9
- Exploration: ε = 1.0 → 0.1

**Checkpoints:** Auto-saved every 500,000 steps to `checkpoints/`

**Training Time:** ~20 hours (GPU) or ~80 hours (CPU)

## How It Works

The DQN algorithm in [agent.py](agent.py):
1. Sample random batch from replay buffer
2. Estimate Q(s,a) using online network
3. Compute target: r + γ·max Q(s',a') using target network
4. Minimize loss via backpropagation
5. Periodically sync online → target network

## Resources

- [Original PyTorch MadMario Tutorial](https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html)
- [DQN Paper](https://www.nature.com/articles/nature14236) - Mnih et al., Nature 2015
- [Double DQN](https://arxiv.org/abs/1509.06461) - van Hasselt et al., 2015
- [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/)

## License

For educational purposes only. Original tutorial and code by PyTorch.

