import torch
import random, numpy as np
from pathlib import Path

from neural import MarioNet
from collections import deque


class Mario:
    """
    Deep Q-Network (DQN) Agent for playing Super Mario Bros

    This agent learns to play Mario through Deep Reinforcement Learning.
    Key components:
    - act(): Choose actions (epsilon-greedy exploration/exploitation)
    - cache(): Store experiences in replay buffer
    - recall(): Sample random batch of experiences
    - learn(): Update Q-network by minimizing TD error (MAIN LEARNING METHOD)
    - td_estimate(): Compute current Q-value predictions
    - td_target(): Compute target Q-values using Bellman equation
    - update_Q_online(): Backpropagate loss and update network weights
    """

    def __init__(self, state_dim, action_dim, save_dir, checkpoint=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=150000)
        self.batch_size = 512

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.9999995
        self.exploration_rate_min = 0.1
        self.gamma = 0.99

        self.curr_step = 0
        self.burnin = 1e5
        self.learn_every = 1
        self.sync_every = 1e4

        self.save_every = 5e5
        self.save_dir = save_dir

        # Device selection: CUDA > MPS > CPU
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.use_cuda = True  # Keep for backward compatibility
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
            self.use_cuda = False
        else:
            self.device = torch.device('cpu')
            self.use_cuda = False

        print(f"Using device: {self.device}")

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
            print(f"CUDA Version: {torch.version.cuda}")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("TF32 enabled for faster training")

        self.net = MarioNet(self.state_dim, self.action_dim).float()
        self.net = self.net.to(self.device)

        self.learning_rate = 0.0001
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.loss_fn = torch.nn.SmoothL1Loss()
        self.max_grad_norm = 10.0

        if checkpoint:
            self.load(checkpoint)

        print("\n" + "="*60)
        print("HYPERPARAMETERS")
        print("="*60)
        print(f"Replay Buffer Size:      {self.memory.maxlen:,}")
        print(f"Batch Size:              {self.batch_size}")
        print(f"Gamma (Discount):        {self.gamma}")
        print(f"Learning Rate:           {self.learning_rate}")
        print(f"Exploration Decay:       {self.exploration_rate_decay}")
        print(f"Learn Every N Steps:     {self.learn_every}")
        print(f"Sync Target Every:       {int(self.sync_every):,} steps")
        print(f"Gradient Clip Norm:      {self.max_grad_norm}")
        print(f"Burn-in Period:          {int(self.burnin):,} steps")
        print("="*60 + "\n")


    def act(self, state):
        """
        Given a state, choose an epsilon-greedy action and update value of step.

        Inputs:
        state(LazyFrame): A single observation of the current state, dimension is (state_dim)
        Outputs:
        action_idx (int): An integer representing which action Mario will perform
        """
        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        # EXPLOIT
        else:
            state_array = np.array(state)
            state_tensor = torch.from_numpy(state_array).float().to(self.device)
            state_tensor = state_tensor.unsqueeze(0)
            action_values = self.net(state_tensor, model='online')
            action_idx = torch.argmax(action_values, axis=1).item()

        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        self.curr_step += 1
        return action_idx

    def cache(self, state, next_state, action, reward, done):
        """
        Store the experience to self.memory (replay buffer)

        Inputs:
        state (LazyFrame),
        next_state (LazyFrame),
        action (int),
        reward (float),
        done(bool))
        """
        state_array = np.array(state)
        next_state_array = np.array(next_state)

        state = torch.from_numpy(state_array).float().to(self.device)
        next_state = torch.from_numpy(next_state_array).float().to(self.device)
        action = torch.tensor([action], dtype=torch.long, device=self.device)
        reward = torch.tensor([reward], dtype=torch.float, device=self.device)
        done = torch.tensor([done], dtype=torch.bool, device=self.device)

        self.memory.append( (state, next_state, action, reward, done,) )


    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()


    def td_estimate(self, state, action):
        current_Q = self.net(state, model='online')[np.arange(0, self.batch_size), action]
        return current_Q


    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model='online')
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model='target')[np.arange(0, self.batch_size), best_action]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()


    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return loss.item()


    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())


    def learn(self):
        """
        *** THIS IS WHERE LEARNING HAPPENS! ***

        This method implements the core Deep Q-Learning algorithm (Double DQN).
        It is called after every step in the game to gradually improve Mario's policy.

        The learning process:
        1. Periodically sync target network (for stable learning)
        2. Sample a batch of past experiences (Experience Replay)
        3. Compute TD estimate: Q(s, a) - what we currently predict
        4. Compute TD target: r + γ·max Q(s', a') - what we should predict
        5. Minimize the difference (loss) between estimate and target
        6. Backpropagate gradients to update the neural network weights

        This is the heart of Reinforcement Learning - learning from experience!
        """

        # Sync target network periodically for stable learning
        # Target network provides stable Q-value targets
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        # Save model checkpoint periodically
        if self.curr_step % self.save_every == 0:
            self.save()

        # Don't learn until we have enough experiences in memory (burnin period)
        # This ensures we have diverse experiences to learn from
        if self.curr_step < self.burnin:
            return None, None

        # Only learn every N steps (not every single step)
        # This is more computationally efficient
        if self.curr_step % self.learn_every != 0:
            return None, None

        # === STEP 1: Sample a batch of experiences from memory ===
        # Experience Replay: Learn from past experiences randomly
        # This breaks correlation between consecutive experiences
        state, next_state, action, reward, done = self.recall()

        # === STEP 2: Compute TD Estimate ===
        # TD = Temporal Difference
        # Estimate Q(s, a) using our online network
        td_est = self.td_estimate(state, action)

        # === STEP 3: Compute TD Target ===
        # Target = r + γ·max Q(s', a')
        # This is what the Q-value should be according to the Bellman equation
        td_tgt = self.td_target(reward, next_state, done)

        # === STEP 4: Update the Q-network ===
        # Minimize loss = (TD estimate - TD target)²
        # This backpropagates gradients and updates network weights
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)


    def save(self):
        # Save with step number
        save_path = self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt"

        # Also save to a fixed "latest" checkpoint for easy resuming
        latest_path = self.save_dir / "mario_net_latest.chkpt"

        checkpoint_data = dict(
            model=self.net.state_dict(),
            exploration_rate=self.exploration_rate,
            exploration_rate_decay=self.exploration_rate_decay,
            optimizer=self.optimizer.state_dict(),
            curr_step=self.curr_step,
            learning_rate=self.learning_rate
        )

        torch.save(checkpoint_data, save_path)
        torch.save(checkpoint_data, latest_path)
        print(f"MarioNet saved to {save_path} at step {self.curr_step}")


    def load(self, load_path):
        if not load_path.exists():
            raise ValueError(f"{load_path} does not exist")

        ckp = torch.load(load_path, map_location=self.device, weights_only=False)
        state_dict = ckp.get('model')

        # Load model weights
        self.net.load_state_dict(state_dict)

        # Load training state
        self.exploration_rate = ckp.get('exploration_rate', 1.0)
        self.exploration_rate_decay = ckp.get('exploration_rate_decay', 0.99999975)
        self.curr_step = ckp.get('curr_step', 0)

        # Load optimizer state if available
        if 'optimizer' in ckp:
            self.optimizer.load_state_dict(ckp['optimizer'])
            print(f"Loaded optimizer state")

        # Load learning rate if available
        if 'learning_rate' in ckp:
            self.learning_rate = ckp['learning_rate']
            # Update optimizer learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate

        print(f"Loading model at {load_path}")
        print(f"  Exploration rate: {self.exploration_rate:.6f}")
        print(f"  Exploration decay: {self.exploration_rate_decay}")
        print(f"  Current step: {self.curr_step}")
        print(f"  Learning rate: {self.learning_rate}")
