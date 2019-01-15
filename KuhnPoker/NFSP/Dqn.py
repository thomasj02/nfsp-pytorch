import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from collections import deque, namedtuple
import numpy as np
import random
from KuhnPoker.Device import device
from KuhnPoker.PolicyWrapper import infoset_to_state
from KuhnPoker.Policies import Policy
from KuhnPoker.KuhnPokerGame import KuhnInfoset


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()

        self.state_size = state_size
        self.action_size = action_size

        fc1_units = 64
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc1_activation = nn.LeakyReLU()

        # fc2_units = 64
        # self.fc2 = nn.Linear(fc1_units, fc2_units)

        self.fc3 = nn.Linear(fc1_units, action_size)

        torch.nn.init.orthogonal_(self.fc1.weight, torch.nn.init.calculate_gain('relu'))
        # torch.nn.init.orthogonal_(self.fc2.weight, torch.nn.init.calculate_gain('relu'))
        torch.nn.init.orthogonal_(self.fc3.weight, torch.nn.init.calculate_gain('linear'))

    def forward(self, state):
        """Build a network that maps state -> action values."""

        state = self.fc1_activation(self.fc1(state))
        # state = F.relu(self.fc2(state))
        state = self.fc3(state)

        return state


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class QPolicyParameters(object):
    def __init__(
            self,
            buffer_size: int,
            batch_size: int,
            gamma: float,
            tau: float,
            epsilon: float,
            learning_rate: float
    ):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon
        self.learning_rate = learning_rate


class QPolicy(object):
    def __init__(
            self,
            nn_local: QNetwork,
            nn_target: QNetwork,
            parameters: QPolicyParameters):
        self.qnetwork_local = nn_local
        self.qnetwork_target = nn_target
        self._copy_weights()
        self.parameters = parameters

        self.memory = ReplayBuffer(
            buffer_size=self.parameters.buffer_size,
            batch_size=self.parameters.batch_size,
            seed=42
        )

        self.optimizer = optim.SGD(self.qnetwork_local.parameters(), lr=parameters.learning_rate)

    def _copy_weights(self):
        for target_param, param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(param.data)

    def add_sars(self, state, action, reward, next_state, is_terminal):
        self.memory.add(state, action, reward, next_state, is_terminal)

    def _get_action_values(self, state, network=None):
        if network is None:
            network = self.qnetwork_local

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        network.eval()
        with torch.no_grad():
            action_values = network(state)
            network.train()

        return action_values

    def act(self, state, greedy: bool):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        action_values = self._get_action_values(state)

        # Epsilon-greedy action selection
        if greedy or random.random() > self.parameters.epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            action_size = 2  # Kuhn poker
            return random.choice(np.arange(action_size))

    def learn(self, epochs: int):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        if len(self.memory) < self.parameters.batch_size:
            return

        self.qnetwork_local.train()
        self.qnetwork_target.eval()
        for _ in range(epochs):
            experiences = self.memory.sample()
            states, actions, rewards, next_states, dones = experiences

            q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
            q_targets = rewards + (self.parameters.gamma * q_targets_next * (1 - dones))

            q_local = self.qnetwork_local(states)
            q_local = q_local.gather(1, actions)

            loss = F.mse_loss(q_targets, q_local)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # ------------------- update target network ------------------- #
            self._soft_update()

    def _soft_update(self):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(
                self.parameters.tau * local_param.data + (1.0 - self.parameters.tau) * target_param.data)


class KuhnQPolicy(Policy):
    def __init__(self, q_policy: QPolicy):
        self.q_policy = q_policy

    def aggressive_action_prob(self, infoset: KuhnInfoset):
        raise RuntimeError("Q Policies don't have aggressive action probs")

    def get_action(self, infoset: KuhnInfoset):
        state = infoset_to_state(infoset)
        q_policy_action = self.q_policy.act(state, greedy=False)
        return q_policy_action


