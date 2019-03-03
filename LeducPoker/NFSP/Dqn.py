import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from collections import deque, namedtuple
import numpy as np
import random
from Device import device
from LeducPoker.PolicyWrapper import infoset_to_state
from LeducPoker.Policies import Policy
from LeducPoker.LeducPokerGame import LeducInfoset, PlayerActions
from typing import List
from torchsummary import summary


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size,  hidden_units: List[int]):
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

        input_size = state_size
        self.layers = []
        for layer_hidden_units in hidden_units:
            layer = nn.Linear(input_size, layer_hidden_units)
            bound = 1 / np.sqrt(input_size)
            with torch.no_grad():
                layer.weight.uniform_(-bound, bound)

            self.layers.append(layer)
            self.layers.append(nn.ReLU())
            input_size = layer_hidden_units

        final_units = action_size
        self.layers.append(nn.Linear(input_size, final_units))
        self.layers = nn.ModuleList(self.layers)

        # for layer in self.layers:
        #     if layer[1] is None:
        #         torch.nn.init.orthogonal_(layer[0].weight, torch.nn.init.calculate_gain('linear'))
        #     else:
        #         torch.nn.init.orthogonal_(layer[0].weight, torch.nn.init.calculate_gain('relu'))
        print("QNetwork:")
        summary(self, (self.state_size,), device=device)

    def forward(self, state):
        """Build a network that maps state -> action values."""

        for layer in self.layers:
            state = layer(state)

        return state


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.

        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        reward *= 0.0385  # Arbitrary reward scaling from Heinrich Lua code

        e = self.experience(state.copy(), action.copy(), reward.copy(), next_state.copy(), done)
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

        if self.qnetwork_target is not None:
            self._copy_weights()
        else:
            self.qnetwork_target = self.qnetwork_local

        self.parameters = parameters

        self.memory = ReplayBuffer(
            buffer_size=self.parameters.buffer_size,
            batch_size=self.parameters.batch_size)

        # self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=parameters.learning_rate)
        self.optimizer = optim.SGD(self.qnetwork_local.parameters(), lr=parameters.learning_rate)
        self.epoch_num = 0
        self.last_loss = None

    def eval(self):
        self.qnetwork_local.eval()
        self.qnetwork_target.eval()

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

    def get_action_probs(self, state):
        action_values = self._get_action_values(state)
        valid_action_values = action_values.cpu().data.numpy()[0]
        max_val_action = np.argmax(valid_action_values)

        probs = [self.parameters.epsilon / 3] * 3
        probs[max_val_action] += 1 - self.parameters.epsilon

        return probs

    def act(self, state, greedy: bool):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """

        # Epsilon-greedy action selection
        if greedy:
            action_values = self._get_action_values(state)
            valid_action_values = action_values.cpu().data.numpy()[0]
            max_val_action = np.argmax(valid_action_values)
            retval = max_val_action
            did_greedy = True
        else:
            act_probs = self.get_action_probs(state)
            retval = np.random.choice([PlayerActions.FOLD, PlayerActions.CHECK_CALL, PlayerActions.BET_RAISE], 1,
                p=act_probs)
            did_greedy = False

        return retval, did_greedy

    def learn(self, epochs: int):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        if len(self.memory) < 1000:
            return

        self.qnetwork_target.train()  # ???
        for _ in range(epochs):
            experiences = self.memory.sample()
            states, actions, rewards, next_states, dones = experiences

            if self.qnetwork_target is not None:
                q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
            else:
                q_targets_next = self.qnetwork_local(next_states).detach().max(1)[0].unsqueeze(1)
            q_targets = rewards + (self.parameters.gamma * q_targets_next * (1 - dones))

            self.qnetwork_local.train()
            q_local = self.qnetwork_local(states)
            q_local = q_local.gather(1, actions)

            loss = F.mse_loss(q_local, q_targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.last_loss = loss.float()

            self.epoch_num += 1
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

        if not self.qnetwork_target or self.qnetwork_local == self.qnetwork_target:
            return

        if self.parameters.tau < 1:
            for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
                target_param.data.copy_(
                    self.parameters.tau * local_param.data + (1.0 - self.parameters.tau) * target_param.data)
        elif self.epoch_num % self.parameters.tau == 0:
            self._copy_weights()


class LeducQPolicy(Policy):
    def __init__(self, q_policy: QPolicy):
        self.q_policy = q_policy
        self.last_action_greedy = False

    def action_prob(self, infoset: LeducInfoset):
        state = infoset_to_state(infoset)
        return self.q_policy.get_action_probs(state)

    def get_action(self, infoset: LeducInfoset) -> PlayerActions:
        state = infoset_to_state(infoset)
        q_policy_action, self.last_action_greedy = self.q_policy.act(state, greedy=False)
        return q_policy_action
