import torch
import torch.optim as optim
import torch.nn as nn
from Device import device
from LeducPoker.NFSP.ReservoirSampling import Reservoir
import numpy as np
from LeducPoker.Policies import Policy
from LeducPoker.PolicyWrapper import infoset_to_state
from LeducPoker.LeducPokerGame import LeducInfoset
from typing import List
from torchsummary import summary


class SupervisedNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_units: List[int]):
        super(SupervisedNetwork, self).__init__()

        self.state_size = state_size
        self.action_size = action_size

        input_size = state_size
        self.layers = []
        for layer_hidden_units in hidden_units:
            self.layers.append(nn.Linear(input_size, layer_hidden_units))
            self.layers.append(nn.LeakyReLU())
            input_size = layer_hidden_units

        if action_size <= 2:
            final_units = 1
        else:
            final_units = action_size
        self.layers.append(nn.Linear(input_size, final_units))
        self.layers.append(nn.LeakyReLU())
        self.layers = nn.ModuleList(self.layers)

        self.softmax = nn.Softmax(dim=1)

        # for layer in self.layers:
        #     torch.nn.init.orthogonal_(layer[0].weight, torch.nn.init.calculate_gain('relu'))
        print("SupervisedNetwork:")
        print(self)
        summary(self, (self.state_size,), device=device)


    def forward(self, state):
        """Build a network that maps state -> action values."""

        for layer in self.layers:
            state = layer(state)

        return state


class SupervisedPolicy(Policy):
    def __init__(self, network: SupervisedNetwork):
        self.network = network

    def action_prob(self, infoset: LeducInfoset):
        state = infoset_to_state(infoset)
        state = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(device)
        nn_retval = self.network.forward(state).cpu().detach()
        nn_retval = self.network.softmax(nn_retval)
        retval = nn_retval.cpu().detach().numpy()[0]

        if not infoset.can_fold:
            retval[0] = 0
        if not infoset.can_raise:
            retval[2] = 0
        retval /= retval.sum()

        return retval


class SupervisedTrainerParameters(object):
    def __init__(self, buffer_size: int, batch_size: int, learning_rate: float):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate


class SupervisedTrainer(object):
    def __init__(self, supervised_trainer_parameters: SupervisedTrainerParameters, network: SupervisedNetwork):
        self.parameters = supervised_trainer_parameters
        self.reservoir = Reservoir(self.parameters.buffer_size)
        self.network = network.to(device)

        # self.optimizer = optim.Adam(self.network.parameters(), lr=self.parameters.learning_rate)
        self.optimizer = optim.SGD(self.network.parameters(), lr=self.parameters.learning_rate)
        self.loss_fn = nn.CrossEntropyLoss()
        # self.loss_fn = nn.BCELoss()
        self.last_loss = None

    def add_observation(self, state, action):
        self.reservoir.add_sample((state, action))

    def learn(self, epochs):
        self.network.train()

        for _ in range(epochs):
            self.optimizer.zero_grad()

            samples = self.reservoir.sample(self.parameters.batch_size)

            inputs = np.array([s[0] for s in samples])
            targets = np.array([s[1] for s in samples])

            inputs = torch.from_numpy(inputs).float().to(device)
            targets = torch.from_numpy(targets).long().to(device)

            outputs = self.network.forward(inputs)
            loss = self.loss_fn(outputs, targets)
            self.last_loss = loss
            loss.backward()
            self.optimizer.step()

            self.last_loss = loss.float()


