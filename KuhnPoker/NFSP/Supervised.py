import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from KuhnPoker.Device import device
from KuhnPoker.NFSP.ReservoirSampling import Reservoir
import numpy as np


class SupervisedTrainerParameters(object):
    def __init__(self, buffer_size: int, batch_size: int, learning_rate: float):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate


class SupervisedTrainer(object):
    def __init__(self, supervised_trainer_parameters: SupervisedTrainerParameters, network: SupervisedNetwork):
        self.parameters = supervised_trainer_parameters
        self.resevoir = Reservoir(self.parameters.buffer_size)
        self.network = network.to(device)

        self.optimizer = optim.Adam(lr=self.parameters.learning_rate)
        self.loss_fn = nn.MSELoss()
        self.last_loss = None

    def add_observation(self, state, action):
        self.resevoir.add_sample((state, action))

    def learn(self, epochs):
        for _ in range(epochs):
            self.optimizer.zero_grad()

            samples = self.resevoir.sample(self.parameters.batch_size)

            inputs = np.array([s[0] for s in samples])
            targets = np.array([s[1] for s in samples])

            inputs = torch.from_numpy(inputs).float().to(device)
            targets = torch.from_numpy(targets).float().to(device)

            outputs = network.forward(inputs)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()

            self.last_loss = loss.float()


class SupervisedNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(SupervisedNetwork, self).__init__()

        fc1_units = 64
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.bn1 = nn.BatchNorm1d(fc1_units)

        fc2_units = 64
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.bn2 = nn.BatchNorm1d(fc2_units)

        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""

        state = F.relu(self.bn1(self.fc1(state)))
        state = F.relu(self.bn2(self.fc2(state)))
        state = self.fc3(state)

        return state
