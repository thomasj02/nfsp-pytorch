import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from KuhnPoker.Device import device
from KuhnPoker.NFSP.ReservoirSampling import Reservoir
import numpy as np
from KuhnPoker.Policies import Policy
from KuhnPoker.PolicyWrapper import infoset_to_state
from KuhnPoker.KuhnPokerGame import KuhnInfoset


class SupervisedNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(SupervisedNetwork, self).__init__()

        self.state_size = state_size
        self.action_size = action_size

        fc1_units = 64
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc1_activation = nn.LeakyReLU()

        if action_size <= 2:
            fc2_units = 1
        else:
            fc2_units = action_size
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc2_activation = nn.LeakyReLU()

        # self.final_activation = nn.Softmax(dim=1)
        self.final_activation = nn.Sigmoid()

        torch.nn.init.orthogonal_(self.fc1.weight, torch.nn.init.calculate_gain('relu'))
        torch.nn.init.orthogonal_(self.fc2.weight, torch.nn.init.calculate_gain('relu'))

    def forward(self, state):
        """Build a network that maps state -> action values."""

        state = self.fc1_activation(self.fc1(state))
        state = self.fc2_activation(self.fc2(state))
        state = self.final_activation(state)

        return state


class SupervisedPolicy(Policy):
    def __init__(self, network: SupervisedNetwork):
        self.network = network

    def aggressive_action_prob(self, infoset: KuhnInfoset):
        state = infoset_to_state(infoset)
        state = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(device)
        nn_retval = self.network.forward(state).cpu().detach()
        retval = nn_retval.cpu().detach().numpy()[0][0]
        return retval


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

        #self.optimizer = optim.SGD(self.network.parameters(), lr=self.parameters.learning_rate)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.parameters.learning_rate)
        # self.loss_fn = nn.CrossEntropyLoss()
        self.loss_fn = nn.BCELoss()
        self.last_loss = None

    def add_observation(self, state, action):
        self.resevoir.add_sample((state, action))

    def learn(self, epochs):
        self.network.train()

        for _ in range(epochs):
            self.optimizer.zero_grad()

            samples = self.resevoir.sample(self.parameters.batch_size)

            inputs = np.array([s[0] for s in samples])
            targets = np.array([s[1] for s in samples])

            inputs = torch.from_numpy(inputs).float().to(device)
            targets = torch.from_numpy(targets).float().to(device).unsqueeze(1)

            outputs = self.network.forward(inputs)
            loss = self.loss_fn(outputs, targets)
            self.last_loss = loss
            loss.backward()
            self.optimizer.step()

            self.last_loss = loss.float()


