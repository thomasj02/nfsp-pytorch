import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import KuhnPoker.KuhnPokerGame as KuhnPokerGame
import KuhnPoker.Policies as Policies
import KuhnPoker.Exploitability as Exploitability
import numpy as np
import random
from typing import List
from tqdm import tqdm
from tensorboardX import SummaryWriter


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("Using GPU :-)")
    device = torch.device("cuda:0")
else:
    print("Using CPU :-(")
    device = torch.device("cpu")


def infoset_to_state(infoset: KuhnPokerGame.KuhnInfoset) -> np.ndarray:
    state = np.zeros(3 + 2 * 3)  # 9 total

    # 3 one-hots for card
    state[infoset.card] = 1

    # 2 rounds, 3 one-hot each
    # Rounds are encoded as one of [didn't happen yet; check/call; bet/raise]
    if len(infoset.bet_sequence) >= 1:
        if infoset.bet_sequence[0] == 0:
            state[4] = 1
        else:
            state[5] = 1
    else:
        state[3] = 1
    if len(infoset.bet_sequence) == 2:
        if infoset.bet_sequence[1] == 0:
            state[7] = 1
        else:
            state[8] = 1
    else:
        state[6] = 1

    return state


class PlayerTrajectories(object):
    def __init__(self):
        self.states = []
        self.rewards = []
        self.probs = []
        self.actions = []

    def add_transition(self, state, action, prob, reward):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(prob)
        self.rewards.append(reward)


def collect_trajectories(policy: Policies.Policy, num_games: int):
    player_trajectories = [PlayerTrajectories(), PlayerTrajectories()]

    for _ in range(num_games):
        game = KuhnPokerGame.KuhnPokerGame()

        while not game.game_state.is_terminal:
            player_to_act = game.game_state.player_to_act
            infoset = game.game_state.infosets[player_to_act]
            infoset_state = infoset_to_state(infoset)
            infoset_state = torch.from_numpy(np.array(infoset_state)).float().to(device)
            aggressive_action_prob = policy.forward(infoset_state).cpu().detach()
            state = infoset_to_state(infoset)
            # Manually calculate the action so we don't have to re-evaluate the infoset
            action = int(random.random() < aggressive_action_prob.numpy()[0])

            new_bet_sequence = game.game_state.bet_sequence + (action,)
            game.game_state.bet_sequence = new_bet_sequence
            if game.game_state.is_terminal:
                game_rewards = game.game_state.get_payoffs()
            else:
                game_rewards = 0, 0

            player_trajectories[player_to_act].add_transition(
                state, action, aggressive_action_prob, game_rewards[player_to_act])

            if game.game_state.is_terminal:
                other_player = (player_to_act + 1) % 2
                player_trajectories[other_player].rewards[-1] = game_rewards[other_player]

    return player_trajectories


class Policy(nn.Module):
    def __init__(self, state_size: int):
        super(Policy, self).__init__()

        # two fully connected layer
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

        self.lr = nn.LeakyReLU()

        self.sig = nn.Sigmoid()

        torch.nn.init.orthogonal_(self.fc1.weight, torch.nn.init.calculate_gain('relu'))
        torch.nn.init.orthogonal_(self.fc2.weight, torch.nn.init.calculate_gain('relu'))

    def forward(self, x):
        x = self.lr(self.fc1(x))
        x = self.lr(self.fc2(x))
        x = self.lr(self.fc3(x))
        return self.sig(x)


class NnPolicyWrapper(Policies.Policy):
    def __init__(self, nn_policy: Policy):
        self.nn_policy = nn_policy

    def aggressive_action_prob(self, infoset: KuhnPokerGame.KuhnInfoset):
        state = infoset_to_state(infoset)
        state = torch.from_numpy(np.array(state)).float().to(device)
        retval = self.nn_policy.forward(state)
        return retval.cpu().detach().numpy()[0]


# convert states to probability, passing through the policy
def states_to_prob(policy, states):
    states = torch.stack(states)
    #policy_input = states.view(-1,*states.shape[-3:])
    #return policy(policy_input).view(states.shape[:-3])
    retval = policy(states)
    return retval.view(retval.numel())


def clipped_surrogate(policy: Policy, player_trajectories: List[PlayerTrajectories], discount=0.995, epsilon=0.1, beta=0.01):
    clipped_surrogates = torch.Tensor().to(device)

    for player_trajectory in player_trajectories:
        rewards = player_trajectory.rewards
        old_probs = player_trajectory.probs
        states = player_trajectory.states

        reward_discounts = discount ** np.arange(len(rewards))
        rewards = np.asarray(rewards) * reward_discounts[:, np.newaxis]

        # convert rewards to future rewards
        rewards_future = rewards[::-1].cumsum(axis=0)[::-1]

        mean = np.mean(rewards_future, axis=1)
        std = np.std(rewards_future, axis=1) + 1.0e-10

        rewards_normalized = (rewards_future - mean[:, np.newaxis]) / std[:, np.newaxis]

        # convert everything into pytorch tensors and move to gpu if available
        old_probs = torch.tensor(old_probs, dtype=torch.float, device=device)
        rewards = torch.tensor(rewards_normalized, dtype=torch.float, device=device)

        # convert states to policy (or probability)
        new_probs = states_to_prob(policy, tuple(torch.from_numpy(s).to(device, dtype=torch.float32) for s in states))

        # ratio for clipping
        ratio = new_probs / old_probs

        # clipped function
        clip = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
        clipped_surrogate = torch.min(ratio * rewards, clip * rewards)

        # include a regularization term
        # this steers new_policy towards 0.5
        # add in 1.e-10 to avoid log(0) which gives nan
        entropy = -(new_probs * torch.log(old_probs + 1.e-10) + (1.0 - new_probs) * torch.log(1.0 - old_probs + 1.e-10))

        trajectory_clipped_surrogates = clipped_surrogate + beta * entropy
        clipped_surrogates = torch.cat((clipped_surrogates, torch.flatten(trajectory_clipped_surrogates)), 0)

    # this returns an average of all the entries of the tensor
    # effective computing L_sur^clip / T
    # averaged over time-step and number of trajectories
    # this is desirable because we have normalized our rewards
    return torch.mean(clipped_surrogates)


if __name__ == "__main__":
    policy = Policy(9).to(device)
    writer = SummaryWriter(log_dir="./log_dir")
    print("Init exploitability:", Exploitability.get_exploitability(NnPolicyWrapper(policy)))
    optimizer = optim.Adam(policy.parameters(), lr=1e-4)

    episodes = 10000
    sgd_epochs = 4
    _beta = 0.
    _epsilon = 0.1

    with tqdm(range(episodes)) as t:
        for e in t:
            traj = collect_trajectories(policy, 100)

            for _ in range(sgd_epochs):
                L = -clipped_surrogate(policy, traj, epsilon=_epsilon, beta=0.01)
                optimizer.zero_grad()
                L.backward()
                optimizer.step()
                del L

            exploitability = Exploitability.get_exploitability(NnPolicyWrapper(policy))
            writer.add_scalar("exploitability", exploitability, global_step=e)
            t.set_postfix({"exploitability": exploitability})
            _beta *= 0.995
            _epsilon *= 0.999

    writer.close()