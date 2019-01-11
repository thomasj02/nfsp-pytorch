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
from KuhnPoker.Device import device
from KuhnPoker.PolicyWrapper import infoset_to_state


class PlayerTrajectories(object):
    def __init__(self):
        self.states = []
        self.rewards = []
        self.probs = []
        self.actions = []

        self._states_in_progress = []
        self._rewards_in_progress = []
        self._probs_in_progress = []
        self._actions_in_progress = []

    def add_transition(self, state, action, prob, reward):
        self._states_in_progress.append(state)
        self._actions_in_progress.append(action)
        self._probs_in_progress.append(prob)
        self._rewards_in_progress.append(reward)

    def amend_last_reward(self, new_reward):
        self._rewards_in_progress[-1] = new_reward

    def complete_trajectory(self):
        # TODO Discount rate
        rewards = np.cumsum(self._rewards_in_progress[::-1])[::-1]
        self.rewards.extend(rewards)
        self.states.extend(self._states_in_progress)
        self.probs.extend(self._probs_in_progress)
        self.actions.extend(self._actions_in_progress)

        self._states_in_progress = []
        self._rewards_in_progress = []
        self._probs_in_progress = []
        self._actions_in_progress = []


def collect_trajectories(policy: Policies.Policy, num_games: int):
    nash_policy = Policies.NashPolicy(0)
    nash_player = 0
    player_trajectories = [PlayerTrajectories(), PlayerTrajectories()]

    for _ in range(num_games):
        game = KuhnPokerGame.KuhnPokerGame()

        while not game.game_state.is_terminal:
            player_to_act = game.game_state.player_to_act
            infoset = game.game_state.infosets[player_to_act]

            if player_to_act == nash_player:
                action = nash_policy.get_action(infoset)
            else:
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

            if player_to_act != nash_player:
                player_trajectories[player_to_act].add_transition(
                    state, action, aggressive_action_prob, game_rewards[player_to_act])

            if game.game_state.is_terminal:
                other_player = (player_to_act + 1) % 2
                if other_player != nash_player:
                    player_trajectories[other_player].amend_last_reward(game_rewards[other_player])
                player_trajectories[(nash_player + 1) % 2].complete_trajectory()

    return player_trajectories

# def collect_trajectories(policy: Policies.Policy, num_games: int):
#     player_trajectories = [PlayerTrajectories(), PlayerTrajectories()]
#
#     for _ in range(num_games):
#         game = KuhnPokerGame.KuhnPokerGame()
#
#         while not game.game_state.is_terminal:
#             player_to_act = game.game_state.player_to_act
#             infoset = game.game_state.infosets[player_to_act]
#             infoset_state = infoset_to_state(infoset)
#             infoset_state = torch.from_numpy(np.array(infoset_state)).float().to(device)
#             aggressive_action_prob = policy.forward(infoset_state).cpu().detach()
#             state = infoset_to_state(infoset)
#
#             # Manually calculate the action so we don't have to re-evaluate the infoset
#             action = int(random.random() < aggressive_action_prob.numpy()[0])
#
#             new_bet_sequence = game.game_state.bet_sequence + (action,)
#             game.game_state.bet_sequence = new_bet_sequence
#             if game.game_state.is_terminal:
#                 game_rewards = game.game_state.get_payoffs()
#             else:
#                 game_rewards = 0, 0
#
#             player_trajectories[player_to_act].add_transition(
#                 state, action, aggressive_action_prob, game_rewards[player_to_act])
#
#             if game.game_state.is_terminal:
#                 other_player = (player_to_act + 1) % 2
#                 player_trajectories[other_player].amend_last_reward(game_rewards[other_player])
#                 player_trajectories[0].complete_trajectory()
#                 player_trajectories[1].complete_trajectory()
#
#     return player_trajectories


class Policy(nn.Module):
    def __init__(self, state_size: int):
        super(Policy, self).__init__()

        # two fully connected layer
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 1)

        self.lr = nn.ReLU()

        self.final = nn.Sigmoid()

        torch.nn.init.orthogonal_(self.fc1.weight, torch.nn.init.calculate_gain('relu'))
        torch.nn.init.orthogonal_(self.fc2.weight, torch.nn.init.calculate_gain('relu'))
        torch.nn.init.orthogonal_(self.fc3.weight, torch.nn.init.calculate_gain('relu'))
        torch.nn.init.orthogonal_(self.fc4.weight, torch.nn.init.calculate_gain('sigmoid'))

    def forward(self, x):
        x = self.lr(self.fc1(x))
        x = self.lr(self.fc2(x))
        x = self.lr(self.fc3(x))
        x = self.lr(self.fc4(x))
        return self.final(x)


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


class Trainer(object):
    def __init__(self, policy: Policy, sgd_epochs: int, beta: float, epsilon: float):
        self.reward_cnt = 0
        self.reward_sum = 0.
        self.reward_sum_sq = 0.
        self.reward_mean = 0
        self.reward_stdev = 0

        self.policy = policy
        self.optimizer = optim.Adam(policy.parameters(), lr=1e-4)
        self.sgd_epochs = sgd_epochs
        self.beta = beta
        self.epsilon = epsilon

        self.epoch = 0

    def _add_rewards(self, reward_list: List[float]):
        self.reward_cnt += len(reward_list)
        self.reward_sum += sum(reward_list)
        self.reward_sum_sq += sum([r*r for r in reward_list])

        self.reward_mean = self.reward_sum / self.reward_cnt

        # https://math.stackexchange.com/a/102982/90263
        self.reward_stdev = (1.0 / self.reward_cnt) * np.sqrt(
            self.reward_cnt * self.reward_sum_sq - self.reward_sum * self.reward_sum)

    def _norm_reward(self, reward):
        return reward
        #return (reward - self.reward_mean) / self.reward_stdev

    def clipped_surrogate(self, policy: Policy, player_trajectories: List[PlayerTrajectories]):
        self._add_rewards(player_trajectories[0].rewards + player_trajectories[1].rewards)

        clipped_surrogates = torch.Tensor().to(device)

        rewards = []
        old_probs = []
        states = []

        for player_trajectory in player_trajectories:
            rewards.extend(player_trajectory.rewards)
            old_probs.extend(player_trajectory.probs)
            states.extend(player_trajectory.states)

        rewards = [self._norm_reward(r) for r in rewards]

        rewards = np.asarray(rewards)

        # convert everything into pytorch tensors and move to gpu if available
        old_probs = torch.tensor(old_probs, dtype=torch.float, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float, device=device)

        # convert states to policy (or probability)
        new_probs = states_to_prob(policy, tuple(torch.from_numpy(s).to(device, dtype=torch.float32) for s in states))

        # ratio for clipping
        ratio = new_probs / old_probs

        # clipped function
        clip = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
        clipped_surrogate = torch.min(ratio * rewards, clip * rewards)

        # include a regularization term
        # this steers new_policy towards 0.5
        # add in 1.e-10 to avoid log(0) which gives nan
        entropy = -(new_probs * torch.log(old_probs + 1.e-10) + (1.0 - new_probs) * torch.log(1.0 - old_probs + 1.e-10))

        trajectory_clipped_surrogates = clipped_surrogate + self.beta * entropy
        clipped_surrogates = torch.cat((clipped_surrogates, torch.flatten(trajectory_clipped_surrogates)), 0)

        # this returns an average of all the entries of the tensor
        # effective computing L_sur^clip / T
        # averaged over time-step and number of trajectories
        # this is desirable because we have normalized our rewards
        return torch.mean(clipped_surrogates)

    def train(self):
        traj = collect_trajectories(self.policy, 1000)

        for _ in range(self.sgd_epochs):
            L = -self.clipped_surrogate(self.policy, traj)
            self.optimizer.zero_grad()
            L.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.1)
            self.optimizer.step()
            del L

        self.beta *= 0.995
        self.epsilon *= 0.999
        self.epoch += 1


def card_to_str(card: int):
    card_map = {0: "J", 1: "Q", 2: "K"}
    return card_map[card]


def log_strategy(writer: SummaryWriter, policy: NnPolicyWrapper, global_step: int):
    infoset = KuhnPokerGame.KuhnInfoset(0, ())

    for card in range(3):
        infoset.card = card

        infoset.bet_sequence = ()
        aggressive_action_prob = policy.aggressive_action_prob(infoset)
        node_name = "strategy/%s/p0_open" % card_to_str(card)
        writer.add_scalar(node_name, aggressive_action_prob, global_step=global_step)

        infoset.bet_sequence = (0,)
        aggressive_action_prob = policy.aggressive_action_prob(infoset)
        node_name = "strategy/%s/p0_check/p1" % card_to_str(card)
        writer.add_scalar(node_name, aggressive_action_prob, global_step=global_step)

        infoset.bet_sequence = (0, 1)
        aggressive_action_prob = policy.aggressive_action_prob(infoset)
        node_name = "strategy/%s/p0_check/p1_bet/p0" % card_to_str(card)
        writer.add_scalar(node_name, aggressive_action_prob, global_step=global_step)

        infoset.bet_sequence = (1,)
        aggressive_action_prob = policy.aggressive_action_prob(infoset)
        node_name = "strategy/%s/p0_bet/p1" % card_to_str(card)
        writer.add_scalar(node_name, aggressive_action_prob, global_step=global_step)


if __name__ == "__main__":
    _policy = Policy(9).to(device)
    _writer = SummaryWriter(log_dir="./log_dir")

    _episodes = 10000

    _trainer = Trainer(_policy, sgd_epochs=4, beta=0, epsilon=0.01)
    print("Init exploitability:", Exploitability.get_exploitability(NnPolicyWrapper(_trainer.policy)))

    with tqdm(range(_episodes)) as t:
        for e in t:
            _trainer.train()

            exploitability = Exploitability.get_exploitability(NnPolicyWrapper(_trainer.policy))
            _writer.add_scalar("exploitability/exploitability", exploitability["exploitability"], global_step=e)
            _writer.add_scalar("exploitability/p0_value", exploitability["p0_value"], global_step=e)
            _writer.add_scalar("exploitability/p1_value", exploitability["p1_value"], global_step=e)
            t.set_postfix({"exploitability": exploitability})
            log_strategy(_writer, NnPolicyWrapper(_trainer.policy), e)

    _writer.close()