import KuhnPoker.Policies
import torch.nn as nn
import KuhnPoker.KuhnPokerGame
from KuhnPoker.Device import device
import torch
import numpy as np


def infoset_to_state(infoset: KuhnPoker.KuhnPokerGame.KuhnInfoset) -> np.ndarray:
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
        state[4] = 0.5
        state[5] = 0.5
    if len(infoset.bet_sequence) == 2:
        if infoset.bet_sequence[1] == 0:
            state[7] = 1
        else:
            state[8] = 1
    else:
        state[6] = 1
        state[7] = 0.5
        state[8] = 0.5

    return state


class NnPolicyWrapper(KuhnPoker.Policies.Policy):
    def __init__(self, nn_policy: nn.Module):
        self.nn_policy = nn_policy

    def aggressive_action_prob(self, infoset: KuhnPoker.KuhnPokerGame.KuhnInfoset):
        state = infoset_to_state(infoset)
        state = torch.from_numpy(np.array(state)).float().to(device)
        retval = self.nn_policy.forward(state)
        return retval.cpu().detach().numpy()[0]

