import KuhnPoker.Policies
import torch.nn as nn
import KuhnPoker.KuhnPokerGame
from KuhnPoker.Device import device
import torch
import numpy as np
from typing import Optional


def infoset_to_state(infoset: Optional[KuhnPoker.KuhnPokerGame.KuhnInfoset]) -> np.ndarray:
    state = np.zeros(3 + 2 * 2)  # 7 total

    if infoset is None:
        return state

    # 3 one-hots for card
    state[infoset.card] = 1

    # 2 rounds, 2 one-hot each
    # Rounds are encoded as one of [check/call; bet/raise]
    if len(infoset.bet_sequence) >= 1:
        if infoset.bet_sequence[0] == 0:
            state[3] = 1
        else:
            state[4] = 1
    if len(infoset.bet_sequence) == 2:
        if infoset.bet_sequence[1] == 0:
            state[5] = 1
        else:
            state[6] = 1

    if len(infoset.bet_sequence) > 2:
        raise RuntimeError

    return state


class NnPolicyWrapper(KuhnPoker.Policies.Policy):
    def __init__(self, nn_policy: nn.Module):
        self.nn_policy = nn_policy

    def aggressive_action_prob(self, infoset: KuhnPoker.KuhnPokerGame.KuhnInfoset):
        state = infoset_to_state(infoset)
        state = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(device)
        retval = self.nn_policy.forward(state)
        return retval.cpu().detach().numpy()[0]

