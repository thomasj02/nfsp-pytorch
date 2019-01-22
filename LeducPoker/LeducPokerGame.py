from typing import Tuple, Optional, List
import random
import numpy as np


class PlayerActions:
    BET_RAISE = 2
    CHECK_CALL = 1
    FOLD = 0


class LeducNode(object):
    def __init__(self, bet_sequences: List[Tuple], board_card: Optional[int] = None):
        assert len(bet_sequences) == 2
        self.bet_sequences = bet_sequences
        self.board_card = board_card

    @property
    def is_terminal(self):
        if len(self.bet_sequences[0]) > 0 and self.bet_sequences[0][-1] == PlayerActions.FOLD:
            return True

        if len(self.bet_sequences[1]) <= 1:
            return False

        if self.bet_sequences[1][-1] != PlayerActions.BET_RAISE:
            return True

        return False

    @property
    def player_to_act(self):
        return (len(self.bet_sequences[0]) + len(self.bet_sequences[1])) % 2

    