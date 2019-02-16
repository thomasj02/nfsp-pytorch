from LeducPoker.NFSP.Agent import NfspAgent, collect_trajectories, CompositePolicy
from LeducPoker.NFSP.Dqn import QPolicy, QPolicyParameters, QNetwork
from LeducPoker.NFSP.Supervised import SupervisedTrainer, SupervisedTrainerParameters, SupervisedNetwork
from Device import device
from LeducPoker.PolicyWrapper import NnPolicyWrapper, infoset_to_state
import LeducPoker.Exploitability as Exploitability
from LeducPoker.Policies import Policy, NashPolicy
from LeducPoker.LeducPokerGame import LeducInfoset, PlayerActions
from tqdm import tqdm
from tensorboardX import SummaryWriter
import torch
import datetime
import math
from typing import Optional
import copy
import sys


import logging
logger = logging.getLogger(__name__)


def card_to_str(card: int):
    card_map = {0: "J", 1: "Q", 2: "K"}
    return card_map[card]


class StrategyLogger(object):
    def __init__(self, writer: SummaryWriter, nash_policy: Policy, text_only: bool):
        self.writer = writer
        self.nash_policy = nash_policy
        self.text_only = text_only

        self.total_error = 0
        self.state_cnt = 0

    def log_strategy(self, policy: NnPolicyWrapper, global_step: int):
        self.total_error = 0
        self.state_cnt = 0
        self._log_strategy(policy, None, global_step)
        self.total_error /= self.state_cnt

        if not self.text_only:
            self.writer.add_scalar("losses/strategy_error", self.total_error, global_step=global_step)
        logger.debug("Total strategy error: %s", self.total_error)

    def _log_strategy(
            self,
            policy: NnPolicyWrapper,
            infoset: Optional[LeducInfoset],
            global_step: int):
        def recurse(new_action):
            after_action_infoset = copy.deepcopy(infoset)
            after_action_infoset.add_action(new_action)
            self._log_strategy(policy, after_action_infoset, global_step)

        if infoset is None:
            for card in range(3):
                infoset = LeducInfoset(card, bet_sequences=[(), ()], board_card=None)
                self._log_strategy(policy, infoset, global_step)
        elif infoset.player_to_act == -1:
            for board_card in range(3):
                infoset = LeducInfoset(card=infoset.card, bet_sequences=infoset.bet_sequences, board_card=board_card)
                self._log_strategy(policy, infoset, global_step)
        elif infoset.is_terminal:
            return
        else:
            action_probs = policy.action_prob(infoset)
            nash_action_probs = self.nash_policy.action_prob(infoset)
            action_probs -= nash_action_probs

            node_name = "strategy/" + str(infoset)
            node_name = node_name.replace(":", "_")
            for action in PlayerActions.ALL_ACTIONS:
                if action == PlayerActions.FOLD and infoset.can_fold:
                    if not self.text_only:
                        self.writer.add_scalar(node_name+"/f", action_probs[action], global_step=global_step)
                    logger.debug("Epoch %s Strategy %s %s", e, node_name+"/f", action_probs[action])
                    self.total_error += abs(action_probs[action])
                    self.state_cnt += 1
                    recurse(action)
                elif action == PlayerActions.BET_RAISE and infoset.can_raise:
                    if not self.text_only:
                        self.writer.add_scalar(node_name+"/r", action_probs[action], global_step=global_step)
                    logger.debug("Epoch %s Strategy %s %s", e, node_name+"/r", action_probs[action])
                    self.total_error += abs(action_probs[action])
                    self.state_cnt += 1
                    recurse(action)
                elif action == PlayerActions.CHECK_CALL:
                    if not self.text_only:
                        self.writer.add_scalar(node_name + "/c", action_probs[action], global_step=global_step)
                    logger.debug("Epoch %s Strategy %s %s", e, node_name+"/c", action_probs[action])
                    self.total_error += abs(action_probs[action])
                    self.state_cnt += 1
                    recurse(action)


def log_qvals(
        writer: SummaryWriter,
        policy: QPolicy,
        infoset: Optional[LeducInfoset],
        global_step: int,
        text_only: bool):
    def recurse(new_action):
        after_action_infoset = copy.deepcopy(infoset)
        after_action_infoset.add_action(new_action)
        log_qvals(writer, policy, after_action_infoset, global_step, text_only)

    if infoset is None:
        for card in range(3):
            infoset = LeducInfoset(card, bet_sequences=[(), ()], board_card=None)
            log_qvals(writer, policy, infoset, global_step, text_only)
    elif infoset.player_to_act == -1:
        for board_card in range(3):
            infoset = LeducInfoset(card=infoset.card, bet_sequences=infoset.bet_sequences, board_card=board_card)
            log_qvals(writer, policy, infoset, global_step, text_only)
    elif infoset.is_terminal:
        return
    else:
        state = infoset_to_state(infoset)
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        q_vals = policy.qnetwork_local.forward(state).cpu().numpy()[0]

        node_name = "q_vals/" + str(infoset)
        node_name = node_name.replace(":", "_")

        for action in PlayerActions.ALL_ACTIONS:
            if action == PlayerActions.FOLD and infoset.can_fold:
                if not text_only:
                    writer.add_scalar(node_name+"/f", q_vals[action], global_step=global_step)
                logger.debug("Epoch %s QValue %s %s", e, node_name+"/f", q_vals[action])
                recurse(action)
            elif action == PlayerActions.BET_RAISE and infoset.can_raise:
                if not text_only:
                    writer.add_scalar(node_name+"/r", q_vals[action], global_step=global_step)
                logger.debug("Epoch %s QValue %s %s", e, node_name+"/r", q_vals[action])
                recurse(action)
            elif action == PlayerActions.CHECK_CALL:
                if not text_only:
                    writer.add_scalar(node_name + "/c", q_vals[action], global_step=global_step)
                logger.debug("Epoch %s QValue %s %s", e, node_name+"/c", q_vals[action])
                recurse(action)


def make_agent(q_policy_parameters, supervised_trainer_parameters, nu):
    network_units = [64]
    state_size = infoset_to_state(LeducInfoset(card=0, bet_sequences=[(), ()], board_card=None)).shape[0]
    q_network_local = QNetwork(state_size=state_size, action_size=3, hidden_units=network_units).to(device)
    #q_network_target = QNetwork(state_size=state_size, action_size=3, hidden_units=network_units).to(device)
    q_network_target = None

    q_policy = QPolicy(
        nn_local=q_network_local,
        nn_target=q_network_target,
        parameters=q_policy_parameters)

    supervised_network = SupervisedNetwork(state_size=state_size, action_size=3, hidden_units=network_units).to(device)
    supervised_trainer = SupervisedTrainer(
        supervised_trainer_parameters=supervised_trainer_parameters, network=supervised_network)

    return NfspAgent(q_policy=q_policy, supervised_trainer=supervised_trainer, nu=nu)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train NFSP')
    parser.add_argument('-t', '--tag', help="Tensorboard directory name tag", default="")
    parser.add_argument('-l', '--logdir', help="Tensorboard log directory", default="./log_dir")
    parser.add_argument('--tau', help="Q network tau", default=0.03, type=float)
    parser.add_argument('--gamma', help="Q network gamma", default=0.99, type=float)
    parser.add_argument('--log_q', help="Log q values in tensorboard", action="store_true")
    parser.add_argument('--log_strategy', help="Log strategy values in tensorboard", action="store_true")
    parser.add_argument('--log_text_only', help="Log strategy and Q only to logfile", action="store_true")
    parser.add_argument('-e', '--epochs', help="Number of epochs", type=int, default=10000)
    _args = parser.parse_args()

    import os
    try:
        os.mkdir(_args.logdir)
    except FileExistsError:
        pass

    _stdout_handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)s - %(message)s',
        handlers=[
            logging.FileHandler(
                _args.logdir + '/LeducTrainer.%s.log' % datetime.datetime.now().strftime("%Y-%m-%d.%H:%M:%S")),
            _stdout_handler])
    _stdout_handler.setLevel(logging.INFO)

    logger.info("Starting run with args: %s", _args)

    if len(_args.tag) > 0:
        _writer = SummaryWriter(log_dir=_args.logdir + "/" + _args.tag + "_" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    else:
        _writer = SummaryWriter(log_dir=_args.logdir + "/" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    _initial_episilon = 0.06
    _q_policy_parameters = QPolicyParameters(
        buffer_size=200_000,
        batch_size=128,
        gamma=_args.gamma,
        tau=_args.tau,
        epsilon=_initial_episilon,
        learning_rate=0.1)

    _supervised_trainer_parameters = SupervisedTrainerParameters(
        buffer_size=2_000_000,
        batch_size=128,
        learning_rate=0.005
    )

    _nu = 0.1
    # _single_agent = make_agent(_q_policy_parameters, _supervised_trainer_parameters, _nu)
    _nfsp_agents = [
        make_agent(_q_policy_parameters, _supervised_trainer_parameters, _nu),
        make_agent(_q_policy_parameters, _supervised_trainer_parameters, _nu)
        # _single_agent,
        # _single_agent
    ]
    _composite_supervised_policy = CompositePolicy([agent.leduc_supervised_policy for agent in _nfsp_agents])

    [agent.leduc_supervised_policy.network.eval() for agent in _nfsp_agents]
    logger.info("P0 Init exploitability: %s", Exploitability.get_exploitability(_nfsp_agents[0].leduc_supervised_policy))
    logger.info("P1 Init exploitability: %s", Exploitability.get_exploitability(_nfsp_agents[1].leduc_supervised_policy))

    _nash_policy = NashPolicy(
        p0_strat_filename="/home/tjohnson/PycharmProjects/PokerRL/LeducPoker/fullgame_strats/strat1",
        p1_strat_filename="/home/tjohnson/PycharmProjects/PokerRL/LeducPoker/fullgame_strats/strat2")

    _strategy_logger = StrategyLogger(writer=_writer, nash_policy=_nash_policy, text_only=_args.log_text_only)
    # while any(
    #         len(a.supervised_trainer.reservoir.samples) < _supervised_trainer_parameters.batch_size
    #         for a in _nfsp_agents):
    #     collect_trajectories(_nfsp_agents, num_games=128)
    #     logger.info("128 warmup games")

    _episodes = _args.epochs

    with tqdm(range(_episodes)) as t:
        for e in t:
            logger.debug("Epoch begins: %s", e)
            collect_trajectories(_nfsp_agents, num_games=1000, max_samples=128)

            for _agent in _nfsp_agents:
                _agent.q_policy.learn(1)
                _agent.supervised_trainer.learn(1)
            # _single_agent.q_policy.learn(2)
            # _single_agent.supervised_trainer.learn(2)

            with torch.no_grad():
                [agent.leduc_supervised_policy.network.eval() for agent in _nfsp_agents]

                exploitability = Exploitability.get_exploitability(_composite_supervised_policy)
                _writer.add_scalar("exploitability/exploitability", exploitability["exploitability"], global_step=e)
                _writer.add_scalar("exploitability/p0_value", exploitability["p0_value"], global_step=e)
                _writer.add_scalar("exploitability/p1_value", exploitability["p1_value"], global_step=e)

                if _nfsp_agents[0].supervised_trainer.last_loss is not None:
                    _writer.add_scalar("losses/supervised/p0", _nfsp_agents[0].supervised_trainer.last_loss,
                                       global_step=e)
                if _nfsp_agents[1].supervised_trainer.last_loss is not None:
                    _writer.add_scalar("losses/supervised/p1", _nfsp_agents[1].supervised_trainer.last_loss,
                                   global_step=e)

                if _nfsp_agents[0].q_policy.last_loss is not None:
                    _writer.add_scalar("losses/rl/p0", _nfsp_agents[0].q_policy.last_loss,
                                       global_step=e)
                if _nfsp_agents[1].q_policy.last_loss is not None:
                    _writer.add_scalar("losses/rl/p1", _nfsp_agents[1].q_policy.last_loss,
                                   global_step=e)

                _writer.add_scalar("globals/epsilon", _q_policy_parameters.epsilon, global_step=e)
                _writer.add_scalar("globals/rl_mem_p0", len(_nfsp_agents[0].q_policy.memory), global_step=e)
                _writer.add_scalar("globals/rl_mem_p1", len(_nfsp_agents[1].q_policy.memory), global_step=e)
                _writer.add_scalar("globals/sl_mem_p0", len(_nfsp_agents[0].supervised_trainer.reservoir.samples),
                                   global_step=e)
                _writer.add_scalar("globals/sl_mem_p1", len(_nfsp_agents[1].supervised_trainer.reservoir.samples),
                                   global_step=e)

                t.set_postfix({"exploitability": exploitability, "epsilon": _q_policy_parameters.epsilon})

                logger.debug(
                    "Epoch: %s Exploitability: %s p0_value: %s p1_value: %s epsilon: %s",
                    e, exploitability["exploitability"], exploitability["p0_value"], exploitability["p1_value"],
                    _q_policy_parameters.epsilon)

                if _args.log_strategy:
                    _strategy_logger.log_strategy(policy=_composite_supervised_policy, global_step=e)

                if _args.log_q:
                    log_qvals(
                        _writer,
                        _nfsp_agents[0].q_policy,
                        infoset=None,
                        global_step=e,
                        text_only=_args.log_text_only)

            _q_policy_parameters.epsilon = _initial_episilon
            _q_policy_parameters.epsilon = _initial_episilon / (1 + 1e-2 * math.sqrt(e))

    _writer.close()
    print("Done")
