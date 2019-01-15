from KuhnPoker.NFSP.Agent import NfspAgent, collect_trajectories
from KuhnPoker.NFSP.Dqn import QPolicy, QPolicyParameters, KuhnQPolicy, QNetwork
from KuhnPoker.NFSP.Supervised import SupervisedTrainer, SupervisedTrainerParameters, SupervisedNetwork
from KuhnPoker.Device import device
from KuhnPoker.PolicyWrapper import NnPolicyWrapper, infoset_to_state
import KuhnPoker.Exploitability as Exploitability
from KuhnPoker.KuhnPokerGame import KuhnInfoset
from tqdm import tqdm
from tensorboardX import SummaryWriter
import torch
import datetime
import math


def card_to_str(card: int):
    card_map = {0: "J", 1: "Q", 2: "K"}
    return card_map[card]


def log_strategy(writer: SummaryWriter, policy: NnPolicyWrapper, global_step: int):
    infoset = KuhnInfoset(0, ())

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


def log_qvals(writer: SummaryWriter, policy: QPolicy, global_step: int):
    infoset = KuhnInfoset(0, ())

    for card in range(3):
        infoset.card = card

        infoset.bet_sequence = ()
        state = torch.from_numpy(infoset_to_state(infoset)).float().unsqueeze(0).to(device)
        q_vals = policy.qnetwork_local.forward(state).cpu().numpy()[0]
        node_name = "q_vals/%s/p0_open" % card_to_str(card)
        writer.add_scalar(node_name, q_vals[1] - q_vals[0], global_step=global_step)

        infoset.bet_sequence = (0,)
        state = torch.from_numpy(infoset_to_state(infoset)).float().unsqueeze(0).to(device)
        q_vals = policy.qnetwork_local.forward(state).cpu().numpy()[0]
        node_name = "q_vals/%s/p0_check/p1" % card_to_str(card)
        writer.add_scalar(node_name, q_vals[1] - q_vals[0], global_step=global_step)

        infoset.bet_sequence = (0, 1)
        state = torch.from_numpy(infoset_to_state(infoset)).float().unsqueeze(0).to(device)
        q_vals = policy.qnetwork_local.forward(state).cpu().numpy()[0]
        node_name = "q_vals/%s/p0_check/p1_bet/p0" % card_to_str(card)
        writer.add_scalar(node_name, q_vals[1] - q_vals[0], global_step=global_step)

        infoset.bet_sequence = (1,)
        state = torch.from_numpy(infoset_to_state(infoset)).float().unsqueeze(0).to(device)
        q_vals = policy.qnetwork_local.forward(state).cpu().numpy()[0]
        node_name = "q_vals/%s/p0_bet/p1" % card_to_str(card)
        writer.add_scalar(node_name, q_vals[1] - q_vals[0], global_step=global_step)


if __name__ == "__main__":
    _writer = SummaryWriter(log_dir="./log_dir/" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    initial_episilon = 0.01
    _q_policy_parameters = QPolicyParameters(
        buffer_size=2_000,
        batch_size=128,
        gamma=0.99,
        tau=0.03,
        epsilon=initial_episilon,
        learning_rate=1e-1)
    _q_network_local = QNetwork(state_size=7, action_size=2).to(device)
    _q_network_target = QNetwork(state_size=7, action_size=2).to(device)

    _q_policy = QPolicy(
        nn_local=_q_network_local,
        nn_target=_q_network_target,
        parameters=_q_policy_parameters)

    _supervised_trainer_parameters = SupervisedTrainerParameters(
        buffer_size=200_000,
        batch_size=128,
        learning_rate=0.005
    )
    _supervised_network = SupervisedNetwork(state_size=7, action_size=2).to(device)
    _supervised_trainer = SupervisedTrainer(
        supervised_trainer_parameters=_supervised_trainer_parameters, network=_supervised_network)

    _nu = 0.1
    _nfsp_agents = [
        NfspAgent(q_policy=_q_policy, supervised_trainer=_supervised_trainer, nu=_nu),
        NfspAgent(q_policy=_q_policy, supervised_trainer=_supervised_trainer, nu=_nu)
    ]

    _supervised_network.eval()
    print("Init exploitability:", Exploitability.get_exploitability(_nfsp_agents[0].kuhn_supervised_policy))
    print("Init exploitability:", Exploitability.get_exploitability(_nfsp_agents[1].kuhn_supervised_policy))

    while len(_supervised_trainer.resevoir.samples) < _supervised_trainer_parameters.batch_size:
        collect_trajectories(_nfsp_agents, num_games=128)
        print("128 warmup games")

    _episodes = 10_000
    with tqdm(range(_episodes)) as t:
        for e in t:
            collect_trajectories(_nfsp_agents, num_games=64)
            _q_policy.learn(epochs=2)
            _supervised_trainer.learn(epochs=2)

            with torch.no_grad():
                _supervised_network.eval()

                exploitability = Exploitability.get_exploitability(_nfsp_agents[0].kuhn_supervised_policy)
                _writer.add_scalar("exploitability/exploitability", exploitability["exploitability"], global_step=e)
                _writer.add_scalar("exploitability/p0_value", exploitability["p0_value"], global_step=e)
                _writer.add_scalar("exploitability/p1_value", exploitability["p1_value"], global_step=e)
                _writer.add_scalar("losses/supervised", _supervised_trainer.last_loss, global_step=e)
                t.set_postfix({"exploitability": exploitability, "epsilon": _q_policy_parameters.epsilon})


                log_strategy(_writer, _nfsp_agents[0].kuhn_supervised_policy, e)
                log_qvals(_writer, _nfsp_agents[0].q_policy, e)

            _q_policy_parameters.epsilon = initial_episilon - initial_episilon * math.sqrt(e) / math.sqrt(_episodes)

    _writer.close()
    print("Done")
