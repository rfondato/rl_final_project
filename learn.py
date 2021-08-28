import os
from argparse import ArgumentParser

from custom_features_extractor import CustomBoardExtractor, CNNFeaturesExtractor
from players import RandomPlayer, GreedyPlayer, TorchPlayer
from reversi_model import CustomReversiModel
from print_utils import print_bold, print_green, print_cyan

players = {
    "RandomPlayer": RandomPlayer,
    "GreedyPlayer": GreedyPlayer,
    "TorchPlayer": TorchPlayer
}


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("-t", "--timesteps", type=int,
                        help="Amount of timesteps to train the model. Default: 100000", default=10e4)
    parser.add_argument("-m", "--model_path", type=str,
                        help="Path of a saved model to continue training. If omitted it will train a new model.")
    parser.add_argument("-b", "--board_shape", type=int,
                        help="Shape of board to train on (b x b). Default: 8", default=8)
    parser.add_argument("-e", "--envs", type=int,
                        help="Amount of parallel envs for training and evaluating. Default: 8", default=8)
    parser.add_argument("-o", "--opponent", type=str,
                        help="Class of Opponent to train versus, or 'multiple' to train vs different opponents. "
                             "If 'multiple' is chosen, each parallel env instance will create a random opponent between "
                             "RandomPlayer, GreedyPlayer and models (TorchPlayer) specified in the opp_path folder. "
                             "Default: RandomPlayer",
                        default="RandomPlayer", choices=["RandomPlayer", "GreedyPlayer", "TorchPlayer", "multiple"])
    parser.add_argument("--opp_path", type=str,
                        help="Path of saved model file for opponent if it's TorchPlayer, or "
                             "path of folder containing various models to pick if opponent is 'multiple'.")
    parser.add_argument("--opp_device", type=str, default="auto",
                        help="Device where to run the opponent if it's a TorchPlayer (cpu, cuda, auto). Default: auto")
    parser.add_argument("-n", "--num_steps", type=int,
                        help="Number of steps in buffer (epoch size). Default: 2048", default=2048)
    parser.add_argument("-ep", "--epochs", type=int,
                        help="Number of epochs. Default: 10", default=10)
    parser.add_argument("-bs", "--batch_size", type=int,
                        help="Batch size. Default: 64", default=64)
    parser.add_argument("-lr", "--learning_rate", type=float,
                        help="Learning Rate. Default: 2e-4", default=2e-4)
    parser.add_argument("-fe", "--features_extractor", type=str, choices=["MLP", "CNN"], default="MLP",
                        help="Features extractor: CNN or MLP. Default: MLP")
    parser.add_argument("-nn", "--num_neurons", type=int, default=32,
                        help="Number of neurons on each layer of the policy and value networks. Default: 32.")
    parser.add_argument("-nl", "--num_layers", type=int, default=2,
                        help="Number of layers of the policy and value networks. Default: 2.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose mode")

    return parser.parse_args()


def print_args(args):
    print(f"""
Training the model with the following parameters:
Time steps: {args.timesteps}
Model's path: {args.model_path if args.model_path is not None else print_bold(print_green("<NEW MODEL>"))}
Board shape: {args.board_shape} x {args.board_shape}
Number of envs: {args.envs}

Opponent:
    * Class: {args.opponent}
    * Path: {args.opp_path or "None"}
    * Device: {args.opp_device}
    
Num of steps per epoch: {args.num_steps}
Number of epochs: {args.epochs}
Batch size: {args.batch_size}
Learning rate: {args.learning_rate}
Number of neurons: {args.num_neurons}
Number of layers: {args.num_layers}
Features extractor: {args.features_extractor}

""")


def learn(args):
    print_args(args)
    features_extractor = CNNFeaturesExtractor if (args.features_extractor == "CNN") else CustomBoardExtractor
    nn = [args.num_neurons for _ in range(args.num_layers)]
    net_arch = [dict(pi=nn, vf=nn)]
    model = CustomReversiModel(board_shape=args.board_shape,
                               n_envs=args.envs,
                               local_player=players[args.opponent] if args.opponent in players else args.opponent,
                               n_steps=args.num_steps,
                               n_epochs=args.epochs,
                               learning_rate=args.learning_rate,
                               ent_coef=0.001,
                               gae_lambda=0.9,
                               batch_size=args.batch_size,
                               net_arch=net_arch,
                               load_from_path=args.model_path,
                               use_previous_saved_params=False,
                               features_extractor=features_extractor,
                               path_local_player=args.opp_path,
                               device_local_player=args.opp_device,
                               verbose=args.verbose)
    print()
    print(print_cyan(f"New model will be saved in the following path: {os.path.abspath(model.get_new_model_save_path())}"))
    print()
    model.learn(total_timesteps=args.timesteps)


def validate_args(args):
    if args.opponent == "TorchPlayer" and not args.opp_path:
        print(
            "Param --opp_path needed when opponent is of type TorchPlayer. "
            "Run 'python learn.py -h' for more information.")
        exit(1)


if __name__ == "__main__":
    args = parse_arguments()
    validate_args(args)
    learn(args)
