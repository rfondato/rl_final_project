import os
from argparse import ArgumentParser

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
                        help="Amount of parallel envs for training. Default: 8", default=8)
    parser.add_argument("-o", "--opponent", type=str,
                        help="Class of Opponent to train versus. Default: RandomPlayer",
                        default="RandomPlayer", choices=["RandomPlayer", "GreedyPlayer", "TorchPlayer"])
    parser.add_argument("--opp_path", type=str,
                        help="Path of saved model for opponent if it's TorchPlayer")
    parser.add_argument("--opp_device", type=str, default="auto",
                        help="Device where to run the opponent if it's TorchPlayer (cpu, cuda, auto). Default: auto")
    parser.add_argument("-n", "--num_steps", type=int,
                        help="Number of steps in buffer (epoch size). Default: 2048", default=2048)
    parser.add_argument("-ep", "--epochs", type=int,
                        help="Number of epochs. Default: 10", default=10)
    parser.add_argument("-bs", "--batch_size", type=int,
                        help="Batch size. Default: 64", default=64)

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

""")


def learn(args):
    print_args(args)
    model = CustomReversiModel(board_shape=args.board_shape,
                               n_envs=args.envs,
                               local_player=players[args.opponent],
                               n_steps=args.num_steps,
                               n_epochs=args.epochs,
                               batch_size=args.batch_size,
                               load_from_path=args.model_path,
                               use_previous_saved_params=True,
                               path_local_player=args.opp_path,
                               device_local_player=args.opp_device)
    print()
    print(print_cyan(f"New model will be saved in the following path: {os.path.abspath(model.get_new_model_save_path())}"))
    print()
    model.learn(total_timesteps=args.timesteps)


def validate_args(args):
    if args.opponent == "TorchPlayer" and not args.opp_path:
        print(
            "Param --opp_path needed when opponent is of type TorchPlayer. Run learn -h for more information.")
        exit(1)


if __name__ == "__main__":
    args = parse_arguments()
    validate_args(args)
    learn(args)
