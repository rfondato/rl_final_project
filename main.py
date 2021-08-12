from argparse import ArgumentParser

from boardgame2 import ReversiEnv

from arena import Arena
from players import RandomPlayer, GreedyPlayer, TorchPlayer
from reversi_model import CustomReversiModel

players = {
    "RandomPlayer": RandomPlayer,
    "GreedyPlayer": GreedyPlayer,
    "TorchPlayer": TorchPlayer
}


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("action", help="Action to execute", choices=["learn", "play"])
    parser.add_argument("-t", "--timesteps", type=int,
                        help="[learn action only]: Amount of timesteps to train the model. Default: 10e5", default=10e5)
    parser.add_argument("-e", "--envs", type=int,
                        help="[learn action only]: Amount of parallel envs for training. Default: 8", default=8)
    parser.add_argument("-p", "--player", type=str,
                        help="[play action only]: Class of Player 1. Default: RandomPlayer",
                        default="RandomPlayer", choices=["RandomPlayer", "GreedyPlayer", "TorchPlayer"])
    parser.add_argument("-o", "--opponent", type=str,
                        help="[learn|play]: Class of Player 2 (play) / Opponent to train with (learn). Default: RandomPlayer",
                        default="RandomPlayer", choices=["RandomPlayer", "GreedyPlayer", "TorchPlayer"])
    parser.add_argument("-b", "--board_shape", type=int,
                        help="[learn action only]: Shape of board to train on (b x b). Default: 8", default=8)
    parser.add_argument("-n", "--num_steps", type=int,
                        help="[learn action only]: Number of steps in buffer (epoch size). Default: 2048", default=2048)
    parser.add_argument("-ep", "--epochs", type=int,
                        help="[learn action only]: Number of epochs. Default: 10", default=10)
    parser.add_argument("-f", "--file_path", type=str,
                        help="[learn|play]: Path of saved model to continue training (learn) / for player 1 (play) if it's TorchPlayer")
    parser.add_argument("--path_opp", type=str,
                        help="[learn|play]: Path of saved model for opponent (learn) / for player 2 (play) if it's TorchPlayer")
    parser.add_argument("-g", "--games", type=int, help="[play action only]: Number of games to play", default=500)

    return parser.parse_args()


def learn(args):
    model = CustomReversiModel(board_shape=args.board_shape,
                               n_envs=args.envs,
                               local_player=players[args.opponent],
                               n_steps=args.num_steps,
                               n_epochs=args.epochs,
                               load_from_path=args.file_path,
                               use_previous_saved_params=False)
    model.learn(total_timesteps=args.timesteps)


def play(args):
    print(f"Playing {args.games} games...")
    env = ReversiEnv(board_shape=args.board_shape)
    player_1 = players[args.player](player=1, env=env)
    player_2 = players[args.opponent](player=2, env=env)
    arena = Arena(player_1, player_2, env, verbose=True)
    arena.play(args.games)
    arena.print_players_stats()


if __name__ == "__main__":
    args = parse_arguments()
    if args.action == "learn":
        learn(args)
        exit(0)
    elif args.action == "play":
        play(args)
        exit(0)
    else:
        print(f"Invalid action: {args.action}")
        exit(1)
