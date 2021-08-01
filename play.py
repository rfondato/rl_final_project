from argparse import ArgumentParser

from boardgame2 import ReversiEnv

from arena import Arena
from players import RandomPlayer, GreedyPlayer, TorchPlayer

players = {
    "RandomPlayer": RandomPlayer,
    "GreedyPlayer": GreedyPlayer,
    "TorchPlayer": TorchPlayer
}


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("-p1", "--player1", type=str,
                        help="Class of Player 1. Default: RandomPlayer",
                        default="RandomPlayer", choices=["RandomPlayer", "GreedyPlayer", "TorchPlayer"])
    parser.add_argument("-p2", "--player2", type=str,
                        help="Class of Player 2. Default: RandomPlayer",
                        default="RandomPlayer", choices=["RandomPlayer", "GreedyPlayer", "TorchPlayer"])
    parser.add_argument("-b", "--board_shape", type=int,
                        help="Shape of board to play on (b x b). Default: 8", default=8)
    parser.add_argument("-f1", "--file_path_1", type=str,
                        help="Path of saved model for player 1 if it's TorchPlayer")
    parser.add_argument("-f2", "--file_path_2", type=str,
                        help="Path of saved model for player 2 if it's TorchPlayer")
    parser.add_argument("-g", "--games", type=int, help="Number of games to play. Default: 500", default=500)
    parser.add_argument("-s", "--stats", action="store_true", help="Show player stats")

    return parser.parse_args()


def play(args):
    print(f"Playing {args.games} games...")
    env = ReversiEnv(board_shape=args.board_shape)
    player_1 = players[args.player1](player=1, env=env, model_path=args.file_path_1)
    player_2 = players[args.player2](player=-1, env=env, model_path=args.file_path_2)
    arena = Arena(player_1, player_2, env, verbose=True)
    arena.play(args.games)
    if args.stats:
        arena.print_players_stats()


def validate_args(args):
    if args.player1 == "TorchPlayer" and not args.file_path_1:
        print(
            "Param -f1 (--file_path_1) needed when Player 1 is of type TorchPlayer. Run play -h for more information.")
        exit(1)
    if args.player2 == "TorchPlayer" and not args.file_path_2:
        print(
            "Param -f2 (--file_path_2) needed when Player 2 is of type TorchPlayer. Run play -h for more information.")
        exit(1)


if __name__ == "__main__":
    args = parse_arguments()
    validate_args(args)
    play(args)
