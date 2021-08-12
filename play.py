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
    parser.add_argument("-d1", "--device_1", type=str,
                        help="Device to run the network on for player 1 if it's TorchPlayer. Default: auto",
                        choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("-d2", "--device_2", type=str,
                        help="Device to run the network on for player 2 if it's TorchPlayer. Default: auto",
                        choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("-mc1", "--use_monte_carlo_1", action="store_true",
                        help="Use monte carlo tree search on player 1 if it's a TorchPlayer")
    parser.add_argument("-mc2", "--use_monte_carlo_2", action="store_true",
                        help="Use monte carlo tree search on player 2 if it's a TorchPlayer")
    parser.add_argument("--mcts_depth_level", type=int,
                        help="Number of levels of depth to explore on the tree for monte carlo. Default: 1",
                        default=1)
    parser.add_argument("--mcts_n_processes", type=int,
                        help="Number of processes to use for monte carlo. Default: 1",
                        default=1)
    parser.add_argument("-g", "--games", type=int, help="Number of games to play. Default: 500", default=500)
    parser.add_argument("-s", "--stats", action="store_true", help="Show player stats")

    return parser.parse_args()


def print_args(args):
    print(f"""
Playing games with the following parameters:
Number of games: {args.games}
Board shape: {args.board_shape} x {args.board_shape}
Player 1:
    * Class: {args.player1}
    * Model path: {args.file_path_1 or "None"}
    * Device: {args.device_1}
    * Using monte carlo: {"YES" if args.use_monte_carlo_1 else "NO"}
    * Monte carlo's depth level: {args.mcts_depth_level if args.use_monte_carlo_1 else "DOESN'T APPLY"}
    * Monte carlo's Number of processes: {args.mcts_n_processes if args.use_monte_carlo_1 else "DOESN'T APPLY"}
Player 2:
    * Class: {args.player2}
    * Model path: {args.file_path_2 or "None"}
    * Device: {args.device_2}
    * Using monte carlo: {"YES" if args.use_monte_carlo_2 else "NO"}
    * Monte carlo's depth level: {args.mcts_depth_level if args.use_monte_carlo_2 else "DOESN'T APPLY"}
    * Monte carlo's Number of processes: {args.mcts_n_processes if args.use_monte_carlo_2 else "DOESN'T APPLY"}
    
""")


def play(args):
    print_args(args)
    env = ReversiEnv(board_shape=args.board_shape)
    player_1 = players[args.player1](player=1, env=env, model_path=args.file_path_1, device=args.device_1,
                                     mcts=args.use_monte_carlo_1, levelLimit=args.mcts_depth_level,
                                     mtcs_n_processes=args.mcts_n_processes)
    player_2 = players[args.player2](player=-1, env=env, model_path=args.file_path_2, device=args.device_2,
                                     mcts=args.use_monte_carlo_2, levelLimit=args.mcts_depth_level,
                                     mtcs_n_processes=args.mcts_n_processes)
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
