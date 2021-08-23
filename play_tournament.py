import os
from argparse import ArgumentParser

from tournament import Tournament


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("-f", "--models_folder", type=str, default="./models/",
                        help="Path of folder where model files are located. Default: ./models/")
    parser.add_argument("-b", "--board_shape", type=int,
                        help="Shape of board to play on (b x b). Default: 8", default=8)
    parser.add_argument("-g", "--games", type=int, default=100,
                        help="Number of games to play in each match between players. Default: 100. ")
    parser.add_argument("-d", "--deterministic", action="store_true",
                        help="Run models in deterministic mode. Default is non-deterministic.")
    parser.add_argument("-dv", "--device", type=str, default="auto",
                        help="Device where players should run (auto, cuda, cpu). Default: auto")
    parser.add_argument("-n", "--n_processes", type=int,
                        help="Number of processes to use to play the tournament. Default: Number of logical cores")
    parser.add_argument("-def", "--default_players", action="store_true",
                        help="Add default Random and Greedy players to the tournament")
    parser.add_argument("--log_dir", type=str, default=None,
                        help="Directory where to store the logs of the tournament. Default is no logging.")

    return parser.parse_args()


def print_args(args):
    print(f"""
Playing a tournament with the following parameters:
Model's folder: {args.models_folder}
Board shape: {args.board_shape} x {args.board_shape}
Number of games per match: {args.games}
Deterministic mode: {"ON" if args.deterministic else "OFF"}
Device: {args.device}
Number of processes: {args.n_processes if args.n_processes is not None else os.cpu_count()}
{"Added RandomPlayer and GreedyPlayer" if args.default_players else ""}

""")


def validate_args(args):
    f = args.models_folder
    if (f is None) or (not os.path.exists(f)) or (not os.path.isdir(f)):
        print("Error: Invalid models folder")
        exit(1)


def play_tournament(args):
    print_args(args)
    t = Tournament(
        models_folder=args.models_folder,
        games_per_match=args.games,
        board_shape=args.board_shape,
        deterministic=args.deterministic,
        device=args.device,
        verbose=True,
        n_processes=args.n_processes,
        add_default_players=args.default_players,
        log_folder=args.log_dir
    )
    t.play()


if __name__ == "__main__":
    args = parse_arguments()
    validate_args(args)
    play_tournament(args)
