import os
import sys
from datetime import datetime
from multiprocessing import Pool, cpu_count
from typing import Callable, Any

import dill
from boardgame2 import ReversiEnv

from arena import Arena
from players import BasePlayer, TorchPlayer, RandomPlayer, GreedyPlayer
from print_utils import print_cyan, print_bold, print_green


def run_dill_encoded(encoded_payload):
    _f, _args = dill.loads(encoded_payload)
    return _f(*_args)


def run_async(pool: Pool, f: Callable, args: Any):
    payload = dill.dumps((f, args))
    return pool.apply_async(run_dill_encoded, (payload,))


class Match:

    def __init__(self,
                 player1: BasePlayer,
                 player2: BasePlayer,
                 games: int = 1,
                 board_shape: int = 8,
                 log_folder: str = None
                 ):
        if (log_folder is not None) and (not os.path.exists(log_folder) or not os.path.isdir(log_folder)):
            raise Exception("Invalid log folder")

        env = ReversiEnv(board_shape=board_shape)
        self.player1 = player1
        self.player2 = player2
        self.games = games
        self.player1.env = env
        self.player2.env = env
        self.log_folder = log_folder
        self.log = log_folder is not None
        self.arena = Arena(player1, player2, env, verbose=False)

    def play(self):
        self.arena.play(self.games)
        if self.log:
            self._log_match()

        return self.get_points()

    def _log_match(self):
        stats1, stats2 = self.arena.get_players_stats()
        log_file_name = f"{self.log_folder}/match_{self.player1.name}_vs_{self.player2.name}.log"
        winner = self.arena.get_winner().name if self.arena.get_winner() is not None else "TIE"
        with open(log_file_name, "w") as log_file:
            log_file.write(f"MATCH: {self.player1.name} vs {self.player2.name} \n\n")
            log_file.write(f"WINNER: {winner}\n\n")
            log_file.write(str(stats1))
            log_file.write(str(stats2))

    def get_winner(self):
        return self.arena.get_winner()

    def get_points(self):
        player1_points = 1 if self.arena.get_winner() is None else 3 if self.player1 == self.arena.get_winner() else 0
        player2_points = 1 if self.arena.get_winner() is None else 3 if self.player2 == self.arena.get_winner() else 0

        return self.player1.name, player1_points, self.player2.name, player2_points

    def get_player_stats(self):
        return self.arena.get_players_stats()


class Tournament:

    def __init__(self,
                 models_folder: str,
                 games_per_match: int = 100,
                 board_shape: int = 8,
                 deterministic: bool = False,
                 device: str = "auto",
                 n_processes: int = None,
                 verbose: bool = True,
                 add_default_players: bool = False,
                 log_folder: str = None
                 ):

        if (models_folder is None) or (not os.path.exists(models_folder)) or (not os.path.isdir(models_folder)):
            raise Exception("Invalid models folder")

        if (log_folder is not None) and (not os.path.exists(log_folder) or not os.path.isdir(log_folder)):
            raise Exception("Invalid log folder")

        self.models_folder = models_folder
        self.games = games_per_match
        self.board_shape = board_shape
        self.deterministic = deterministic
        self.device = device
        self.players = []
        self.positions_table = dict()
        self.n_processes = n_processes if (n_processes is not None) else cpu_count()
        self.verbose = verbose
        self.add_default_players = add_default_players
        self.log_folder = log_folder
        self.finished_matches = 0

    def play(self):
        self.finished_matches = 0

        if self.log_folder is not None:
            self._create_log_folder()

        if self.verbose:
            print("Initiating Tournament: \n")
            if self.log_folder is not None:
                print(f"Logging matches in the following folder: {os.path.abspath(self.log_folder)} \n")

        self._load_players()
        self._create_positions_table()

        matches = self._create_matches()

        if self.verbose:
            print(f"Playing {len(matches)} matches in {self.n_processes} processes...")
            print()
            self.print_match_count(self.finished_matches, len(matches))

        if self.n_processes > 1:
            self._play_async(matches)
        else:
            self._play_sync(matches)

        self.positions_table = sorted(self.positions_table.items(), key=lambda x: x[1], reverse=True)

        if self.verbose:
            self.print_positions()

    def _play_async(self, matches):
        with Pool(self.n_processes) as p:
            jobs = []
            for match in matches:
                jobs.append(run_async(p, match.play, []))

            for job in jobs:
                self._process_match_results(job.get(), len(matches))

    def _play_sync(self, matches):
        for match in matches:
            self._process_match_results(match.play(), len(matches))

    def _process_match_results(self, results, total_matches):
        p1, points1, p2, points2 = results
        self.finished_matches += 1
        self.positions_table[p1] += points1
        self.positions_table[p2] += points2
        if self.verbose:
            self.print_match_count(self.finished_matches, total_matches)

    def _create_log_folder(self):
        tournament_folder = self.log_folder + "/Tournament_" + str(datetime.now()).replace(" ", "_")
        os.mkdir(tournament_folder)
        self.log_folder = tournament_folder

    def print_match_count(self, finished, total):
        print("\r", end="")
        sys.stdout.write("\033[K")
        print(f"Finished {finished} of {total}", end="")

    def get_positions(self):
        return self.positions_table

    def print_positions(self):
        groups = self.group_players_by_points()
        winner_word = "WINNER" if len(groups[0]) == 1 else "WINNERS"
        winners = ",".join([i[0] for i in groups[0]])
        print()
        print(print_bold(print_cyan(f"{winner_word}: {winners}")))
        print()
        position = 1
        for group in groups:
            players = ",".join([i[0] for i in group])
            points = group[0][1]
            print(f"{position} - {players} - {points} points")
            position += 1

    def _load_players(self):
        self.players = []
        models = [f for f in os.listdir(self.models_folder) if os.path.isfile(os.path.join(self.models_folder, f))]
        env = ReversiEnv(board_shape=self.board_shape)
        for m in models:
            self.players.append(
                TorchPlayer(
                    env=env,
                    model_path=os.path.join(self.models_folder, m),
                    deterministic=self.deterministic,
                    device=self.device
                )
            )

        if self.add_default_players:
            self.players.append(RandomPlayer(env=env))
            self.players.append(GreedyPlayer(env=env))

        if self.verbose:
            print(print_green("Competitors: "))
            for p in self.players:
                print(print_green(f"* {p.name}"))
            print()

    def _create_positions_table(self):
        self.positions_table = dict()
        for p in self.players:
            self.positions_table[p.name] = 0

    def _create_matches(self):
        remaining_players = self.players.copy()
        matches = []
        for p1 in self.players:
            remaining_players.remove(p1)
            for p2 in remaining_players:
                matches.append(Match(p1, p2, self.games, self.board_shape, self.log_folder))

        return matches

    def group_players_by_points(self):
        current_points = self.positions_table[0][1]
        current_group = []
        groups = []
        for item in self.positions_table:
            if item[1] != current_points:
                groups.append(current_group)
                current_group = [item]
                current_points = item[1]
            else:
                current_group.append(item)

        groups.append(current_group)
        return groups
