from reversi_model import CustomReversiModel
from players import RandomPlayer

if __name__ == "__main__":
    model = CustomReversiModel(board_shape=8, n_envs=8, local_player=RandomPlayer)
    model.learn(total_timesteps=int(1e5))
