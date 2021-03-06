import argparse
import math
import numpy as np
import random
import time
from tqdm import tqdm

import ai
import observations
import tetris_boost
import tetris_boost_5


class Tetris:
    def __init__(self, five_piece=False):
        self.score = 0
        self.move_columns = 4
        self.board_height = 20
        self.board_width = 10
        self.state_shape = (self.board_height, self.board_width)
        self.boardArray = np.zeros(self.state_shape, dtype=bool)
        if five_piece:
            self.board = tetris_boost_5.Board()
        else:
            self.board = tetris_boost.Board()
        self.numberOfMoves = 0
        self.movesArray = np.zeros((0, 0), dtype=int)
        self.movesPlayed = 0
        self.linesCleared = np.zeros(4)

    def getMovesArray(self):
        moves = self.board.getMoves()
        self.numberOfMoves = self.board.getNumberOfMoves()
        self.movesArray = np.zeros((self.numberOfMoves, self.move_columns), dtype=int)

        for i in range(self.numberOfMoves):
            for j in range(self.move_columns):
                self.movesArray[i, j] = int(self.board.getValueOfVectorInts(
                    moves, i, j))
        return self.movesArray

    def getBoardArray(self, move):
        rend = self.board.rend(int(move[0]), int(move[1]), int(move[2]), int(move[3]))
        for i in range(self.board_height):
            for j in range(self.board_width):
                self.boardArray[i, j] = self.board.getValueOfVectorBools(
                    rend, i, j)
        return self.boardArray

    def place(self, move):
        self.board.place(int(move[0]), int(move[1]), int(move[2]), int(move[3]))
        old_score = self.score
        self.score = self.board.getScore()

        if self.score > old_score + 300:
            self.linesCleared[3] += 1
        elif self.score > old_score + 100:
            self.linesCleared[2] += 1
        elif self.score > old_score + 40:
            self.linesCleared[1] += 1
        elif self.score > old_score:
            self.linesCleared[0] += 1

        self.getBoardArray(move)
        self.movesPlayed += 1

    def render(self):
        board_str = ""
        for r in range(self.boardArray.shape[0]):
            for c in range(self.boardArray.shape[1]):
                if self.boardArray[r][c] == 0:
                    board_str += "-"
                else:
                    board_str += "X"
            board_str += "\n"
        print(board_str)

    def reset(self):
        self.board.reset()
        self.boardArray = np.zeros(self.state_shape, dtype=bool)
        self.score = 0
        self.movesPlayed = 0
        self.linesCleared = np.zeros(4)


class Game:
    def __init__(self, model, five_piece=False):
        self.model = model
        self.tetris = Tetris(five_piece=five_piece)

    def play(self, n_moves=1, skip_render=False, sleep=1):
        for n in range(n_moves):
            moves = self.tetris.getMovesArray()
            best_move = None
            best_score = -math.inf
            for move in moves:
                board = self.tetris.getBoardArray(move)
                obs = np.array(observations.compressed(observations.from_board(board)))
                # Max compressed obs
                score = self.model.score(obs)
                if score > best_score:
                    best_move = move
                    best_score = score

            if best_move is None:
                return self.tetris.score

            old_score = self.tetris.score
            self.tetris.place(best_move)
            if not skip_render:
                print(best_move)
                self.tetris.render()
                if self.tetris.score > old_score:
                    print("Scored", self.tetris.score - old_score)

            time.sleep(sleep)

    def render(self):
        self.tetris.render()

    def reset(self):
        self.tetris.reset()


def hmm_model(n=8):
    return ai.load_hmm(n)


def random_model():
    class Random:
        def score(self, _):
            return random.random()
    return Random()


def test_model(model, n_iter=1000, save_filename=None):
    game = Game(model)

    stats = [[], [], []]
    for n in tqdm(range(n_iter)):
        # Play until out of moves
        game.play(n_moves=123456789, skip_render=True, sleep=0)

        stats[0].append(game.tetris.score)
        stats[1].append(game.tetris.movesPlayed)
        stats[2].append(game.tetris.linesCleared)
        if save_filename is not None:
            np.save(save_filename, stats)

        game.reset()

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", "-f", help="File to save the scores in", default="hmm_stats.npy", type=str)
    parser.add_argument("--model", "-m", help="Model to play ('hmm', or 'random')", default="hmm", type=str)
    parser.add_argument("--n", "-n", help="Number of hidden states", default=4, type=int)
    parser.add_argument("--n_iter", "-i", help="Number of games to play", default=1000, type=int)
    args = parser.parse_args()

    if args.model == "hmm":
        model = hmm_model(args.n)
    else:
        model = random_model()

    test_model(model, n_iter=args.n_iter, save_filename=args.filename)
