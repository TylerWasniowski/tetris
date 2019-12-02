import math
import numpy as np
import random
from tqdm import tqdm

import ai
import observations
import tetris_boost


class Tetris:
    def __init__(self):
        self.score = 0
        self.previous_score = 0
        self.move_columns = 4
        self.board_height = 20
        self.board_width = 10
        self.state_shape = (self.board_height, self.board_width)
        self.boardArray = np.zeros(self.state_shape, dtype=bool)
        self.board = tetris_boost.Board()
        self.numberOfMoves = 0
        self.movesArray = np.zeros((0, 0), dtype=int)
        self.movesPlayed = 0

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

    def getReward(self, score, previous_score):
        reward = score - previous_score
        return reward

    def place(self, move):
        self.board.place(int(move[0]), int(move[1]), int(move[2]), int(move[3]))
        return self.getBoardArray(move)

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
        self.boardArray = np.zeros(self.state_shape)
        self.previous_score = 0
        self.score = 0
        self.movesPlayed = 0


class Game:
    def __init__(self, model):
        self.model = model
        self.tetris = Tetris()

    def play(self, n_moves=1, skip_render=False):
        for n in range(n_moves):
            moves = self.tetris.getMovesArray()
            best_move = None
            best_score = -math.inf
            for move in moves:
                board = self.tetris.getBoardArray(move)
                obs = np.array(observations.compressed(observations.from_board(board)))
                score = self.model.score(obs)
                if score > best_score:
                    best_move = move
                    best_score = score

            if best_move is None:
                return

            self.tetris.place(best_move)
            if not skip_render:
                print(best_move)
                self.tetris.render()

    def render(self):
        self.tetris.render()


def hmm(n=8):
    return Game(ai.load_hmm(n))


def test_hmm(n=8, iter=1000):
    game = hmm(n)

    avg_score = 0
    for n in tqdm(range(iter)):
        # Play until out of moves
        game.play(n_moves=123456789, skip_render=True)
        avg_score += game.tetris.score / iter

    return avg_score









