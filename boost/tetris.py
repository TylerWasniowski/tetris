import tetris
import random
import numpy as np
import pickle
from tqdm import tqdm
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class Tetris:
    def __init__(self):
        self.score = 0
        self.previous_score = 0
        self.move_columns = 4
        self.board_height = 20
        self.board_width = 10
        self.boardArray = np.zeros((self.board_height, self.board_width))
        self.randomMove = np.zeros((1, 4))
        self.board = tetris.Board()
        self.current_state = self.getBoardArray(self.board.rend(0, 0, 0, 0))
        self.next_state = self.getBoardArray(self.board.rend(0, 0, 0, 0))
        self.numberOfMoves = 0
        self.movesArray = np.zeros((0, 0))

    def getMovesArray(self, moves):
        self.numberOfMoves = self.board.getNumberOfMoves()
        self.movesArray = np.zeros((self.numberOfMoves, self.move_columns))

        for i in range(self.numberOfMoves):
            for j in range(self.move_columns):
                self.movesArray[i, j] = self.board.getValueOfVectorInts(
                    moves, i, j)
        return self.movesArray

    def getBoardArray(self, rend):
        for i in range(self.board_height):
            for j in range(self.board_width):
                self.boardArray[i, j] = self.board.getValueOfVectorBools(
                    rend, i, j)
        return self.boardArray

    def getReward(self, score, previous_score):
        reward = score - previous_score
        return reward

    # def get_next_states(self):
    #     states = {}

    def reset(self):
        self.board.reset()
        self.boardArray = np.zeros((self.board_height, self.board_width))
        self.current_state = self.getBoardArray(self.board.rend(0, 0, 0, 0))
        self.next_state = self.getBoardArray(self.board.rend(0, 0, 0, 0))
        self.previous_score = 0
        self.score = 0


class DQN:
    def __init__(self, state_shape, numberOfExperiences, numberOfGames, discount, epsilon):
        self.state_shape = state_shape
        self.experiences = []
        self.numberOfExperiences = numberOfExperiences
        self.numberOfGames = numberOfGames
        self.discount = discount
        self.epsilon = epsilon
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(32, input_shape=self.state_shape, activation="relu", batch_size=None))
        model.add(Dense(32, activation="relu", batch_size=None))
        model.add(Dense(1, activation="linear", batch_size=None))
        model.compile(loss="mse", optimizer="adam")
        return model

    def add_experience(self, current_state, next_state, action, reward):
        self.experiences.append((current_state, next_state, action, reward))

    def train(self, batch_size):
        batch = random.sample(self.experiences, batch_size)
        next_states = []
        for _, next_state, _, _ in batch:
            next_states.append(next_state)

        print("next_states:\n", next_states)


    def writeExperiencesToFile(self, filename):
        with open(f"{filename}", "ab") as file:
            pickle.dump(self.experiences, file)

    def readExperiencesFromFile(self, filename):
        with open(f"{filename}", "rb") as file:
            self.experiences = pickle.load(file)


def collect_experiences(tetris, dqn):
    for i in tqdm(range(dqn.numberOfExperiences)):
        tetris.current_state = tetris.next_state
        # print("current_state =\n", tetris.current_state)
        tetris.movesArray = tetris.getMovesArray(tetris.board.getMoves())

        if tetris.board.getNumberOfMoves() > 0:
            rowIndex = random.randint(0, tetris.board.getNumberOfMoves() - 1)
        else:
            tetris.reset()
            continue

        action = tetris.movesArray[rowIndex]
        pieceIndex, row, col, rot = int(action[0]), int(
            action[1]), int(action[2]), int(action[3])
        # print(f"pieceIndex: {pieceIndex}, row: {row}, col: {col}, rot: {rot}")

        notGameOver = tetris.board.isValid(pieceIndex, row, col, rot)
        # print("notGameOver: ", notGameOver)

        if notGameOver:
            tetris.board.place(pieceIndex, row, col, rot)
            tetris.boardArray = tetris.getBoardArray(
                tetris.board.rend(pieceIndex, row, col, rot))
            tetris.next_state = tetris.boardArray
            tetris.previous_score = tetris.score
            tetris.score = tetris.board.getScore()
            reward = tetris.getReward(tetris.score, tetris.previous_score)
            # print("next_state =\n", tetris.next_state)
            # print("action = ", action)
            # print("previous_score = ", tetris.previous_score)
            # print("reward = ", reward)
            # print("score = ", tetris.score)
            dqn.add_experience(tetris.current_state,
                               tetris.next_state, action, reward)
        else:
            tetris.reset()


def main():
    tetris = Tetris()
    dqn = DQN(state_shape=(tetris.board_height, tetris.board_width),numberOfExperiences=100, numberOfGames=10, discount=0.95, epsilon=1)

    # dqn.readExperiencesFromFile("experiences")
    collect_experiences(tetris, dqn)
    # dqn.writeExperiencesToFile("experiences")

    # for i in tqdm(range(dqn.numberOfGames)):
        # print(f"Game #{i}")
        # current_state = tetris.reset()
        # gameOver = False

        # while not gameOver:
        #     next_states = tetris.get_next_states()
        #     best_state = dqn.best_state(next_states.values())

        # dqn.train(batch_size=32)

if __name__ == "__main__":
    main()
