import tetris
import random
import math
import numpy as np
import pickle
from tqdm import tqdm
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam


class Tetris:
    def __init__(self):
        self.score = 0
        self.previous_score = 0
        self.move_columns = 4
        self.board_height = 20
        self.board_width = 10
        self.state_shape = (self.board_height, self.board_width)
        self.boardArray = np.zeros(self.state_shape)
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

    def best_action(self):
        # Return index of movesArray that gives the max reward
        action = 0
        return action

    def get_next_states(self):
        # Returns a dictionary that stores state/action pairs
        next_states = []
        temp_board = self.board
        temp_movesArray = self.movesArray
        temp_boardArray = self.boardArray
        temp_next_state = self.next_state
        temp_previous_score = self.previous_score
        temp_score = self.score

        for move in self.movesArray:
            pieceIndex, row, col, rot = int(move[0]), int(move[1]), int(move[2]), int(move[3])
            self.board.place(pieceIndex, row, col, rot)
            render = self.board.rend(pieceIndex, row, col, rot)
            self.boardArray = self.getBoardArray(render)
            self.next_state = self.boardArray

            self.previous_score = self.score
            self.score = self.board.getScore()
            reward = self.getReward(self.score, self.previous_score)
            next_states.append(self.next_state)

            self.board = temp_board
            self.movesArray = temp_movesArray
            self.boardArray = temp_boardArray
            self.next_state = temp_next_state
            self.previous_score = temp_previous_score
            self.score = temp_score

        return next_states

    def reset(self):
        self.board.reset()
        self.boardArray = np.zeros(self.state_shape)
        self.current_state = self.getBoardArray(self.board.rend(0, 0, 0, 0))
        self.next_state = self.getBoardArray(self.board.rend(0, 0, 0, 0))
        self.previous_score = 0
        self.score = 0


class DQN:
    def __init__(self, state_shape, experience_size=200, discount=0.95, epsilon=1, epsilon_min=0, epsilon_stop_episode=50):
        self.state_shape = state_shape
        self.experiences = []
        self.experience_size = experience_size
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = (
            self.epsilon - self.epsilon_min) / (epsilon_stop_episode)
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(32, input_shape=self.state_shape,
                        activation="relu", batch_size=None))
        model.add(Dense(32, activation="relu", batch_size=None))
        model.add(Dense(10, activation="linear", batch_size=None))
        model.compile(loss="mse", optimizer="adam")
        return model

    def add_experience(self, current_state, next_state, action, reward):
        self.experiences.append((current_state, next_state, action, reward))

    def best_state(self, states):
        max_value = None
        best_state = None

        for state in states:
            value = self.model.predict(state)[0]
            if not max_value or value > max_value:
                max_value = value
                best_state = state

        return best_state

    def train(self, batch_size=32, epochs=3):
        batch = random.sample(self.experiences, batch_size)
        next_states = []

        for (_, next_state, _, _) in batch:
            next_states.append(next_state)

        next_states = np.asarray(next_states)
        # print("next_states:\n", next_states)
        # print("next_states.shape: ", next_states.shape)

        # q_values = []
        # for x in self.model.predict(next_states):
        #     # print("x.shape:", x.shape)
        #     q_values.append(x[0])

        # print("q_values: ", q_values)

        x = []
        y = []

        for i, (current_state, next_state, action, reward) in enumerate(batch):
            # q = reward + self.discount * q_values[i]
            x.append(current_state)
            y.append(next_state)

        x = np.asarray(x)
        y = np.asarray(y)
        # print("x: ", x)
        # print("y: ", y)
        print("x.shape: ", x.shape)
        print("y.shape: ", y.shape)

        self.model.fit(x, y, batch_size=batch_size, epochs=epochs, verbose=1)

        print("model.layers[-1].get_weights()[1]:\n",
              self.model.layers[-1].get_weights()[1])

        max_q = -math.inf
        for i in range(len(self.model.layers[-1].get_weights()[1])):
            print("max_q: ", max_q)
            value = self.model.layers[-1].get_weights()[1][i]
            if max_q < value:
                max_q = value

        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

    def writeExperiencesToFile(self, filename):
        with open(f"{filename}", "ab") as file:
            pickle.dump(self.experiences, file)

    def readExperiencesFromFile(self, filename):
        with open(f"{filename}", "rb") as file:
            self.experiences = pickle.load(file)


def collect_experiences(tetris, dqn):
    # Collect experiences where each experience consists of a tuple:
    # (current_state, next_state, action, reward)
    # These experiences are then used to train the DQN.

    for i in tqdm(range(dqn.experience_size)):
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
            render = tetris.board.rend(pieceIndex, row, col, rot)
            tetris.boardArray = tetris.getBoardArray(render)
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


def train_model(tetris, dqn):
    episodes = 20
    train_every = 5
    scores = []

    for episode in tqdm(range(episodes)):
        print(f"Game #{episode}")
        current_state = tetris.reset()
        notGameOver = True

        while notGameOver:
            next_states = tetris.get_next_states()
            # print("next_states:\n", next_states)
            tetris.movesArray = tetris.getMovesArray(tetris.board.getMoves())
            # best_action = tetris.best_action(current_state)
            # action = tetris.movesArray[best_action]

            # pieceIndex, row, col, rot = int(action[0]), int(
            #     action[1]), int(action[2]), int(action[3])

            # tetris.board.place(pieceIndex, row, col, rot)
            # render = tetris.board.rend(pieceIndex, row, col, rot)
            # tetris.boardArray = tetris.getBoardArray(render)
            # tetris.next_state = tetris.boardArray

            # tetris.previous_score = tetris.score
            # tetris.score = tetris.board.getScore()
            # reward = tetris.getReward(tetris.score, tetris.previous_score)

            # notGameOver = tetris.board.isValid(pieceIndex, row, col, rot)

            # dqn.add_experience(current_state, next_states[best_action], action, reward)

            # current_state = next_states[best_action]
            # best_action = 0
            notGameOver = False

        scores.append(tetris.board.getScore())

        if episode % train_every == 0:
            dqn.train(batch_size=32, epochs=3)


def main():
    tetris = Tetris()
    dqn = DQN(state_shape=tetris.state_shape, experience_size=100, discount=0.95, epsilon=1, epsilon_min=0, epsilon_stop_episode=50)

    # dqn.readExperiencesFromFile("experiences")
    collect_experiences(tetris, dqn)
    # dqn.writeExperiencesToFile("experiences")
    train_model(tetris, dqn)
    # dqn.train(batch_size=32, epochs=5)


if __name__ == "__main__":
    main()
