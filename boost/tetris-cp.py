import tetris
import random
import math
from statistics import mean
import numpy as np
import pickle
from tqdm import tqdm
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import tensorflow as tf
from keras import backend as K
from keras.utils import multi_gpu_model


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
        self.movesPlayed = 0

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

    def reset(self):
        self.board.reset()
        self.boardArray = np.zeros(self.state_shape)
        self.current_state = self.getBoardArray(self.board.rend(0, 0, 0, 0))
        self.next_state = self.getBoardArray(self.board.rend(0, 0, 0, 0))
        self.previous_score = 0
        self.score = 0
        self.movesPlayed = 0


class DQN:
    def __init__(self, state_shape, experience_size, discount, epsilon, epsilon_min, epsilon_stop_episode):
        self.state_shape = state_shape
        self.experiences = deque(maxlen=experience_size)
        self.experience_size = experience_size
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_stop_episode = epsilon_stop_episode
        self.epsilon_decay = (
            self.epsilon - self.epsilon_min) / (epsilon_stop_episode)
        self.model = self.build_model()
        self.parallel_model = multi_gpu_model(self.model, gpus=2)
        self.parallel_model.compile(loss="mse", optimizer="adam")

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

    def best_action(self, state):
        # Return index of movesArray that gives the max reward
        # if np.random.rand() <= self.epsilon:
        #     return random.randrange(self.action_size)
        state = np.expand_dims(state, axis=0)
        act_values = self.model.predict(state)[0][0]
        # print("act_values:\n", act_values)
        # print("np.argmax(act_values):", np.argmax(act_values))
        action = np.argmax(act_values)
        # action = 0
        return action
        # return np.argmax(act_values[0])

        # action = 0
        # return action

    def train(self, batch_size=32, epochs=3):
        # if len(self.experiences) >= batch_size:
        batch = random.sample(self.experiences, batch_size)

        for current_state, next_state, action, reward in batch:
            next_state = np.expand_dims(next_state, axis=0)
            target = reward + self.discount * \
                np.amax(self.model.predict(next_state)[0])
            # print("target: ", target)
            current_state = np.expand_dims(current_state, axis=0)
            target_f = self.model.predict(current_state)
            # print("target_f: ", target_f)
            target_f[0][0] = target
            # target_f[0][action] = target
            # self.model.fit(current_state, target_f, epochs=epochs, verbose=0)
            self.parallel_model.fit(current_state, target_f, epochs=epochs, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

    def writeExperiencesToFile(self, filename):
        with open(f"{filename}", "ab") as file:
            pickle.dump(self.experiences, file)

    def readExperiencesFromFile(self, filename):
        with open(f"{filename}", "rb") as file:
            self.experiences = pickle.load(file)


def print_stats(array, label, episodes):
    print(f"{label}:\n", array)
    print(f"min({label}):", min(array), "reached on game #",
          np.argmin(array) + 1, "/", episodes)
    print(f"max({label}):", max(array), "reached on game #",
          np.argmax(array) + 1, "/", episodes)
    print(f"mean({label}):", mean(array))


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
            tetris.movesPlayed += 1
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


def train_model(tetris, dqn, batch_size, epochs, episodes, train_every):
    scores = []
    movesPlayed = []

    for episode in tqdm(range(episodes)):
        tetris.current_state = tetris.reset()
        notGameOver = True
        while notGameOver:
            tetris.current_state = tetris.next_state
            # print("current_state =\n", tetris.current_state)
            tetris.movesArray = tetris.getMovesArray(tetris.board.getMoves())
            # best_action = 0

            if tetris.board.getNumberOfMoves() > 0:
                best_action = dqn.best_action(tetris.current_state)
            else:
                tetris.reset()
                continue

            if best_action >= np.size(tetris.movesArray, 0):
                best_action = 0

            # print("movesArray:\n", tetris.movesArray)

            action = tetris.movesArray[best_action]

            pieceIndex, row, col, rot = int(action[0]), int(
                action[1]), int(action[2]), int(action[3])
            # print(f"pieceIndex: {pieceIndex}, row: {row}, col: {col}, rot: {rot}")

            notGameOver = tetris.board.isValid(pieceIndex, row, col, rot)
            # print("notGameOver: ", notGameOver)

            tetris.board.place(pieceIndex, row, col, rot)
            tetris.movesPlayed += 1
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

        scores.append(tetris.score)
        movesPlayed.append(tetris.movesPlayed)

        if episode % train_every == 0:
            dqn.train(batch_size=batch_size, epochs=epochs)

    print_stats(scores, "scores", episodes)
    print_stats(movesPlayed, "movesPlayed", episodes)


def main():
    session_gpu = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    print("Print:", session_gpu)

    print("available gpus:", K.tensorflow_backend._get_available_gpus())

    tetris = Tetris()

    dqn = DQN(state_shape=tetris.state_shape, experience_size=500,
              discount=0.95, epsilon=1, epsilon_min=0, epsilon_stop_episode=350)

    collect_experiences(tetris, dqn)

    train_model(tetris, dqn, batch_size=32,
                epochs=3, episodes=50, train_every=5)


if __name__ == "__main__":
    main()
