import tetris_boost as tetris
import random
import math
import numpy as np
import pickle
from tqdm import tqdm
from collections import deque
import tensorflow as tf
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
        self.state_shape = (self.board_height, self.board_width)
        self.boardArray = np.zeros(self.state_shape)
        self.randomMove = np.zeros((1, 4))
        self.board = tetris.Board()
        self.current_state = self.get_board_array(self.board.rend(0, 0, 0, 0))
        self.next_state = self.get_board_array(self.board.rend(0, 0, 0, 0))
        self.temp_state = self.get_board_array(self.board.rend(0, 0, 0, 0))
        self.numberOfMoves = 0
        self.movesArray = np.zeros((0, 0))

    def get_moves_array(self, moves):
        self.numberOfMoves = self.board.getNumberOfMoves()
        self.movesArray = np.zeros((self.numberOfMoves, self.move_columns))

        for i in range(self.numberOfMoves):
            for j in range(self.move_columns):
                self.movesArray[i, j] = self.board.getValueOfVectorInts(
                    moves, i, j)
        return self.movesArray

    def get_board_array(self, rend):
        for i in range(self.board_height):
            for j in range(self.board_width):
                self.boardArray[i, j] = self.board.getValueOfVectorBools(
                    rend, i, j)
        return self.boardArray

    def print_board(self):
        state_str = ""
        for r in range(self.current_state.shape[0]):
            for c in range(self.current_state.shape[1]):
                if self.current_state[r][c] == 0:
                    state_str += "-"
                else:
                    state_str += "X"
            state_str += "\n"
        print(state_str)

    def get_reward(self, score, previous_score):
        reward = score - previous_score
        return reward

    def get_next_states(self):
        next_states = []

        moves = self.get_moves_array(self.board.getMoves())

        for move in moves:
            pieceIndex, row, col, rot = int(move[0]), int(
                move[1]), int(move[2]), int(move[3])

            render = self.board.rend(pieceIndex, row, col, rot)
            temp_state = self.get_board_array(render)

            next_states.append([temp_state, move])
            self.reset_temp_state()

        return next_states

    def reset_temp_state(self):
        self.temp_state = self.get_board_array(self.board.rend(0, 0, 0, 0))

    def reset(self):
        self.board.reset()
        self.boardArray = np.zeros(self.state_shape)
        self.current_state = self.get_board_array(self.board.rend(0, 0, 0, 0))
        self.next_state = self.get_board_array(self.board.rend(0, 0, 0, 0))
        self.temp_state = self.get_board_array(self.board.rend(0, 0, 0, 0))
        self.previous_score = 0
        self.score = 0


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

    def build_model(self):
        model = Sequential()
        model.add(Dense(32, input_shape=self.state_shape,
                        activation="relu", batch_size=None))
        model.add(Dense(32, activation="relu", batch_size=None))
        model.add(Dense(1, activation="linear", batch_size=None))
        model.compile(loss="mse", optimizer="adam")
        return model

    def add_experience(self, current_state, next_state, action, reward):
        self.experiences.append((current_state, next_state, action, reward))

    def best_action(self, states):
        max_value = None
        best_action = None

        for state in states:
            state[0] = np.expand_dims(state[0], axis=0)

            prediction = self.model.predict(state[0])
            # print("prediction:", prediction)

            value = np.max(prediction)
            # print("value:", value)

            best_action = state[1]
            # print("best_action:", best_action)

            if not max_value or value > max_value:
                max_value = value
                best_action = state[1]

        # print("best_action:", best_action)

        return best_action

        # return states[-1][1]

    def train(self, batch_size=32, epochs=3):
        batch = random.sample(self.experiences, batch_size)

        next_states = np.array([x[1] for x in batch])
        next_qs = [x[0] for x in self.model.predict(next_states)]

        x = []
        y = []

        for i, (state, _, reward, done) in enumerate(batch):
            if not done:
                new_q = reward + self.discount * next_qs[i]
            else:
                new_q = reward

            x.append(state)
            y.append(new_q)

        x = np.array(x)
        y = np.array(y)

        print("x.shape:", x.shape)
        print("y.shape:", y.shape)

        y = np.expand_dims(y, axis=0)
        print("y.shape:", y.shape)

        self.model.fit(x, y, batch_size=batch_size, epochs=epochs, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

    # def train(self, batch_size=32, epochs=3):
    #     batch = random.sample(self.experiences, batch_size)

        # for current_state, next_state, action, reward in batch:
        #     if current_state is None:
        #         continue
        #     next_state = np.expand_dims(next_state, axis=0)
        #     target = reward + self.discount * \
        #         np.amax(self.model.predict(next_state)[0])
        #     # print("target: ", target)
        #     # print("current_state:", current_state)
        #     current_state = np.expand_dims(current_state, axis=0)
        #     print("current_state:\n", current_state)
        #     # print("current_state1:", current_state)
        #     target_f = self.model.predict(current_state)
        #     # print("target_f: ", target_f)
        #     target_f[0][0] = target
        #     # target_f[0][action] = target
        #     # self.model.fit(current_state, target_f, epochs=epochs, verbose=0)
        #     self.model.fit(current_state, target_f, epochs=epochs, verbose=0)

        # if self.epsilon > self.epsilon_min:
        #     self.epsilon -= self.epsilon_decay

    def write_experiences_to_file(self, filename):
        with open(f"{filename}", "ab") as file:
            pickle.dump(self.experiences, file)

    def read_experiences_from_file(self, filename):
        with open(f"{filename}", "rb") as file:
            self.experiences = pickle.load(file)


def print_stats(array, label, episodes):
    print(f"{label}:\n", array)
    print(f"min({label}):", min(array))
    print(f"max({label}):", max(array))
    print(f"mean({label}):", np.mean(array))


def collect_experiences(tetris, dqn):
    # Collect experiences where each experience consists of a tuple:
    # (current_state, next_state, action, reward)
    # These experiences are then used to train the DQN.

    for i in tqdm(range(dqn.experience_size)):
        tetris.current_state = tetris.next_state
        # print("current_state =\n", tetris.current_state)
        tetris.movesArray = tetris.get_moves_array(tetris.board.getMoves())

        if tetris.board.getNumberOfMoves() > 0:
            moveIndex = random.randint(0, tetris.board.getNumberOfMoves() - 1)
        else:
            tetris.reset()
            continue

        action = tetris.movesArray[moveIndex]
        pieceIndex, row, col, rot = int(action[0]), int(
            action[1]), int(action[2]), int(action[3])
        # print(f"pieceIndex: {pieceIndex}, row: {row}, col: {col}, rot: {rot}")

        notGameOver = tetris.board.isValid(pieceIndex, row, col, rot)
        # print("notGameOver: ", notGameOver)

        if notGameOver:
            tetris.board.place(pieceIndex, row, col, rot)
            render = tetris.board.rend(pieceIndex, row, col, rot)
            tetris.boardArray = tetris.get_board_array(render)
            tetris.next_state = tetris.boardArray

            tetris.previous_score = tetris.score
            tetris.score = tetris.board.getScore()
            reward = tetris.get_reward(tetris.score, tetris.previous_score)
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
        tetris.reset()
        notGameOver = True
        numberOfMovesPlayed = 0

        # print(f"Game #{episode + 1} started.")

        while notGameOver:
            tetris.current_state = tetris.boardArray

            next_states = tetris.get_next_states()

            best_action = dqn.best_action(next_states)

            # print("best_action:", best_action)

            if best_action is None:
                break

            pieceIndex, row, col, rot = int(best_action[0]), int(
                best_action[1]), int(best_action[2]), int(best_action[3])

            notGameOver = tetris.board.isValid(pieceIndex, row, col, rot)

            tetris.board.place(pieceIndex, row, col, rot)
            render = tetris.board.rend(pieceIndex, row, col, rot)
            tetris.boardArray = tetris.get_board_array(render)

            tetris.previous_score = tetris.score
            tetris.score = tetris.board.getScore()
            reward = tetris.get_reward(tetris.score, tetris.previous_score)

            # tetris.print_board()
            # print("score = ", tetris.score)

            numberOfMovesPlayed += 1

            dqn.add_experience(tetris.current_state,
                               tetris.boardArray, best_action, reward)

            tetris.current_state = tetris.boardArray

        # print("score:", tetris.score)

        scores.append(tetris.score)
        movesPlayed.append(numberOfMovesPlayed)

        # if episode % train_every == 0:
        #     dqn.train(batch_size=batch_size, epochs=epochs)

        print_to_file(scores, "scores.csv")
        print_to_file(movesPlayed, "movesPlayed.csv")

    print_stats(scores, "scores", episodes)
    print_stats(movesPlayed, "movesPlayed", episodes)


def print_to_file(array, filename):
    with open(f"{filename}", "a") as file:
        np.savetxt(filename, array, delimiter=",")


def read_from_file(filename):
    with open(f"{filename}", "r") as file:
        return np.loadtxt(filename, delimiter=",")


def main():
    tetris = Tetris()

    dqn = DQN(state_shape=tetris.state_shape, experience_size=100,
              discount=0.95, epsilon=1, epsilon_min=0, epsilon_stop_episode=75)

    collect_experiences(tetris, dqn)

    train_model(tetris, dqn, batch_size=32,
                epochs=3, episodes=50, train_every=5)

    print("scores from file:", read_from_file("scores.csv"))
    print("movesPlayed from file:", read_from_file("movesPlayed.csv"))

    dqn.model.save("dqn_model.h5")


if __name__ == "__main__":
    main()
