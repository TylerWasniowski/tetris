import tetris
import random
# import math
import numpy as np
# import pickle
# from tqdm import tqdm
# from collections import deque
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.optimizers import Adam


class Tetris:
    def __init__(self):
        print("init")
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
        print("Get moves array")
        self.numberOfMoves = self.board.getNumberOfMoves()
        self.movesArray = np.zeros((self.numberOfMoves, self.move_columns))

        for i in range(self.numberOfMoves):
            for j in range(self.move_columns):
                self.movesArray[i, j] = self.board.getValueOfVectorInts(
                    moves, i, j)
        return self.movesArray

    def getBoardArray(self, rend):
        print("get board array")
        for i in range(self.board_height):
            for j in range(self.board_width):
                self.boardArray[i, j] = self.board.getValueOfVectorBools(
                    rend, i, j)
        return self.boardArray

    def getReward(self, score, previous_score):
        print("get reward")
        reward = score - previous_score
        return reward

    def best_action(self, state):
        print("best_action")
        # Return index of movesArray that gives the max reward
        action = 0
        return action

    def reset(self):
        print("reset")
        self.board.reset()
        self.boardArray = np.zeros(self.state_shape)
        self.current_state = self.getBoardArray(self.board.rend(0, 0, 0, 0))
        self.next_state = self.getBoardArray(self.board.rend(0, 0, 0, 0))
        self.previous_score = 0
        self.score = 0


# class DQN:
#     def __init__(self, state_shape, experience_size=200, discount=0.95, epsilon=1, epsilon_min=0, epsilon_stop_episode=50):
#         self.state_shape = state_shape
#         self.experiences = []
#         self.experience_size = experience_size
#         self.discount = discount
#         self.epsilon = epsilon
#         self.epsilon_min = epsilon_min
#         self.epsilon_decay = (self.epsilon - self.epsilon_min) / (epsilon_stop_episode)
#         self.model = self.build_model()
#
#     def build_model(self):
#         model = Sequential()
#         model.add(Dense(32, input_shape=self.state_shape,
#                         activation="relu", batch_size=None))
#         model.add(Dense(32, activation="relu", batch_size=None))
#         model.add(Dense(10, activation="linear", batch_size=None))
#         model.compile(loss="mse", optimizer="adam")
#         return model
#
#     def add_experience(self, current_state, next_state, action, reward):
#         self.experiences.append((current_state, next_state, action, reward))
#
#     def train(self, batch_size=32, epochs=3):
#         batch = random.sample(self.experiences, batch_size)
#
#         for current_state, next_state, action, reward in batch:
#             next_state = np.expand_dims(next_state, axis=0)
#             target = reward + self.discount * np.amax(self.model.predict(next_state)[0])
#             # print("target: ", target)
#             current_state = np.expand_dims(current_state, axis=0)
#             target_f = self.model.predict(current_state)
#             # print("target_f: ", target_f)
#             target_f[0][0] = target
#             # target_f[0][action] = target
#             self.model.fit(current_state, target_f, epochs=epochs, verbose=0)
#
#         if self.epsilon > self.epsilon_min:
#             self.epsilon -= self.epsilon_decay
#
#     def writeExperiencesToFile(self, filename):
#         with open(f"{filename}", "ab") as file:
#             pickle.dump(self.experiences, file)
#
#     def readExperiencesFromFile(self, filename):
#         with open(f"{filename}", "rb") as file:
#             self.experiences = pickle.load(file)


def collect_experiences(tetris):
    # Collect experiences where each experience consists of a tuple:
    # (current_state, next_state, action, reward)
    # These experiences are then used to train the DQN.

    for i in range(1000):
        print("current to next")
        tetris.current_state = tetris.next_state
        print("post current to next")
        state_str = ""
        for r in len(tetris.current_state):
            for c in len(tetris.current_state[r]):
                if tetris.current_state[r][c] == 0:
                    state_str += "-"
                else:
                    state_str += "X"
            state_str += "\n"
        print("getting moves array")
        tetris.movesArray = tetris.getMovesArray(tetris.board.getMoves())
        print("post getting moves array")

        if tetris.board.getNumberOfMoves() > 0:
            rowIndex = random.randint(0, tetris.board.getNumberOfMoves() - 1)
        else:
            tetris.reset()
            continue

        print("selecting action")
        action = tetris.movesArray[rowIndex]
        pieceIndex, row, col, rot = int(action[0]), int(
            action[1]), int(action[2]), int(action[3])
        # print(f"pieceIndex: {pieceIndex}, row: {row}, col: {col}, rot: {rot}")

        print("checking is valid")
        notGameOver = tetris.board.isValid(pieceIndex, row, col, rot)
        print("post is valid")
        # print("notGameOver: ", notGameOver)

        if notGameOver:
            tetris.board.place(pieceIndex, row, col, rot)
            render = tetris.board.rend(pieceIndex, row, col, rot)
            tetris.boardArray = tetris.getBoardArray(render)
            tetris.next_state = tetris.boardArray

            tetris.previous_score = tetris.score
            print("get score")
            tetris.score = tetris.board.getScore()
            print("score is:")
            print(tetris.score)
            print("prev score is:")
            print(tetris.previous_score)
            reward = tetris.getReward(tetris.score, tetris.previous_score)
            print("post get reward")
            # print("next_state =\n", tetris.next_state)
            # print("action = ", action)
            # print("previous_score = ", tetris.previous_score)
            # print("reward = ", reward)
            # print("score = ", tetris.score)
            # dqn.add_experience(tetris.current_state,
            #                    tetris.next_state, action, reward)
        else:
            tetris.reset()

        print()


# def train_model(tetris, dqn, batch_size, epochs, episodes, train_every):
#     scores = []
#
#     for episode in tqdm(range(episodes)):
#         tetris.current_state = tetris.reset()
#         notGameOver = True
#         while notGameOver:
#             tetris.current_state = tetris.next_state
#             # print("current_state =\n", tetris.current_state)
#             tetris.movesArray = tetris.getMovesArray(tetris.board.getMoves())
#
#             if tetris.board.getNumberOfMoves() > 0:
#                 best_action = tetris.best_action(tetris.current_state)
#             else:
#                 tetris.reset()
#                 continue
#
#             action = tetris.movesArray[best_action]
#             pieceIndex, row, col, rot = int(action[0]), int(
#                 action[1]), int(action[2]), int(action[3])
#             # print(f"pieceIndex: {pieceIndex}, row: {row}, col: {col}, rot: {rot}")
#
#             notGameOver = tetris.board.isValid(pieceIndex, row, col, rot)
#             # print("notGameOver: ", notGameOver)
#
#             tetris.board.place(pieceIndex, row, col, rot)
#             render = tetris.board.rend(pieceIndex, row, col, rot)
#             tetris.boardArray = tetris.getBoardArray(render)
#             tetris.next_state = tetris.boardArray
#
#             tetris.previous_score = tetris.score
#             tetris.score = tetris.board.getScore()
#             reward = tetris.getReward(tetris.score, tetris.previous_score)
#             # print("next_state =\n", tetris.next_state)
#             # print("action = ", action)
#             # print("previous_score = ", tetris.previous_score)
#             # print("reward = ", reward)
#             # print("score = ", tetris.score)
#             dqn.add_experience(tetris.current_state, tetris.next_state, action, reward)
#
#         scores.append(tetris.score)
#
#         if episode % train_every == 0:
#             dqn.train(batch_size=batch_size, epochs=epochs)
#
#     print("scores:\n", scores)


def main():
    tetris = Tetris()

    collect_experiences(tetris)

    # train_model(tetris, dqn, batch_size=32, epochs=3, episodes=20, train_every=5)


if __name__ == "__main__":
    main()
