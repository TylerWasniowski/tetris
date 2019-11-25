import tetris
import random
import numpy as np
import pickle
from tqdm import tqdm


class DQN:
    def __init__(self):
        self.experiences = []
        self.score = 0
        self.previous_score = 0
        self.reward = 0
        self.action = 0
        self.move_columns = 4
        self.experience_size = 1000
        self.board_height = 20
        self.board_width = 10
        self.boardArray = np.zeros((self.board_height, self.board_width))
        self.randomMove = np.zeros((1, 4))
        self.board = tetris.Board()
        self.current_state = self.getBoardArray(self.board.rend(0, 0, 0, 0))
        self.next_state = self.getBoardArray(self.board.rend(0, 0, 0, 0))
        self.numberOfMoves = 0
        self.movesArray = np.zeros((0, 0))

    def save_experience(self, current_state, next_state, action, reward):
        self.experiences.append((current_state, next_state, action, reward))

    def getMovesArray(self, moves):
        self.numberOfMoves = self.board.getNumberOfMoves()
        # print("numberOfMoves = ", self.numberOfMoves)

        self.movesArray = np.zeros((self.numberOfMoves, self.move_columns))

        for i in range(self.numberOfMoves):
            for j in range(self.move_columns):
                self.movesArray[i, j] = self.board.getValueOfVectorInts(moves, i, j)
        return self.movesArray

    def getBoardArray(self, rend):
        for i in range(self.board_height):
            for j in range(self.board_width):
                self.boardArray[i, j] = self.board.getValueOfVectorBools(rend, i, j)
        return self.boardArray

    def getReward(self, score, previous_score):
        self.reward = score - previous_score
        return self.reward

    def reset(self):
        self.board.reset()
        self.boardArray = np.zeros((self.board_height, self.board_width))
        self.current_state = self.getBoardArray(self.board.rend(0, 0, 0, 0))
        self.next_state = self.getBoardArray(self.board.rend(0, 0, 0, 0))
        self.previous_score = 0
        self.score = 0

    # def writeToFile(self, filename, array):
    #     with open(f"{filename}", "a") as file:
    #         np.savetxt(file, array, fmt="%i")
    #         file.write("\n")

    def writeExperiencesToFile(self, filename):
        with open(f"{filename}", "ab") as file:
            pickle.dump(self.experiences, file)

    def readExperiencesFromFile(self, filename):
        with open(f"{filename}", "rb") as file:
            self.experiences = pickle.load(file)


def run():
    dqn = DQN()

    # dqn.readExperiencesFromFile("experiences.txt")
    # print("experiences:\n", dqn.experiences)

    for i in tqdm(range(dqn.experience_size)):
        dqn.current_state = dqn.next_state

        # print("current_state =\n", dqn.current_state)

        dqn.movesArray = dqn.getMovesArray(dqn.board.getMoves())

        if dqn.board.getNumberOfMoves() > 0:
            rowIndex = random.randint(0, dqn.board.getNumberOfMoves() - 1)
        else:
            dqn.reset()
            continue

        dqn.randomMove = dqn.movesArray[rowIndex]
        dqn.action = dqn.randomMove

        pieceIndex = int(dqn.randomMove[0])
        # print("pieceIndex: ", pieceIndex)
        row = int(dqn.randomMove[1])
        # print("row: ", row)
        col = int(dqn.randomMove[2])
        # print("col: ", col)
        rot = int(dqn.randomMove[3])
        # print("rot: ", rot)

        notGameOver = dqn.board.isValid(pieceIndex, row, col, rot)
        # print("notGameOver: ", notGameOver)

        if notGameOver:
            dqn.board.place(pieceIndex, row, col, rot)
            dqn.boardArray = dqn.getBoardArray(dqn.board.rend(pieceIndex, row, col, rot))
            dqn.next_state = dqn.boardArray
            dqn.previous_score = dqn.score
            dqn.score = dqn.board.getScore()
            dqn.reward = dqn.getReward(dqn.score, dqn.previous_score)

            # print("next_state =\n", dqn.next_state)
            # print("action = ", dqn.action)
            # print("previous_score = ", dqn.previous_score)
            # print("reward = ", dqn.reward)
            # print("score = ", dqn.score)

            dqn.save_experience(dqn.current_state, dqn.next_state, dqn.action, dqn.reward)
        else:
            dqn.reset()

    # print("experiences =\n", dqn.experiences)
    # dqn.writeExperiencesToFile("experiences.txt")

if __name__ == "__main__":
    run()
