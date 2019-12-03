import tetris_boost as tetris
import random
import numpy as np
from tqdm import tqdm


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
        self.singleLineClear = 0
        self.doubleLineClear = 0
        self.tripleLineClear = 0
        self.quadLineClear = 0

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
        print("score:", self.score)

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

    def best_action(self, states):
        if len(states) > 0:
            return states[-1][1]
        return None

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
        self.singleLineClear = 0
        self.doubleLineClear = 0
        self.tripleLineClear = 0
        self.quadLineClear = 0


def play_random(tetris, moves):
    for move in tqdm(range(moves)):
        tetris.current_state = tetris.next_state

        tetris.movesArray = tetris.get_moves_array(tetris.board.getMoves())

        if tetris.board.getNumberOfMoves() > 0:
            moveIndex = random.randint(0, tetris.board.getNumberOfMoves() - 1)
        else:
            tetris.reset()
            continue

        action = tetris.movesArray[moveIndex]

        pieceIndex, row, col, rot = int(action[0]), int(
            action[1]), int(action[2]), int(action[3])

        notGameOver = tetris.board.isValid(pieceIndex, row, col, rot)

        if notGameOver:
            tetris.board.place(pieceIndex, row, col, rot)
            render = tetris.board.rend(pieceIndex, row, col, rot)
            tetris.boardArray = tetris.get_board_array(render)
            tetris.next_state = tetris.boardArray

            tetris.previous_score = tetris.score
            tetris.score = tetris.board.getScore()
            reward = tetris.get_reward(tetris.score, tetris.previous_score)
            
            tetris.print_board()
        else:
            tetris.reset()


def play(tetris, games):
    scores = []
    movesPlayed = []
    singleLineClears = []
    doubleLineClears = []
    tripleLineClears = []
    quadLineClears = []

    for game in tqdm(range(games)):
        tetris.reset()
        notGameOver = True
        numberOfMovesPlayed = 0

        print(f"Game #{game + 1} started.")

        while notGameOver:
            tetris.current_state = tetris.boardArray

            next_states = tetris.get_next_states()

            best_action = tetris.best_action(next_states)

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

            if reward == 40:
                tetris.singleLineClear += 1
            elif reward == 100:
                tetris.doubleLineClear += 1
            elif reward == 300:
                tetris.tripleLineClear += 1
            elif reward == 1200:
                tetris.quadLineClear += 1

            tetris.print_board()

            numberOfMovesPlayed += 1

            tetris.current_state = tetris.boardArray

        scores.append(tetris.score)
        movesPlayed.append(numberOfMovesPlayed)


        singleLineClears.append(tetris.singleLineClear)
        doubleLineClears.append(tetris.doubleLineClear)
        tripleLineClears.append(tetris.tripleLineClear)
        quadLineClears.append(tetris.quadLineClear)

        print_to_file(scores, "scores.csv")
        print_to_file(movesPlayed, "movesPlayed.csv")

        print_to_file(singleLineClears, "singleLineClears.csv")
        print_to_file(doubleLineClears, "doubleLineClears.csv")
        print_to_file(tripleLineClears, "tripleLineClears.csv")
        print_to_file(quadLineClears, "quadLineClears.csv")

    print_stats(scores, "scores")
    print_stats(movesPlayed, "movesPlayed")

    print_stats(singleLineClears, "singleLineClears")
    print_stats(doubleLineClears, "doubleLineClears")
    print_stats(tripleLineClears, "tripleLineClears")
    print_stats(quadLineClears, "quadLineClears")


def print_stats(array, label):
    print(f"{label}:\n", array)
    print(f"min({label}):", min(array))
    print(f"max({label}):", max(array))
    print(f"mean({label}):", np.mean(array))


def print_to_file(array, filename):
    with open(f"{filename}", "a") as file:
        np.savetxt(filename, array, delimiter=",")


def read_from_file(filename):
    with open(f"{filename}", "r") as file:
        return np.loadtxt(filename, delimiter=",")


def main():
    tetris = Tetris()
    
    play_random(tetris, moves=1000)

    play(tetris, games=10)

    # print("scores from file:", read_from_file("scores.csv"))
    # print("movesPlayed from file:", read_from_file("movesPlayed.csv"))
    
    # print("singleLineClears from file:", read_from_file("singleLineClears.csv"))
    # print("doubleLineClears from file:", read_from_file("doubleLineClears.csv"))
    # print("tripleLineClears from file:", read_from_file("tripleLineClears.csv"))
    # print("quadLineClears from file:", read_from_file("quadLineClears.csv"))


if __name__ == "__main__":
    main()
