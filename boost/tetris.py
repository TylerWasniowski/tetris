import tetris

board = tetris.Board()
# print("type(board) = ", type(board))
print("board", board)


moves = board.getMoves()
print("moves = ", moves)


printMoves1 = board.printMoves(moves)
print("printMoves1 = ", printMoves1)


moves_rows = 34
moves_cols = 4

for i in range(moves_rows):
    for j in range(moves_cols):
        movesValues = board.getValueOfVectorInts(moves, i, j)
        print(f"movesValues[{i}][{j}] = ", movesValues)


val1, val2, val3, val4 = 6, 2, 1, 1

place1 = board.place(val1, val2, val3, val4)

render1 = board.rend(val1, val2, val3, val4)

printRender1 = board.printRend(render1)


# class DQNAgent:
#     def __init__(self, experiences):
#         self.experiences = []

    # def add_experience(current_state, next_state, reward, done):
    #     experiences.append((current_state, next_state, reward, done))


# def run():
