import tetris

board = tetris.Board()
print("type(board) = ", type(board))
print("board", board)

moves = board.getMoves()
print("type(moves) = ", type(moves))
print("moves = ", moves)

printMoves1 = board.printMoves(moves)
print("type(printMoves1) = ", type(printMoves1))
print("printMoves1 = ", printMoves1)

# val1, val2, val3, val4 = 6, 3, 4, 1

# place1 = board.place(val1, val2, val3, val4)
# print("type(place1) = ", type(place1))
# print("place1 = ", place1)

# render1 = board.rend(val1, val2, val3, val4)
# print("type(render1) = ", type(render1))
# print("render1 = ", render1)

# printRender1 = board.printRend(render1)
# print("printRend1 = ", printRender1)
