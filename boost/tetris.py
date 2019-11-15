import tetris

t = tetris.Board()

p0 = tetris.Piece(0, 0 , 0 , 0)
p1 = tetris.Piece(1, 1, 1, 1)

# print("t.getMoves(p0): ", t.getMoves(p0))

# print("t.getMoves(p0): ", t.getMoves(p0))

print("t.place(p0): ", t.place(p0))
print("t.render(p0): ", t.render(p0))

# print("t.getMoves(p1): ", t.getMoves(p1))

print("t.place(p1): ", t.place(p1))
print("t.render(p1): ", t.render(p1))
