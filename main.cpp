// TODO: Move hardcoded values to header file
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <vector>
#include "main.h"

using namespace std;
using namespace std::chrono;
using namespace tetris;

class Piece {
private:
    array<array<bool, PIECE_SIZE>, PIECE_SIZE> shape = {{false}};
    int row;
    int col;
    int rot;

public:
    explicit Piece(array<array<bool, PIECE_SIZE>, PIECE_SIZE> shape) : Piece(shape, 0, 3) {}

    explicit Piece(array<array<bool, PIECE_SIZE>, PIECE_SIZE> shape, int row, int col) {
        this->shape = shape;
        this->row = row;
        this->col = col;
        this->rot = 0;
    }

    // Rotates the piece to the right n times
    void rotate() {
        for (int r = 0; r < PIECE_SIZE / 2; r++) {
            int rowInverse = PIECE_SIZE - r - 1;
            for (int c = r; c < rowInverse; c++) {
                int colInverse = PIECE_SIZE - c - 1;

                int temp = shape[r][c];
                shape[r][c] = shape[c][rowInverse];
                shape[c][rowInverse] = shape[rowInverse][colInverse];
                shape[rowInverse][colInverse] = shape[colInverse][r];
                shape[colInverse][r] = temp;
            }
        }

        rot = (rot + 1) % 4;
    }

    array<array<bool, PIECE_SIZE>, PIECE_SIZE> getShape() {
        return shape;
    }

    int getRow() {
        return row;
    }

    int getCol() {
        return col;
    }

    void setRow(int newRow) {
        row = newRow;
    }

    void setCol(int newCol) {
        col = newCol;
    }

    void setPos(int newRow, int newCol) {
        setRow(newRow);
        setCol(newCol);
    }

    void setRot(int newRot) {
        while (rot != newRot) rotate();
    }
};

class Board {
private:
    bool board[BOARD_HEIGHT][BOARD_WIDTH] = {{false}};
    // Keeps track of visited branches (prevents infinite recursion)
    bool visited[BOARD_HEIGHT + PIECE_SIZE][BOARD_WIDTH + PIECE_SIZE][4] = {{false}};
    // Keeps track of which branches are valid: 0 means not yet determined, -1 means invalid, 1 means valid
    int memo[BOARD_HEIGHT + PIECE_SIZE][BOARD_WIDTH + PIECE_SIZE][4] = {{0}};

    vector<array<int, 2>> getMoves(Piece *piece, int row, int col, int rot) {
        resetMemo();
        resetVisited();
        piece->setRot(rot);

        array<array<bool, PIECE_SIZE>, PIECE_SIZE> shape = piece->getShape();
        vector<array<int, 2>> moves;

        for (int r = 0; r < PIECE_SIZE; r++) {
            for (int c = 0; c < PIECE_SIZE; c++) {
                if (shape[r][c]) {
                    if (isMoveValid(piece, row - r, col - c, 0)) {
                        moves.push_back({{row - r, col - c}});
                    }
                    resetVisited();
                    piece->setRot(rot);
                }
            }
        }


        return moves;
    }

    void resetMemo() {
        for (int i = 0; i < BOARD_HEIGHT + PIECE_SIZE; i++) {
            for (int j = 0; j < BOARD_WIDTH + PIECE_SIZE; j++) {
                for (int k = 0; k < 4; k++) {
                    memo[i][j][k] = 0;
                }
            }
        }
    }

    void resetVisited() {
        for (int i = 0; i < BOARD_HEIGHT + PIECE_SIZE; i++) {
            for (int j = 0; j < BOARD_WIDTH + PIECE_SIZE; j++) {
                for (int k = 0; k < 4; k++) {
                    visited[i][j][k] = false;
                }
            }
        }
    }

    bool isMoveValid(Piece *piece, int row, int col, int rot) {
        if (row == 0 && col == 3) return true;
        // Already visited this branch, terminate
        else if (visited[row + PIECE_SIZE][col + PIECE_SIZE][rot]) return false;
        else if (memo[row + PIECE_SIZE][col + PIECE_SIZE][rot] < 0) return false;
        else if (memo[row + PIECE_SIZE][col + PIECE_SIZE][rot] > 0) return true;

        visited[row + PIECE_SIZE][col + PIECE_SIZE][rot] = true;

        array<array<bool, PIECE_SIZE>, PIECE_SIZE> shape = piece->getShape();
        vector<array<int, 2>> moves;

        for (int r = 0; r < PIECE_SIZE; r++) {
            for (int c = 0; c < PIECE_SIZE; c++) {
                if (shape[r][c] && (
                        isOutOfBounds(row + r, col + c) ||
                        board[row + r][col + c]
                )) {
                    memo[row + PIECE_SIZE][col + PIECE_SIZE][rot] = -1;
                    return false;
                }
            }
        }

        bool canAccess = isMoveValid(piece, row - 1, col, rot) ||
                         isMoveValid(piece, row, col - 1, rot) ||
                         isMoveValid(piece, row, col + 1, rot);

        if (canAccess) {
            memo[row + PIECE_SIZE][col + PIECE_SIZE][rot] = 1;
            return true;
        }

        for (int n = 1; n < 4; n++) {
            int newRot = (rot + n) % 4;
            piece->setRot(newRot);
            if (isMoveValid(piece, row, col, newRot)) {
                memo[row + PIECE_SIZE][col + PIECE_SIZE][rot] = 1;
                return true;
            }
        }

        return false;
    }

    static bool isOutOfBounds(int row, int col) {
        return row < 0 || row >= BOARD_HEIGHT ||
               col < 0 || col >= BOARD_WIDTH;
    }

public:
    vector<array<int, 2>> getMoves(Piece *piece) {
        vector<array<int, 2>> moves;

        for (int r = 0; r < BOARD_HEIGHT; r++) {
            for (int c = 0; c < BOARD_WIDTH; c++) {
                if (!board[r][c] && (r + 1 >= BOARD_HEIGHT || board[r + 1][c])) {
                    // TODO: Check for all rotations of piece
                    vector<array<int, 2>> partialMoves = getMoves(piece, r, c, 0);
                    moves.insert(moves.end(), partialMoves.begin(), partialMoves.end());
                }
            }
        }

        return moves;
    }

    // Returns true if piece can fit on board
    bool isValid(Piece *piece) {
        array<array<bool, PIECE_SIZE>, PIECE_SIZE> shape = piece->getShape();
        int row = piece->getRow();
        int col = piece->getCol();

        for (int r = 0; r < PIECE_SIZE; r++) {
            for (int c = 0; c < PIECE_SIZE; c++) {

                if (shape[r][c] && (
                        isOutOfBounds(row + r, col + c) ||
                        board[row + r][col + c]
                )) {
                    return false;
                }

            }
        }

        return true;
    }

    void place(Piece *piece) {
        array<array<bool, PIECE_SIZE>, PIECE_SIZE> shape = piece->getShape();
        int row = piece->getRow();
        int col = piece->getCol();

        for (int r = 0; r < PIECE_SIZE; r++) {
            for (int c = 0; c < PIECE_SIZE; c++) {
                board[row + r][col + c] |= shape[r][c];
            }
        }
    }

    void render(Piece *piece) {
        array<array<bool, PIECE_SIZE>, PIECE_SIZE> shape = piece->getShape();
        int row = piece->getRow();
        int col = piece->getCol();

        for (int r = 0; r < BOARD_HEIGHT; r++) {
            for (int c = 0; c < BOARD_WIDTH; c++) {
                bool renderPiece = r - row >= 0 && r - row < PIECE_SIZE &&
                                   c - col >= 0 && c - col < PIECE_SIZE &&
                                   shape[r - row][c - col];

                cout << ((board[r][c] || renderPiece) ? "X" : "-");
            }
            cout << "\n";
        }
        cout << endl;
    }
};


int main() {
    auto *board = new Board;

    array<array<bool, PIECE_SIZE>, PIECE_SIZE> straightShape = {{
                                                                        {false, false, false, false},
                                                                        {true, true, true, true},
                                                                        {false, false, false, false},
                                                                        {false, false, false, false}
                                                                }};
    auto *piece = new Piece(straightShape);

    board->render(piece);

    milliseconds start = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
    for (int n = 0; n < 20; n++) {
        vector<array<int, 2>> moves = board->getMoves(piece);
        if (moves.empty()) break;
        array<int, 2> move = moves[rand() % moves.size()];
        piece->setPos(move[0], move[1]);
        piece->setRot(0);
        board->place(piece);

//        board->render(piece);
//
//        this_thread::sleep_for(chrono::milliseconds(1000));
    }
    milliseconds end = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
    cout << (end.count() - start.count()) << "\n";

    cout << "\n" << "\n" << "\n" << "MOVES:\n";
    vector<array<int, 2>> moves = board->getMoves(piece);
    for (array<int, 2> move : moves) {
        piece->setPos(move[0], move[1]);
        piece->setRot(0);
        board->render(piece);
    }

    return 0;
}