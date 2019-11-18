#include <chrono>
#include <cstdlib>
#include <iostream>
#include <set>
#include <thread>
#include <vector>
#include <boost/python.hpp>

using namespace std;
using namespace std::chrono;

namespace tetris {
    const int BOARD_WIDTH = 10;
    const int BOARD_HEIGHT = 20;
    const int PIECE_SIZE = 4;
    const int ROTATIONS = 4;

    // Scores for 1, 2, ..., n line clears with a single placement
    const std::array<int, PIECE_SIZE> SCORES = {{40, 100, 300, 1200}};

    const std::array<std::array<bool, PIECE_SIZE>, PIECE_SIZE>
        BLANK_SHAPE = {{{false, false, false, false},
                        {false, false, false, false},
                        {false, false, false, false},
                        {false, false, false, false}}};
    const std::array<std::array<bool, PIECE_SIZE>, PIECE_SIZE>
        STRAIGHT_SHAPE = {{{false, false, false, false},
                        {true, true, true, true},
                        {false, false, false, false},
                        {false, false, false, false}}};
    const std::array<std::array<bool, PIECE_SIZE>, PIECE_SIZE>
        L_SHAPE = {{{false, false, false, false},
                    {false, true, true, false},
                    {false, true, false, false},
                    {false, true, false, false}}};
    const std::array<std::array<bool, PIECE_SIZE>, PIECE_SIZE>
        FLIPPED_L_SHAPE = {{{false, false, false, false},
                            {false, true, true, false},
                            {false, false, true, false},
                            {false, false, true, false}}};
    const std::array<std::array<bool, PIECE_SIZE>, PIECE_SIZE>
        ZIG_ZAG_SHAPE = {{{false, false, true, false},
                        {false, true, true, false},
                        {false, true, false, false},
                        {false, false, false, false}}};
    const std::array<std::array<bool, PIECE_SIZE>, PIECE_SIZE>
        FLIPPED_ZIG_ZAG_SHAPE = {{{false, true, false, false},
                                {false, true, true, false},
                                {false, false, true, false},
                                {false, false, false, false}}};
    const std::array<std::array<bool, PIECE_SIZE>, PIECE_SIZE>
        SQUARE_SHAPE = {{{false, false, false, false},
                        {false, true, true, false},
                        {false, true, true, false},
                        {false, false, false, false}}};
    const std::array<std::array<bool, PIECE_SIZE>, PIECE_SIZE>
        T_SHAPE = {{{false, false, false, false},
                    {true, true, true, false},
                    {false, true, false, false},
                    {false, false, false, false}}};
}

using namespace tetris;

class Piece {
private:
    array<array<bool, PIECE_SIZE>, PIECE_SIZE> shape = {{false}};
    int row;
    int col;
    int rot;
    int pieceIndex;

public:
    explicit Piece(array<array<bool, PIECE_SIZE>, PIECE_SIZE> shape) : Piece(shape, 0, 3) {}

    explicit Piece(array<array<bool, PIECE_SIZE>, PIECE_SIZE> shape, int row, int col) {
        this->shape = shape;
        this->row = row;
        this->col = col;
        this->rot = 0;
    }

    explicit Piece(int pieceIndex, int row, int col, int rot) {
        this->pieceIndex = pieceIndex;
        this->row = row;
        this->col = col;
        this->rot = rot;
    }

    explicit Piece() { }

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

        rot = (rot + 1) % ROTATIONS;
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
    int score = 0;
    // Keeps track of visited branches (prevents infinite recursion)
    bool visited[BOARD_HEIGHT + PIECE_SIZE][BOARD_WIDTH + PIECE_SIZE][ROTATIONS] = {{false}};
    // Keeps track of which branches are valid: 0 means not yet determined, -1 means invalid, 1 means valid
    int memo[BOARD_HEIGHT + PIECE_SIZE][BOARD_WIDTH + PIECE_SIZE][ROTATIONS] = {{0}};

    set<array<int, 3>> getMoves(Piece *piece, int row, int col, int rot) {
        resetVisited();
        piece->setRot(rot);

        array<array<bool, PIECE_SIZE>, PIECE_SIZE> shape = piece->getShape();
        set<array<int, 3>> moves;

        for (int r = 0; r < PIECE_SIZE; r++) {
            for (int c = 0; c < PIECE_SIZE; c++) {
                if (shape[r][c]) {
                    if (isMoveValid(piece, row - r, col - c, rot)) {
                        moves.insert({{row - r, col - c, rot}});
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
                for (int k = 0; k < ROTATIONS; k++) {
                    memo[i][j][k] = 0;
                }
            }
        }
    }

    void resetVisited() {
        for (int i = 0; i < BOARD_HEIGHT + PIECE_SIZE; i++) {
            for (int j = 0; j < BOARD_WIDTH + PIECE_SIZE; j++) {
                for (int k = 0; k < ROTATIONS; k++) {
                    visited[i][j][k] = false;
                }
            }
        }
    }

    // Recursively checks if the piece would be able to reach the given position from the starting position (0, 3)
    bool isMoveValid(Piece *piece, int row, int col, int rot) {
        // Already visited this branch, terminate
        if (visited[row + PIECE_SIZE][col + PIECE_SIZE][rot]) return false;
        else if (memo[row + PIECE_SIZE][col + PIECE_SIZE][rot] < 0) return false;
        else if (memo[row + PIECE_SIZE][col + PIECE_SIZE][rot] > 0) return true;

        visited[row + PIECE_SIZE][col + PIECE_SIZE][rot] = true;

        array<array<bool, PIECE_SIZE>, PIECE_SIZE> shape = piece->getShape();

        bool foundFirstBlock = false;
        for (int r = 0; r < PIECE_SIZE; r++) {
            for (int c = 0; c < PIECE_SIZE; c++) {
                if (shape[r][c]) {
                    // This is where the piece always starts in Tetris
                    if (!foundFirstBlock && row + r == 0 && col + c == 3) {
                        memo[row + PIECE_SIZE][col + PIECE_SIZE][rot] = 1;
                        return true;
                    }
                    foundFirstBlock = true;
                }

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

        for (int n = 1; n < ROTATIONS; n++) {
            int newRot = (rot + n) % ROTATIONS;
            piece->setRot(newRot);
            if (isMoveValid(piece, row, col, newRot)) {
                memo[row + PIECE_SIZE][col + PIECE_SIZE][rot] = 1;
                return true;
            }
        }

        return false;
    }

    void deleteRow(int row) {
        for (int r = row; r > 0; r--) {
            for (int c = 0; c < BOARD_WIDTH; c++) {
                board[r][c] = board[r - 1][c];
            }
        }
    }

    bool isRowCompleted(int row) {
        for (int c = 0; c < BOARD_WIDTH; c++) {
            if (!board[row][c]) return false;
        }

        return true;
    }

    static bool isOutOfBounds(int row, int col) {
        return row < 0 || row >= BOARD_HEIGHT ||
               col < 0 || col >= BOARD_WIDTH;
    }

public:

    // array<Piece, >
    //wrap
    // Returns a set of possible moves in format: (row, col, rot)
    set<array<int, 3>> getMoves(Piece *piece) {
        resetMemo();

        set<array<int, 3>> moves;

        for (int r = 0; r < BOARD_HEIGHT; r++) {
            for (int c = 0; c < BOARD_WIDTH; c++) {
                if (!board[r][c] && (r + 1 >= BOARD_HEIGHT || board[r + 1][c])) {
                    for (int rot = 0; rot < ROTATIONS; rot++) {
                        set<array<int, 3>> partialMoves = getMoves(piece, r, c, rot);
                        moves.insert(partialMoves.begin(), partialMoves.end());
                    }
                }
            }
        }

        return moves;
    }

    // set<array<int, 3>> getMoves()
    // {
    //     resetMemo();

    //     set<array<int, 3>> moves;

    //     for (int r = 0; r < BOARD_HEIGHT; r++)
    //     {
    //         for (int c = 0; c < BOARD_WIDTH; c++)
    //         {
    //             if (!board[r][c] && (r + 1 >= BOARD_HEIGHT || board[r + 1][c]))
    //             {
    //                 for (int rot = 0; rot < ROTATIONS; rot++)
    //                 {
    //                     set<array<int, 3>> partialMoves = getMoves(piece, r, c, rot);
    //                     moves.insert(partialMoves.begin(), partialMoves.end());
    //                 }
    //             }
    //         }
    //     }

    //     return moves;
    // }

    //wrap
    // Returns a set of possible moves in format: (pieceIndex, row, col, rot)
    set<array<int, 4>> getMoves(array<Piece*, 7> pieces) {
        set<array<int, 4>> moves;

        int i = 0;
        for (Piece* piece : pieces) {
            set<array<int, 3>> partialMoves = getMoves(piece);
            for (array<int, 3> move : partialMoves) {
                moves.insert({i, move[0], move[1], move[2]});
            }
            i++;
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

    //wrap
    void place(Piece *piece) {
        array<array<bool, PIECE_SIZE>, PIECE_SIZE> shape = piece->getShape();
        int row = piece->getRow();
        int col = piece->getCol();

        int rowsCompleted = 0;
        for (int r = 0; r < PIECE_SIZE; r++) {
            for (int c = 0; c < PIECE_SIZE; c++) {
                if (isOutOfBounds(row + r, col + c))
                    board[row + r][col + c] |= shape[r][c];
            }

            if (isRowCompleted(row + r)) {
                deleteRow(row + r);
                rowsCompleted++;
            }
        }

        score += SCORES[rowsCompleted];
    }

    //wrap - return boolean array for new board
    void render(Piece *piece) {
        array<array<bool, PIECE_SIZE>, PIECE_SIZE> shape = piece->getShape();
        int row = piece->getRow();
        int col = piece->getCol();

        for (int r = 0; r < BOARD_HEIGHT; r++) {
            for (int c = 0; c < BOARD_WIDTH; c++) {
                bool renderPiece = r - row >= 0 && r - row < PIECE_SIZE &&
                                   c - col >= 0 && c - col < PIECE_SIZE &&
                                   shape[r - row][c - col];

                if (renderPiece) {
                    cout << "O";
                } else if (board[r][c]) {
                    cout << "X";
                } else {
                    cout << "-";
                }
            }
            cout << "\n";
        }
        cout << endl;
    }
};

int main() {
    srand(23477846);

    auto *board = new Board;

    auto *blankPiece = new Piece(BLANK_SHAPE);
    auto *straightPiece = new Piece(STRAIGHT_SHAPE);
    auto *lPiece = new Piece(L_SHAPE);
    auto *flippedLPiece = new Piece(FLIPPED_L_SHAPE);
    auto *zigZagPiece = new Piece(ZIG_ZAG_SHAPE);
    auto *flippedZigZagPiece = new Piece(FLIPPED_ZIG_ZAG_SHAPE);
    auto *squarePiece = new Piece(SQUARE_SHAPE);
    auto *tPiece = new Piece(T_SHAPE);

    array<Piece *, 7> pieces = {straightPiece,
                                lPiece,
                                flippedLPiece,
                                zigZagPiece,
                                flippedZigZagPiece,
                                squarePiece,
                                tPiece};

    board->render(blankPiece);

    milliseconds start = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
    for (int n = 0; n < 10; n++) {
        set<array<int, 4>> moves = board->getMoves(pieces);
        if (moves.empty()) break;
        int selectedMove = rand() % moves.size();
        int i = 0;
        for (array<int, 4> move : moves) {
            if (i++ == selectedMove) {
                cout << move[0] << ", " << move[1] << ", " << move[2] << ", " << move[3] << "\n";
                pieces[move[0]]->setPos(move[1], move[2]);
                pieces[move[0]]->setRot(move[3]);
                board->place(pieces[move[0]]);
                array<array<bool, PIECE_SIZE>, PIECE_SIZE> shape = pieces[move[0]]->getShape();
                for (int r = 0; r < PIECE_SIZE; r++) {
                    for (int c = 0; c < PIECE_SIZE; c++) {
                        cout << (shape[r][c] ? "X" : "-");
                    }
                    cout << "\n";
                }
                board->render(pieces[move[0]]);
                break;
            }
        }

    //        this_thread::sleep_for(chrono::milliseconds(1000));
    }

    milliseconds end = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
    cout << (end.count() - start.count()) << "\n";

    cout << "\n" << "\n" << "\n" << "Next moves:\n";
    set<array<int, 4>> moves = board->getMoves(pieces);
    for (array<int, 4> move : moves) {
        cout << move[0] << ", " << move[1] << ", " << move[2] << ", " << move[3] << "\n";
        pieces[move[0]]->setPos(move[1], move[2]);
        pieces[move[0]]->setRot(move[3]);
        array<array<bool, PIECE_SIZE>, PIECE_SIZE> shape = straightPiece->getShape();
        for (int r = 0; r < PIECE_SIZE; r++) {
            for (int c = 0; c < PIECE_SIZE; c++) {
                cout << (shape[r][c] ? "X" : "-");
            }
            cout << "\n";
        }
        // board->render(straightPiece);
        board->render(pieces[move[0]]);
    }

    return 0;
}

BOOST_PYTHON_MODULE(tetris)
{
    using namespace boost::python;

    // Overloaded functions
    set<array<int, 3>> (Board::*getMoves)(Piece *piece) = &Board::getMoves;

    // set<array<int, 4>> (Board::*getMoves_b)(array<Piece *, 7> pieces) = &Board::getMoves;

    class_<Board>("Board")
        .def("place", &Board::place)
        .def("render", &Board::render)
        .def("getMoves", getMoves);

    class_<Piece>("Piece")
        .def(init<int, int, int, int>())
        .def(init<array<array<bool, PIECE_SIZE>, PIECE_SIZE>>())
        .def(init<array<array<bool, PIECE_SIZE>, PIECE_SIZE>, int, int>());
}
