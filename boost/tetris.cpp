#include <array>
#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <cstdlib>
#include <iostream>
#include <random>
#include <set>

using namespace std;

namespace tetris {
  const int BOARD_WIDTH = 10;
  const int BOARD_HEIGHT = 20;
  const int PIECE_SIZE = 4;
  const int ROTATIONS = 4;
  const int NUM_SHAPES = 8;

  // Scores for 1, 2, ..., n line clears with a single placement
  const array<int, PIECE_SIZE> SCORES = {{40, 100, 300, 1200}};

  const array<array<bool, PIECE_SIZE>, PIECE_SIZE> BLANK_SHAPE = {
      {{false, false, false, false},
      {false, false, false, false},
      {false, false, false, false},
      {false, false, false, false}}};
  const array<array<bool, PIECE_SIZE>, PIECE_SIZE> STRAIGHT_SHAPE = {
      {{false, false, false, false},
      {true, true, true, true},
      {false, false, false, false},
      {false, false, false, false}}};
  const array<array<bool, PIECE_SIZE>, PIECE_SIZE> L_SHAPE = {
      {{false, false, false, false},
      {false, true, true, false},
      {false, true, false, false},
      {false, true, false, false}}};
  const array<array<bool, PIECE_SIZE>, PIECE_SIZE> FLIPPED_L_SHAPE = {
      {{false, false, false, false},
      {false, true, true, false},
      {false, false, true, false},
      {false, false, true, false}}};
  const array<array<bool, PIECE_SIZE>, PIECE_SIZE> ZIG_ZAG_SHAPE = {
      {{false, false, true, false},
      {false, true, true, false},
      {false, true, false, false},
      {false, false, false, false}}};
  const array<array<bool, PIECE_SIZE>, PIECE_SIZE> FLIPPED_ZIG_ZAG_SHAPE = {
      {{false, true, false, false},
      {false, true, true, false},
      {false, false, true, false},
      {false, false, false, false}}};
  const array<array<bool, PIECE_SIZE>, PIECE_SIZE> SQUARE_SHAPE = {
      {{false, false, false, false},
      {false, true, true, false},
      {false, true, true, false},
      {false, false, false, false}}};
  const array<array<bool, PIECE_SIZE>, PIECE_SIZE> T_SHAPE = {
      {{false, false, false, false},
      {true, true, true, false},
      {false, true, false, false},
      {false, false, false, false}}};
    
  const array<array<array<bool, PIECE_SIZE>, PIECE_SIZE>, NUM_SHAPES> SHAPES = {
    BLANK_SHAPE, STRAIGHT_SHAPE, L_SHAPE, FLIPPED_L_SHAPE, ZIG_ZAG_SHAPE, FLIPPED_L_SHAPE, SQUARE_SHAPE, T_SHAPE
  };
}

using namespace tetris;

typedef vector<int> VectorInt;
typedef vector<bool> VectorBool;

class Piece {
 private:
  array<array<bool, PIECE_SIZE>, PIECE_SIZE> shape;
  int row;
  int col;
  int rot;
  int pieceIndex;

 public:
  explicit Piece(array<array<bool, PIECE_SIZE>, PIECE_SIZE> shape)
      : Piece(shape, 0, 3) { }

  explicit Piece(array<array<bool, PIECE_SIZE>, PIECE_SIZE> shape, int row,
                 int col) {
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

  explicit Piece() {}

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

  array<array<bool, PIECE_SIZE>, PIECE_SIZE> getShape() { return shape; }

  int getRow() { return row; }

  int getCol() { return col; }

  void setRow(int newRow) { row = newRow; }

  void setCol(int newCol) { col = newCol; }

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
  vector<VectorInt> currentMoves;
  array<Piece *, 7> pieces;
  bool board[BOARD_HEIGHT][BOARD_WIDTH] = {{false}};
  //bool board[BOARD_HEIGHT][BOARD_WIDTH] = {false};
  int score = 0;
  // Keeps track of visited branches (prevents infinite recursion)
  bool visited[BOARD_HEIGHT + PIECE_SIZE][BOARD_WIDTH + PIECE_SIZE][ROTATIONS] = {{{false}}};
  // Keeps track of which branches are valid: 0 means not yet determined, -1
  // means invalid, 1 means valid
  int memo[BOARD_HEIGHT + PIECE_SIZE][BOARD_WIDTH + PIECE_SIZE][ROTATIONS] = {{{0}}};

  set<array<int, 3>> getMoves(Piece *piece, int row, int col, int rot) {
      cout << "resetting visited\n";
    resetVisited();
      cout << "post resetting visited\n";
    piece->setRot(rot);

    array<array<bool, PIECE_SIZE>, PIECE_SIZE> shape = piece->getShape();
    set<array<int, 3>> moves;

    for (int r = 0; r < PIECE_SIZE; r++) {
      for (int c = 0; c < PIECE_SIZE; c++) {
        if (shape[r][c]) {
            cout << "checking move valid\n";
            if (isMoveValid(piece, row - r, col - c, rot)) {
            moves.insert({{row - r, col - c, rot}});
          }
            cout << "post checking move valid\n";

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

  // Recursively checks if the piece would be able to reach the given position
  // from the starting position (0, 3)
  bool isMoveValid(Piece *piece, int row, int col, int rot) {
    // Already visited this branch, terminate
    if (visited[row + PIECE_SIZE][col + PIECE_SIZE][rot])
      return false;
    else if (memo[row + PIECE_SIZE][col + PIECE_SIZE][rot] < 0)
      return false;
    else if (memo[row + PIECE_SIZE][col + PIECE_SIZE][rot] > 0)
      return true;

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

        if (shape[r][c] &&
            (isOutOfBounds(row + r, col + c) || board[row + r][col + c])) {
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
    return row < 0 || row >= BOARD_HEIGHT || col < 0 || col >= BOARD_WIDTH;
  }

 public:
  Board() {
    auto *blankPiece = new Piece(BLANK_SHAPE);
    auto *straightPiece = new Piece(STRAIGHT_SHAPE);
    auto *lPiece = new Piece(L_SHAPE);
    auto *flippedLPiece = new Piece(FLIPPED_L_SHAPE);
    auto *zigZagPiece = new Piece(ZIG_ZAG_SHAPE);
    auto *flippedZigZagPiece = new Piece(FLIPPED_ZIG_ZAG_SHAPE);
    auto *squarePiece = new Piece(SQUARE_SHAPE);
    auto *tPiece = new Piece(T_SHAPE);

    pieces = { straightPiece, lPiece, flippedLPiece, zigZagPiece,
        flippedZigZagPiece, squarePiece, tPiece
    };

//    for (int r = 0; r < BOARD_HEIGHT; r++) {
//      for (int c = 0; c < BOARD_WIDTH; c++) {
//        board[r][c] = false;
//      }
//    }
  }

  // Returns a set of possible moves in format: (row, col, rot)
  set<array<int, 3>> getMoves(Piece *piece) {
    cout << "get moves piece\n";
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

    cout << "post moves piece\n";
    return moves;
  }

  // Returns a set of possible moves in format: (pieceIndex, row, col, rot)
  set<array<int, 4>> getMoves(array<Piece *, 7> pieces) {
    set<array<int, 4>> moves;

    int i = 0;
    for (Piece *piece : pieces) {
      set<array<int, 3>> partialMoves = getMoves(piece);
      for (array<int, 3> move : partialMoves) {
        moves.insert({i, move[0], move[1], move[2]});
      }
      i++;
    }

    return moves;
  }

  void place(Piece *piece) {
    array<array<bool, PIECE_SIZE>, PIECE_SIZE> shape = piece->getShape();
    int row = piece->getRow();
    int col = piece->getCol();

    int rowsCompleted = 0;
    for (int r = 0; r < PIECE_SIZE; r++) {
      for (int c = 0; c < PIECE_SIZE; c++) {
        // if (isOutOfBounds(row + r, col + c))
        board[row + r][col + c] |= shape[r][c];
      }

      if (isRowCompleted(row + r)) {
        deleteRow(row + r);
        rowsCompleted++;
      }
    }

    score += SCORES[rowsCompleted];
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

  //wrap
  vector<VectorInt> getMoves() {
    currentMoves.clear();
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> distribution(0, 6);
    int randomIndex = distribution(gen);
    auto *piece = pieces[randomIndex];

    // vector<VectorInt> moves;
    set<array<int, 3>> partialMoves = getMoves(piece);

    for (array<int, 3> move : partialMoves) {
      currentMoves.push_back({randomIndex, move[0], move[1], move[2]});
    }

    return currentMoves;
  }

  int getNumberOfMoves() {
    return currentMoves.size();
  }

  //wrap
  void printMoves(const vector<VectorInt> &vvi) {
    for (int i = 0; i < vvi.size(); i++) {
      for (int j = 0; j < vvi[i].size(); j++) {
        cout << vvi[i][j] << " ";
      }
      cout << endl;
    }
  }

  //wrap
  void place(int pieceIndex, int row, int col, int rot) {
    auto *piece = pieces[pieceIndex];
    piece->setRow(row);
    piece->setCol(col);
    piece->setRot(rot);
    place(piece);
  }

  //wrap
  vector<VectorBool> rend(int pieceIndex, int row, int col, int rot) {
    array<array<bool, PIECE_SIZE>, PIECE_SIZE> shape = SHAPES[pieceIndex];
    auto *piece = pieces[pieceIndex];

    vector<VectorBool> newBoard;
    VectorBool vb;

    for (int r = 0; r < BOARD_HEIGHT; r++) {
      for (int c = 0; c < BOARD_WIDTH; c++) {
        bool renderPiece = r - row >= 0 && r - row < PIECE_SIZE &&
                           c - col >= 0 && c - col < PIECE_SIZE &&
                           shape[r - row][c - col];

        vb.push_back(renderPiece || board[r][c]);
      }
      newBoard.push_back(vb);
      vb.clear();
    }

    return newBoard;
  }

  int getValueOfVectorInts(vector<VectorInt> &vvi, int i, int j) {
    return vvi[i][j];
  }

  bool getValueOfVectorBools(vector<VectorBool> &vvb, int i, int j) {
    return vvb[i][j];
  }

  int getScore() {
    return score;
  }

  void reset() {
    resetMemo();
    resetVisited();
    score = 0;

    for (int row = 0; row < BOARD_HEIGHT; row++) {
      for (int col = 0; col < BOARD_WIDTH; col++) {
        board[row][col] = false;
      }
    }
  }

  // Returns true if piece can fit on board
  bool isValid(Piece *piece) {
    array<array<bool, PIECE_SIZE>, PIECE_SIZE> shape = piece->getShape();
    int row = piece->getRow();
    int col = piece->getCol();

    for (int r = 0; r < PIECE_SIZE; r++) {
      for (int c = 0; c < PIECE_SIZE; c++) {
        if (shape[r][c] &&
            (isOutOfBounds(row + r, col + c) || board[row + r][col + c])) {
          return false;
        }
      }
    }

    return true;
  }

  // Returns true if piece can fit on board
  bool isValid(int pieceIndex, int row, int col, int rot) {
    auto *piece = pieces[pieceIndex];
    piece->setRow(row);
    piece->setCol(col);
    piece->setRot(rot);
    return isValid(piece);
  }

  //wrap
  void printRend(const vector<VectorBool> &vvb) {
    for (int i = 0; i < vvb.size(); i++) {
      for (int j = 0; j < vvb[i].size(); j++) {
        // cout << "vvb[" << i << "][" << j << "] = " << vvb[i][j] << " ";
        cout << vvb[i][j] << " ";
      }
       cout << endl;
    }
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

  array<Piece *, 7> pieces = {
      straightPiece, lPiece, flippedLPiece, zigZagPiece,
      flippedZigZagPiece, squarePiece, tPiece
  };

  board->render(blankPiece);

  for (int n = 0; n < 10; n++) {
    set<array<int, 4>> moves = board->getMoves(pieces);
    if (moves.empty()) break;
    int selectedMove = rand() % moves.size();
    int i = 0;
    for (array<int, 4> move : moves) {
      if (i++ == selectedMove) {
        cout << move[0] << ", " << move[1] << ", " << move[2] << ", " << move[3]
             << "\n";
        pieces[move[0]]->setPos(move[1], move[2]);
        pieces[move[0]]->setRot(move[3]);
        board->place(pieces[move[0]]);
        array<array<bool, PIECE_SIZE>, PIECE_SIZE> shape =
            pieces[move[0]]->getShape();
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
  }

  cout << "\n"
       << "\n"
       << "\n"
       << "Next moves:\n";
  set<array<int, 4>> moves = board->getMoves(pieces);
  for (array<int, 4> move : moves) {
    cout << move[0] << ", " << move[1] << ", " << move[2] << ", " << move[3]
         << "\n";
    pieces[move[0]]->setPos(move[1], move[2]);
    pieces[move[0]]->setRot(move[3]);
    array<array<bool, PIECE_SIZE>, PIECE_SIZE> shape =
        straightPiece->getShape();
    for (int r = 0; r < PIECE_SIZE; r++) {
      for (int c = 0; c < PIECE_SIZE; c++) {
        cout << (shape[r][c] ? "X" : "-");
      }
      cout << "\n";
    }
    board->render(pieces[move[0]]);
  }

  return 0;
}

BOOST_PYTHON_MODULE(tetris) {
  using namespace boost::python;

  void (Board::*place)(int pieceIndex, int row, int col, int rot) = &Board::place;

  vector<VectorInt> (Board::*getMoves)() = &Board::getMoves;

  bool (Board::*isValid)(int pieceIndex, int row, int col, int rot) = &Board::isValid;

  class_<vector<VectorBool>>("vector<VectorBool>")
      .def(vector_indexing_suite<vector<VectorBool>>());

  class_<vector<VectorInt>>("vector<VectorInt>")
      .def(vector_indexing_suite<vector<VectorInt>>());

  class_<Board>("Board")
      .def("place", place)
      .def("getMoves", getMoves)
      .def("rend", &Board::rend)
      .def("printRend", &Board::printRend)
      .def("printMoves", &Board::printMoves)
      .def("getValueOfVectorInts", &Board::getValueOfVectorInts)
      .def("getValueOfVectorBools", &Board::getValueOfVectorBools)
      .def("getScore", &Board::getScore)
      .def("reset", &Board::reset)
      .def("isValid", isValid)
      .def("getNumberOfMoves", &Board::getNumberOfMoves);
}
