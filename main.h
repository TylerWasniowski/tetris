//
// Created by rooke on 11/10/2019.
//

#ifndef TETRIS_MAIN_H
#define TETRIS_MAIN_H

namespace tetris {
    const int BOARD_WIDTH = 10;
    const int BOARD_HEIGHT = 20;
    const int PIECE_SIZE = 4;
    const int ROTATIONS = 4;

    // Scores for 1, 2, ..., n line clears with a single placement
    const std::array<int, PIECE_SIZE> SCORES = {{40, 100, 300, 1200}};

    const std::array<std::array<bool, PIECE_SIZE>, PIECE_SIZE>
            BLANK_SHAPE = {{
                                   {false, false, false, false},
                                   {false, false, false, false},
                                   {false, false, false, false},
                                   {false, false, false, false}
                           }};
    const std::array<std::array<bool, PIECE_SIZE>, PIECE_SIZE>
            STRAIGHT_SHAPE = {{
                                      {false, false, false, false},
                                      {true, true, true, true},
                                      {false, false, false, false},
                                      {false, false, false, false}
                              }};
    const std::array<std::array<bool, PIECE_SIZE>, PIECE_SIZE>
            L_SHAPE = {{
                                      {false, false, false, false},
                                      {false, true, true, false},
                                      {false, true, false, false},
                                      {false, true, false, false}
                              }};
    const std::array<std::array<bool, PIECE_SIZE>, PIECE_SIZE>
            FLIPPED_L_SHAPE = {{
                                      {false, false, false, false},
                                      {false, true, true, false},
                                      {false, false, true, false},
                                      {false, false, true, false}
                              }};
    const std::array<std::array<bool, PIECE_SIZE>, PIECE_SIZE>
            ZIG_ZAG_SHAPE = {{
                                      {false, false, true, false},
                                      {false, true, true, false},
                                      {false, true, false, false},
                                      {false, false, false, false}
                              }};
    const std::array<std::array<bool, PIECE_SIZE>, PIECE_SIZE>
            FLIPPED_ZIG_ZAG_SHAPE = {{
                                      {false, true, false, false},
                                      {false, true, true, false},
                                      {false, false, true, false},
                                      {false, false, false, false}
                              }};
    const std::array<std::array<bool, PIECE_SIZE>, PIECE_SIZE>
            SQUARE_SHAPE = {{
                                      {false, false, false, false},
                                      {false, true, true, false},
                                      {false, true, true, false},
                                      {false, false, false, false}
                              }};
    const std::array<std::array<bool, PIECE_SIZE>, PIECE_SIZE>
            T_SHAPE = {{
                                    {false, false, false, false},
                                    {true, true, true, false},
                                    {false, true, false, false},
                                    {false, false, false, false}
                            }};
}

#endif //TETRIS_MAIN_H
