import numpy as np


BOARD_SIZE = 20, 10


def raw(file_name="../full_observations_A=0.0_U=20.0.txt", start_char='A', end_char='Z'):
    with open(file_name, "r") as file:
        # Filters out EOF newline and anything else that's not an observation
        observations = list(filter(lambda ch: start_char <= ch <= end_char, file.read()))
        symbols = np.unique(observations)
        # Map each symbol to a number in [0, 1, ..., m - 1] where is the number of symbols
        observations = list(map(lambda ch: symbols.searchsorted(ch), observations))
        # Put it in format for hmm library [[obs1], [obs2], ...]
        return np.array([observations]).T


def compressed(raw_obs=None):
    if not raw_obs:
        raw_obs = raw().T[0]

    compressed_obs = []

    symbols = np.unique(raw_obs)
    for start_index in range(0, len(raw_obs), BOARD_SIZE[1]):
        end_index = start_index + BOARD_SIZE[1]
        board = raw_obs[start_index:end_index]
        min_obs = min(board)
        for i in range(0, len(board)):
            # This represents a column with no blocks in it
            if board[i] == symbols[-1]:
                # Give this column a unique symbol
                board[i] = 0
            else:
                board[i] -= min_obs - 1
        compressed_obs.extend(board)

    return np.array([compressed_obs]).T


# hmmlearn expects no intermediate symbols skipped (if 20 is in the list, 0-19 should be too), this adds missing symbols
def to_training_input(observations):
    symbols = np.unique(observations.T[0])
    full_symbols = list(range(0, BOARD_SIZE[0] + 1))
    missing_symbols = np.setdiff1d(full_symbols, symbols)
    print(missing_symbols)
    print(observations.T[0].shape)
    print(missing_symbols.shape)
    return np.array([np.concatenate((observations.T[0], missing_symbols))]).T


# Returns an array of 10 numbers, each number represents the location of the highest occupied block in each column
def from_board(board, start_row=0):
    obs = np.zeros(BOARD_SIZE[1])
    for col in range(0, BOARD_SIZE[1]):
        obs[col] = BOARD_SIZE[0]
        for row in range(start_row, BOARD_SIZE[0]):
            if board[row][col]:
                obs[col] = row
                break

    return obs
