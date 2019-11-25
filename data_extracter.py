# Extracts observation data from this video: https://www.youtube.com/watch?v=DdfRQjb5o9k
# Followed https://www.learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/ for working with video in opencv
import cv2 as cv
import numpy as np

# x, y
BOARD_START_1 = 412, 190
LINE_SCORE_START_1 = 412, 122
LINE_SCORE_END_1 = 503, 171
NEXT_PIECE_START_1 = 522, 122
NEXT_PIECE_END_1 = 612, 171

BOARD_START_2 = 670, 190
LINE_SCORE_START_2 = 670, 122
LINE_SCORE_END_2 = 760, 171
NEXT_PIECE_START_2 = 779, 122
NEXT_PIECE_END_2 = 870, 171

BLOCK_SIZE = 20

# rows, columns
BOARD_SIZE = 20, 10

# The grayscale (max of RGB) value above which is classified as a block 
BLOCK_COLOR_THRESHOLD = 63

# After a new piece enters the board, how many frames should we rewind to get the board state
LOOK_BACK = 3
# Number of frames a line clear lasts until the line score updates
LINE_CLEAR_DURATION = 6

TEXT_START = 0, 100
TEXT_SPACING = 50
FONT_SIZE = 1
FONT_THICKNESS = 2
FONT = cv.FONT_HERSHEY_SIMPLEX
FONT_COLOR = 0, 255, 0


# Returns a 2D boolean array containing true if a block space is occupied, and false if not
def get_blocks(frame, board_start):
    blocks = np.zeros(BOARD_SIZE, dtype=bool)

    row_end = board_start[1] + BOARD_SIZE[0] * BLOCK_SIZE
    col_end = board_start[0] + BOARD_SIZE[1] * BLOCK_SIZE
    cropped_frame = frame[board_start[1]:row_end, board_start[0]:col_end]
    i = 0
    for y in range(BLOCK_SIZE // 2, cropped_frame.shape[0], BLOCK_SIZE):
        row = cropped_frame[y]
        j = 0
        for x in range (BLOCK_SIZE // 2, row.shape[0], BLOCK_SIZE):
            grayscale = np.max(row[x])
            blocks[i, j] = grayscale > BLOCK_COLOR_THRESHOLD
            j += 1

        i += 1

    return blocks


# Draws white blocks over occupied spaces and black blocks over non-occupied spaces
def draw_blocks(frame, blocks, board_start):
    for i in range(0, blocks.shape[0]):
        for j in range(0, blocks.shape[1]):
            if blocks[i][j]:
                top_left = (board_start[0] + j * BLOCK_SIZE, board_start[1] + i * BLOCK_SIZE)
                bottom_right = (board_start[0] + (j + 1) * BLOCK_SIZE, board_start[1] + (i + 1) * BLOCK_SIZE)
                color = 255, 255, 255
                frame = cv.rectangle(frame, top_left, bottom_right, color, cv.FILLED)

    return frame


# Returns an array of 10 numbers, each number represents the location of the highest occupied block in each column
def get_observations(blocks, start_row=0):
    observations = np.zeros(blocks.shape[1])
    for col in range(0, blocks.shape[1]):
        observations[col] = BOARD_SIZE[0]
        for row in range(start_row, blocks.shape[0]):
            if blocks[row][col]:
                observations[col] = row
                break

    return observations


def skip_frames(video, frames):
    # Subtracting 1 because video.read() increments 1
    new_frame_pos = max(0, video.get(cv.CAP_PROP_POS_FRAMES) + frames - 1)
    video.set(cv.CAP_PROP_POS_FRAMES, new_frame_pos)
    ret, frame = video.read()
    return ret, frame, new_frame_pos


def write(frame, text, i):
    pos = (TEXT_START[0], TEXT_START[1] + i * TEXT_SPACING)
    return cv.putText(frame, text, pos, FONT, FONT_SIZE, FONT_COLOR, FONT_THICKNESS)


board_start = BOARD_START_2
line_score_start = LINE_SCORE_START_2
line_score = LINE_SCORE_END_2
next_piece_start = NEXT_PIECE_START_2
next_piece_end = NEXT_PIECE_END_2

frame_start = 0


video = cv.VideoCapture("Finals_2016_Classic_Tetris_World_Championship_clipped.mp4")


video.set(cv.CAP_PROP_POS_FRAMES, frame_start)

frame_pos = frame_start
blocks_count = -1
all_observations = [np.zeros(BOARD_SIZE[1]) + BOARD_SIZE[0]]
while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    blocks = get_blocks(frame, board_start)

    # State change happens when a new piece enters the board space or a line clear occurs
    # Due to rotations, some blocks may be hidden above the board space so we check if 3 or more blocks were added
    # Line clears result in at least 2 less blocks
    blocks_count_new = np.sum(blocks)
    if blocks_count >= 0 and (blocks_count_new > blocks_count + 2 or blocks_count_new < blocks_count - 1):
        ret, frame, frame_pos = skip_frames(video, -LOOK_BACK)
        if not ret:
            break

        blocks = get_blocks(frame, board_start)
        all_observations.append(get_observations(blocks))
        print(str(frame_pos) + ": " + str(all_observations[-1]))
        
        ret, frame, frame_pos = skip_frames(video, LOOK_BACK)
        if not ret:
            break

    # Line clear
    # Many white squares means a tetris has occurred (flashing white board), skip animation
    if blocks_count >= 0 and (blocks_count_new > blocks_count + 4 or blocks_count_new < blocks_count - 1):
        next_piece = frame[next_piece_start[1]:next_piece_end[1], next_piece_start[0]:next_piece_end[0]]
        ret, frame, frame_pos = skip_frames(video, LINE_CLEAR_DURATION)
        if not ret:
            break
        # Wait until next piece (because line clear cooldown time is unpredictable)
        while np.allclose(next_piece, frame[next_piece_start[1]:next_piece_end[1], next_piece_start[0]:next_piece_end[0]], 0, 30):
            ret, frame, frame_pos = skip_frames(video, 1)
            if not ret:
                break

        # Move one frame to have a buffer
        ret, frame, frame_pos = skip_frames(video, 1)
        if not ret:
            break

        # Record new observations
        blocks = get_blocks(frame, board_start)
        blocks_count_new = np.sum(blocks)
        all_observations.append(get_observations(blocks, 5))
        print(str(frame_pos) + ": " + str(all_observations[-1]))
    blocks_count = blocks_count_new

    frame = draw_blocks(frame, blocks, board_start)
    frame = write(frame, "Frame: " + str(frame_pos), 0)
    frame = write(frame, "# Blocks: " + str(blocks_count), 1)
    frame = write(frame, "State #: " + str(len(all_observations)), 2)
    frame = write(frame, "Observations: " + str(all_observations[-1]), 3)

    # cv.imshow('Frame', frame)

    # # Wait 1ms, exit on q
    # if cv.waitKey(1) & 0xFF == ord('q'):
    #     break
    frame_pos += 1

np.save("observations_2_Finals_2016_Classic_Tetris_World_Championship_clipped.mp4.npy", all_observations)

filtered_observations = filter(lambda obs: not np.array_equal(obs, np.zeros(BOARD_SIZE[1])), all_observations)
filtered_observations_txt = ""
for observations in filtered_observations:
    for observation in observations:
        filtered_observations_txt += str(observation) + ","
f = open("filtered_observations_2.txt", "w")
f.write(filtered_observations_txt + "\n")
f.close()

video.release()
cv.destroyAllWindows()
