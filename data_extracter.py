# Extracts observation data from this video: https://www.youtube.com/watch?v=DdfRQjb5o9k
# Followed https://www.learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/ for working with video in opencv
import cv2 as cv
import numpy as np

# x, y
board_start_1 = 412, 190
line_score_start_1 = 412, 122
line_score_end_1 = 503, 171
next_piece_start_1 = 522, 122
next_piece_end_1 = 612, 171

board_start_2 = 670, 190
line_score_start_2 = 670, 122
line_score_end_2 = 760, 171
next_piece_start_2 = 779, 122
next_piece_end_2 = 870, 171

block_size = 20

# rows, columns
board_size = 20, 10

# The grayscale (max of RGB) value above which is classified as a block 
block_color_threshold = 63

text_start = 0, 100
text_spacing = 50
font_size = 1
font_thickness = 2
font = cv.FONT_HERSHEY_SIMPLEX
font_color = 0, 255, 0


def get_blocks(frame, board_start):
    blocks = np.zeros(board_size, dtype=bool)

    row_end = board_start[1] + board_size[0] * block_size
    col_end = board_start[0] + board_size[1] * block_size
    cropped_frame = frame[board_start[1]:row_end, board_start[0]:col_end]
    i = 0
    for y in range(block_size // 2, cropped_frame.shape[0], block_size):
        row = cropped_frame[y]
        j = 0
        for x in range (block_size // 2, row.shape[0], block_size):
            grayscale = np.max(row[x])
            blocks[i, j] = grayscale > block_color_threshold
            j += 1

        i += 1

    return blocks


# Draws white blocks over occupied spaces and black blocks over non-occupied spaces
def draw_blocks(frame, blocks, board_start):
    for i in range(0, blocks.shape[0]):
        for j in range(0, blocks.shape[1]):
            top_left = (board_start[0] + j * block_size, board_start[1] + i * block_size)
            bottom_right = (board_start[0] + (j + 1) * block_size, board_start[1] + (i + 1) * block_size)
            color = 0, 0, 0
            if blocks[i][j]:
                color = 255, 255, 255

            frame = cv.rectangle(frame, top_left, bottom_right, color, cv.FILLED)

    return frame


def write(frame, text, i):
    pos = (text_start[0], text_start[1] + i * text_spacing)
    return cv.putText(frame, text, pos, font, font_size, font_color, font_thickness)


board_start = board_start_1

video = cv.VideoCapture("Finals_2016_Classic_Tetris_World_Championship_clipped.mp4")
video.set(cv.CAP_PROP_POS_FRAMES, 25250)

# Capture frame-by-frame
while video.isOpened():
    ret, frame = video.read()

    if ret == True:
        blocks_1 = get_blocks(frame, board_start_1)
        blocks_2 = get_blocks(frame, board_start_2)
        frame = draw_blocks(frame, blocks_1, board_start_1)
        frame = draw_blocks(frame, blocks_2, board_start_2)
        frame = write(frame, "# Blocks board 1: " + str(np.sum(blocks_1)), 0)
        frame = write(frame, "# Blocks board 2: " + str(np.sum(blocks_2)), 1)

        cv.imshow('Frame', frame)

        # Exit on q
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

video.release()
cv.destroyAllWindows()
