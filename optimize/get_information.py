import numpy as np
"""
[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]]
"""

def get_info(board):
    height_col = [0]*10
    holes = [0]*10
    for row in range(10):
        for col in range(20):
            if board[row][col] == 1:
                height_col[row] = (20-col)
                for check in range(col+1, 20):
                    if board[row][check] == 0:
                        holes[row] += 1
                break

    # get max height col
    height_max_return = max(height_col)

    # get the different col sum
    diff = [0]*9
    for row in range(1, 10):
        diff[row] = abs(height_col[row] - height_col[row-1])
    diff_return = sum(diff)

    # get height sum
    height_sum_return = sum(height_col)

    # get sum hole
    holes_return = sum(holes)

    # get well col
    well_col_return = 10 - np.count_nonzero(height_col)

    return height_max_return, height_sum_return, diff_return, holes_return, well_col_return




