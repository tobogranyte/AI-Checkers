import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy import optimize
import numpy as np
from Board import Board
from Piece import Piece

board = Board()
board.start_game()

board.visual_state()
print(board.red_numbers())

board.move_piece("Red", 10, 1)

board.visual_state()
print(board.red_numbers())

board.move_piece("Black", 8, 1)

board.visual_state()
print(board.red_numbers())

board.move_piece("Red", 10, 1)

board.visual_state()
print(board.red_numbers())
