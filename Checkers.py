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
board.test_setup()

print(board.red_state())

piece = Piece("Black", king = False, x = 0, y = 4)

print(piece.legal_moves(board))

