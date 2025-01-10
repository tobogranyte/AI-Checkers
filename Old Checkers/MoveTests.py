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

piece = Piece("Red", king = False, xPosition = 0, yPosition = 1)

