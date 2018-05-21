import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy import optimize
import numpy as np
from initializeBoard import initializeBoard
from Board import Board

board = Board()
board.start_game()

print(board.state)