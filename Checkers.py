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
#print(board.red_numbers())

board.move_piece("Red", 10, 1)

board.visual_state()
#print(board.red_numbers())

board.move_piece("Black", 8, 1)

board.visual_state()
#print(board.red_numbers())

board.move_piece("Red", 10, 1)

board.visual_state()
#print(board.red_numbers())

board.move_piece("Black", 9, 0)

board.visual_state()
#print(board.red_numbers())

board.move_piece("Black", 9, 0)

board.visual_state()
#print(board.red_numbers())

board.move_piece("Black", 4, 1)

board.visual_state()
#print(board.red_numbers())

board.move_piece("Black", 4, 0)

board.visual_state()
#print(board.red_numbers())

board.move_piece("Black", 1, 0)

board.visual_state()
#print(board.red_numbers())

board.move_piece("Black", 1, 1)

board.visual_state()
#print(board.red_numbers())

board.move_piece("Red", 10, 0)

board.visual_state()
#print(board.red_numbers())

board.move_piece("Red", 10, 0)

board.visual_state()
#print(board.red_numbers())

board.move_piece("Red", 10, 3)

board.visual_state()
#print(board.red_numbers())

board.move_piece("Red", 10, 2)

board.visual_state()
#print(board.red_numbers())

board.move_piece("Red", 10, 0)

board.visual_state()
#print(board.red_numbers())

board.move_piece("Red", 11, 0)

board.visual_state()
#print(board.red_numbers())

board.move_piece("Red", 6, 0)

board.visual_state()
#print(board.red_numbers())

board.move_piece("Red", 6, 0)

board.visual_state()
#print(board.red_numbers())

board.move_piece("Red", 2, 1)

board.visual_state()
#print(board.red_numbers())

board.move_piece("Red", 2, 0)

board.visual_state()
#print(board.red_numbers())

board.move_piece("Black", 9, 1)

board.visual_state()
#print(board.red_numbers())

board.move_piece("Black", 9, 1)

board.visual_state()
#print(board.red_numbers())

board.move_piece("Black", 9, 1)

board.visual_state()
#print(board.red_numbers())

board.move_piece("Red", 6, 0)

board.visual_state()
#print(board.red_numbers())

board.move_piece("Black", 9, 2)

board.visual_state()
#print(board.red_numbers())

board.move_piece("Black", 9, 3)

board.visual_state()
#print(board.red_numbers())

board.move_piece("Black", 9, 2)

board.visual_state()
#print(board.red_numbers())

for p in board.red_piece:
	print(p.color, p.number, p.number_one_hot(), p.in_play)

for p in board.black_piece:
	print(p.color, p.number, p.number_one_hot(), p.in_play)


print(board.red_state())
print(board.black_flattened_home_view())
print(board.red_flattened_home_view())

