import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy import optimize
import numpy as np
from Board import Board
from Piece import Piece
from Player import Player
from Game import Game

if input("Symmetric Models [Y/n]?") == "Y":
	s_model = input("Model name:")
	import_string = 'from ' + s_model + ' import ' + s_model + ' as sm' # create symmetric model import string
	exec(import_string, globals())
	symmetric_model = sm()
	red_player = Player(model = symmetric_model, color = "Red")
	black_player = Player(model = symmetric_model, color = "Black")
else:
	r_model = input("Red player model:")
	b_model = input("Black player model:")
	red_import_string = 'from ' + r_model + ' import ' + r_model + ' as rm' # create red model import string
	black_import_string = 'from ' + b_model + ' import ' + b_model + ' as bm' # create black model import string
	exec(red_import_string, globals())
	exec(black_import_string, globals())
	red_model = rm()
	black_model = bm()
	red_player = Player(model = red_model)
	black_player = Player(model = black_model)

while True:
	game = Game(red_player = red_player, black_player = black_player)
	#game.first_moves()
	win, side, red_piece_count, black_piece_count, red_player_count, black_player_count = game.play_game()
	print(win, side, red_piece_count, black_piece_count, red_player_count, black_player_count)
	red_player.reset()
	black_player.reset()

