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
from shutil import copyfile

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
	red_player = Player(model = red_model, color = "Red")
	black_player = Player(model = black_model, color = "Black")

if input("Mandatory jumps [Y/n]?") == "Y":
	jump_rule = True
else:
	jump_rule = False

red_wins = 1
black_wins = 1
games = 0
highest_margin = 0

while True:
	game = Game(red_player = red_player, black_player = black_player, jump_rule = jump_rule)
	games += 1
	win, side, red_piece_count, black_piece_count, red_player_count, black_player_count, red_player_illegal_count, black_player_illegal_count = game.play_game()
	if side == "Red":
		red_wins += 1
	else:
		black_wins += 1
	margin = abs(red_piece_count - black_piece_count)
	if margin > highest_margin:
		highest_margin = margin
		copyfile('game_history.txt', 'best_game.txt')
	print(games, side, red_piece_count, black_piece_count, red_player_count, black_player_count, red_player_illegal_count, black_player_illegal_count, red_wins/black_wins, highest_margin)
	red_player.reset()
	black_player.reset()

