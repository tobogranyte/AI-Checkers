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
import matplotlib.pyplot as plt

train = ''
train_red = 'n'
train_black = 'n'
red_wins = 0
black_wins = 0
games = 0
red_illegal_total = 0
red_move_total = 0
black_illegal_total = 0
black_move_total = 0
red_win_pct_hist = []
black_win_pct_hist = []
red_illegal_pct_hist = []
black_illegal_pct_hist = []
max_hist0 = []
max_hist1 = []
max_hist2 = []
max_hist3 = []
max_hist4 = []
max_hist5 = []
max_hist6 = []
min_hist0 = []
min_hist1 = []
min_hist2 = []
min_hist3 = []
min_hist4 = []
min_hist5 = []
min_hist6 = []
games_hist = []
cost_hist = []
params = {}

if input("Symmetric Models [Y/n]?") == "Y":
	symmetric = True
	s_model = input("Model name:")
	train = input("Train models?")
	if train == "Y":
		train_games = int(input("Number of games before training:"))
	import_string = 'from ' + s_model + ' import ' + s_model + ' as sm' # create symmetric model import string
	exec(import_string, globals())
	symmetric_model = sm()
	red_player = Player(model = symmetric_model, color = "Red")
	black_player = Player(model = symmetric_model, color = "Black")
else:
	symmetric = False
	r_model = input("Red player model:")
	b_model = input("Black player model:")
	if r_model != "RMM":
		train_red = input("Train Red?")
	if b_model != "RMM":
		train_black = input("Train Black?")
	if train_red == "Y" or train_black == "Y":
		train_games = int(input("Number of games before training:"))
	plot_interval = int(input("Plot interval:"))
	red_import_string = 'from ' + r_model + ' import ' + r_model + ' as rm' # create red model import string
	black_import_string = 'from ' + b_model + ' import ' + b_model + ' as bm' # create black model import string
	exec(red_import_string, globals())
	exec(black_import_string, globals())
	red_model = rm()
	black_model = bm()
	red_player = Player(model = red_model, color = "Red") # create the red player assigning model and color
	black_player = Player(model = black_model, color = "Black") # create the black player assigning model and color

if input("Play game [Y/n]:") == "Y":

	if input("Mandatory jumps [Y/n]?") == "Y":
		jump_rule = True
	else:
		jump_rule = False

	plt.figure(1, dpi=75, figsize=(16,16))
	plt.ion()
	ax1 = plt.subplot2grid((40, 1), (0, 0), colspan=2, rowspan=8)
	ax2 = plt.subplot2grid((40, 1), (12, 0), colspan=2, rowspan=8)
	ax3 = plt.subplot2grid((40, 1), (22, 0), colspan=2, rowspan=8)
	ax4 = plt.subplot2grid((40, 1), (32, 0), colspan=2, rowspan=8)
	#ax1.set_title('Win Percentage')
	#ax1.set_xlabel('Games')
	#ax1.set_ylabel('Percentage')
	ax1.set_title('Illegal/Legal Means')
	ax1.set_xlabel('Games')
	ax1.set_ylabel('Percentage')
	ax2.set_title('Illegal Move Percentage')
	ax2.set_xlabel('Games')
	ax2.set_ylabel('Percentage')
	ax3.set_title('Cost')
	ax3.set_xlabel('Games')
	ax3.set_ylabel('Cost')
	ax4.set_title('Min/Max')
	ax4.set_xlabel('Games')
	ax4.set_ylabel('Min/Max')
	plt.show()


	while True:
		game_batch = []
		for count in range(0,train_games)
			game_batch.append(Game(red_player = red_player, black_player = black_player, jump_rule = jump_rule))
			game_batch[count].start_game()
		games += train_games
		win, side, red_piece_count, black_piece_count, red_move_count, black_move_count, red_illegal_count, black_illegal_count = game.play_game()
		red_illegal_total += red_illegal_count
		black_illegal_total += black_illegal_count
		red_move_total += red_move_count
		black_move_total += black_move_count
		if side == "Red":
			red_wins += 1
		else:
			black_wins += 1
		if side == "Red":
			p_side = "Red  "
		else:
			p_side = "Black"
		margin = abs(red_piece_count - black_piece_count)
		red_illegal_pct = (red_illegal_total * 100)/(red_move_total + red_illegal_total)
		black_illegal_pct = (black_illegal_total * 100)/(black_move_total + black_illegal_total)
		red_win_pct = (red_wins * 100)/(red_wins + black_wins)
		black_win_pct = (black_wins * 100)/(red_wins + black_wins)
		print(games, p_side, "%.2f" % red_illegal_pct, "%.2f" % black_illegal_pct, "%.2f" % red_win_pct, "%.2f" % black_win_pct)
		if (train == "Y" or train_red == "Y" or train_black == "Y") and (games % train_games == 0):
			if symmetric:
				print("Training model...")
			else:
				if train_red == "Y":
					print("Training Red...")
					cost, params = red_player.train_model()
					illegal_means = params["illegal_means"]
					legal_means = params["legal_means"]
					minimums = params["mins"]
					maximums = params["maxes"]
					red_player.save_parameters()
				if train_black == "Y":
					print("Training Black...")
					black_player.train_model()
					black_player.save_parameters()
			red_win_pct_hist.append(red_win_pct)
			black_win_pct_hist.append(black_win_pct)
			red_illegal_pct_hist.append(red_illegal_pct)
			black_illegal_pct_hist.append(black_illegal_pct)
			max_hist0.append(maximums[0])
			max_hist1.append(maximums[1])
			max_hist2.append(maximums[2])
			max_hist3.append(maximums[3])
			max_hist4.append(maximums[4])
			max_hist5.append(maximums[5])
			min_hist0.append(minimums[0])
			min_hist1.append(minimums[1])
			min_hist2.append(minimums[2])
			min_hist3.append(minimums[3])
			min_hist4.append(minimums[4])
			min_hist5.append(minimums[5])
			cost_hist.append(cost)
			games_hist.append(games)
			if (params["trainings"] % plot_interval == 0) or params["trainings"] < 100:
				#ax1.plot(games_hist, red_win_pct_hist, 'r-', games_hist, black_win_pct_hist, 'k-')
				ax1.plot(games_hist, illegal_means, 'r-', games_hist, legal_means, 'g-')
				ax2.plot(games_hist, red_illegal_pct_hist, 'r-', games_hist, black_illegal_pct_hist, 'k-')
				ax3.semilogy(games_hist, cost_hist, 'k-')
				ax4.plot(games_hist, max_hist0, 'r-')
				ax4.plot(games_hist, max_hist1, 'g-')
				ax4.plot(games_hist, max_hist2, 'b-')
				ax4.plot(games_hist, max_hist3, 'c-')
				ax4.plot(games_hist, max_hist4, 'm-')
				ax4.plot(games_hist, max_hist5, 'k-')
				ax4.plot(games_hist, min_hist0, 'r-')
				ax4.plot(games_hist, min_hist1, 'g-')
				ax4.plot(games_hist, min_hist2, 'b-')
				ax4.plot(games_hist, min_hist3, 'c-')
				ax4.plot(games_hist, min_hist4, 'm-')
				ax4.plot(games_hist, min_hist5, 'k-')
				plt.draw()
				plt.pause(0.001)
			red_wins = 0
			black_wins = 0
			red_illegal_total = 0
			red_move_total = 0
			black_illegal_total = 0
			black_move_total = 0
			red_player.reset()
			black_player.reset()

