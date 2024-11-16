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
import time
import csv
import pdb

train = ''
train_red = 'n'
train_black = 'n'
red_wins = 0
black_wins = 0
batch_count = 0
games_total = 0
red_win_pct = 0
black_win_pct = 0
red_illegal_total = 0
red_move_total = 0
black_illegal_total = 0
black_move_total = 0
red_win_pct_hist = []
black_win_pct_hist = []
red_illegal_pct_hist = []
black_illegal_pct_hist = []
params = {}

def tally_and_print_stats(game):
	global red_wins
	global black_wins
	global batch_count
	global red_illegal_total
	global red_move_total
	global black_illegal_total
	global black_move_total
	global red_win_pct
	global black_win_pct
	global red_illegal_pct
	global black_illegal_pct

	win, side, red_piece_count, black_piece_count, red_move_count, black_move_count, red_illegal_count, black_illegal_count = game.stats()
	red_move_total += game.red_moves
	black_move_total += game.black_moves
	if side == "Red":
		red_wins += 1
	else:
		black_wins += 1
	if side == "Red":
		p_side = "Red  "
	else:
		p_side = "Black"
	margin = abs(red_piece_count - black_piece_count)
	red_win_pct = (red_wins * 100)/(red_wins + black_wins)
	black_win_pct = (black_wins * 100)/(red_wins + black_wins)
	print("%8d" % game.number, p_side, "%.2f" % red_win_pct, "%.2f" % black_win_pct)

def add_point(line, x, y):
	line.set_xdata(list(line.get_xdata()) + [x])
	line.set_ydata(list(line.get_ydata()) + [y])

# Main loop where you update data
def update_plots(new_data):
	# Example: Assume new_data is a dictionary with new points for each dataset
	add_point(lines["win_pct"], new_data["games"], new_data["win_pct"])
	add_point(lines["cost_hist"], new_data["games"], new_data["cost_hist"])
	add_point(lines["max_hist0"], new_data["games"], new_data["max_hist0"])
	add_point(lines["max_hist1"], new_data["games"], new_data["max_hist1"])
	add_point(lines["max_hist2"], new_data["games"], new_data["max_hist2"])
	add_point(lines["max_hist3"], new_data["games"], new_data["max_hist3"])
	add_point(lines["max_hist4"], new_data["games"], new_data["max_hist4"])
	add_point(lines["min_hist0"], new_data["games"], new_data["min_hist0"])
	add_point(lines["min_hist1"], new_data["games"], new_data["min_hist1"])
	add_point(lines["min_hist2"], new_data["games"], new_data["min_hist2"])
	add_point(lines["min_hist3"], new_data["games"], new_data["min_hist3"])
	add_point(lines["min_hist4"], new_data["games"], new_data["min_hist4"])

	# Update plot limits and redraw
	for ax in [ax1, ax2, ax3, ax4]:
		ax.relim()
		ax.autoscale_view()

	plt.draw()
	plt.pause(0.001)

if input("Self play [Y/n]?") == "Y":
	self_play = True
	s_model = input("Model name:")
	train = input("Train models?")
	train_games = int(input("Number of games before training:"))
	bootstrap_threshold = int(input("Bootstrap threshold:"))
	import_string = 'from ' + s_model + ' import ' + s_model + ' as sm' # create self_play model import string
	exec(import_string, globals())
	red_model = sm()
	black_model = sm()
else:
	self_play = False
	r_model = input("Red player model:")
	b_model = input("Black player model:")
	if r_model != "RMM":
		train_red = input("Train Red?")
	if b_model != "RMM":
		train_black = input("Train Black?")
	if train_red == "Y" or train_black == "Y":
		train_games = int(input("Number of games before training:"))
	red_import_string = 'from ' + r_model + ' import ' + r_model + ' as rm' # create red model import string
	black_import_string = 'from ' + b_model + ' import ' + b_model + ' as bm' # create black model import string
	exec(red_import_string, globals())
	exec(black_import_string, globals())
	red_model = rm()
	black_model = bm()
plot_interval = int(input("Plot interval:"))
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
	ax1.set_title('Red Win Percentage')
	ax1.set_xlabel('Games')
	ax1.set_ylabel('Percentage')
	ax3.set_title('Cost')
	ax3.set_xlabel('Games')
	ax3.set_ylabel('Cost')
	ax4.set_title('Min/Max')
	ax4.set_xlabel('Games')
	ax4.set_ylabel('Min/Max')
	lines = {
		"win_pct": ax1.plot([], [], 'r-')[0],
		"cost_hist": ax3.plot([], [], 'k-')[0],
		"max_hist0": ax4.plot([], [], 'r-')[0],
		"max_hist1": ax4.plot([], [], 'g-')[0],
		"max_hist2": ax4.plot([], [], 'b-')[0],
		"max_hist3": ax4.plot([], [], 'c-')[0],
		"max_hist4": ax4.plot([], [], 'm-')[0],
		"min_hist0": ax4.plot([], [], 'r--')[0],
		"min_hist1": ax4.plot([], [], 'g--')[0],
		"min_hist2": ax4.plot([], [], 'b--')[0],
		"min_hist3": ax4.plot([], [], 'c--')[0],
		"min_hist4": ax4.plot([], [], 'm--')[0]
	}	

	plt.show()
	
	stats = ["Create", "Play", "Train", "Main"]
	training_stats = ["Init", "Forward", "Backward", "Bookkeeping", "Total"]
	with open("stats.csv", mode="a", newline="") as file:
		writer = csv.writer(file)
		writer.writerow(stats)  # Write the list as a new row
	with open("training_stats.csv", mode="a", newline="") as file:
		writer = csv.writer(file)
		writer.writerow(training_stats)  # Write the list as a new row

	while True:
		stats = []
		main_loop_start = time.time()
		red_game_set = []
		black_game_set = []
		games = []
		done = False
		create_games_start = time.time()
		for count in range(0,train_games): # create a batch-sized array of games and "start" each game
			game = Game(red_player = red_player, black_player = black_player, jump_rule = jump_rule, number = count)
			"""
			Create a game. This creates a game, setting all game parameters and also choosing the starting
			player (red or black) at random.
			"""
			games.append(game)
			# append the game to the batch of games
		for n, game in enumerate(games):
			if game.player_color() == "Red": # game with a red move
				red_game_set.append(game) # append to list of games with red moves
			else: #game with a black move
				black_game_set.append(game) #append to list of games with black moves
		games_total += train_games

		batch_count += 1 # this is the first batch
		red_X_parallel_batch = []
		black_X_parallel_batch = []
		red_mask_parallel_batch = []
		black_mask_parallel_batch = []
		red_moves_parallel_batch = []
		black_moves_parallel_batch = []
		black_game_numbers_batch = []
		red_game_numbers_batch = []
		create_games_end = time.time()
		create_games_time = create_games_end - create_games_start
		stats.append(create_games_time)
		play_games_start = time.time()
		while not done:
			red_X_parallel = np.zeros((red_model.layers_dims[0], len(red_game_set))) # X values for all games where it's a red move (column vector * number of red move games)
			black_X_parallel = np.zeros((black_model.layers_dims[0], len(black_game_set))) # X values for all games where it's a black move (column vector * number of black move games)
			red_Y_parallel = np.zeros((96, len(red_game_set))) # unit normalized legal moves label
			black_Y_parallel = np.zeros((96, len(black_game_set))) # unit normalized legal moves label
			red_mask_parallel = np.zeros((96, len(red_game_set))) # non-normalized legal moves label
			black_mask_parallel = np.zeros((96, len(black_game_set))) # non-normalized legal moves label
			red_moves_parallel = np.zeros((96, len(red_game_set)))
			black_moves_parallel = np.zeros((96, len(black_game_set)))
			black_game_numbers = np.zeros((1, len(black_game_set)))
			red_game_numbers = np.zeros((1, len(red_game_set)))
			for n, game in enumerate(red_game_set):
				"""
				Step through each game in the red training batch. For each game, get the input vector (X) from
				the model and add it to the appropriate location in the red_X_batch.
				Also for each game, get the Y (legal moves) and add it to the red_Y_batch.
				"""
				X, mask = game.generate_X_mask()
				"""
				X: (397) Input vector for the model
				Y: (96) Unit normalized vector with illegal moves all zero and legal moves summing to 1
				mask: (96) Illegal moves 0, Legal moves 1
				"""
				red_X_parallel[:,n] = X # insert X as a column to the parallel input
				red_mask_parallel[:,n] = mask # insert mask as a column to the parallel non-normalized label
				red_game_numbers[:,n] = game.number # the game number in the batch of games being played
			for n, game in enumerate(black_game_set):
				"""
				Step through each game in the black training batch. For each game, get the input vector (X) from
				the model and add it to the appropriate location in the black_X_batch.
				Also for each game, get the Y (legal moves) and add it to the black_Y_batch.
				"""
				X, mask = game.generate_X_mask()
				black_X_parallel[:,n] = X
				black_mask_parallel[:,n] = mask
				black_game_numbers[:,n] = game.number
			red_AL = red_model.forward_pass(red_X_parallel, red_mask_parallel) # get matrix of vector probabilities for the next move in all red games
			black_AL = black_model.forward_pass(black_X_parallel, black_mask_parallel) # get matrix of vector probabilities for the next move in all black games

			for n, game in enumerate(red_game_set):
				"""
				Step through each game in the red game batch. Get a one hot move by rolling the dice according to probabilities.
				Keep rolling the dice until a legal move is generated.
				Make the move.
				"""
				if np.count_nonzero(red_mask_parallel[:,n]) != 0: # if there are any legal moves at all
					one_hot_move, piece_number, move = red_model.generate_move(red_AL[:,n]) # make a dice-roll attempt
					while np.count_nonzero(one_hot_move * red_mask_parallel[:,n]) == 0: # check if the current proposed move is illegal
						print('Illegal move')
						one_hot_move, piece_number, move = red_model.generate_move(red_AL[:,n]) # roll the dice again
					red_moves_parallel[:,n] = one_hot_move # log the move made
					win, stalemate = game.make_move(move, piece_number) # have the game make the move
					if win or stalemate: # if it was a win or stalemate (game over)
						tally_and_print_stats(game)
			for n, game in enumerate(black_game_set):
				"""
				Step through each game in the black game batch. Get a one hot move by rolling the dice according to probabilities.
				Keep rolling the dice until a legal move is generated.
				Make the move.
				"""
				if np.count_nonzero(black_mask_parallel[:,n]) != 0: # if there are any legal moves at all
					one_hot_move, piece_number, move = black_model.generate_move(black_AL[:,n]) # make a dice-roll attempt
					while np.count_nonzero(one_hot_move * black_mask_parallel[:,n]) == 0: # check if the current proposed move is illegal
						print('Illegal move')
						one_hot_move, piece_number, move = black_model.generate_move(black_AL[:,n]) # roll the dice again
					black_moves_parallel[:,n] = one_hot_move # log the move made
					win, stalemate = game.make_move(move, piece_number) # have the game make the move
					if win or stalemate: # if it was a win or stalemate (game over)
						tally_and_print_stats(game)
			red_X_parallel_batch.append(red_X_parallel)
			black_X_parallel_batch.append(black_X_parallel)
			red_mask_parallel_batch.append(red_mask_parallel)
			black_mask_parallel_batch.append(black_mask_parallel)
			red_moves_parallel_batch.append(red_moves_parallel)
			black_moves_parallel_batch.append(black_moves_parallel)
			black_game_numbers_batch.append(black_game_numbers)
			red_game_numbers_batch.append(red_game_numbers)

			red_game_set = []
			black_game_set = []
			done = True
			for n, game in enumerate(games):
				if not game.win:
					# print("Not done", n, game.player_color(), game.player_count(), game.other_player_color(), game.other_player_count())
					# print(game.board.visual_state())
					done = False
					if game.player_color() == "Red": # game with a red move
						red_game_set.append(game) # append to list of games with red moves
					else: #game with a black move
						black_game_set.append(game) #append to list of games with black moves
		play_games_end = time.time()
		play_games_time = play_games_end - play_games_start
		stats.append(play_games_time)

		train_model_start = time.time()
		if (train == "Y" or train_red == "Y" or train_black == "Y"):
			if self_play:
				print("Training model...")
				print("Training Red...")
				cost, params = red_model.train_model(Y = np.hstack(red_moves_parallel_batch), X = np.hstack(red_X_parallel_batch), mask = np.hstack(red_mask_parallel_batch))
				minimums = params["mins"]
				maximums = params["maxes"]
				red_model.save_parameters()
			else:
				if train_red == "Y":
					print("Training Red...")
					cost, params = red_model.train_model(Y = np.hstack(red_moves_parallel_batch), X = np.hstack(red_X_parallel_batch), mask = np.hstack(red_mask_parallel_batch))
					minimums = params["mins"]
					maximums = params["maxes"]
					red_model.save_parameters()
				if train_black == "Y":
					print("Training Black...")
					black_player.train_model()
					black_player.save_parameters()
			new_data = {
				"games": games_total,  # Example game count
				"win_pct": red_win_pct,
				"cost_hist": cost,
				"max_hist0": maximums[0], "max_hist1": maximums[1], "max_hist2": maximums[2], "max_hist3": maximums[3], "max_hist4": maximums[4],
				"min_hist0": minimums[0], "min_hist1": minimums[1], "min_hist2": minimums[2], "min_hist3": minimums[3], "min_hist4": minimums[4],
			}
			update_plots(new_data)
			if (params["trainings"] % plot_interval == 0) or params["trainings"] < 100:
				plt.show()
				plt.pause(0.001)
			red_wins = 0
			black_wins = 0
			red_illegal_total = 0
			red_move_total = 0
			black_illegal_total = 0
			black_move_total = 0
			red_player.reset()
			black_player.reset()
		train_model_end = time.time()
		train_model_time = train_model_end - train_model_start
		stats.append(train_model_time)

		main_loop_end = time.time()
		main_loop_time = main_loop_end - main_loop_start
		stats.append(main_loop_time)
		with open("stats.csv", mode="a", newline="") as file:
			writer = csv.writer(file)
			writer.writerow(stats)  # Write the list as a new row
