import matplotlib.pyplot as plt
import numpy as np
from Board import Board
from Piece import Piece
from Player import Player
from Game import Game
from shutil import copyfile
import time
import csv
import pdb
import json
from datetime import datetime
import os

def tally_and_print_stats(game):
	global red_wins
	global black_wins
	global red_illegal_total
	global red_move_total
	global black_illegal_total
	global black_move_total
	global red_win_pct
	global black_win_pct

	win, draw, side, red_piece_count, black_piece_count, red_move_count, black_move_count, red_illegal_count, black_illegal_count = game.stats()
	red_move_total += game.red_moves
	black_move_total += game.black_moves
	if win:
		if side == "Red":
			red_wins += 1
		else:
			black_wins += 1
		if side == "Red":
			p_side = "Red  "
		else:
			p_side = "Black"
	else:
		p_side = "Draw "
	margin = abs(red_piece_count - black_piece_count)
	red_win_pct = (red_wins * 100)/(red_wins + black_wins)
	black_win_pct = (black_wins * 100)/(red_wins + black_wins)
	print("{:8d} {} {:.2f} {:.2f} {:4d} {:2d} {:2d}".format(game.number, p_side, red_win_pct, black_win_pct, game.moves, red_piece_count, black_piece_count))

def write_stats(stats, identifier):
		with open(f"Torch_{identifier}/evaluation.csv", mode="a", newline="") as file:
			writer =		 csv.writer(file)
			writer.writerow(stats)  # Write the list as a new row


identifier = input("Identifier:")
bootstrap_version = input("Bootstrap version:")
bootstrap_version = int(bootstrap_version)
if input("Mandatory jumps [Y/n]?") == "Y":
	jump_rule = True
else:
	jump_rule = False
s_model = input("Model name:")
import_string = 'from ' + s_model + ' import ' + s_model + ' as sm' # create self_play model import string
exec(import_string, globals())
red_model = sm("red_model", identifier)
black_model = sm("black_model", identifier)
red_player = Player(model = red_model, color = "Red") # create the red player assigning model and color
black_player = Player(model = black_model, color = "Black") # create the black player assigning model and color
stats = [f"Model Temperature {red_model.temperature}"]
write_stats(stats, identifier)
stats = ['','','Red']
write_stats(stats, identifier)

stats = ['', '']
for black_version in range(0, bootstrap_version):
	stats.append(black_version)
write_stats(stats, identifier)

for black_version in range(0, bootstrap_version):
	if black_version == 0:
		stats = ['Black', black_version]
	else:
		stats = ['', black_version]
	for red_version in range(0, bootstrap_version):
		red_wins = 0
		black_wins = 0
		red_illegal_total = 0
		red_move_total = 0
		black_illegal_total = 0
		black_move_total = 0
		red_model.load_checkpoint(f"black_model_{red_version}", identifier)
		black_model.load_checkpoint(f"black_model_{black_version}", identifier)
		red_game_set = []
		black_game_set = []
		games = []
		done = False
		for count in range(0,1000): # create a batch-sized array of games and "start" each game
			if count%2 == 0:
				game = Game(red_player = red_player, black_player = black_player, jump_rule = jump_rule, number = count, side = "red")
			else:
				game = Game(red_player = red_player, black_player = black_player, jump_rule = jump_rule, number = count, side = "black")	
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
		while not done:
			red_X_parallel = np.zeros((red_model.layers_dims[0], len(red_game_set))) # X values for all games where it's a red move (column vector * number of red move games)
			black_X_parallel = np.zeros((black_model.layers_dims[0], len(black_game_set))) # X values for all games where it's a black move (column vector * number of black move games)
			red_mask_parallel = np.zeros((96, len(red_game_set))) # non-normalized legal moves label
			black_mask_parallel = np.zeros((96, len(black_game_set))) # non-normalized legal moves label
			black_game_numbers = np.zeros((1, len(black_game_set)), dtype=int)
			red_game_numbers = np.zeros((1, len(red_game_set)), dtype=int)
			for n, game in enumerate(red_game_set):
				X, mask = game.generate_X_mask()
				red_X_parallel[:,n] = X # insert X as a column to the parallel input
				red_mask_parallel[:,n] = mask # insert mask as a column to the parallel non-normalized label
				red_game_numbers[:,n] = game.number # the game number in the batch of games being played
			for n, game in enumerate(black_game_set):
				X, mask = game.generate_X_mask()
				black_X_parallel[:,n] = X # insert X as a column to the parallel input
				black_mask_parallel[:,n] = mask # insert mask as a column to the parallel non-normalized label
				black_game_numbers[:,n] = game.number # the game number in the batch of games being played
			red_AL = red_model.forward_pass(red_X_parallel, red_mask_parallel) # get matrix of vector probabilities for the next move in all red games
			black_AL = black_model.forward_pass(black_X_parallel, black_mask_parallel) # get matrix of vector probabilities for the next move in all black games
			for n, game in enumerate(red_game_set):
				one_hot_move, piece_number, move = red_model.generate_move(red_AL[:,n]) # make a dice-roll attempt
				win, draw = game.make_move(move, piece_number) # have the game make the move
				if win: # if it was a win or draw (game over)
					tally_and_print_stats(game)
				elif draw:
					tally_and_print_stats(game)
			for n, game in enumerate(black_game_set):
				one_hot_move, piece_number, move = black_model.generate_move(black_AL[:,n]) # make a dice-roll attempt
				win, draw = game.make_move(move, piece_number) # have the game make the move
				if win: # if it was a win or draw (game over)
					tally_and_print_stats(game)
				elif draw:
					tally_and_print_stats(game)
			red_game_set = []
			black_game_set = []
			done = True
			for n, game in enumerate(games):
				if (not game.win) and (not game.draw):
					# print("Not done", n, game.player_color(), game.player_count(), game.other_player_color(), game.other_player_count())
					# print(game.board.visual_state())
					done = False
					if game.player_color() == "Red": # game with a red move
						red_game_set.append(game) # append to list of games with red moves
					else: #game with a black move
						black_game_set.append(game) #append to list of games with black moves
		stats.append(red_win_pct)
	write_stats(stats, identifier)
