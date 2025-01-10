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
import time
import csv
import pdb
import json
from datetime import datetime
import os
from SimulationEdge import SimulationEdge

def datetime_to_string(dt: datetime) -> str:
	return dt.strftime('%Y%m%d%H%M%S')

def tally_and_print_stats(game):
	global red_wins
	global black_wins
	global batch_count
	global red_illegal_total
	global red_move_total
	global black_illegal_total
	global black_move_total
	global red_win_pct
	global draw_pct
	global black_win_pct
	global draws
	global game_red_moves
	global game_black_moves

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
		draws += 1
	game_red_moves[:, game.number] = game.red_moves
	game_black_moves[:, game.number] = game.black_moves
	margin = abs(red_piece_count - black_piece_count)
	red_win_pct = (red_wins * 100)/(red_wins + black_wins)
	black_win_pct = (black_wins * 100)/(red_wins + black_wins)
	draw_pct = (draws * 100) / (red_wins + black_wins + draws)
	print("{:8d} {} {:.2f} {:.2f} {:4d} {:2d} {:2d}".format(game.number, p_side, red_win_pct, black_win_pct, game.moves, red_piece_count, black_piece_count))

def get_identifier(dt: datetime = None) -> str:
	if dt is None:
		dt = datetime.now()
	return dt.strftime('%Y%m%d%H%M%S')

def add_point(line, x, y):
	line.set_xdata(list(line.get_xdata()) + [x])
	line.set_ydata(list(line.get_ydata()) + [y])

# Main loop where you update data
def update_plots(new_data):
	# Example: Assume new_data is a dictionary with new points for each dataset
	add_point(lines["win_pct"], new_data["games"], new_data["win_pct"])
	add_point(lines["boot_avg"], new_data["games"], new_data["boot_avg"])
	add_point(lines["boot_ver"], new_data["games"], new_data["boot_ver"])
	add_point(lines["cost_hist"], new_data["games"], new_data["cost_hist"])
	add_point(lines["draw_pct"], new_data["games"], new_data["draw_pct"])
	add_point(lines["weights0"], new_data["games"], new_data["weights0"])
	add_point(lines["weights1"], new_data["games"], new_data["weights1"])
	add_point(lines["weights2"], new_data["games"], new_data["weights2"])
	add_point(lines["weights3"], new_data["games"], new_data["weights3"])

	# Update plot limits and redraw
	for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax6]:
		ax.relim()
		ax.autoscale_view()

	plt.draw()
	plt.pause(0.01)

def save_plots(filename):
	'''
	Save the current plot data to a file.
	'''
	plot_data = {}
	for name, line in lines.items():
		try:
			# Convert NumPy arrays to lists explicitly
			x_data = line.get_xdata()
			y_data = line.get_ydata()
			
			if isinstance(x_data, np.ndarray):
				x_data = x_data.tolist()
			if isinstance(y_data, np.ndarray):
				y_data = y_data.tolist()

			plot_data[name] = {
				"x": x_data,
				"y": y_data
			}
		except Exception as e:
			print(f"Error processing line '{name}': {e}")
			continue

	try:
		with open(f"Torch_{identifier}/{filename}", 'w') as f:
			json.dump(plot_data, f)
		print(f"Plots saved to Torch_{identifier}/{filename}")
	except Exception as e:
		print(f"Error saving plots to Torch_{identifier}/{filename}: {e}")

def load_plots(filename):
	'''
	Load plot data from a file and update the lines.
	'''
	try:
		with open(f"Torch_{identifier}/{filename}", 'r') as f:
			plot_data = json.load(f)
	except Exception as e:
		print(f"Error loading file 'Torch_{identifier}/{filename}': {e}")
		return

	for name, data in plot_data.items():
		if name in lines:
			try:
				lines[name].set_xdata(data["x"])
				lines[name].set_ydata(data["y"])
			except Exception as e:
				print(f"Error updating line '{name}': {e}")
		else:
			print(f"Warning: Line {name} not found in the current plot.")

	# Update plot limits and redraw
	for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
		ax.relim()
		ax.autoscale_view()

	plt.draw()
	print(f"Plots loaded from Torch_{identifier}/{filename}")


def save_data():
	data = {
    "bootstrap_version": bootstrap_version,
    "bootstrap_average": bootstrap_average,
    "games_total": games_total,
	"identifier": identifier
	}

	with open("data.json", "w") as f:
		json.dump(data, f)

	print("Data saved to data.json")

def load_data():
	global bootstrap_version, bootstrap_average, games_total, identifier  # Declare these as global
	with open("data.json", "r") as f:
		data = json.load(f)

	bootstrap_version = data["bootstrap_version"]
	bootstrap_average = data["bootstrap_average"]
	games_total = data["games_total"]
	identifier = data["identifier"]
	print(f"Data loaded from data.json")

def set_outputs(sim_edge_set, policy_output, value_output, depth):
	for n, sim_edge in enumerate(sim_edge_set):
		sim_edge.set_policy_output(policy_output[:, n])
		sim_edge.set_value_output(value_output[:, n])

def set_child_outputs(sim_edge_set, policy_output, value_output, number, index, depth):
	for n, sim_edge in enumerate(sim_edge_set):
		if not sim_edge.singleton:
			policy_edge = policy_output[:, number == sim_edge.number] # create trimmed policy output just for the one parent game/sim edge
			value_edge = value_output[:, number == sim_edge.number] # create trimmed value output just for the one parent game/sim edge
			index_edge = index[number == sim_edge.number] # create trimmed index just for the one parent game/sim edge
			sim_edge.set_child_outputs(policy_edge, value_edge, index_edge, depth) # send trimmed arrays to parent

def create_children(sim_edge_set, depth):
	for n, sim_edge in enumerate(sim_edge_set):
		if sim_edge.mask.sum() > 1:
			sim_edge.create_children(depth)
		else:
			sim_edge.set_singleton()

def set_child_UCBs(sim_edge_set, depth):
	for n, sim_edge in enumerate(sim_edge_set):
		if not sim_edge.singleton:
			sim_edge.set_child_UCBs(depth)


def get_child_inputs(sim_edge_set, depth):
	board_parallel = []
	pieces_parallel = []
	mask_parallel = []
	game_parallel = []
	index_parallel = []
	for n, sim_edge in enumerate(sim_edge_set):
		if not sim_edge.singleton:
			board, pieces, mask, number, index = sim_edge.get_child_inputs(depth)
			board_parallel.append(board)
			pieces_parallel.append(pieces)
			mask_parallel.append(mask)
			game_parallel.append(number)
			index_parallel.append(index)
	if len(board_parallel) > 0:
		return np.vstack(board_parallel), np.hstack(pieces_parallel), np.hstack(mask_parallel), np.hstack(game_parallel), np.hstack(index_parallel)
	else:
		return None, None, None, None, None

def update_N(sim_edge_set):
	for n, sim_edge in enumerate(sim_edge_set):
		if not sim_edge.singleton:
			sim_edge.update_N()

def update_Q(sim_edge_set):
	for n, sim_edge in enumerate(sim_edge_set):
		if not sim_edge.singleton:
			sim_edge.update_Q()

def get_Y(sim_edge_set):
	Y = np.zeros((96, len(sim_edge_set)))
	N = np.zeros((96, len(sim_edge_set)))
	for n, sim_edge in enumerate(sim_edge_set):
		Y[:, n], N[:, n] = sim_edge.get_Y()
	return Y, N




identifier = get_identifier()
train = ''
train_red = 'n'
train_black = 'n'
red_wins = 0
black_wins = 0
draws = 0
batch_count = 0
games_total = 0
batch_total_games = 0
red_win_pct = 0
black_win_pct = 0
draw_pct = 0
red_illegal_total = 0
red_move_total = 0
black_illegal_total = 0
black_move_total = 0
bootstrap_version = 0
bootstrap_average = 0
discount_factor = 0.95
loss_training_percentage = .001
exploration_depth = 10
exploration_traversals = 10
params = {}
resume = input("Resume [Y/n]?")

if resume == "Y":
	load_data()
	print(f"Bootstrap version:{bootstrap_version}")
	print(f"Bootstrap average:{bootstrap_average}")
	print(f"Games total:{games_total}")
	print(f"Identifier:{identifier}")
else:
	os.mkdir(f"Torch_{identifier}")

if input("Self play [Y/n]?") == "Y":
	self_play = True
	s_model = input("Model name:")
	train = input("Train models?")
	train_games = int(input("Number of games before training:"))
	bootstrap_threshold = int(input("Bootstrap threshold:"))
	import_string = 'from Models.' + s_model + ' import ' + s_model + ' as sm' # create self_play model import string
	exec(import_string, globals())
	red_model = sm("red_model", identifier)
	red_model_lr = red_model.learning_rate
	black_model = sm("black_model", identifier)

	if resume == "Y":
		black_model.load_checkpoint(f'black_model_{bootstrap_version}', identifier)
	else:
		black_model.save_obj('black_model_' + str(bootstrap_version), identifier)

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
	red_model = rm("red_model", identifier)
	black_model = bm("black_model", identifier)
plot_interval = int(input("Plot interval:"))
red_player = Player(model = red_model, color = "Red") # create the red player assigning model and color
black_player = Player(model = black_model, color = "Black") # create the black player assigning model and color

if input("Play game [Y/n]:") == "Y":

	if input("Mandatory jumps [Y/n]?") == "Y":
		jump_rule = True
	else:
		jump_rule = False
	plt.figure(1, dpi=75, figsize=(20,16))
	plt.ion()
	ax1 = plt.subplot2grid((78, 2), (0, 0), colspan=2, rowspan=9)
	ax2 = plt.subplot2grid((78, 2), (15, 0), colspan=2, rowspan=9)
	ax3 = plt.subplot2grid((78, 2), (28, 0), colspan=2, rowspan=9)
	ax4 = plt.subplot2grid((78, 2), (41, 0), colspan=2, rowspan=9)
	ax5 = plt.subplot2grid((78, 2), (54, 0), colspan=2, rowspan=9)
	ax6 = plt.subplot2grid((78, 2), (67, 0), colspan=2, rowspan=9)
	#ax1.set_title('Win Percentage')
	#ax1.set_xlabel('Games')
	#ax1.set_ylabel('Percentage')
	ax1.set_title('Red Win Percentage')
	ax1.set_xlabel('Games')
	ax1.set_ylabel('Percentage')
	ax2.set_title('Red Win Rolling')
	ax2.set_xlabel('Games')
	ax2.set_ylabel('Percentage')
	ax3.set_title('Bootstrap Version')
	ax3.set_xlabel('Games')
	ax3.set_ylabel('Version')
	ax4.set_title('Cost')
	ax4.set_xlabel('Games')
	ax4.set_ylabel('Cost')
	ax5.set_title('Draw Percentage')
	ax5.set_xlabel('Games')
	ax5.set_ylabel('Draw Percentage')
	ax6.set_title('Min/Max')
	ax6.set_xlabel('Games')
	ax6.set_ylabel('Min/Max')
	lines = {
		"win_pct": ax1.plot([], [], 'r-')[0],
		"boot_avg": ax2.plot([], [], 'g-')[0],
		"boot_ver": ax3.plot([], [], 'b-')[0],
		"cost_hist": ax4.plot([], [], 'k-')[0],
		"draw_pct": ax5.plot([], [], 'k-')[0],
		"weights0": ax6.plot([], [], 'r-')[0],
		"weights1": ax6.plot([], [], 'g-')[0],
		"weights2": ax6.plot([], [], 'b-')[0],
		"weights3": ax6.plot([], [], 'c-')[0]
	}	

	plt.show()
	if resume == "Y":
		load_plots(red_model.__class__.__name__ + ".json")
	
	stats = ["Create", "Play", "Train", "Main"]
	with open("stats.csv", mode="a", newline="") as file:
		writer = csv.writer(file)
		writer.writerow(stats)  # Write the list as a new row

	training_stats = ["Init", "Forward", "Backward", "Bookkeeping", "Total"]
	with open("training_stats.csv", mode="a", newline="") as file:
		writer = csv.writer(file)
		writer.writerow(training_stats)  # Write the list as a new row

	simulation_stats = ["Traversal", "Depth", "Create Children", "Get Inputs", "Create Outputs", "Set UCBs"]
	with open("simulation_stats.csv", mode="a", newline="") as file:
		writer = csv.writer(file)
		writer.writerow(simulation_stats)  # Write the list as a new row

	while True:
		stats = []
		main_loop_start = time.time()
		done = False
		full_batch = False
		purge = False
		red_pieces_parallel_batch = []
		black_pieces_parallel_batch = []
		red_board_parallel_batch = []
		black_board_parallel_batch = []
		red_mask_parallel_batch = []
		black_mask_parallel_batch = []
		red_moves_parallel_batch = []
		black_moves_parallel_batch = []
		black_game_numbers_batch = []
		red_game_numbers_batch = []
		black_move_numbers_batch = []
		red_move_numbers_batch = []
		red_Y_parallel_batch = []
		black_Y_parallel_batch = []
		red_N_parallel_batch = []
		black_N_parallel_batch = []
		games_set = train_games
		batch_total_games = 0
		game_value = np.full((1,0), 0.5)
		game_red_moves = np.zeros((1,0), dtype=int)
		game_black_moves = np.zeros((1,0), dtype=int)
		batch_count = 0
		depth = 0
		'''Full training and game-play loop'''
		while not full_batch:
			create_games_start = time.time()
			games = []
			red_game_set = []
			black_game_set = []
			red_sim_edge_set = []
			black_sim_edge_set = []
			game_value = np.hstack([game_value, np.full((1, games_set), 0.5)])
			game_red_moves = np.hstack([game_red_moves, np.zeros((1, games_set), dtype=int)]) # Total number of red moves for each game in a game set
			game_black_moves = np.hstack([game_black_moves, np.zeros((1, games_set), dtype=int)]) # total number of black moves for each game in a game set
			print("Creating games")
			'''
			Create games for the set of games to add to the batch. When wins and losses are balanced
			game_set will be the entire batch. When losses go down, games will be added to the batch
			to increase the number of losses. Excess wins will be randomly purged.
			'''
			for count in range(0, games_set): # create a batch-sized array of games and "start" each game
				'''
				Create a game. This creates a game, setting all game parameters and also choosing the starting
				player (red or black) at random.
				'''
				if count % 2 == 0:
					side = "Red"
				else:
					side = "Black"
				game = Game(red_player = red_player, black_player = black_player, jump_rule = jump_rule, number = (batch_total_games) + count, side = side)
				games.append(game)
				# append the game to the batch of games
			'''
			Divide games into two sets. One set is the games with a red move, the other is the games with
			a black move.
			'''
			for n, game in enumerate(games):
				if game.player_color() == "Red": # game with a red move
					red_game_set.append(game) # append to list of games with red moves
				else: #game with a black move
					black_game_set.append(game) #append to list of games with black moves

			start_time = time.time()
			for n, game in enumerate(red_game_set):
				red_sim_edge_set.append(game.make_simulation_edge(depth, game.player_color))

			for n, game in enumerate(black_game_set):
				black_sim_edge_set.append(game.make_simulation_edge(depth, game.player_color))
			end_time = time.time()
			total = end_time - start_time
			print(total)
			input("pause")

			create_games_end = time.time()
			create_games_time = create_games_end - create_games_start
			stats.append(create_games_time)

			play_games_start = time.time()
			'''
			Play all games in each set in parallel, moving through each step of making a move
			in parallel across each game set
			'''
			print("Playing games")
			while not done:
				print("New move")
				red_pieces_parallel = np.zeros((409, len(red_game_set))) # X values for all games where it's a red move (column vector * number of red move games)
				black_pieces_parallel = np.zeros((409, len(black_game_set))) # X values for all games where it's a black move (column vector * number of black move games)
				red_board_parallel = np.zeros((len(red_game_set), 4, 8, 4)) 
				black_board_parallel = np.zeros((len(black_game_set), 4, 8, 4)) 
				red_Y_parallel = np.zeros((96, len(red_game_set))) # red probabilities target
				black_Y_parallel = np.zeros((96, len(black_game_set))) # black probabilities target
				red_N_parallel = np.zeros((96, len(red_game_set))) # red probabilities target
				black_N_parallel = np.zeros((96, len(black_game_set))) # black probabilities target
				red_mask_parallel = np.zeros((96, len(red_game_set))) # non-normalized legal moves label
				black_mask_parallel = np.zeros((96, len(black_game_set))) # non-normalized legal moves label
				red_moves_parallel = np.zeros((96, len(red_game_set)))
				black_moves_parallel = np.zeros((96, len(black_game_set)))
				black_game_numbers = np.zeros((1, len(black_game_set)), dtype=int) # game number for this black move across all the games in the black game set
				red_game_numbers = np.zeros((1, len(red_game_set)), dtype=int) # game number for this red move across all the games in the red game set
				black_move_numbers = np.zeros((1, len(black_game_set)), dtype=int) # red move number (zero indexed) for this move across all games in the red game set.
				red_move_numbers = np.zeros((1, len(red_game_set)), dtype=int) # red move number (zero indexed) for this move across all games in the red game set.
				red_singletons = np.zeros((1, len(red_game_set)), dtype=int)
				black_singletons = np.zeros((1, len(black_game_set)), dtype=int)
				traversal = 0 # First traversal for this move
				depth = 0 # First traversal for this move
					
				'''Get input and mask vectors for all games for forward pass'''
				sim_setup_start = time.time()
				for n, sim_edge in enumerate(red_sim_edge_set):
					'''
					Step through each sim_edge in the red training batch. For each sim_edge, get the input vector (X) from
					the model and add it to the appropriate location in the red_X_batch.
					Also for each sim_edge, get the Y (legal moves) and add it to the red_Y_batch.
					'''
					red_move_numbers[:,n] = sim_edge.get_red_moves(depth) + 1 # add what move number this is to this sample in the training batch. For training.
					red_game_numbers[:,n] = sim_edge.get_number() # the sim_edge number of this sample in the batch of sim_edges being played
					board, pieces, mask = sim_edge.generate_X_mask(depth) # For training and forward pass.
					red_pieces_parallel[:,n] = pieces # insert X as a column to the parallel input
					red_board_parallel[n] = board
					red_mask_parallel[:,n] = mask # insert mask as a column to the parallel non-normalized label
				for n, sim_edge in enumerate(black_sim_edge_set):
					'''
					Step through each sim_edge in the black training batch. For each sim_edge, get the input vector (X) from
					the model and add it to the appropriate location in the black_X_batch.
					Also for each sim_edge, get the Y (legal moves) and add it to the black_Y_batch.
					'''
					black_move_numbers[:,n] = sim_edge.get_black_moves(depth) + 1
					black_game_numbers[:,n] = sim_edge.get_number()
					board, pieces, mask = sim_edge.generate_X_mask(depth)
					black_board_parallel[n] = board
					black_pieces_parallel[:,n] = pieces
					black_mask_parallel[:,n] = mask

				'''Do forward pass (get probabilities) for next move across all red games'''
				if len(red_game_set) > 0:
					red_model_policy_output, red_model_value_output = red_model.forward_pass(red_board_parallel, red_pieces_parallel, red_mask_parallel) # get matrix of vector probabilities for the next move in all red games

				'''Do forward pass (get probabilities) for next move across all black games'''
				if len(black_game_set) > 0:
					black_model_policy_output, black_model_value_output = black_model.forward_pass(black_board_parallel, black_pieces_parallel, black_mask_parallel) # get matrix of vector probabilities for the next move in all black games

				'''
				Send data back to the sim edge for update
				'''
				set_outputs(red_sim_edge_set, red_model_policy_output, red_model_value_output, depth)
				set_outputs(black_sim_edge_set, black_model_policy_output, black_model_value_output, depth)
				sim_setup_end = time.time()
				sim_setup_time = sim_setup_end - sim_setup_start

				while traversal < exploration_traversals: # not yet enough traversals
					depth = 0
					while depth < exploration_depth: # not yet deep enough on this traversal
						simulation_stats = []
						simulation_stats.append(traversal)
						simulation_stats.append(depth)
						'''
						with ProcessPoolExecutor(max_workers=12) as executor:
							executor.map(create_children, red_game_set, itertools.repeat(depth))
						with ProcessPoolExecutor(max_workers=12) as executor:
							executor.map(create_children, black_game_set, itertools.repeat(depth))
						'''
						create_children_start = time.time()

						create_children(red_sim_edge_set, depth)
						create_children(black_sim_edge_set, depth)

						create_children_end = time.time()
						create_children_time = create_children_end - create_children_start
						simulation_stats.append(create_children_time)

						get_inputs_start = time.time()

						red_edge_board_parallel, red_edge_pieces_parallel, red_edge_mask_parallel, red_number_parallel, red_index_parallel = get_child_inputs(red_sim_edge_set, depth)
						black_edge_board_parallel, black_edge_pieces_parallel, black_edge_mask_parallel, black_number_parallel, black_index_parallel = get_child_inputs(black_sim_edge_set, depth)

						get_inputs_end = time.time()
						get_inputs_time = get_inputs_end - get_inputs_start
						simulation_stats.append(get_inputs_time)

						set_outputs_start = time.time()

						if red_edge_board_parallel is not None and not np.array_equal(red_edge_board_parallel, np.array([[None]])) and red_edge_board_parallel.size != 0:
							se_red_model_policy_output, se_red_model_value_output = red_model.forward_pass(red_edge_board_parallel, red_edge_pieces_parallel, red_edge_mask_parallel) # get matrix of vector probabilities for the next move in all red games
							set_child_outputs(red_sim_edge_set, se_red_model_policy_output, se_red_model_value_output, red_number_parallel, red_index_parallel, depth)
						if black_edge_board_parallel is not None and not np.array_equal(black_edge_board_parallel, np.array([[None]])) and black_edge_board_parallel.size != 0:
							se_black_model_policy_output, se_black_model_value_output = black_model.forward_pass(black_edge_board_parallel, black_edge_pieces_parallel, black_edge_mask_parallel) # get matrix of vector probabilities for the next move in all red games
							set_child_outputs(black_sim_edge_set, se_black_model_policy_output, se_black_model_value_output, black_number_parallel, black_index_parallel, depth)

						set_outputs_end = time.time()
						set_outputs_time = set_outputs_end - set_outputs_start
						simulation_stats.append(set_outputs_time)

						set_UCBs_start = time.time()

						set_child_UCBs(red_sim_edge_set, depth)
						set_child_UCBs(black_sim_edge_set, depth)

						set_UCBs_end = time.time()
						set_UCBs_time = set_UCBs_end - set_UCBs_start
						simulation_stats.append(set_UCBs_time)

						with open("simulation_stats.csv", mode="a", newline="") as file:
							writer = csv.writer(file)
							writer.writerow(simulation_stats)  # Write the list as a new row

						depth += 1

					update_N_start = time.time()

					update_N(red_sim_edge_set)
					update_N(black_sim_edge_set)

					update_N_end = time.time()
					update_N_time = update_N_end - update_N_start

					update_Q_start = time.time()

					update_Q(red_sim_edge_set)
					update_Q(black_sim_edge_set)

					update_Q_end = time.time()
					update_Q_time = update_Q_end - update_Q_start

					traversal += 1
				
				
				red_Y_parallel, red_N_parallel = get_Y(red_sim_edge_set)
				black_Y_parallel, black_N_parallel = get_Y(black_sim_edge_set)
				'''Make move for all games based on probabilities'''
				for n, game in enumerate(red_game_set):
					'''
					Step through each game in the red game batch. Get a one hot move by rolling the dice according to probabilities.
					Keep rolling the dice until a legal move is generated.
					Make the move.
					'''
					if np.count_nonzero(red_mask_parallel[:,n]) != 0: # if there are any legal moves at all
						one_hot_move, piece_number, move = red_model.generate_move(red_model_policy_output[:,n]) # make a dice-roll attempt
						while np.count_nonzero(one_hot_move * red_mask_parallel[:,n]) == 0: # check if the current proposed move is illegal
							pdb.set_trace()
							print('Illegal move')
							one_hot_move, piece_number, move = red_model.generate_move(red_model_policy_output[:,n]) # roll the dice again
						red_moves_parallel[:,n] = one_hot_move # log the move made
						win, draw = game.make_move(move, piece_number) # have the game make the move
						if win: # if it was a win or draw (game over)
							game_value[:, game.number] = 1 # positive reward for this game because red won
							tally_and_print_stats(game)
						elif draw:
							tally_and_print_stats(game)
				for n, game in enumerate(black_game_set):
					'''
					Step through each game in the black game batch. Get a one hot move by rolling the dice according to probabilities.
					Keep rolling the dice until a legal move is generated.
					Make the move.
					'''
					if np.count_nonzero(black_mask_parallel[:,n]) != 0: # if there are any legal moves at all
						one_hot_move, piece_number, move = black_model.generate_move(black_model_policy_output[:,n]) # make a dice-roll attempt
						while np.count_nonzero(one_hot_move * black_mask_parallel[:,n]) == 0: # check if the current proposed move is illegal
							print('Illegal move')
							one_hot_move, piece_number, move = black_model.generate_move(black_model_policy_output[:,n]) # roll the dice again
						black_moves_parallel[:,n] = one_hot_move # log the move made
						win, draw = game.make_move(move, piece_number) # have the game make the move
						if win: # if it was a win or draw (game over)
							game_value[:,game.number] = 0 # negative reward for this game because red lost
							tally_and_print_stats(game)
						elif draw:
							tally_and_print_stats(game)
				red_pieces_parallel_batch.append(red_pieces_parallel)
				black_pieces_parallel_batch.append(black_pieces_parallel)
				red_board_parallel_batch.append(red_board_parallel)
				black_board_parallel_batch.append(black_board_parallel)
				red_mask_parallel_batch.append(red_mask_parallel)
				black_mask_parallel_batch.append(black_mask_parallel)
				red_moves_parallel_batch.append(red_moves_parallel)
				black_moves_parallel_batch.append(black_moves_parallel)
				black_game_numbers_batch.append(black_game_numbers)
				red_game_numbers_batch.append(red_game_numbers) # A master list of the game numbers of every red move in the batch
				black_move_numbers_batch.append(black_move_numbers)
				red_move_numbers_batch.append(red_move_numbers)
				red_Y_parallel_batch.append(red_Y_parallel)
				black_Y_parallel_batch.append(black_Y_parallel)
				red_N_parallel_batch.append(red_N_parallel)
				black_N_parallel_batch.append(black_N_parallel)

				red_game_set = []
				black_game_set = []
				red_sim_edge_set = []
				black_sim_edge_set = []
				number = None
				done = True
				for n, game in enumerate(games):
					if (not game.win) and (not game.draw):
						number = game.number
						# print("Not done", n, game.player_color(), game.player_count(), game.other_player_color(), game.other_player_count())
						# print(game.board.visual_state())
						done = False
						if game.player_color() == "Red": # game with a red move
							red_game_set.append(game) # append to list of games with red moves
							red_sim_edge_set.append(game.make_simulation_edge(0, game.player_color))
						else: #game with a black move
							black_game_set.append(game) #append to list of games with black moves
							black_sim_edge_set.append(game.make_simulation_edge(0, game.player_color))
				play_games_end = time.time()
			loss_total = (np.squeeze(game_value) == 0).sum()
			win_total = (np.squeeze(game_value) == 1).sum()
			batch_total_games += games_set
			if batch_total_games == train_games:
				draw_total = (np.squeeze(game_value) == 0).sum()
			if loss_total < int((train_games - draw_total) * loss_training_percentage):
				print("Not enough losses")
				batch_count += 1
				done = False
				games_set = int((((train_games - draw_total) * loss_training_percentage * 1.1) - loss_total) / (loss_total / batch_total_games)) # Set a new games set that will be sufficient to generate enough losses to make the balance
				if games_set > train_games:
					games_set = train_games # But don't go over the total batch size in any one set.
			else:
				full_batch = True
				target_win_total = int((1 - loss_training_percentage) * loss_total / loss_training_percentage)
				win_purge = win_total - target_win_total
				if win_purge > 0 and batch_total_games != train_games:
					purge = True
					winning_game_list_mask = np.squeeze(game_value, axis=0) == 1
					game_number_array = np.arange(batch_total_games)
					winning_game_numbers = np.random.permutation(game_number_array[winning_game_list_mask])
					winning_game_purge = winning_game_numbers[:win_purge]
					winning_games_mask = ~np.isin(np.squeeze(np.hstack(red_game_numbers_batch)), winning_game_purge)
				
			play_games_time = play_games_end - play_games_start
			stats.append(play_games_time)
		V = game_value[:, np.squeeze(np.hstack(red_game_numbers_batch))] #assign a win/loss value to each game
		data = {
			"red_pieces_parallel_batch": np.hstack(red_pieces_parallel_batch),
			"black_pieces_parallel_batch": np.hstack(black_pieces_parallel_batch),
			"red_board_parallel_batch": np.vstack(red_board_parallel_batch),
			"black_board_parallel_batch": np.vstack(black_board_parallel_batch),
			"red_mask_parallel_batch": np.hstack(red_mask_parallel_batch),
			"black_mask_parallel_batch": np.hstack(black_mask_parallel_batch),
			"red_moves_parallel_batch": np.hstack(red_moves_parallel_batch),
			"black_moves_parallel_batch": np.hstack(black_moves_parallel_batch),
			"red_game_numbers_batch": np.hstack(red_game_numbers_batch),
			"black_game_numbers_batch": np.hstack(black_game_numbers_batch),
			"red_move_numbers_batch": np.hstack(red_move_numbers_batch),
			"black_move_numbers_batch": np.hstack(black_move_numbers_batch),
			"red_Y_parallel_batch": np.hstack(red_Y_parallel_batch),
			"black_Y_parallel_batch": np.hstack(black_Y_parallel_batch),
			"red_N_parallel_batch": np.hstack(red_N_parallel_batch),
			"black_N_parallel_batch": np.hstack(black_N_parallel_batch),
			"V": V,
			"game_value": game_value
			}
		np.savez("arrays.npz", **data)

		train_model_start = time.time()
		if (train == "Y" or train_red == "Y" or train_black == "Y"):
			if self_play:
				print("Training model...")
				print("Training Red...")
				Y = np.hstack(red_Y_parallel_batch)
				pieces = np.hstack(red_pieces_parallel_batch)
				board = np.vstack(red_board_parallel_batch)
				mask = np.hstack(red_mask_parallel_batch)
				V = game_value[:, np.squeeze(np.hstack(red_game_numbers_batch))] #assign a win/loss value to each game
				game_red_moves_batch = game_red_moves[:, np.squeeze(np.hstack(red_game_numbers_batch))] # total number of red moves for the game that this move was part of
				if purge:
					Y = Y[:, winning_games_mask]
					V = V[:, winning_games_mask]
					pieces = pieces[:, winning_games_mask]
					board = board[winning_games_mask]
					mask = mask[:, winning_games_mask]
				cost, params = red_model.train_model(Y, V, board, pieces, mask)
				minimums = params["mins"]
				maximums = params["maxes"]
				weights = params["weights"]
				red_model.save_parameters(identifier)
			else:
				if train_red == "Y":
					print("Training Red...")
					cost, params = red_model.train_model(Y = np.hstack(red_moves_parallel_batch), X = np.hstack(red_pieces_parallel_batch), mask = np.hstack(red_mask_parallel_batch))
					minimums = params["mins"]
					maximums = params["maxes"]
					red_model.save_parameters(identifier)
				if train_black == "Y":
					print("Training Black...")
					black_player.train_model()
					black_player.save_parameters(identifier)
			if bootstrap_average == 0:
				bootstrap_average = red_win_pct
			else:
				bootstrap_average = 0.9 * bootstrap_average + 0.1 * red_win_pct
			games_total += batch_total_games
			print(f"Games total:{games_total}")
			new_data = {
				"games": games_total,  # Example game count
				"win_pct": red_win_pct,
				"boot_avg": bootstrap_average,
				"boot_ver": bootstrap_version,
				"cost_hist": float(cost),
				"draw_pct": float(draw_pct),
				"weights0": float(weights[0]), "weights1": float(weights[1]), "weights2": float(weights[2]), "weights3": float(weights[3]),
			}
			print(f"weights[0]: {weights[0]}, weights[1]: {weights[1]}, weights[2]: {weights[2]}, weights[3]: {weights[3]}, ")
			update_plots(new_data)
			save_plots(red_model.__class__.__name__ + ".json")
			save_data()
			if (params["trainings"] % plot_interval == 0) or params["trainings"] < 100:
				plt.show()
				plt.pause(0.001)
			red_wins = 0
			black_wins = 0
			draws = 0
			red_illegal_total = 0
			red_move_total = 0
			black_illegal_total = 0
			black_move_total = 0
			red_player.reset()
			black_player.reset()
		train_model_end = time.time()
		train_model_time = train_model_end - train_model_start
		stats.append(train_model_time)
		if bootstrap_average >= bootstrap_threshold:
			print(f"Bootstrap average: {bootstrap_average}")
			print(f"Bootstrap threshold: {bootstrap_threshold}")
			print('BOOTSTRAP')
			black_model.load_checkpoint('red_model', identifier)
			bootstrap_average = 0
			bootstrap_version += 1
			black_model.save_obj('black_model_' + str(bootstrap_version), identifier)
			red_model.update_learning_rate(0.5)
			save_data()
		main_loop_end = time.time()
		main_loop_time = main_loop_end - main_loop_start
		stats.append(main_loop_time)
		with open("stats.csv", mode="a", newline="") as file:
			writer = csv.writer(file)
			writer.writerow(stats)  # Write the list as a new row
