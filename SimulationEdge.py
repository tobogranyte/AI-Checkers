import numpy as np
from Board import Board
import random
from Game import Game
import copy

class SimulationEdge:

	def __init__(self, game, move_vector = None, P = None, depth = 0):
		self.game = copy.deepcopy(game)
		self.depth = depth
		self.P = 0
		self.N = 0
		self.V = 0
		self.Q = 0
		self.c = 1
		self.UCB = 0
		self.singleton = False
		self.has_children = False
		self.child_sim_edges = {}
		self.policy_output = np.array((96))
		self.value_output = np.array((1))
		self.legal_moves = np.array((96))
		self.active_child_index = 0
		if move_vector != None:
			self.move_vector = move_vector
			piece_number, move = self.onehot_to_moveandpiece(move_vector)
			'''Have the game make the move handed to it in move'''
			self.win, self.draw = self.game.make_move(move, piece_number)

		'''
		Create required game outputs from game state then delete the game
		'''
		self.board, self.pieces, self.mask = self.game.generate_X_mask()
		self.red_moves = self.game.red_moves
		self.black_moves = self.game.black_moves
		self.number = self.game.number
		self.player_color = self.game.player_color()
		self.other_player_color = self.game.other_player_color()


	def onehot_to_moveandpiece(self, one_hot_move):

		piece_number = int(np.argmax(one_hot_move)/8)
		move = one_hot_move[(8 * piece_number):((8 * piece_number) + 8)]

		return piece_number, move
	
	def moveandpiece_to_onehot(self, piece_number, move):

		one_hot_move = np.zeros()
		index = piece_number * 8 + move
		one_hot_move[index] = 1

		return one_hot_move
	
	'''
	Set of functions that either retrun the game variable from this sim edge, or pass the request
	along to the next sim edge in the branch path
	'''
	
	def get_red_moves(self, depth):
		if depth == self.depth:
			return self.red_moves
		else:
			return self.child_sim_edges[self.active_child_index].get_red_moves(depth)

	def get_black_moves(self, depth):
		if depth == self.depth:
			return self.black_moves
		else:
			return self.child_sim_edges[self.active_child_index].get_black_moves(depth)

	def generate_X_mask(self, depth):
		if depth == self.depth:
			return self.board, self.pieces, self.mask
		else:
			return self.child_sim_edges[self.active_child_index].generate_X_mask(depth)
		
	def get_child_inputs(self, depth):
		if depth == self.depth:
			board_parallel = []
			pieces_parallel = []
			mask_parallel = []
			game_parallel = []
			index_parallel = []
			for index, sim_edge in self.child_sim_edges:
				board, pieces, mask = self.generate_X_mask(self, depth + 1) # Get values for the level below the current one
				board_parallel.append(board)
				pieces_parallel.append(pieces)
				mask_parallel.append(mask)
				game_parallel.append(self.number)
				index_parallel.append(index)
			return np.vstack(board_parallel), np.hstack(pieces_parallel), np.hstack(mask_parallel), np.hstack(game_parallel), np.hstack(index_parallel)
		else:
			return  self.child_sim_edges[self.active_child_index].get_child_output(depth)
		

	def get_number(self, depth):
		if depth == self.depth:
			return self.number
		else:
			return self.child_sim_edges[self.active_child_index].get_number(depth)
		
	def set_policy_output(self, policy_output, depth):
		if depth == self.depth:
			self.policy_output = policy_output
		else:
			self.child_sim_edges[self.active_child_index].set_policy_output(policy_output, depth)

	def set_value_output(self, value_output, depth):
		if depth == self.depth:
			self.value_output = value_output
		else:
			self.child_sim_edges[self.active_child_index].set_value_output(value_output, depth)

	def create_children(self, depth):
		if depth == self.depth:
			if not self.has_children:
				legal_indices = np.where(self.mask == 1)[0]
				for index in legal_indices:
					move_vector = np.zeros(96)
					move_vector[index] = 1
					self.child_sim_edges = {index: self.game.make_simulation_edge(move_vector = move_vector, depth = self.depth + 1) }
		else:
			self.child_sim_edges[self.active_child_index].create_children(depth)
		self.game = None

	def set_child_UCBs(self, depth):
		if depth == self.depth:
			for index, sim_edge in self.child_sim_edges:
				sim_edge.set_UCB(self.policy_output[index], self.N)


			
	def set_singleton(self):
		self.singleton = True






