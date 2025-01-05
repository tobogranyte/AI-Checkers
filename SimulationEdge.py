import numpy as np
from Board import Board
import random
import copy
import pdb
import sys

class SimulationEdge:

	def __init__(self, game, depth, root_color, move_vector = None, P = None):
		from Game import Game
		game_state = game.get_game_state()
		self.game = Game(game.red_player, game.black_player, game.jump_rule, game.number, game.player.color)
		self.game.set_game_state(game_state)
		self.win = game.win
		self.side = game.side
		if self.win:
			print(f"{id(self)} Simulated win. {self.side}")
		self.depth = depth
		self.root_color = root_color
		self.P = 0
		self.N = 0
		self.Q = 0
		self.v = 0
		self.c = 1
		self.UCB = 0
		self.singleton = False
		self.has_children = False
		self.has_child_outputs = False
		self.child_sim_edges = {}
		self.policy_output = np.array((96))
		self.value_output = np.array((1))
		self.active_child_index = 0
		if move_vector is not None:
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
		if depth == self.depth and not self.win:
			board_parallel = []
			pieces_parallel = []
			mask_parallel = []
			game_parallel = []
			index_parallel = []
			for index, sim_edge in self.child_sim_edges.items():
				if not sim_edge.win:
					board, pieces, mask = sim_edge.generate_X_mask(depth + 1) # Get values for the level below the current one
					board_parallel.append(board.reshape(-1, *board.shape))
					pieces_parallel.append(pieces.reshape(-1, 1))
					mask_parallel.append(mask.reshape(-1, 1))
					game_parallel.append(self.number)
					index_parallel.append(index)
				else:
					if sim_edge.side == self.root_color:
						sim_edge.v = 1
					else:
						sim_edge.v = 0
			if board_parallel != []:
				return np.vstack(board_parallel), np.hstack(pieces_parallel), np.hstack(mask_parallel), np.hstack(game_parallel), np.hstack(index_parallel)
			else:
				return np.zeros((0, 4, 8, 4)), np.zeros((409, 0)), np.zeros((96, 0)), np.array([]), np.array([])
		elif not self.win and self.has_children:
			return self.child_sim_edges[self.active_child_index].get_child_inputs(depth)
		else:
			return np.zeros((0, 4, 8, 4)), np.zeros((409, 0)), np.zeros((96, 0)), np.array([]), np.array([])
		

	def get_number(self):
		
		return self.number
		
	def set_policy_output(self, policy_output):
		self.policy_output = policy_output

	def set_value_output(self, value_output):
		self.value_output = value_output
		if not self.win:
			self.v = value_output.item() # start Q as the value output
		else:
			if self.side == self.root_color:
				self.v = 1
			else:
				self.v = 0

	def create_children(self, depth):
		if depth == self.depth:
			if not self.has_children:
				if not self.win:
					legal_indices = np.where(self.mask == 1)[0]
					for index in legal_indices:
						move_vector = np.zeros(96)
						move_vector[index] = 1
						new_edge = self.game.make_simulation_edge(root_color = self.root_color, move_vector = move_vector, depth = self.depth + 1)
						self.child_sim_edges.update({index: new_edge})
					self.has_children = True
					self.game = None
		elif not self.win:
			self.child_sim_edges[self.active_child_index].create_children(depth)
		

	def set_child_UCBs(self, depth):
		max_UCB = -sys.float_info.max
		if depth == self.depth:
			for index, sim_edge in self.child_sim_edges.items():
				UCB = sim_edge.set_UCB(self.policy_output[index], self.N)
				if UCB > max_UCB:
					max_UCB = UCB
					self.active_child_index = index
		elif not self.win:
			self.child_sim_edges[self.active_child_index].set_child_UCBs(depth)

	def set_UCB(self, policy, sum_N):
		if not self.has_children or self.N == 0:
			self.UCB = self.v + self.c * policy * (sum_N ** 0.5)/(self.N + 1)
		else:
			self.UCB = self.Q + self.c * policy * (sum_N ** 0.5)/(self.N + 1)
		return self.UCB

	def set_child_outputs(self, policy_array, value_array, index_array, depth):
		if depth == self.depth:
			if not self.has_child_outputs:
				for index, sim_edge in self.child_sim_edges.items():
					sim_edge.set_policy_output(policy_array[:, index_array == index]) # get policy output just for the child that matches the index
					sim_edge.set_value_output(value_array[:, index_array == index]) # get value output just for the child that matches the index
				self.has_child_outputs = True
		elif not self.win:
			self.child_sim_edges[self.active_child_index].set_child_outputs(policy_array, value_array, index_array, depth)

	def set_singleton(self):
		self.singleton = True

	def update_N(self):
		self.N += 1
		if self.has_children:
			self.child_sim_edges[self.active_child_index].update_N()

	def update_Q(self):
		if self.has_children:
			child_Q = self.child_sim_edges[self.active_child_index].update_Q()
			self.Q = self.Q + (child_Q - self.Q)/ self.N
			return self.Q
		else:
			return self.v
	
	def softmax(self, x):
		e_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
		return e_x / np.sum(e_x, axis=-1, keepdims=True)
		
	def get_Y(self):
		Y = np.zeros((96))
		N = np.zeros((96))
		if not self.singleton:
			for index, sim_edge in self.child_sim_edges.items():
				Y[index] = sim_edge.N
				N[index] = Y[index]
		else:
			for index, sim_edge in self.child_sim_edges.items():
				Y[index] = 1
		Y = np.where(self.mask == 0, float('-inf'), Y)
		Y = self.softmax(Y)
		return Y, N
	
	def get_child(self, depth):
		if depth == self.depth:
			return self
		else:
			return self.child_sim_edges[self.active_child_index].get_child(depth)






