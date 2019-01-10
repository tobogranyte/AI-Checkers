import numpy as np
from Board import Board
"""
Player class manages all moves for a given player. At the beginning of
a session, two players are created, one for each side. Each player
is assigned a model which handles deciding what move to make.
Once a game (Game) is started, the game lets the player know when
it is its turn to move by invoking the make_move method which includes
passing the board state. The player, in turn, passes that board state
to the model which returns what move it would like to make. The player
verifies the move is legal. If not, it asks the model for another move and
continues to do so until the model generates a legal move. Once the model
generates a legal move, the player hands that move back to the game which
ultimately moves the piece.
"""

class Player:
	
	def __init__(self, model, color):
		self.model = model
		self.color = color
		self.reset()
		if self.color == "Red":
			self.other_color = "Black"
		else:
			self.other_color = "Red"

	def make_move(self, board, jump_piece_number = None, jump_rule = True):
		model_move = np.zeros((48), dtype = 'int')
		move = np.zeros((4), dtype = 'int')
		piece_number = -1 # this will remain -1 until it's determined there are legal moves
		model_move, board_legal_moves = self.model.move(board, color = self.color, jump_piece_number = jump_piece_number, jump_rule = jump_rule) # retrieve one hot move and legal moves array
		if np.count_nonzero(board_legal_moves) != 0: # if there are any legal moves at all
			piece_number = int(np.argmax(model_move)/4)
			move = model_move[(4 * piece_number):((4 * piece_number) + 4)]
			while np.count_nonzero(model_move * board_legal_moves) == 0: # check if the cuurent proposed move is illegal
				self.illegal_move_count += 1
				model_move, _ = self.model.move(board, color = self.color, jump_piece_number = jump_piece_number, jump_rule = jump_rule, illegal = True) # retrieve one hot move
				piece_number = int(np.argmax(model_move)/4)
				move = model_move[(4 * piece_number):((4 * piece_number) + 4)]
			self.model.complete_move()
		return move, piece_number

	def train_model(self):
		cost, params = self.model.train()
		return cost, params

	def save_parameters(self):
		self.model.save_parameters()

	def increment_move_count(self):
		self.move_count += 1

	def reset(self):
		self.move_count = 0
		self.illegal_move_count = 0