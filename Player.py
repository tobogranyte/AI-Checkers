import numpy as np
from Board import Board

# Player class manages all moves for a given player. At the beginning of
# a session, two players is be created, one for each side. Each player
# is be assigned a model which handles deciding what move to make.
# Once a game (Game) is started, the game lets the player know when
# it is its turn to move by issuing the [name TBD] command which includes
# passing the board state. The player, in turn, passes that board state
# to the model which returns what move it would like to make. The player
# hands that back to the game which ultimately moves the piece.

class Player:
	
	def __init__(self, model, color):
		self.model = model
		self.color = color
		self.reset()
		if self.color == "Red":
			self.other_color = "Black"
		else:
			self.other_color = "Red"

	def make_move(self, board, jump_piece_number = None):
		model_move = np.zeros((48))
		move = np.zeros((4))
		piece_number = -1
		if np.count_nonzero(board.legal_moves(color = self.color, jump_piece_number = jump_piece_number)) != 0:
			model_move = self.model.move(board, color = self.color, jump_piece_number = jump_piece_number) # retrieve one hot move
			piece_number = int(np.argmax(model_move)/4)
			move = model_move[(4 * piece_number):((4 * piece_number) + 4)]
			while np.count_nonzero(model_move * board.legal_moves(color = self.color, jump_piece_number = jump_piece_number)) == 0: # check if the cuurent proposed move is legal
				self.illegal_move_count += 1
				model_move = self.model.move(board, color = self.color, jump_piece_number = jump_piece_number) # retrieve one hot move
				piece_number = int(np.argmax(model_move)/4)
				move = model_move[(4 * piece_number):((4 * piece_number) + 4)]
				#print(model_move)
				#print(board.legal_moves(color = self.color))
				#print(piece_number)
				#print(move)
		return move, piece_number

	def increment_move_count(self):
		self.move_count += 1

	def reset(self):
		self.move_count = 0
		self.illegal_move_count = 0