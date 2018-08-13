import numpy as np

class RMM:
	
	def __init__(self):
		print("New Random Moves Model")

	def move(self, board, color, jump_piece_number = None, jump_rule = True, illegal = False):
		move = np.random.randint(48)
		one_hot_move = np.eye(48, dtype = 'int')[move]
		if illegal == False:
			self.board_legal_moves = board.legal_moves(color = color, jump_piece_number = jump_piece_number, jump_rule = jump_rule)

		return one_hot_move, self.board_legal_moves