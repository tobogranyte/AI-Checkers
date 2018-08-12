import numpy as np

class RMM:
	
	def __init__(self):
		print("New Random Moves Model")

	def move(self, board, color, jump_piece_number = None, jump_rule = True, illegal = False):
		move = np.random.randint(48)
		one_hot_move = np.eye(48, dtype = 'int')[move]

		return one_hot_move