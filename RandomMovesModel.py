import numpy as np

class RandomMovesModel:
	
	def __init__(self):
		print("Hello model world!")

	def move(self, board, color, jump_piece_number = None):
		move = np.random.randint(48)
		one_hot_move = np.eye(48, dtype = 'int')[move]

		return one_hot_move