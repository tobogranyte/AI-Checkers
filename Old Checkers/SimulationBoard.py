import numpy as np
from Piece import Piece

class SimulationBoard:

	def __init__(self, state, red_pieces, black_pieces): # blank board: no pieces
		self.state = state.copy()
		self.red_numbers = np.zeros((8, 4), int) - 1
		self.black_numbers = np.zeros((8, 4), int) - 1
		self.jump_mask = np.array([1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1], dtype = 'int')
		self.red_pieces = list(red_pieces)
		self.black_pieces = list(black_pieces)

