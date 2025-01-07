import numpy as np
from Piece import Piece

class Board:

	def __init__(self): # blank board: no pieces
		self.state = np.zeros((4, 8, 4), int) #set all board positions to [0,0,0,0] vectors
		self.red_numbers = np.zeros((8, 4), int) - 1
		self.black_numbers = np.zeros((8, 4), int) - 1
		self.jump_mask = np.array([1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1], dtype = 'int')
