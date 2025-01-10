import numpy as np

class GameState:
	def __init__(self, state, red_pieces, black_pieces):
		self.state = state
		self.red_pieces = red_pieces
		self.black_pieces = black_pieces
