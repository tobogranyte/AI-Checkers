import numpy as np

class Board:

	def __init__(self):
		self.state = np.zeros((4, 8, 4), int)
		print("Initialize")

	def start_game(self):
		self.state[0] = np.concatenate((np.zeros((5,4), int), np.ones((3,4), int)), axis = 0)
		self.state[2] = np.concatenate((np.ones((3,4), int), np.zeros((5,4), int)), axis = 0)

	def red_state(self):
		return self.state

	def black_state(self):
		return np.flip(np.flip(self.state, axis = 1), axis = 2)

	# return array in a given x, y position from the vantage point of player
	def position(self, y, x, player):
		if player == "Red":
			return self.red_state()[:, 7-y, x]
		elif player == "Black":
			return self.black_state()[:, 7-y, x]


	# return the color in a given x, y position from the vantage point of player
	def color(self, y, x, player):
		if (self.position(y, x, player)[0] == 1) or (self.position(y, x, player)[1] == 1):
			return "Red"
		elif (self.position(y, x, player)[2] == 1) or (self.position(y, x, player)[3] == 1):
			return "Black"
		else:
			return "O"

	# def legal_moves(self, position):


