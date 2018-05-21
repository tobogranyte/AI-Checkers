import numpy as np

class Board:

	def __init__(self):
		self.state = np.zeros((4, 8, 4), int)
		print("Initialize")

	def start_game(self):
		self.state[0] = np.concatenate((np.zeros((5,4), int), np.ones((3,4), int)), axis = 0)
		self.state[2] = np.concatenate((np.ones((3,4), int), np.zeros((5,4), int)), axis = 0)
		print("Start Game")

	def status(self):
		return self.state

