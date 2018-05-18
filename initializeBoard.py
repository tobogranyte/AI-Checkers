import numpy as np

def initializeBoard ():

	# create a matrix for an 8 x 4 board with four layers
	# [0] - Red piece
	# [1] - Red King
	# [2] - Black piece
	# [3] - Black king
	Board = np.zeros((4, 8, 4), int)

	# Initialize the board with Red and Black pieces
	Board[0] = np.concatenate((np.zeros((5,4), int), np.ones((3,4), int)), axis = 0)
	Board[2] = np.concatenate((np.ones((3,4), int), np.zeros((5,4), int)), axis = 0)

	return Board