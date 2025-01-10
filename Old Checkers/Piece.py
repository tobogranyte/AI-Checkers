import numpy as np

class Piece:

	def __init__(self, number, color, king = False, x = 0, y = 0, in_play = True):
		self.xPosition = x
		self.yPosition = y
		self.number = number
		self.color = color
		self.king = king
		self.in_play = in_play

		# Generate an array for this piece which will be able to map z values to whatever board position this piece occupies
		self.update_position_array()
	# def place_piece(board)

	def update_position_array(self):
		# [R, RK, B, BK]
		self.position_array = np.array([(self.color == "Red") and not (self.king), (self.color == "Red") and self.king, (self.color == "Black") and not (self.king), (self.color == "Black") and self.king]) * 1 * self.in_play

	def number_one_hot(self):
		one_hot = np.zeros((12), int)
		one_hot[self.number] = 1
		return one_hot


	def make_king(self):
		self.king = True
		self.update_position_array()



