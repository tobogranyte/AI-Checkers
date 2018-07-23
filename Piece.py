import numpy as np

class Piece:

	def __init__(self, number, color, king = False, x = 0, y = 0, in_play = True):
		self.xPosition = x
		self.yPosition = y
		self.number = number
		self.color = color
		self.king = king
		self.in_play = in_play
		self.flatten_position()

		# Generate an array for this piece which will be able to map z values to whatever board position this piece occupies
		self.update_position_array()
	# def place_piece(board)

	def update_position_array(self):
		self.position_array = np.array([(self.color == "Red") and not (self.king), (self.color == "Red") and self.king, (self.color == "Black") and not (self.king), (self.color == "Black") and self.king]) * 1 * self.in_play

	def number_one_hot(self):
		one_hot = np.zeros((12), int)
		one_hot[self.number] = 1
		return one_hot


	def flatten_position(self):
		# return a flattened array with 32 elements reflecting 
		# 32 positinos on the board. The bottom left (z, 7, 0)
		# maps to position 0. All will be 0s except the position
		# where the piece is, which will be 1. Bottom left is always
		# bottom left of a board oriented with player's (red or black) home side
		# facing the player.

		self.flattened_position = np.zeros((32,1), dtype = int)
		self.flattened_position[(7 - self.yPosition) * 4 + self.xPosition] = 1

	def legal_moves(self, board):
		m = np.zeros(4, dtype = int).reshape(2,2) # default 2x2 matrix is all zeros until a move is deterimined
		if self.in_play: # piece is still on the board
			m[0, :] = np.array([self.forward_left(board), self.forward_right(board)]) # update values for forward moves
			if self.king:
				m[1, :] = np.array([self.backward_left(board), self.backward_right(board)]) # update values for backward moves if piece is king

		return m

	def make_king(self):
		self.king = True
		self.update_position_array()

	def forward_left(self, board): # determine move value for forward left (0 = illegal, 1 = move, 2 = jump)
		if self.yPosition%2 == 0: # even row
			if self.xPosition == 0: # leftmost position
				return 0 # can't move forward left
			else: #not the leftmost position
				c = board.color((self.yPosition + 1), (self.xPosition - 1), self.color) # color in forward left position
				if self.yPosition != 6: # can move up to two positions forward
					c2 = board.color((self.yPosition + 2), (self.xPosition - 1), self.color) #color in forward left jump position
					if c  == self.color:
						return 0
					elif c == "O":
						return 1
					elif c2 == "O":
						return 2
					else:
						return 0
				else:
					if c != "O":
						return 0
					else:
						return 1
		elif  self.yPosition == 7: # odd and last row
			return 0 # no possible forward moves
		else: # not the last row
			c = board.color((self.yPosition + 1), (self.xPosition), self.color)
			if self.xPosition != 0: # odd and not the leftmost position
				c2 = board.color((self.yPosition + 2), (self.xPosition - 1), self.color)
				if c  == self.color:
					return 0
				elif c == "O":
					return 1
				elif c2 == "O":
					return 2
				else:
					return 0
			else: # odd and leftmost position
				if c != "O":
					return 0
				else:
					return 1

	def forward_right(self, board): # determine move value for forward right (0 = illegal, 1 = move, 2 = jump)
		if self.yPosition%2 != 0: # odd row
			if  self.yPosition == 7: # last row
				return 0 # no possible forward moves
			else:
				if self.xPosition == 3: # odd and rightmost position
					return 0 # can't move forward right
				else: # odd and not the rightmost position
					c = board.color((self.yPosition + 1), (self.xPosition + 1), self.color)
					c2 = board.color((self.yPosition + 2), (self.xPosition + 1), self.color)
					if c  == self.color:
						return 0
					elif c == "O":
						return 1
					elif c2 == "O":
						return 2
					else:
						return 0
		else: #even row
			c = board.color((self.yPosition + 1), (self.xPosition), self.color)
			if self.xPosition != 3: # even and not rightmost position
				c2 = board.color((self.yPosition + 2), (self.xPosition + 1), self.color)
				if c  == self.color:
					return 0
				elif c == "O":
					return 1
				elif c2 == "O":
					return 2
				else:
					return 0
			else: # even and rightmost position
				if c != "O":
					return 0
				else:
					return 1

	def backward_left(self, board): # determine move value for backward left (0 = illegal, 1 = move, 2 = jump)
		if self.yPosition%2 == 0: # even row
			if  self.yPosition == 0: # even and first row
				return 0 # no possible backward moves
			else: #even and not first row
				if self.xPosition == 0: # leftmost position
					return 0 # can't move backward left
				else: #not the leftmost position
					c = board.color((self.yPosition - 1), (self.xPosition - 1), self.color) # color in backward left position
					c2 = board.color((self.yPosition - 2), (self.xPosition - 1), self.color) #color in forward left jump position
					if c  == self.color:
						return 0
					elif c == "O":
						return 1
					elif c2 == "O":
						return 2
					else:
						return 0
		else: # odd row
			c = board.color((self.yPosition - 1), (self.xPosition), self.color)
			if self.xPosition != 0: # odd and not the leftmost position
				if self.yPosition != 1: # can move up to two positions backward
					c2 = board.color((self.yPosition - 2), (self.xPosition - 1), self.color)
					if c  == self.color:
						return 0
					elif c == "O":
						return 1
					elif c2 == "O":
						return 2
					else:
						return 0
				else:
					if c != "O":
						return 1
					else:
						return 0
			else: # odd and leftmost position
				if c != "O":
					return 0
				else:
					return 1

	def backward_right(self, board): # determine move value for backward right (0 = illegal, 1 = move, 2 = jump)
		if self.yPosition%2 != 0: # odd row
			if self.xPosition == 3: # odd and rightmost position
				return 0 # can't move backward right
			else: # odd and not the rightmost position
				c = board.color((self.yPosition - 1), (self.xPosition + 1), self.color)
				if self.yPosition != 1: # odd and can move up to two positions backward 
					c2 = board.color((self.yPosition - 2), (self.xPosition + 1), self.color)
					if c  == self.color:
						return 0
					elif c == "O":
						return 1
					elif c2 == "O":
						return 2
					else:
						return 0
				else:
					if c != "O":
						return 0
					else:
						return 1
		else: # even row
			if self.yPosition == 0: # first row
				return 0
			c = board.color((self.yPosition - 1), (self.xPosition), self.color)
			if self.xPosition != 3: # even and not rightmost position
				c2 = board.color((self.yPosition - 2), (self.xPosition + 1), self.color)
				if c  == self.color:
					return 0
				elif c == "O":
					return 1
				elif c2 == "O":
					return 2
				else:
					return 0
			else: # even and rightmost position
				if c != "O":
					return 0
				else:
					return 1



