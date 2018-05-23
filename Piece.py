import numpy as np

class Piece:

	def __init__(self, color, king = False, x = 0, y = 0):
		self.xPosition = x
		self.yPosition = y
		self.flatten_position()
		self.color = color

	def flatten_position(self):
		# return a scalar integer reflecting which of the board's
		# 32 positinos the piece occupies
		# bottom left (z, 7, 0) maps to position 0

		self.flattened_position = np.zeros((32,1), dtype = int)
		self.flattened_position[(7 - self.yPosition) * 4 + self.xPosition] = 1

	def legal_moves(self, board):
		return np.array([[self.forward_left(board), self.forward_right(board)], [self.backward_left(board), self.backward_right(board)]])
		#position the board to have the correct color "facing" the player


	def forward_left(self, board):
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

	def forward_right(self, board):
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

	def backward_left(self, board):
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

	def backward_right(self, board):
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



