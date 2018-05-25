import numpy as np
from Piece import Piece

class Board:

	def __init__(self):
		self.state = np.zeros((4, 8, 4), int)
		self.numbers = np.zeros((8, 4), int) - 1
		print("Initialize")

	def start_game(self):
		self.red_piece = []
		self.black_piece = []
		for p in range (0,12):
			self.red_piece.append(Piece(p, "Red", king = False, x = p%4, y = int(p/4), in_play = True))
			self.place_piece(self.red_piece[p])
			self.black_piece.append(Piece(p, "Black", king = False, x = p%4, y = int(p/4), in_play = True))
			self.place_piece(self.black_piece[p])

	def red_state(self):
		return self.state

	def red_numbers(self):
		return self.numbers

	def black_state(self):
		return np.flip(np.flip(self.state, axis = 1), axis = 2)

	def black_numbers(self):
		return np.flip(np.flip(self.numbers, axis = 0), axis = 1)

	def visual_state(self):
		p = ["O", "r", "R", "b", "B"]
		s = np.argmax(np.concatenate((np.zeros((1,8,4), dtype = int), self.red_state()), axis = 0), axis = 0)
		i = 1
		for y in s:
			for x in y:
				if i == 1:
					print(' ', p[x], end='')
				else:
					print(p[x], ' ', end='')
			i = -i
			print()
		print()
		#print(s)

	def place_piece(self, piece):
		if piece.color == "Red":
			self.red_state()[:, 7 - piece.yPosition, piece.xPosition] = piece.position_array
			self.red_numbers()[7 - piece.yPosition, piece.xPosition] = piece.number
		elif piece.color == "Black":
			self.black_state()[:, 7 - piece.yPosition, piece.xPosition] = piece.position_array
			self.black_numbers()[7 - piece.yPosition, piece.xPosition] = piece.number

	def remove_piece(self, piece):
		if piece.color == "Red":
			self.red_state()[:, 7 - piece.yPosition, piece.xPosition] = np.zeros(4, dtype = int)
			self.red_numbers()[7 - piece.yPosition, piece.xPosition] = -1
		elif piece.color == "Black":
			self.black_state()[:, 7 - piece.yPosition, piece.xPosition] = np.zeros(4, dtype = int)
			self.black_numbers()[7 - piece.yPosition, piece.xPosition] = -1
		piece.in_play = False

	def move_piece(self, color, number, move):
		if color == "Red":
			piece = self.red_piece[number]
			other_piece = self.black_piece
			state = self.red_state()
			numbers = self.red_numbers()
		else:
			piece = self.black_piece[number]
			other_piece = self.red_piece
			state = self.black_state()
			numbers = self.black_numbers()
		m = piece.legal_moves(self).flatten()[move]
		y1 = int(move / 2) * (-2) + 1
		yDest = piece.yPosition + (y1) * m
		x1 = move%2 + piece.yPosition % 2 - 1
		xDest = piece.xPosition + (x1 + ((move % 2 + ((piece.yPosition + 1 ) % 2 - 1)) * (m - 1))) * (m == 1 or m == 2)
		print("Shift: ", piece.yPosition % 2 - 1)
		print("x1: ", x1)
		print("y1: ", y1)
		print("Number: ", numbers[7 - (piece.yPosition + y1), piece.xPosition + x1])
		print("x: ", piece.xPosition + x1)
		print("y: ", 7 - piece.yPosition + y1)
		if m == 2:
			self.remove_piece(other_piece[numbers[7 - (piece.yPosition + y1), piece.xPosition + x1]])
		state[:, 7 - piece.yPosition, piece.xPosition] = np.zeros(4, dtype=int)
		numbers[7 - piece.yPosition, piece.xPosition] = -1
		piece.yPosition = yDest
		piece.xPosition = xDest
		state[:, 7 - piece.yPosition, piece.xPosition] = piece.position_array
		numbers[7 - piece.yPosition, piece.xPosition] = piece.number

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


