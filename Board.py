import numpy as np
from Piece import Piece

class Board:

	def __init__(self): # blank board: no pieces
		self.state = np.zeros((4, 8, 4), int) #set all board positions to [0,0,0,0] vectors
		self.red_numbers = np.zeros((8, 4), int) - 1 
		self.black_numbers = np.zeros((8, 4), int) - 1 

	def setup(self): # set up all pieces in starting positions
		self.red_piece = []
		self.black_piece = []
		for p in range (0,12):
			self.red_piece.append(Piece(p, "Red", king = False, x = p%4, y = int(p/4), in_play = True))
			self.place_piece(self.red_piece[p])
			self.black_piece.append(Piece(p, "Black", king = False, x = p%4, y = int(p/4), in_play = True))
			self.place_piece(self.black_piece[p])

	def red_state(self): # return the objective board state
		return self.red_state

	def red_home_view(self): # return the red state view of the board for the model
		return self.state

	def black_state(self): # return the board state viewed "from the other side of the board"
		return np.flip(np.flip(self.state, axis = 1), axis = 2)

	def get_piece_vector(self, color):
		if color == 'Red':
			numbers = self.red_numbers # red piece numbers for all board locations (red vantage point)
			piece = self.red_piece				
		else:
			numbers = self.black_numbers # black
			piece = self.black_piece
		v = np.zeros((396))
		in_play = np.zeros((12))
		eye = np.eye(12)
		for i in range(numbers.size):
			n = numbers.flatten()[i]
			if n != -1:
				v[(i * 12):(i * 12) + 12] = eye[n]
		for p in piece:
			if p.in_play == True:
				in_play[p.number] = 1
		v[384:396] = in_play
		return v

	def black_home_view(self): 
		# return the black state view ("from the other side of the board"), and with the red/black
		# dimension flipped for the model. This is so that no matter what
		# model I'm training, even if I'm training the same model to play both sides, I can present
		# an entirely red/black neutral version of the game. The model simply learns its home side and
		# home pieces regardless of red/black. Essentially, this switches the red/black grid so even though
		# it's nominally playing black, to the model, their pieces are always red.
		state = self.black_state()
		return np.append(state[2:,:,:], state[:2,:,:], axis = 0)

	def legal_moves(self, color, jump_piece_number = None, jump_rule = True):
		# return a 48 element array with all the legal moves for pieces 0-11 consecutively
		# 0 = illegal, 1 = legal, 2 = legal jump
		if color == "Red":
			piece = self.red_piece
		else:
			piece = self.black_piece
		if jump_piece_number: # this is the second, mandatory jump move after a first jump
			moves = np.zeros((48), dtype = 'int')
			moves[(jump_piece_number * 4):((jump_piece_number * 4) + 4)] = (piece[jump_piece_number].legal_moves(self).flatten()/2).astype(int) * 2
		else:
			moves = np.array((), dtype = 'int')
			for p in piece:
				moves = np.append(moves, p.legal_moves(self).flatten())
			if np.max(moves) == 2 and jump_rule:
				moves = (moves / 2).astype(int) * 2
		return moves

	def piece_count(self, color):
		count = 0
		if color == "Red":
			piece = self.red_piece
		else:
			piece = self.black_piece
		for p in piece:
			if p.in_play == True:
				count = count + 1
		return count

	def legal_piece_moves(self, color, piece_number):
		if color == "Red":
			piece = self.red_piece
		else:
			piece = self.black_piece
		legal_moves = piece[piece_number].legal_moves(self)
		return legal_moves

	def black_numbers(self):
		return np.flip(np.flip(self.numbers, axis = 0), axis = 1)

	def print_visual_state(self):
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

	def visual_state(self):
		board = ""
		p = ["O", "r", "R", "b", "B"]
		s = np.argmax(np.concatenate((np.zeros((1,8,4), dtype = int), self.red_state()), axis = 0), axis = 0)
		i = 1
		for y in s:
			for x in y:
				if i == 1:
					board += ' ' + p[x]
				else:
					board += p[x] + ' '
			i = -i
			board += '\n'
		board += '\n'
		return board
		#print(s)

	def place_piece(self, piece):
		if piece.color == "Red":
			self.red_state()[:, 7 - piece.yPosition, piece.xPosition] = piece.position_array
			self.red_numbers[7 - piece.yPosition, piece.xPosition] = piece.number
		elif piece.color == "Black":
			self.black_state()[:, 7 - piece.yPosition, piece.xPosition] = piece.position_array
			self.black_numbers[7 - piece.yPosition, piece.xPosition] = piece.number

	def king_piece(self, piece):
		piece.make_king()
		if piece.color == "Red":
			self.red_state()[:, 7 - piece.yPosition, piece.xPosition] = piece.position_array
		elif piece.color == "Black":
			self.black_state()[:, 7 - piece.yPosition, piece.xPosition] = piece.position_array

	def remove_piece(self, piece):
		if piece.color == "Red":
			self.red_state()[:, 7 - piece.yPosition, piece.xPosition] = np.zeros(4, dtype = int)
			self.red_numbers[7 - piece.yPosition, piece.xPosition] = -1
		elif piece.color == "Black":
			self.black_state()[:, 7 - piece.yPosition, piece.xPosition] = np.zeros(4, dtype = int)
			self.black_numbers[7 - piece.yPosition, piece.xPosition] = -1
		piece.in_play = False

	# move a piece from one position to another
	def move_piece(self, color, number, move):
		if color == "Red": # set the stage for a red move
			piece = self.red_piece[number] # set the piece to move
			opposition_piece = self.black_piece # set the opposition positions
			state = self.red_state() # set the opposition positions
			numbers = self.red_numbers # set the array with all the red piece number positions
			opposition_numbers = np.flip(np.flip(self.black_numbers, axis = 0), axis = 1) # set the opposition array with all the red piece number positions flipped
		else: # set the stage for a black move
			piece = self.black_piece[number] # set the stage for a red move
			opposition_piece = self.red_piece # set the piece to move
			state = self.black_state() # set the opposition positions
			numbers = self.black_numbers # set the array with all the black piece number positions
			opposition_numbers = np.flip(np.flip(self.red_numbers, axis = 0), axis = 1) # set the opposition array with all the red piece number positions flipped
		m = piece.legal_moves(self).flatten()[move] # retrieve the value for the move chosen (0 for illegal, 1 for move, 2 for jump)
		y1 = int(move / 2) * (-2) + 1 # 
		yDest = piece.yPosition + (y1) * m # destination y position
		x1 = move%2 + piece.yPosition % 2 - 1
		xDest = piece.xPosition + (x1 + ((move % 2 + ((piece.yPosition + 1 ) % 2 - 1)) * (m - 1))) * (m == 1 or m == 2) # destination x position
		if m == 2:
			self.remove_piece(opposition_piece[opposition_numbers[7 - (piece.yPosition + y1), piece.xPosition + x1]])
		state[:, 7 - piece.yPosition, piece.xPosition] = np.zeros(4, dtype=int)
		numbers[7 - piece.yPosition, piece.xPosition] = -1
		piece.yPosition = yDest
		piece.xPosition = xDest
		state[:, 7 - piece.yPosition, piece.xPosition] = piece.position_array
		numbers[7 - piece.yPosition, piece.xPosition] = piece.number
		if piece.yPosition == 7:
			self.king_piece(piece)

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


