import numpy as np
from Piece import Piece

class Board:

	def __init__(self): # blank board: no pieces
		self.state = np.zeros((4, 8, 4), int) #set all board positions to [0,0,0,0] vectors
		self.red_numbers = np.zeros((8, 4), int) - 1
		self.black_numbers = np.zeros((8, 4), int) - 1
		self.jump_mask = np.array([1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1], dtype = 'int')

	def setup(self): # set up all pieces in starting positions
		self.red_pieces = []
		self.black_pieces = []
		for p in range (0,12):
			self.red_pieces.append(Piece(p, "Red", king = False, x = p%4, y = int(p/4), in_play = True))
			self.place_piece(self.red_pieces[p])
			self.black_pieces.append(Piece(p, "Black", king = False, x = p%4, y = int(p/4), in_play = True))
			self.place_piece(self.black_pieces[p])

	def red_state(self): # return the objective board state
		return self.state

	def red_home_view(self): # return the red state view of the board for the model
		return self.state

	def black_state(self): # return the board state viewed "from the other side of the board"
		return np.flip(np.flip(self.state, axis = 1), axis = 2)

	def get_piece_vector(self, color):
		if color == 'Red':
			numbers = self.red_numbers # red piece numbers for all board locations (red vantage point)
			piece = self.red_pieces
		else:
			numbers = self.black_numbers # black
			piece = self.black_pieces
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
		return v # 1-dimensional vector

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
		# return a 96 element array with all the legal moves for pieces 0-11 consecutively
		# 0 = illegal, 1 = legal, 2 = legal jump
		
		if color == "Red":
			pieces = self.red_pieces
		else:
			pieces = self.black_pieces
		moves = np.zeros((96), dtype = 'int') # zero array to put the legal moves
		if jump_piece_number: # this is the second, mandatory jump move after a first jump
			moves[(jump_piece_number * 8):((jump_piece_number * 8) + 8)] = pieces[jump_piece_number].jump_moves(self).flatten()
			"""
			If there is a jump piece, it means that this is the second move by this color, because the first was a jump.
			Thus, the only legal move at this point is a jump. So it inserts the legal moves for this piece only into the
			array. The rest will remain zeros.
			"""
		else:
			for p in pieces:
				moves[(p.number * 8):((p.number * 8) + 8)] = p.legal_moves(self).flatten() # insert legal moves for the selected piece into the move array
			if np.max(moves * self.jump_mask) == 1 and jump_rule: # a jump is available and a jump rule (requiring the player to use a jump if availabe) is in effect
				moves = moves * self.jump_mask # mask only jump moves for legal moves
		return moves

	def piece_count(self, color):
		count = 0
		if color == "Red":
			pieces = self.red_pieces
		else:
			pieces = self.black_pieces
		for p in pieces:
			if p.in_play == True:
				count = count + 1
		return count

	def legal_piece_moves(self, color, piece_number):
		if color == "Red":
			pieces = self.red_pieces
		else:
			pieces = self.black_pieces
		legal_moves = pieces[piece_number].legal_moves(self)
		return legal_moves

	def print_visual_state(self):
		p = ["O", "r", "R", "b", "B"]
		s = np.argmax(np.concatenate((np.zeros((1,8,4), dtype = int), self.red_state()), axis = 0), axis = 0)
		own_numbers = self.red_numbers # set the array with all the red piece number positions
		opposition_numbers = np.flip(np.flip(self.black_numbers, axis = 0), axis = 1) # set the opposition array with all the red piece number positions flipped
		numbers = np.max(np.append(own_numbers, opposition_numbers).reshape(2,8,4), axis = 0)
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
		own_numbers = self.red_numbers # set the array with all the red piece number positions
		opposition_numbers = np.flip(np.flip(self.black_numbers, axis = 0), axis = 1) # set the opposition array with all the red piece number positions flipped
		numbers = np.max(np.append(own_numbers, opposition_numbers).reshape(2,8,4), axis = 0)
		i = 1
		for idy, y in enumerate(s):
			for idx, x in enumerate(y):
				if i == 1:
					board += '   ' + p[x]
					if numbers[idy,idx] < 10 and numbers[idy,idx] >= 0:
						board += '0'
						board += numbers[idy,idx].astype(str)
					elif numbers[idy,idx] < 0:
						board += '  '
					else:
						board += numbers[idy,idx].astype(str)
				else:
					board += p[x]
					if numbers[idy,idx] < 10 and numbers[idy,idx] >= 0:
						board += '0'
						board += numbers[idy,idx].astype(str)
					elif numbers[idy,idx] < 0:
						board += '  '
					else:
						board += numbers[idy,idx].astype(str)
					board += '   '
			i = -i
			board += '\n'
		board += '\n'
		return board
		#print(s)

	def pieces_out(self):
		print("B Out:", end='')
		for p in self.black_pieces:
			if p.in_play == False:
				print(p.number, end = ' ')
		print()
		print("R Out:", end='')
		for p in self.red_pieces:
			if p.in_play == False:
				print(p.number, end = ' ')
		print()

	def position_array(self, color, yPosition, xPosition): # return a position array [n,n,n,n] for the xPosition and yPosition from the vantage point of color
		if color == "Red":
			return self.red_home_view()[:, 7-yPosition, xPosition]
		else:
			return self.black_home_view()[:, 7-yPosition, xPosition]

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
			piece = self.red_pieces[number] # set the piece to move
			opposition_pieces = self.black_pieces # set the opposition positions
			state = self.red_state() # set the opposition positions
			numbers = self.red_numbers # set the array with all the red piece number positions
			opposition_numbers = np.flip(np.flip(self.black_numbers, axis = 0), axis = 1) # set the opposition array with all the red piece number positions flipped
		else: # set the stage for a black move
			piece = self.black_pieces[number] # set the stage for a red move
			opposition_pieces = self.red_pieces # set the piece to move
			state = self.black_state() # set the opposition positions
			numbers = self.black_numbers # set the array with all the black piece number positions
			opposition_numbers = np.flip(np.flip(self.red_numbers, axis = 0), axis = 1) # set the opposition array with all the red piece number positions flipped
		masked_move = piece.legal_moves(self).flatten() * move # mask the move with legal moves for that piece
		vertical = np.sum(masked_move.reshape(4,2), axis=1) # vertical array of move in y axis
		horizontal = np.sum(masked_move.reshape(4,2), axis=0)
		left_right = np.argmax(horizontal)
		m = np.array([2,1,1,2])[np.argmax(vertical)] # 2 for jump, 1 for move
		y_offset = (2 - np.argmax(np.insert(vertical, 2, 0)))
		yDest = piece.yPosition + y_offset
		x1 = left_right + piece.yPosition % 2 - 1 # x offset for first row from position
		xDest = piece.xPosition + (x1 + ((left_right + ((piece.yPosition + 1 ) % 2 - 1)) * (m - 1))) * (m == 1 or m == 2) # destination x position
		#                                |   x offset for first row from position    |   |     |
		#                                ---------------------------------------------   -------
		#                                                                             Don't include
		#                                                                               if m == 1 (only moving 1 row)
		if m == 2:
			opposition_number = opposition_numbers[7 - (piece.yPosition + int(y_offset / 2)), piece.xPosition + x1]
			# print("Remove piece:", opposition_pieces[opposition_number].color, opposition_number)
			self.remove_piece(opposition_pieces[opposition_number])
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

	def get_piece_arrays(self, color):
		piece_arrays = np.zeros(384, int)
		if color == "Red":
			for p in range (12):
				piece_arrays[32*p:32*(p+1)] = self.red_pieces[p].spaces(self).flatten()
		else:
			for p in range (12):
				piece_arrays[32*p:32*(p+1)] = self.black_pieces[p].spaces(self).flatten()
		return piece_arrays

	# return the color in a given x, y position from the vantage point of player
	def position_color(self, y, x, player):
		if (self.position(y, x, player)[0] == 1) or (self.position(y, x, player)[1] == 1):
			return "Red"
		elif (self.position(y, x, player)[2] == 1) or (self.position(y, x, player)[3] == 1):
			return "Black"
		else:
			return "O"


