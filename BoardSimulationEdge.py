import numpy as np
from Piece import Piece
from GameState import GameState
from Player import Player


class BoardSimulationEdge:
	def __init__(self, game_state: "GameState", depth, root_color, red_player, black_player, jump_piece_number, red_moves, black_moves, number, player_color, other_player_color, player: "Player"):
		self.game_state = game_state
		self.depth = depth
		self.root_color = root_color
		self.red_player = red_player
		self.black_player = black_player
		self.jump_piece_number = jump_piece_number
		self.win = False
		self.side = None
		self.P = 0
		self.N = 0
		self.Q = 0
		self.v = 0
		self.c = 1
		self.UCB = 0
		self.singleton = False
		self.has_children = False
		self.has_child_outputs = False
		self.child_sim_edges = {}
		self.policy_output = np.array((96))
		self.value_output = np.array((1))
		self.active_child_index = 0
		self.red_moves = red_moves
		self.black_moves = black_moves
		self.number = number
		self.player_color = player_color
		self.other_player_color = other_player_color
		self.player = player
		self.jump_mask = np.array([1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1], dtype = 'int')
		self.mask = self.legal_moves()

	def get_red_moves(self, depth):
		if depth == self.depth:
			return self.red_moves
		else:
			return self.child_sim_edges[self.active_child_index].get_red_moves(depth)

	def get_black_moves(self, depth):
		if depth == self.depth:
			return self.black_moves
		else:
			return self.child_sim_edges[self.active_child_index].get_black_moves(depth)

	def red_home_view(self): # return the red state view of the board for the model
		return self.game_state.state

	def black_state(self): # return the board state viewed "from the other side of the board"
		return np.flip(np.flip(self.game_state.state, axis = 1), axis = 2)

	def legal_moves(self, jump_rule = True):
		# return a 96 element array with all the legal moves for pieces 0-11 consecutively
		# 0 = illegal, 1 = legal, 2 = legal jump
		
		if self.player.color == "Red":
			pieces = self.game_state.red_pieces
		else:
			pieces = self.game_state.black_pieces
		moves = np.zeros((96), dtype = 'int') # zero array to put the legal moves
		if self.jump_piece_number: # this is the second, mandatory jump move after a first jump
			moves[(self.jump_piece_number * 8):((self.jump_piece_number * 8) + 8)] = self.jump_moves(pieces[self.jump_piece_number]).flatten()
			"""
			If there is a jump piece, it means that this is the second move by this color, because the first was a jump.
			Thus, the only legal move at this point is a jump. So it inserts the legal moves for this piece only into the
			array. The rest will remain zeros.
			"""
		else:
			for p in pieces:
				moves[(p.number * 8):((p.number * 8) + 8)] = self.piece_legal_moves(p).flatten() # insert legal moves for the selected piece into the move array
			if np.max(moves * self.jump_mask) == 1 and jump_rule: # a jump is available and a jump rule (requiring the player to use a jump if availabe) is in effect
				moves = moves * self.jump_mask # mask only jump moves for legal moves
		return moves

	def jump_moves(self, piece):
		"""
		Return only the jump rows (first and last rows) from legal moves. The rest are zeros."""
		j = np.zeros(8, dtype = int).reshape(4,2)
		m = self.piece_legal_moves(piece)
		j[0,:] = m[0,:]
		j[3,:] = m[3,:]
		return j
	
	def position_color(self, y, x, player):
		if (self.position(y, x, player)[0] == 1) or (self.position(y, x, player)[1] == 1):
			return "Red"
		elif (self.position(y, x, player)[2] == 1) or (self.position(y, x, player)[3] == 1):
			return "Black"
		else:
			return "O"
		
	def position(self, y, x, player):
		if player == "Red":
			return self.game_state.state[:, 7-y, x]
		elif player == "Black":
			return self.black_state()[:, 7-y, x]

	'''
	Methods that used to be in Piece class
	'''
	def forward_left(self, piece): # determine move value for forward left (0 = illegal, 1 = move, 2 = jump)
		if piece.yPosition%2 == 0: # even row
			if piece.xPosition == 0: # leftmost position
				return np.array([[0],[0]]) # can't move forward left
			else: #not the leftmost position
				c = self.position_color((piece.yPosition + 1), (piece.xPosition - 1), piece.color) # color in forward left position
				if piece.yPosition != 6: # can move up to two positions forward
					c2 = self.position_color((piece.yPosition + 2), (piece.xPosition - 1), piece.color) #color in forward left jump position
					if c  == piece.color: # forward left occupied by same color (no moves available)
						return np.array([[0],[0]])
					elif c == "O": # forward left unoccupied
						return np.array([[0],[1]])
					elif c2 == "O": # forward left jump unoccupied
						return np.array([[1],[0]])
					else: # no moves available
						return np.array([[0],[0]])
				else: # can move only move one position forward
					if c != "O": # forward left occupied
						return np.array([[0],[0]])
					else: #forward left available
						return np.array([[0],[1]])
		elif  piece.yPosition == 7: # odd and last row
			return np.array([[0],[0]]) # no possible forward moves
		else: # not the last row
			c = self.position_color((piece.yPosition + 1), (piece.xPosition), piece.color)
			if piece.xPosition != 0: # odd and not the leftmost position
				c2 = self.position_color((piece.yPosition + 2), (piece.xPosition - 1), piece.color)
				if c  == piece.color:  # forward left occupied by same color (no moves available)
					return np.array([[0],[0]])
				elif c == "O":  # forward left unoccupied
					return np.array([[0],[1]])
				elif c2 == "O": # forward left jump unoccupied
					return np.array([[1],[0]])
				else: 
					return np.array([[0],[0]])
			else: # odd and leftmost position
				if c != "O":
					return np.array([[0],[0]])
				else:
					return np.array([[0],[1]])

	def forward_right(self, piece): # determine move value for forward right (0 = illegal, 1 = move, 2 = jump)
		if piece.yPosition%2 != 0: # odd row
			if  piece.yPosition == 7: # last row
				return np.array([[0],[0]]) # no possible forward moves
			else:
				if piece.xPosition == 3: # odd and rightmost position
					return np.array([[0],[0]]) # can't move forward right
				else: # odd and not the rightmost position
					c = self.position_color((piece.yPosition + 1), (piece.xPosition + 1), piece.color)
					c2 = self.position_color((piece.yPosition + 2), (piece.xPosition + 1), piece.color)
					if c  == piece.color:
						return np.array([[0],[0]])
					elif c == "O":
						return np.array([[0],[1]])
					elif c2 == "O":
						return np.array([[1],[0]])
					else:
						return np.array([[0],[0]])
		else: #even row
			c = self.position_color((piece.yPosition + 1), (piece.xPosition), piece.color)
			if piece.xPosition != 3: # even and not rightmost position
				if piece.yPosition != 6: # can move up to two positions forward
					c2 = self.position_color((piece.yPosition + 2), (piece.xPosition + 1), piece.color)
					if c  == piece.color:
						return np.array([[0],[0]])
					elif c == "O":
						return np.array([[0],[1]])
					elif c2 == "O":
						return np.array([[1],[0]])
					else:
						return np.array([[0],[0]])
				else:
					if c != "O":
						return np.array([[0],[0]])
					else:
						return np.array([[0],[1]])
			else: # even and rightmost position
				if c != "O":
					return np.array([[0],[0]])
				else:
					return np.array([[0],[1]])

	def backward_left(self, piece): # determine move value for backward left (0 = illegal, 1 = move, 2 = jump)
		if piece.yPosition%2 == 0: # even row
			if  piece.yPosition == 0: # even and first row
				return np.array([[0],[0]]) # no possible backward moves
			else: #even and not first row
				if piece.xPosition == 0: # leftmost position
					return np.array([[0],[0]]) # can't move backward left
				else: #not the leftmost position
					c = self.position_color((piece.yPosition - 1), (piece.xPosition - 1), piece.color) # color in backward left position
					c2 = self.position_color((piece.yPosition - 2), (piece.xPosition - 1), piece.color) #color in forward left jump position
					if c  == piece.color:
						return np.array([[0],[0]])
					elif c == "O":
						return np.array([[1],[0]])
					elif c2 == "O":
						return np.array([[0],[1]])
					else:
						return np.array([[0],[0]])
		else: # odd row
			c = self.position_color((piece.yPosition - 1), (piece.xPosition), piece.color)
			if piece.xPosition != 0: # odd and not the leftmost position
				if piece.yPosition != 1: # can move up to two positions backward
					c2 = self.position_color((piece.yPosition - 2), (piece.xPosition - 1), piece.color)
					if c  == piece.color:
						return np.array([[0],[0]])
					elif c == "O":
						return np.array([[1],[0]])
					elif c2 == "O":
						return np.array([[0],[1]])
					else:
						return np.array([[0],[0]])
				else:
					if c != "O":
						return np.array([[1],[0]])
					else:
						return np.array([[0],[0]])
			else: # odd and leftmost position
				if c != "O":
					return np.array([[0],[0]])
				else:
					return np.array([[1],[0]])

	def backward_right(self, piece): # determine move value for backward right (0 = illegal, 1 = move, 2 = jump)
		if piece.yPosition%2 != 0: # odd row
			if piece.xPosition == 3: # odd and rightmost position
				return np.array([[0],[0]]) # can't move backward right
			else: # odd and not the rightmost position
				c = self.position_color((piece.yPosition - 1), (piece.xPosition + 1), piece.color)
				if piece.yPosition != 1: # odd and can move up to two positions backward 
					c2 = self.position_color((piece.yPosition - 2), (piece.xPosition + 1), piece.color)
					if c  == piece.color:
						return np.array([[0],[0]])
					elif c == "O":
						return np.array([[1],[0]])
					elif c2 == "O":
						return np.array([[0],[1]])
					else:
						return np.array([[0],[0]])
				else:
					if c != "O":
						return np.array([[0],[0]])
					else:
						return np.array([[1],[0]])
		else: # even row
			if piece.yPosition == 0: # first row
				return np.array([[0],[0]])
			c = self.position_color((piece.yPosition - 1), (piece.xPosition), piece.color)
			if piece.xPosition != 3: # even and not rightmost position
				c2 = self.position_color((piece.yPosition - 2), (piece.xPosition + 1), piece.color)
				if c  == piece.color:
					return np.array([[0],[0]])
				elif c == "O":
					return np.array([[1],[0]])
				elif c2 == "O":
					return np.array([[0],[1]])
				else:
					return np.array([[0],[0]])
			else: # even and rightmost position
				if c != "O":
					return np.array([[0],[0]])
				else:
					return np.array([[1],[0]])
	
	def piece_legal_moves(self, piece):
		"""
		Returns a [4,2] array in the following form:
		[[0, 0], -> jump forward left, jump forward right
    	[ 0, 0], -> move forward left, move forward right
    	[ 0, 0], -> move backward left, move backward right
    	[ 0, 0]] -> jump backward left, jump backward right
		"""
		m = np.zeros(8, dtype = int).reshape(4,2) # default 2x2 matrix is all zeros until a move is deterimined
		if piece.in_play: # piece is still on the board
			m[0:2, :] = np.append(self.forward_left(piece), self.forward_right(piece), axis=1) # update values for forward moves
			if piece.king:
				m[2:4, :] = np.append(self.backward_left(piece), self.backward_right(piece), axis = 1) # update values for backward moves if piece is king
		return m

