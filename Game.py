import numpy as np
from Board import Board

# Game class manages all game mechanics. The main program creates a new game,
# assigns players to the game (which have already been given models), and issues the
# play_game command to play a game. The game then returns all relevant data
# from the game once it's complete.

class Game:

	def __init__(self, red_player, black_player, jump_rule, number, side = None):
		self.board = Board() # create a new board for the game. Internal representation of the board state.
		self.board.setup() # set up the board with pieces in starting formation
		self.red_player = red_player # assign the red player
		self.black_player = black_player # assign the black player
		self.jump_rule = jump_rule # are available jumps mandatory to make?
		self.number = number
		self.attempts = 0
		self.moves = 0
		self.red_attempts = 0
		self.red_moves = 0
		self.black_attempts = 0
		self.black_moves = 0
		self.moves_since_jump = 0
		self.draw = False
		self.win = False
		self.jump_piece_number = None
		if side == None:
			if np.random.uniform(0, 1) >= .5: # choose a starting player at random
				self.player = self.red_player
			else:
				self.player = self.black_player
		elif side == "red":
			self.player = self.red_player
		else:
			self.player = self.black_player
		

	def get_piece_positions(self):
		red_pieces = {}
		black_pieces = {}
		for n in range(0,12):
			if self.board.red_pieces[n].in_play:
				rcol = (self.board.red_pieces[n].xPosition * 2) + (self.board.red_pieces[n].yPosition % 2)
				rrow = 7 - self.board.red_pieces[n].yPosition
				red_pieces[n] = (rrow, rcol)
			else:
				red_pieces[n] = (-1, -1)
			if self.board.black_pieces[n].in_play:
				bcol = 7 - ((self.board.black_pieces[n].xPosition * 2)  + (self.board.black_pieces[n].yPosition % 2))
				brow = self.board.black_pieces[n].yPosition
				black_pieces[n] = (brow, bcol)
			else:
				black_pieces[n] = (-1, -1)
		return red_pieces, black_pieces

	def get_legal_moves(self):
		red_board_legal_moves = self.board.legal_moves(color = "Red", jump_piece_number = self.jump_piece_number, jump_rule = self.jump_rule)
		print(red_board_legal_moves.reshape(1,-1))
		black_board_legal_moves = self.board.legal_moves(color = "Black", jump_piece_number = self.jump_piece_number, jump_rule = self.jump_rule)
		red_legal = {}
		black_legal = {}
		for n in range(0,12):
			red_legal[n] = red_board_legal_moves[n*8:(n+1)*8].reshape(4,2)
			black_legal[n] = np.flip( np.flip( black_board_legal_moves[n*8:(n+1)*8].reshape(4,2), axis = 0), axis = 1)
		return red_legal, black_legal
	
	def get_in_play(self):
		red_in_play = {}
		black_in_play = {}

		for n in range(0,12):
			red_in_play[n] = self.board.red_pieces[n].in_play
			black_in_play[n] = self.board.black_pieces[n].in_play
		return red_in_play, black_in_play
	
	def get_kings(self):
		red_kings = {}
		black_kings = {}

		for n in range(0,12):
			red_kings[n] = self.board.red_pieces[n].king
			black_kings[n] = self.board.black_pieces[n].king
		return red_kings, black_kings

	def player_color(self):
		return self.player.color

	def other_player_color(self):
		return self.player.other_color
	
	def make_move(self, move, piece_number):
		# print(self.player.color, "move ", piece_number, ":", move)
		board_move = np.zeros((96), dtype = 'int') # create output vector placeholder with zeros
		board_legal_moves = self.board.legal_moves(color = self.player.color, jump_piece_number = self.jump_piece_number, jump_rule = self.jump_rule) # get legal moves from the board
		if np.max(move) != 0: # there are legal moves
			board_move[(piece_number * 8):((piece_number * 8) + 8)] = move # insert an 8 element vector of the moves for the chosen piece into the correct spot in the board move vector
			move_array = board_legal_moves * board_move # generate an array that masks the board move with legal moves to determine whether the move is legal
			while np.count_nonzero(move_array) == 0: # if there are all zeros, it's not legal
				# print("Checkers proposed illegal move!!")
				board_move = np.zeros((96), dtype = 'int')
				board_move[(piece_number * 8):((piece_number * 8) + 8)] = move
				move_array = board_legal_moves * board_move
			self.board.move_piece(self.player.color, piece_number, move) # move the piece on the board
			self.player.increment_move_count()
			self.moves += 1
			# print(self.board.visual_state())
			# print(self.board.pieces_out())
			# input('Resume...')
			#game_history.write(self.board.visual_state())
			if np.max(move_array * self.board.jump_mask) == 1: # it was a jump
				self.moves_since_jump = 0
				count = self.board.piece_count(color = self.player.other_color) # get the number of opposing pieces that are still left on the board for the other player
				if count == 0: # that was the last piece for the other player
					self.win = True # The game is won
					self.side = self.player.color # by the current player
				else: # it wasn't the last piece
					if np.max((self.board.legal_piece_moves(color = self.player.color, piece_number = piece_number).flatten()) * np.array([1,1,0,0,0,0,1,1])) == 1: # There is another available jump for this piece
						self.jump_piece_number = piece_number
					else: # There is not another jump for this piece
						self.jump_piece_number = None # no more jump piece
						if self.player == self.red_player:
							self.player = self.black_player
						else:
							self.player = self.red_player
						# other player's turn
					board_legal_moves = self.board.legal_moves(color = self.player.color, jump_piece_number = self.jump_piece_number, jump_rule = self.jump_rule) 
					if np.max(board_legal_moves) == 0: # Check if next player has any legal moves, if not:
						self.win = True # declare win
						self.side = self.player.other_color # player other that next player wins
			else: # it was not a jump
				self.moves_since_jump += 1 # increment count since last jump
				self.jump_piece_number = None
				if self.player == self.red_player:
					self.player = self.black_player
				else:
					self.player = self.red_player
				board_legal_moves = self.board.legal_moves(color = self.player.color, jump_piece_number = self.jump_piece_number, jump_rule = self.jump_rule)
				if np.max(board_legal_moves) == 0:
					self.win = True
					self.side = self.player.other_color
				elif self.moves_since_jump >= 400:
					self.draw = True
					self.side = self.player.other_color
		else:  # no legal moves
			self.win = True # declare a win
			self.side = self.player.other_color # set the side to the other color (who won)
		#game_history.close()

		return self.win, self.draw

	def other_player_count(self):
		return self.board.piece_count(color = self.player.other_color)

	def player_count(self):
		return self.board.piece_count(color = self.player.color)

	def stats(self):

		return self.win, self.draw, self.side, self.board.piece_count("Red"), self.board.piece_count("Black"), self.red_player.move_count, self.black_player.move_count, self.red_player.illegal_move_count, self.black_player.illegal_move_count
				
	def generate_X_mask(self):
		board_input, pieces_input = self.player.model.get_input_vector(self.board, self.player.color, jump_piece_number = self.jump_piece_number)
		board_legal_moves = self.board.legal_moves(color = self.player.color, jump_piece_number = self.jump_piece_number, jump_rule = self.jump_rule) # get legal moves (48,) for current board position (0: illegal, 1:legal, 2:jump-legal)
		# Can't have this be 0 because the next line divides by zero

		return board_input, pieces_input, board_legal_moves


	def static_playtest(self):
		self.board.visual_state()
		#print(self.board.red_numbers

		self.board.move_piece("Red", 10, 1)

		self.board.visual_state()
		#print(self.board.red_numbers

		self.board.move_piece("Black", 8, 1)

		self.board.visual_state()
		#print(self.board.red_numbers

		self.board.move_piece("Red", 10, 1)

		self.board.visual_state()
		#print(self.board.red_numbers

		self.board.move_piece("Black", 9, 0)

		self.board.visual_state()
		#print(self.board.red_numbers

		self.board.move_piece("Black", 9, 0)

		self.board.visual_state()
		#print(self.board.red_numbers

		self.board.move_piece("Black", 4, 1)

		self.board.visual_state()
		#print(self.board.red_numbers

		self.board.move_piece("Black", 4, 0)

		self.board.visual_state()
		#print(self.board.red_numbers

		self.board.move_piece("Black", 1, 0)

		self.board.visual_state()
		#print(self.board.red_numbers

		self.board.move_piece("Black", 1, 1)

		self.board.visual_state()
		#print(self.board.red_numbers

		self.board.move_piece("Red", 10, 0)

		self.board.visual_state()
		#print(self.board.red_numbers

		self.board.move_piece("Red", 10, 0)

		self.board.visual_state()
		#print(self.board.red_numbers

		self.board.move_piece("Red", 10, 3)

		self.board.visual_state()
		#print(self.board.red_numbers

		self.board.move_piece("Red", 10, 2)

		self.board.visual_state()
		#print(self.board.red_numbers

		self.board.move_piece("Red", 10, 0)

		self.board.visual_state()
		#print(self.board.red_numbers

		self.board.move_piece("Red", 11, 0)

		self.board.visual_state()
		#print(self.board.red_numbers

		self.board.move_piece("Red", 6, 0)

		self.board.visual_state()
		#print(self.board.red_numbers

		self.board.move_piece("Red", 6, 0)

		self.board.visual_state()
		#print(self.board.red_numbers

		self.board.move_piece("Red", 2, 1)

		self.board.visual_state()
		#print(self.board.red_numbers

		self.board.move_piece("Red", 2, 0)

		self.board.visual_state()
		#print(self.board.red_numbers

		self.board.move_piece("Black", 9, 1)

		self.board.visual_state()
		#print(self.board.red_numbers

		self.board.move_piece("Black", 9, 1)

		self.board.visual_state()
		#print(self.board.red_numbers

		self.board.move_piece("Black", 9, 1)

		self.board.visual_state()
		#print(self.board.red_numbers

		self.board.move_piece("Red", 6, 0)

		self.board.visual_state()
		#print(self.board.red_numbers

		self.board.move_piece("Black", 9, 2)

		self.board.visual_state()
		#print(self.board.red_numbers

		self.board.move_piece("Black", 9, 3)

		self.board.visual_state()
		#print(self.board.red_numbers

		self.board.move_piece("Black", 9, 2)

		self.board.visual_state()
		#print(self.board.red_numbers

		for p in self.board.red_piece:
			print(p.color, p.number, p.number_one_hot(), p.in_play)

		for p in self.board.black_piece:
			print(p.color, p.number, p.number_one_hot(), p.in_play)


		print(self.board.red_state())
		print(self.board.black_home_view())
		print(self.board.red_home_view())

	def first_moves(self):
		print(self.board.red_numbers)
		print(self.board.black_numbers)
		self.board.visual_state()
		#print(self.board.red_numbers

		self.board.move_piece("Red", 10, 1)

		self.board.visual_state()
		#print(self.board.red_numbers

		self.board.move_piece("Black", 8, 1)

		self.board.visual_state()

		print(self.board.red_state())
		print(self.board.black_home_view())
		print(self.board.red_home_view())
		print(self.board.red_numbers)
		print(self.board.black_numbers)

