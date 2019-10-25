import numpy as np
from Board import Board

# Game class manages all game mechanics. The main program creates a new game,
# assigns players to the game (which have already been given models), and issues the
# play_game command to play a game. The game then returns all relevant data
# from the game once it's complete.

class Game:

	def __init__(self, red_player, black_player, jump_rule, number):
		self.board = Board() # create a new board for the game. Internal representation of the board state.
		self.board.setup() # set up the board with pieces in starting formation
		self.red_player = red_player # assign the red player
		self.black_player = black_player # assign the black player
		self.jump_rule = jump_rule # are available jumps mandatory to make?
		self.number = number
		self.attempts = 0
		self.moves = 0

	def start(self):
		"""
		Play a single game of checkers. Return the following parameters:
		win -- True if game was won, False if stalemate
		side -- color of player that won the game
		self.board.piece_count("Red") -- number of pieces left on the red side
		self.board.piece_count("Black") -- number of pieces left on the black side
		self.red_player.move_count -- number of red
		"""
		#game_history = open('game_history.txt','w')
		self.red_attempts = 0
		self.red_moves = 0
		self.black_attempts = 0
		self.black_moves = 0
		self.stalemate = False
		self.win = False
		self.jump_piece_number = None
		if np.random.uniform(0, 1) >= .5:
			self.player = self.red_player
		else:
			self.player = self.black_player

		return self.player.color

	def update_attempts_and_moves(self, attempts, color):
		if color == "Red":
			self.red_attempts += attempts
			self.red_moves += 1
		else:
			self.black_attempts += attempts
			self.black_moves += 1

	def player_color(self):
		return self.player.color

	def other_player_color(self):
		return self.player.other_color

	def make_move(self, move, piece_number):
		board_move = np.zeros((96), dtype = 'int') # create output vector placeholder with zeros
		# move, piece_number  = self.player.make_move(self.board, jump_piece_number = self.jump_piece_number, jump_rule = self.jump_rule) # get a move from the player
		board_legal_moves = self.board.legal_moves(color = self.player.color, jump_piece_number = self.jump_piece_number, jump_rule = self.jump_rule) # get legal moves from the board
		if np.max(move) != 0: # there are legal moves
			board_move[(piece_number * 8):((piece_number * 8) + 8)] = move # insert an 8 element vector of the moves for the chosen piece into the correct spot in the board move vector
			move_array = board_legal_moves * board_move # generate an array that masks the board move with legal moves to determine whether the move is legal
			while np.count_nonzero(move_array) == 0: # if there are all zeros, it's not legal
				print("Checkers proposed illegal move!!")
				board_move = np.zeros((96), dtype = 'int')
				board_move[(piece_number * 8):((piece_number * 8) + 8)] = move
				move_array = board_legal_moves * board_move
			self.board.move_piece(self.player.color, piece_number, move) # move the piece on the board
			self.player.increment_move_count()
			#game_history.write(self.board.visual_state())
			if np.max(move_array) == 2: # it was a jump
				count = self.board.piece_count(color = self.player.other_color) # get the number of opposing pieces that are still left on the board
				if count == 0: # that was the last piece
					self.win = True
					self.side = self.player.color
				else: # it wasn't the last piece
					if np.max(self.board.legal_piece_moves(color = self.player.color, piece_number = piece_number).flatten()) == 2:
						self.jump_piece_number = piece_number
					else:
						self.jump_piece_number = None
						if self.player == self.red_player:
							self.player = self.black_player
						else:
							self.player = self.red_player
					board_legal_moves = self.board.legal_moves(color = self.player.color, jump_piece_number = self.jump_piece_number, jump_rule = self.jump_rule)
					if np.max(board_legal_moves) == 0:
						self.win = True
						self.side = self.player.other_color
			else:
				self.jump_piece_number = None
				if self.player == self.red_player:
					self.player = self.black_player
				else:
					self.player = self.red_player
				board_legal_moves = self.board.legal_moves(color = self.player.color, jump_piece_number = self.jump_piece_number, jump_rule = self.jump_rule)
				if np.max(board_legal_moves) == 0:
					self.win = True
					self.side = self.player.other_color
		else:  # no legal moves
			self.win = True # declare a win
			self.side = self.player.other_color # set the side to the other color (who won)
		#game_history.close()

		return self.win, self.stalemate

	def other_player_count(self):
		return self.board.piece_count(color = self.player.other_color)

	def player_count(self):
		return self.board.piece_count(color = self.player.color)

	def stats(self):

		return self.win, self.side, self.board.piece_count("Red"), self.board.piece_count("Black"), self.red_player.move_count, self.black_player.move_count, self.red_player.illegal_move_count, self.black_player.illegal_move_count
				
	def generate_X_Y_mask(self):
		X = self.player.model.get_input_vector(self.board, self.player.color, jump_piece_number = self.jump_piece_number)
		board_legal_moves = self.board.legal_moves(color = self.player.color, jump_piece_number = self.jump_piece_number, jump_rule = self.jump_rule) # get legal moves (48,) for current board position (0: illegal, 1:legal, 2:jump-legal)
		illegal_mask = np.zeros((48)) # create a holder (48,) for the illegal mask (starting filled with zeros)
		illegal_mask[board_legal_moves != 0] = 1 # ones for anything that's legal
		S = np.sum(illegal_mask, axis = 0) # sum of total legal moves
		if S == 0:
			S = 1
		# This seems to be forcing a 1 for the sum when there are no legal moves.
		# But in the fullness of time I'm not sure why a situation with no legal moves would
		# ever make it into the training set.
		Y = illegal_mask / (S) # divide each move mask vector by S to get a unit normal label

		return X, Y, illegal_mask


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

