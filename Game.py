import numpy as np
from Board import Board

# Game class manages all game mechanics. The main program creates a new game,
# assigns players to the game (which have already been given models), and issues the
# play_game command to play a game. The game then returns all relevant data
# from the game once it's complete.

class Game:

	def __init__(self, red_player, black_player, jump_rule):
		self.board = Board() # create a new board for the game
		self.board.setup() # set up the board with pieces in starting formation
		self.red_player = red_player # assign the red player
		self.black_player = black_player # assign the black player
		self.jump_rule = jump_rule # are available jumps mandatory to make?

	def play_game(self):
		#game_history = open('game_history.txt','w')
		game_history = str('')
		stalemate = False
		win = False
		hold = False
		jump_piece_number = None
		no_jump_count = 0 # to track a simplified stalemate rule
		if np.random.uniform(0, 1) >= .5:
			player = self.red_player
		else:
			player = self.black_player
		while not stalemate and not win:
			board_move = np.zeros((48), dtype = 'int')
			move, piece_number  = player.make_move(self.board, jump_piece_number = jump_piece_number, jump_rule = self.jump_rule) # get a move from the player
			board_legal_moves = self.board.legal_moves(color = player.color, jump_piece_number = jump_piece_number, jump_rule = self.jump_rule)
			if np.max(move) != 0:
				board_move[(piece_number * 4):((piece_number * 4) + 4)] = move
				move_array = board_legal_moves * board_move
				while np.count_nonzero(move_array) == 0:
					print("Player proposed illegal move!!")
					board_move = np.zeros((48), dtype = 'int')
					move, piece_number  = player.make_move(self.board, jump_piece_number = jump_piece_number, jump_rule = self.jump_rule)
					board_move[(piece_number * 4):((piece_number * 4) + 4)] = move
					move_array = board_legal_moves * board_move
				self.board.move_piece(player.color, piece_number, np.argmax(move))
				player.increment_move_count()
				#game_history.write(self.board.visual_state())
				game_history += self.board.visual_state()
				if np.max(move_array) == 2:
					count = self.board.piece_count(color = player.other_color)
					if count == 0:
						win = True
						side = player.color
					else:
						if np.max(self.board.legal_piece_moves(color = player.color, piece_number = piece_number).flatten()) == 2:
							jump_piece_number = piece_number
						else:
							jump_piece_number = None
							if player == self.red_player:
								player = self.black_player
							else:
								player = self.red_player
				else:
					jump_piece_number = None
					if player == self.red_player:
						player = self.black_player
					else:
						player = self.red_player
			else:
				win = True
				side = player.other_color
		#game_history.close()
		print(game_history)
		input('Pause...')

		return win, side, self.board.piece_count("Red"), self.board.piece_count("Black"), self.red_player.move_count, self.black_player.move_count, self.red_player.illegal_move_count, self.black_player.illegal_move_count
				
					

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

