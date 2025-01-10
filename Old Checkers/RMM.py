import numpy as np

class RMM:
	
	def __init__(self):
		print("New Random Moves Model")
		self.layers_dims = [1] #  6-layer model

	def move(self, board, color, jump_piece_number = None, jump_rule = True, illegal = False):
		move = np.random.randint(96)
		one_hot_move = np.eye(96, dtype = 'int')[move]
		if illegal == False:
			self.board_legal_moves = board.legal_moves(color = color, jump_piece_number = jump_piece_number, jump_rule = jump_rule)

		return one_hot_move, self.board_legal_moves

	def forward_pass(self, X, game_numbers):
		"""
		in a functioning model, this would generate a matrix of probabilities. For this random number generating "model"
		it will just return a zeros matrix of the correct dimensions that will later be filled (in generate_move) with one_hot
		random values
		"""
		move = np.zeros((96,X.shape[1]))
		return move

	def generate_move(self, AL): # generate a move from a probabilities vector
		choice = np.random.randint(96)
		one_hot_move = np.eye(96, dtype = 'int')[choice] #generate one-hot version
		piece_number = int(np.argmax(one_hot_move)/8) # get the piece number that the move applies to
		move = one_hot_move[(8 * piece_number):((8 * piece_number) + 8)] # generate the move for that piece

		return one_hot_move, piece_number, move

	def complete_move(self):
		pass

	def get_input_vector(self, board, color, jump_piece_number):
		v = np.array([0], dtype=int)
		return v.reshape(v.size, -1)
