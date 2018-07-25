import numpy as np
from Board import Board

# Player class manages all moves for a given player. At the beginning of
# a session, two players is be created, one for each side. Each player
# is be assigned a model which handles deciding what move to make.
# Once a game (Game) is started, the game lets the player know when
# it is its turn to move by issuing the [name TBD] command which includes
# passing the board state. The player, in turn, passes that board state
# to the model which returns what move it would like to make. The player
# hands that back to the game which ultimately moves the piece.

class Player:
	
	def __init__(self, model='RandomMovesModel'):
		import_string = 'from ' + model + ' import Model' # create import string
		exec(import_string, globals())
		self.model = Model()
