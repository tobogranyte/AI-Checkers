import numpy as np
from Board import Board
"""
Player class manages all moves for a given player. At the beginning of
a session, two players are created, one for each side. Each player
is assigned a model which handles deciding what move to make.
Once a game (Game) is started, the game lets the player know when
it is its turn to move by invoking the make_move method which includes
passing the board state. The player, in turn, passes that board state
to the model which returns what move it would like to make. The player
verifies the move is legal. If not, it asks the model for another move and
continues to do so until the model generates a legal move. Once the model
generates a legal move, the player hands that move back to the game which
ultimately moves the piece.
"""

class Player:
	
	def __init__(self, model, color):
		self.model = model
		self.color = color
		self.reset()
		if self.color == "Red":
			self.other_color = "Black"
		else:
			self.other_color = "Red"

	def increment_move_count(self):
		self.move_count += 1

	def reset(self):
		self.move_count = 0
		self.illegal_move_count = 0