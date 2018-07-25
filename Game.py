import numpy as np
from Board import Board

class Game:

	def __init__(self):
		self.board = Board()
		self.board.setup()

	def static_playtest(self):
		self.board.visual_state()
		#print(self.board.red_numbers())

		self.board.move_piece("Red", 10, 1)

		self.board.visual_state()
		#print(self.board.red_numbers())

		self.board.move_piece("Black", 8, 1)

		self.board.visual_state()
		#print(self.board.red_numbers())

		self.board.move_piece("Red", 10, 1)

		self.board.visual_state()
		#print(self.board.red_numbers())

		self.board.move_piece("Black", 9, 0)

		self.board.visual_state()
		#print(self.board.red_numbers())

		self.board.move_piece("Black", 9, 0)

		self.board.visual_state()
		#print(self.board.red_numbers())

		self.board.move_piece("Black", 4, 1)

		self.board.visual_state()
		#print(self.board.red_numbers())

		self.board.move_piece("Black", 4, 0)

		self.board.visual_state()
		#print(self.board.red_numbers())

		self.board.move_piece("Black", 1, 0)

		self.board.visual_state()
		#print(self.board.red_numbers())

		self.board.move_piece("Black", 1, 1)

		self.board.visual_state()
		#print(self.board.red_numbers())

		self.board.move_piece("Red", 10, 0)

		self.board.visual_state()
		#print(self.board.red_numbers())

		self.board.move_piece("Red", 10, 0)

		self.board.visual_state()
		#print(self.board.red_numbers())

		self.board.move_piece("Red", 10, 3)

		self.board.visual_state()
		#print(self.board.red_numbers())

		self.board.move_piece("Red", 10, 2)

		self.board.visual_state()
		#print(self.board.red_numbers())

		self.board.move_piece("Red", 10, 0)

		self.board.visual_state()
		#print(self.board.red_numbers())

		self.board.move_piece("Red", 11, 0)

		self.board.visual_state()
		#print(self.board.red_numbers())

		self.board.move_piece("Red", 6, 0)

		self.board.visual_state()
		#print(self.board.red_numbers())

		self.board.move_piece("Red", 6, 0)

		self.board.visual_state()
		#print(self.board.red_numbers())

		self.board.move_piece("Red", 2, 1)

		self.board.visual_state()
		#print(self.board.red_numbers())

		self.board.move_piece("Red", 2, 0)

		self.board.visual_state()
		#print(self.board.red_numbers())

		self.board.move_piece("Black", 9, 1)

		self.board.visual_state()
		#print(self.board.red_numbers())

		self.board.move_piece("Black", 9, 1)

		self.board.visual_state()
		#print(self.board.red_numbers())

		self.board.move_piece("Black", 9, 1)

		self.board.visual_state()
		#print(self.board.red_numbers())

		self.board.move_piece("Red", 6, 0)

		self.board.visual_state()
		#print(self.board.red_numbers())

		self.board.move_piece("Black", 9, 2)

		self.board.visual_state()
		#print(self.board.red_numbers())

		self.board.move_piece("Black", 9, 3)

		self.board.visual_state()
		#print(self.board.red_numbers())

		self.board.move_piece("Black", 9, 2)

		self.board.visual_state()
		#print(self.board.red_numbers())

		for p in self.board.red_piece:
			print(p.color, p.number, p.number_one_hot(), p.in_play)

		for p in self.board.black_piece:
			print(p.color, p.number, p.number_one_hot(), p.in_play)


		print(self.board.red_state())
		print(self.board.black_home_view())
		print(self.board.red_home_view())
