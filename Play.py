import pygame
import numpy as np
from Board import Board
from Piece import Piece
from Player import Player
from Game import Game


# Initialize Pygame
pygame.init()

# Constants
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
BOARD_SIZE = 8
SQUARE_SIZE = ( WINDOW_WIDTH - 200 ) // BOARD_SIZE
GRAY = (128, 128, 128)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLACK = (0, 0, 0)

# Set up the display
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Checkers Board")

# Dictionaries to track piece positions and identifiers
red_pieces = {}
black_pieces = {}
red_legal = {}
black_legal = {}
red_in_play = {}
black_in_play = {}
red_kings = {}
black_kings = {}

def draw_board():
	for row in range(BOARD_SIZE):
		for col in range(BOARD_SIZE):
			# Alternate colors based on row and column indices
			if (row + col) % 2 == 0:
				color = WHITE
			else:
				color = GRAY
			pygame.draw.rect(screen, color, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

def get_piece_positions(game):
	global red_pieces, black_pieces
	red_pieces, black_pieces = game.get_piece_positions()

def get_legal_moves(game):
	global red_legal, black_legal
	red_legal, black_legal = game.get_legal_moves()

def get_in_play(game):
	global red_in_play, black_in_play
	red_in_play, black_in_play = game.get_in_play()

def get_kings(game):
	global red_kings, black_kings
	red_kings, black_kings = game.get_kings()

def update_game_state(game):
	get_piece_positions(game)
	get_legal_moves(game)
	get_in_play(game)
	get_kings(game)

def draw_pieces():
	"""Draw the pieces dynamically based on their current positions."""
	for piece_id, (row, col) in red_pieces.items():
		if red_in_play[piece_id]:
			x = col * SQUARE_SIZE + SQUARE_SIZE // 2
			y = row * SQUARE_SIZE + SQUARE_SIZE // 2
			pygame.draw.circle(screen, RED, (x, y), SQUARE_SIZE // 3)
		if red_kings[piece_id]:
			draw_king(row, col, WHITE)

	for piece_id, (row, col) in black_pieces.items():
		if black_in_play[piece_id]:
			x = col * SQUARE_SIZE + SQUARE_SIZE // 2
			y = row * SQUARE_SIZE + SQUARE_SIZE // 2
			pygame.draw.circle(screen, BLACK, (x, y), SQUARE_SIZE // 3)
		if black_kings[piece_id]:
			draw_king(row, col, WHITE)

def draw_king(row, col, color):
	center_x = col * SQUARE_SIZE + SQUARE_SIZE // 2
	center_y = row * SQUARE_SIZE + SQUARE_SIZE // 2

	# Draw the circle (for reference, assuming it's already drawn)

	# Create a font object
	font = pygame.font.SysFont('timesnewroman', SQUARE_SIZE // 2)  # Size proportional to the square

	# Render the 'K' text
	text_surface = font.render('K', True, color)  # White "K"
	text_rect = text_surface.get_rect(center=(center_x, center_y))  # Center the text

	# Blit the text onto the screen
	screen.blit(text_surface, text_rect)

def get_square_from_click(pos):
	"""Convert raw x, y position to row, col of the square."""
	x, y = pos  # Raw x, y values from mouse click
	row = y // SQUARE_SIZE
	col = x // SQUARE_SIZE
	return row, col

def red_piece_occupied(row, col):
	for n in range(0, 12):
		if red_pieces[n] == (row, col):
			print(n)
			print(red_legal[n])
			if np.max(red_legal[n]) == 1:
				return True, True, n
			else:
				return True, False, n
	return False, False, -1

def set_highlight(number):
	global selected_highlight, legal_highlights
	legal_highlights = []
	selected_highlight = red_pieces[number]
	for r in range(0,4):
		for c in range(0,2):
			if red_legal[number][r][c] == 1:
				m = 1
				n = 1
				if r == 0 or r == 3:
					m = 2
				if r == 2 or r == 3:
					n = -1
				v_offset = - m * n
				h_offset = (2 * c - 1) * m
				legal_highlights.append((red_pieces[number][0] + v_offset, red_pieces[number][1] + h_offset))

def draw_highlights():
	color = BLACK
	col = selected_highlight[1] * SQUARE_SIZE
	row = selected_highlight[0] * SQUARE_SIZE
	pygame.draw.rect(screen, color, (col, row, SQUARE_SIZE, SQUARE_SIZE), width=2)
	for n in range(0, len(legal_highlights)):
		col = legal_highlights[n][1] * SQUARE_SIZE
		row = legal_highlights[n][0] * SQUARE_SIZE
		pygame.draw.rect(screen, color, (col, row, SQUARE_SIZE, SQUARE_SIZE), width=2)

def is_legal(row, col):
	for n in range(0, len(legal_highlights)):
		if legal_highlights[n] == (row, col):
			return True
	return False

def get_move(row, col):
	move = np.zeros((8), dtype = 'int')
	rp_row = red_pieces[selected_piece][0]
	rp_col = red_pieces[selected_piece][1]
	h_offset_sign = (col -  rp_col)/abs(col -  rp_col)
	if h_offset_sign == -1:
		c = 0
	else:
		c = 1
	v_offset = row -  rp_row
	if v_offset == -2:
		r = 0
	elif v_offset == -1:
		r = 1
	elif v_offset == 1:
		r = 2
	elif v_offset == 2:
		r = 3
	move[r*2 + c] = 1

	return move

def make_black_move(game):
	X, mask = game.generate_X_mask()
	X = X.reshape(-1,1)
	mask = mask.reshape(-1,1)
	black_AL = black_model.forward_pass(X, mask)
	one_hot_move, piece_number, move = black_model.generate_move(black_AL)
	win, draw = game.make_move(move, piece_number)
		
'''identifier = input("Identifier:")
bootstrap_version = input("Bootstrap version:")
bootstrap_version = int(bootstrap_version)
if input("Mandatory jumps [Y/n]?") == "Y":
	jump_rule = True
else:
	jump_rule = False
s_model = input("Model name:")
import_string = 'from ' + s_model + ' import ' + s_model + ' as sm' # create self_play model import string
exec(import_string, globals())
black_model = sm("black_model", identifier)
black_player = Player(model = black_model, color = "Black") # create the black player assigning model and color
'''
identifier = "20241119190000"
bootstrap_version = 7
jump_rule = True
s_model = "PT"
import_string = 'from ' + s_model + ' import ' + s_model + ' as sm' # create self_play model import string
exec(import_string, globals())
red_model = None
black_model = sm("black_model", identifier)
black_model.load_checkpoint(f"black_model_{bootstrap_version}", identifier)
red_player = Player(model = red_model, color = "Red") # create the red player assigning model and color
black_player = Player(model = black_model, color = "Black") # create the black player assigning model and color
game = Game(red_player = red_player, black_player = black_player, jump_rule = jump_rule, number = 0, side = "red")
# Place pieces before entering the game loop
update_game_state(game)
piece_selected = False
selected_highlight = ()
legal_highlights = []


# Game loop
running = True
while running:
	if game.player_color() == "Red":
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				running = False
			elif event.type == pygame.MOUSEBUTTONDOWN:  # Detect mouse click
				# Get the raw x, y coordinates of the click
				x, y = event.pos

				# Convert to row, col
				row, col = get_square_from_click((x, y))
				print(f"Clicked square: row={row}, col={col}")
				
				if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
					occupied, has_legal, number = red_piece_occupied(row, col)
					if occupied and has_legal:
						piece_selected = True
						selected_piece = number
						set_highlight(number)
					elif piece_selected == True and is_legal(row, col):
						move = get_move(row, col)
						game.make_move(move, selected_piece)
						update_game_state(game)
						piece_selected = False
						if game.player_color() == "Black":
							make_black_move(game)
							update_game_state(game)
	else:
		make_black_move(game)
		update_game_state(game)


	# Draw the checkerboard
	draw_board()

	if piece_selected:
		draw_highlights()

	# Draw the pieces
	draw_pieces()

	# Update the display
	pygame.display.flip()

pygame.quit()