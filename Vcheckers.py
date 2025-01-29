import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
batch_size = 2

'''
Create static mask templates

Masks with true for legal moves (jump and regular) for even rows and odd rows
'''
even_mask = torch.tensor([[1, 0, 1],
						  [0, 1, 1],
						  [0, 0, 0],
						  [0, 1, 1],
						  [1, 0, 1]], dtype = int, device=device)
odd_mask = torch.tensor([[1, 0, 1],
						 [1, 1, 0],
						 [0, 0, 0],
						 [1, 1, 0],
						 [1, 0, 1]], dtype = int, device=device)

red_piece = torch.tensor([1,0,0,0], dtype = int, device=device)
red_king = torch.tensor([0,1,0,0], dtype = int, device=device)
black_piece = torch.tensor([0,0,1,0], dtype = int, device=device)
black_king = torch.tensor([0,0,0,1], dtype = int, device=device)

# mask for jump positions. Same for all rows
legal_jumps_mask = torch.tensor([[True, False, True], [False, False, False], [False, False, False], [False, False, False], [True, False, True]], dtype = int, device=device)
# unsqueeze to fit batch of boards
legal_jumps_mask = legal_jumps_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(batch_size, 8, 4, -1, -1)
# create blank board
boards = torch.zeros((batch_size, 4, 8, 4), dtype = int, device=device)


def board_visual(boards):
	board_pieces = torch.full((batch_size, 8, 4), 0, dtype = int, device=device)
	values = torch.tensor([1, 2, 3, 4], dtype = int, device=device)
	hot_indices = torch.argmax(boards, dim=1)  # Shape: (batch_size, H, W)
	valid_mask = boards.sum(dim=1) > 0
	mapped_values = values[hot_indices]
	board_pieces = torch.where(valid_mask, mapped_values, board_pieces)
	visual_board = torch.full((batch_size, 8, 8), 7, dtype = int, device=device)
	even_rows = visual_board[:, ::2, :]
	odd_rows = visual_board[:, 1::2, :]
	visual_board[:, ::2, 1::2] = board_pieces[:, ::2, :]
	visual_board[:, 1::2, ::2] = board_pieces[:, 1::2, :]
	print(visual_board)

def expand_to_5x3(boards):
	return boards.unsqueeze(3).unsqueeze(4).expand(-1, -1, -1, 5, 3).clone()

def edge_pieces(boards):
	boards[0, :, [0, 7], :] = red_piece.view(4, 1, 1).expand(4, 2, 4)
	boards[0, :, 1:7, 0] = red_piece.unsqueeze(1)
	boards[0, :, 1:7, 3] = red_piece.unsqueeze(1)

def edge_kings(boards):
	boards[1, :, [0, 7], :] = red_king.view(4, 1, 1).expand(4, 2, 4)
	boards[1, :, 1:7, 0] = red_king.unsqueeze(1)
	boards[1, :, 1:7, 3] = red_king.unsqueeze(1)

def board_setup(boards):
	boards[1, :, 5:8, :] = red_piece.view(4, 1, 1).expand(4, 3, 4).unsqueeze(0).expand(batch_size, -1, -1, -1)
	boards[1, :, 0:3, :] = black_piece.view(4, 1, 1).expand(4, 3, 4).unsqueeze(0).expand(batch_size, -1, -1, -1)

def get_legal_moves(boards):
	padded_board = torch.zeros((batch_size, 4, 12, 6), dtype = int)
	padded_board[:, 0, :, :] = 1
	padded_board[:, :, 2:10, 1:5] = boards
	'''
	red_pieces(batch_size, 8, 4) Boolean with True if the location has a red_piece and False if not.
	'''
	red_pieces = torch.logical_or(boards[:, 0, :, :], boards[:, 1, :, :])
	red_pieces_masks = expand_to_5x3(red_pieces)

	'''
	empty(batch_size, 12, 6) Padded Boolean with True if the location is empty, False if not.
	'''
	empty = torch.logical_not(padded_board[:, 0, :, :] | padded_board[:, 1, :, :] | padded_board[:, 2, :, :] | padded_board[:, 3, :, :])

	'''
	Available jumps
	'''
	skewed_blacks = torch.zeros((batch_size, 4, 12, 5), dtype = int)
	skewed_blacks[:, 0, :, :] = 1
	skewed_blacks[:, :, 2:10, 0:4] = boards
	skewed_blacks = torch.logical_or(skewed_blacks[:, 2, :, :], skewed_blacks[:, 3, :, :]).float()
	even_rows = torch.arange(skewed_blacks.size(1))[::2]
	skewed_blacks[:, even_rows, 1:] = skewed_blacks[:, even_rows, :-1]
	skewed_blacks[:, even_rows, 0] = 0
	skewed_blacks = skewed_blacks.unsqueeze(1)
	unfold_5_2 = nn.Unfold(kernel_size=(5, 2))
	jumpable_masks = unfold_5_2(skewed_blacks)
	jumpable_masks = jumpable_masks.view(batch_size, 5, 2, 8, 4)
	jumpable_masks = jumpable_masks.permute(0, 3, 4, 1, 2)
	jumpable_masks = F.pad(jumpable_masks, pad=(0,1,0,0), mode='constant', value=1)
	jumpable_masks[:, :, :, :, 2] = jumpable_masks[:, :, :, :, 1]
	jumpable_masks[:, :, :, [0, 4], :] = jumpable_masks[:, :, :, [1, 3], :]
	jumpable_masks[:, :, :, :, 1] = 1
	jumpable_masks[:, :, :, 1:4, :] = 1
	jumpable_masks = jumpable_masks.bool()


	'''
	empty_masks(batch_size, 8, 4, 5, 3) with a (5, 3) array for each location with True for empty and False for not for the (5,3)
	including and surrounding the location.
	'''
	empty = empty.unsqueeze(1).float()
	#print(empty)
	unfold_5_3 = nn.Unfold(kernel_size=(5, 3))
	empty_masks = unfold_5_3(empty)
	empty_masks = empty_masks.view(batch_size, 5, 3, 8, 4)
	empty_masks = empty_masks.permute(0, 3, 4, 1, 2).bool()

	'''
	king(batch_size, 1, 12, 6) Padded Float with 1 if the location is king, 0 if not.
	'''

	king_masks = expand_to_5x3(board[:, 1, :, :])
	king_masks[:, :, :, 0:3, :] = 1
	king_masks = king_masks.bool()


	'''
	empty_masks(batch_size, 8, 4, 5, 3) with a (5, 3) array for each location with True for king-legal destination and False
	for not for the (5,3) including and surrounding the location.
	'''

	legal_masks = torch.zeros((batch_size, 8, 4, 5, 3))
	legal_masks[:, ::2, :, :, :] = even_mask
	legal_masks[:, 1::2, :, :, :] = odd_mask
	legal_masks = legal_masks.bool()
	board_legal_moves = red_pieces_masks & empty_masks & jumpable_masks & king_masks & legal_masks
	jump_moves_only = (board_legal_moves & legal_jumps_mask).any(dim=(1, 2, 3, 4))
	jump_moves_only = jump_moves_only.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4).expand(-1, 8, 4, 3, 3)
	jmo_mask = torch.full((batch_size, 8, 4, 5, 3), False)
	jmo_mask[:, :, :, 1:4, :] = jump_moves_only
	board_legal_moves = torch.where(jmo_mask, False, board_legal_moves)

	return board_legal_moves