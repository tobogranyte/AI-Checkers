import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy import optimize
import numpy as np
from Board import Board
from Piece import Piece
from Player import Player
from Game import Game

if input("Symmetric Models [Y/n]?") == "Y":
	s_model = input("Model name:")
	red_player = Player(model = s_model)
	black_player = Player(model = s_model)
else:
	red_model = input("Red player model:")
	black_model = input("Black player model:")
	red_player = Player(model = red_model)
	black_player = Player(model = black_model)

game = Game()

game.static_playtest()
