import numpy as np
import time
from SimulationBoard import SimulationBoard
from Piece import Piece
from Game import Game
from dataclasses import dataclass, field

class SimulationState:
    def __init__(self, state, red_pieces, black_pieces):
        self.state = state
        self.red_pieces = red_pieces
        self.black_pieces = black_pieces

red_pieces = []
black_pieces = []
for p in range (0,12):
    red_pieces.append(Piece(p, "Red", king = False, x = p%4, y = int(p/4), in_play = True))
    black_pieces.append(Piece(p, "Black", king = False, x = p%4, y = int(p/4), in_play = True))
state = np.zeros((4, 8, 4))

simulation_set = {}

start_time = time.time()
for count in range (1000000):
    #simulation_state = {}
    #simulation_state.update({"state": state.copy(), "red_pieces": list(red_pieces), "black_pieces": list(black_pieces)})
    simulation_state = SimulationState(state=state.copy(), red_pieces=red_pieces, black_pieces=black_pieces)
    
    simulation_set.update({count: simulation_state})
end_time = time.time()
total = end_time - start_time

print(total)
