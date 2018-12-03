This is my AI Checkers project. The bulk of the actual files in this project are not actually the AI model but the checkers mechanics which I had to build from scratch. That includes:

* Checkers.py - The top level class that oversees everything from game-playing through training.
* Game.py - The class that plays a single game.
* Piece.py - The class representing a piece on either side of a game.
* Player.py - The class representing a player. Ιn charge of making individual moves within a game.

The actual AI models are, at this point, in FCN1.py and FCN2.py and RMM.py. The player basically asks the model which move it should make, the model returns probabilities for any of 48 possible moves (12 pieces * four possible moves per piece, not all legal) and then the player uses those probabilities to choose which move. RMM.py is, to date, not a real model. Instead it generates random unit normalized probabilities and it's what the other actual AI models play against--at least for now. Both of my current actual models are fully connected networks, since for the time being, I'm mostly working just on predicting legal vs. illegal moves. FCN1 was the first iteration and FCN2 added some extra features and is what I'm currently in the process of porting over to TensorFlow. That's currently still in an unmerged branch and can be found in FCN_TF.py.