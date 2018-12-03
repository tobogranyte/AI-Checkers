This is my AI Checkers project. The bulk of the actual files in this project are not actually the AI model but the checkers mechanics which I had to build from scratch. That includes:

* Checkers.py - The top level class that oversees everything from game-playing through training.
* Game.py - The class that plays a single game.
* Piece.py - The class representing a piece on either side of a game.
* Player.py - The class representing a player. Ιn charge of making individual moves within a game.

The actual AI models are, at this point, in FCN1.py and FCN2.py and RMM.py. The player basically asks the model which move it should make and then the player makes the move. RMM.py is, to date, not a real model. Instead it makes random moves and it's what the other real models play against. Both of my current models are fully connected networks. FCN1 was the first iteration and FCN2 added some extra features and is what I'm currently in the process of porting over to TensorFlow. That's currently still in an unmerged branch and can be found in FCN_TF.py.