import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import numpy as np

class PT(nn.Module):

	def __init__(self):
		super().__init__()
		self.batch_num = 0
		self.layers_dims = [397, 1024, 512, 256, 128, 96] #  5-layer model
		self.learning_rate = 0.001
		checkpoint = False
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		print(self.device)
		layers = []
		for n in range(len(self.layers_dims) - 2):
			layers.append(nn.Linear(self.layers_dims[n], self.layers_dims[n+1]))
			layers.append(nn.ReLU())
		layers.append(nn.Linear(self.layers_dims[n+1], self.layers_dims[n+2]))
		layers.append(nn.Softmax(dim=1))
		self.model = nn.Sequential(*layers)
		optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
		available = self.check_for_params()
		if available:
			if input("Start from saved?") == "Y":
				# Restore variables from disk.
				self.model.load_state_dict(torch.load("Torch/model_weights.pth"))
				print("Session restored!")
				if input("Make checkpoint from saved?") == "Y":
					name = input("Checkpoint name:")
					self.save_obj(name)
			else:
				if input("Start from checkpoint?") == "Y":
					while checkpoint == False and name != 's':
						name = input("Checkpoint name [s = skip]:")
						if name != "s":
							self.load_checkpoint(name)
						else:
							print("Using initialized parameters.")
				else:
					print("Using initialized parameters.")
		else:
			print("Using initialized parameters.")
		if input("Dice roll or maximize?") == "M":
			self.move_type = "M"
		else:
			self.move_type = "R"
#		if input("Gradient check?") == "Y":
#			self.plot_activations()
#			self.check_gradients()
#			input("Paused.")
		self.initialize_training_batch()
		self.legal_means = []
		self.illegal_means = []
		self.trainings = 0
		self.apply(self._init_weights)
		self.activations = {}  # Dictionary to store activations
		self.hook_handles = []  # List to store hook handles
		self.model = self.model.to(self.device)


	def _init_weights(self, module):
		if isinstance(module, nn.Linear):
			torch.nn.init.xavier_uniform_(module.weight)
			if module.bias is not None:
				torch.nn.init.zeros_(module.bias)

	def add_hooks(self):
		# Register hooks and store handles
		for name, layer in self.named_modules():
			handle = layer.register_forward_hook(self._save_activation(name))
			self.hook_handles.append(handle)

	def remove_hooks(self):
		# Remove all hooks using stored handles
		for handle in self.hook_handles:
			handle.remove()
		self.hook_handles.clear()  # Clear the list after removal

	def forward_pass(self, x):
		x = self.convert(x)
		x = x.to(self.device)
		with torch.no_grad():
			x = self(x)
		x = self.deconvert(x)
		return x

	def forward(self, x, y=None):
		x = self.model(x)
		return x

	def generate_move(self, AL): # generate a move from a probabilities vector
		choice = np.squeeze(np.random.choice(96, 1, p=AL.flatten()/np.sum(AL.flatten()))) # roll the dice and p b
		one_hot_move = np.eye(96, dtype = 'int')[choice] #generate one-hot version
		piece_number = int(np.argmax(one_hot_move)/8) # get the piece number that the move applies to
		move = one_hot_move[(8 * piece_number):((8 * piece_number) + 8)] # generate the move for that piece

		return one_hot_move, piece_number, move

	def train_model(self, y, x, weights, illegal_masks):
		"""
		y: parallel set of unit-normalized legal move vectors to calculate cost.
		x: parallel set of input vectors.
		weights: parallel set of number of attempts at a move to weight the cost.
		illegal_masks: parallel set of non-normalized legal move vectors
		"""
		params = {}
		self.add_hooks()
		self.batch_num += 1
		x = self.convert(x)
		x = x.to(self.device)
		y = self.convert(y)
		y = y.to(self.device)
		weights = self.convert(weights)
		weights = weights.to(self.device)
		x = self(x)
		cost = ((y - x) ** 2) * weights
		cost = cost.mean()
		self.optimizer.zero_grad(set_to_none=True)
		cost.backward()
		self.optimizer.step()
		self.relu_activations = self.get_relu_activations()
		L = len(self.relu_activations)
		mins = []
		maxes = []
		for l in reversed(range(L)):
			mins.append(np.amin(self.relu_activations[l]))
			maxes.append(np.amax(self.relu_activations[l]))
		self.trainings += 1
		self.plot_activations()
		legal_mean, illegal_mean = self.get_means(x, illegal_masks) # use if only training on legal moves, not all moves

		self.legal_means.append(legal_mean)
		self.illegal_means.append(illegal_mean)

		params["illegal_means"] = self.illegal_means
		params["legal_means"] = self.legal_means
		params["trainings"] = self.trainings
		params["mins"] = mins
		params["maxes"] = maxes

		self.initialize_training_batch()

		return cost, params

	def _save_activation(self, name):
		# Hook function to save activations
		def hook(module, input, output):
			layer_type = type(module).__name__
			activation_name = f"{name}_{layer_type}"
			self.activations[activation_name] = output.detach().cpu()
			self.relu_activations = self.get_relu_activations()
		return hook

	def get_relu_activations(self):
		return [activation.numpy() for name, activation in self.activations.items() if '_ReLU' in name]

	def convert(self, x):
		x = torch.from_numpy(np.array(x, dtype=np.float32)).transpose(0,1)
		return x

	def plot_activations(self):
		plt.figure(2)
		plt.ion()
		plt.show()
		L = len(self.relu_activations)
		for l in reversed(range(L)):
			sp = 100 + (10 * (L)) + (l + 1)
			plt.subplot(sp)
			plt.hist(self.relu_activations[l])
		plt.draw()
		plt.pause(0.001)
#		plt.figure(3)
#		plt.ion()
#		plt.show()
#		plt.hist(self.AL)
		plt.figure(4)
		plt.ion()
		plt.show()
		plt.hist(self.num_attempts_batch, bins=np.logspace(np.log10(1), np.log10(300), num=50))
		plt.gca().set_xscale("log")

	def deconvert(self, x):
		x = x.transpose(0,1).detach().cpu().numpy().astype(np.float64)
		return x

	def initialize_training_batch(self):
		self.moves = []
		self.illegal_masks = []
		self.probabilities_batch = []
		self.X_batch = []
		self.attempts = [] # list with attempted (illegal) moves
		#self.attempts_illegal_masks = [] # parallel list with illegal masks for those attempts
		#self.attempts_probabilities = [] # parallel list with probability vectors for those attempts
		#self.attempts_X_batch = [] # parallel list with board inputs for those attempts
		self.num_attempts = 0 # total number of attempts to get to a legal move
		self.num_attempts_batch = []

	def save_parameters(self):
		self.save_obj("model_weights")

	def save_obj(self, name):
		torch.save(self.model.state_dict(), 'Torch/' + name + '.pth')


	def check_for_params(self):
		available = os.path.isfile("Torch/model_weights.pth")

		return available

	def load_checkpoint(self, name):
		try:
			self.model.load_state_dict(torch.load('Torch/' + name + '.pth'))
		except (OSError, IOError) as e:
			checkpoint = False
			print("Can't find that checkpoint...")
		if checkpoint != False:
			print("Checkpoint " + name + ".pth loaded!")
		return checkpoint

	def get_input_vector(self, board, color, jump_piece_number):
		v = board.get_piece_arrays(color)
		# if color == 'Red':
			# v = board.red_home_view().flatten() # get the board state 
		# else:
			# v = board.black_home_view().flatten()
		# v = np.append(v, board.get_piece_vector(color))
		if jump_piece_number != None:
			j_vector = np.eye(12)[jump_piece_number]
			jump = np.array([1])
		else:
			j_vector = np.zeros((12))
			jump = np.array([0])
		v = np.append(v, j_vector)
		v = np.append(v, jump)	
		return v


