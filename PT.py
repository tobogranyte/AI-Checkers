import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import random
import numpy as np
import time
import csv
import pdb

class PT(nn.Module):

	def __init__(self, name):
		super().__init__()
		self.name = name
		self.set_seed(42)
		self.batch_num = 0
		self.layers_dims = [397, 1024, 512, 256, 128, 96] #  5-layer model
		self.learning_rate = 0.0005
		checkpoint = False
		self.from_saved = False
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		print(self.device)
		layers = []
		for n in range(len(self.layers_dims) - 2):
			layers.append(nn.Linear(self.layers_dims[n], self.layers_dims[n+1]))
			layers.append(nn.LayerNorm(self.layers_dims[n+1]))
			layers.append(nn.ReLU())
		layers.append(nn.Linear(self.layers_dims[n+1], self.layers_dims[n+2]))
		self.model = nn.Sequential(*layers)
		self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
		self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=1	)
		available = self.check_for_params()
		if available:
			if input("Start from saved?") == "Y":
				# Restore variables from disk.
				self.load_checkpoint(self.name)

				print("Session restored!")
				if input("Make checkpoint from saved?") == "Y":
					name = input("Checkpoint name:")
					self.save_obj(name)
				self.from_saved = True
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
					self.apply(self._init_weights)
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
		self.legal_means = []
		self.illegal_means = []
		self.trainings = 0
		self.activations = {}  # Dictionary to store activations
		self.hook_handles = []  # List to store hook handles
		self.model = self.model.to(self.device)

	def set_seed(self, seed=42):
		random.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)
		if torch.cuda.is_available():
			torch.cuda.manual_seed_all(seed)

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

	def forward_pass(self, x, mask):
		x = self.convert(x)
		x = x.to(self.device)
		mask = self.convert(mask)
		mask = mask.to(self.device)
		self.model.eval()
		with torch.no_grad():
			x = self(x, mask)
		x = self.deconvert(x)
		return x

	def forward(self, x, mask, y=None):
		logits = self.model(x)
		masked_logits = logits.masked_fill(mask == 0, float('-inf'))  # Mask illegal moves
		masked_probs = F.softmax(masked_logits, dim=1)  # Apply softmax only on legal moves
		if torch.isnan(masked_probs).any():
			pdb.set_trace()

		return masked_probs

	def generate_move(self, AL): # generate a move from a probabilities vector
		choice = np.squeeze(np.random.choice(96, 1, p=AL.flatten()/np.sum(AL.flatten()))) # roll the dice and p b
		one_hot_move = np.eye(96, dtype = 'int')[choice] #generate one-hot version
		piece_number = int(np.argmax(one_hot_move)/8) # get the piece number that the move applies to
		move = one_hot_move[(8 * piece_number):((8 * piece_number) + 8)] # generate the move for that piece

		return one_hot_move, piece_number, move

	def train_model(self, Y, X, mask, reward):
		"""
		y: parallel set of unit-normalized legal move vectors to calculate cost.
		x: parallel set of input vectors.
		weights: parallel set of number of attempts at a move to weight the cost.
		mask: parallel set of non-normalized legal move vectors
		"""
		epsilon = 1e-8
		training_stats = []
		total_start = time.time()
		train_init_start = time.time()
		self.initialize_training_batch()
		self.model.train()
		params = {}
		self.add_hooks()
		self.batch_num += 1
		X = self.convert(X)
		X = X.to(self.device)
		Y = self.convert(Y)
		Y = Y.to(self.device)
		mask_t = self.convert(mask)
		mask_t = mask_t.to(self.device)
		reward = self.convert(reward)
		reward = reward.to(self.device)
		print(mask_t[0])
		train_init_end = time.time()
		train_init_time = train_init_end - train_init_start
		training_stats.append(train_init_time)
		forward_prop_start = time.time()
		print(X.size(0))
		X = self(X, mask_t)
		forward_prop_end = time.time()
		forward_prop_time = forward_prop_end - forward_prop_start
		training_stats.append(forward_prop_time)
		backward_prop_start = time.time()
		log_prob = torch.log(X + epsilon)
		cost = - reward * log_prob[Y == 1]
		cost = cost.sum()
		self.optimizer.zero_grad(set_to_none=True)
		cost.backward()
		self.optimizer.step()
		backward_prop_end = time.time()
		backward_prop_time = backward_prop_end - backward_prop_start
		training_stats.append(backward_prop_time)
		bookkeeping_start = time.time()

		self.save_weights()
		mins = []
		maxes = []
		L = len(self.weights)
		for l in reversed(range(L)):
			mins.append(np.amin(self.weights[l]))
			maxes.append(np.amax(self.weights[l]))
		self.trainings += 1
		self.scheduler.step()
		print("Training batch {}: Current learning rate: {:.6f}".format(self.trainings, self.optimizer.param_groups[0]['lr']))
		# self.plot_activations()


		params["trainings"] = self.trainings
		params["mins"] = mins
		params["maxes"] = maxes

		print(torch.cuda.memory_allocated())
		bookkeeping_end = time.time()
		bookkeeping_time = bookkeeping_end - bookkeeping_start
		training_stats.append(bookkeeping_time)
		total_end = time.time()
		total_time = total_end - total_start
		training_stats.append(total_time)
		self.remove_hooks()
		with open("training_stats.csv", mode="a", newline="") as file:
			writer = csv.writer(file)
			writer.writerow(training_stats)  # Write the list as a new row

		return cost.detach().cpu().numpy().astype(np.float64), params


	def _save_activation(self, name):
		# Hook function to save activations
		def hook(module, input, output):
			layer_type = type(module).__name__
			activation_name = f"{name}_{layer_type}"
			self.activations[activation_name] = output.detach().cpu()
		return hook

	def save_weights(self):
		self.weights = []
		for layer in self.model:
			# Check if the layer is an instance of nn.Linear
			if isinstance(layer, nn.Linear):
				self.weights.append(layer.weight.detach().cpu().numpy())

	def convert(self, x):
		x = torch.from_numpy(np.array(x, dtype=np.float32)).transpose(0,1)
		return x

	def plot_activations(self):
		plt.figure(2)
		plt.ion()
		plt.show()
		L = len(self.weights)
		print(L)
		for l in reversed(range(L)):
			print(l)
			sp = 100 + (10 * (L)) + (l + 1)
			print(sp)
			plt.subplot(sp)
			plt.hist(self.weights[l])
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
		self.save_obj(self.name)

	def save_obj(self, name):
		checkpoint = {
			'model_state_dict': self.model.state_dict(),
			'optimizer_state_dict': self.optimizer.state_dict(),
			'scheduler_state_dict': self.scheduler.state_dict(),  # Save scheduler state
		}

		# Save the checkpoint
		torch.save(checkpoint, 'Torch/' + name + '.pth')
		print(f"Model parameters saved to Torch/{self.name}.pth")


	def check_for_params(self):
		available = os.path.isfile(f"Torch/{self.name}.pth")

		return available

	def load_checkpoint(self, name):
		checkpoint = torch.load(f"Torch/{name}.pth")
		self.model.load_state_dict(checkpoint['model_state_dict'])
		self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
		for state in self.optimizer.state.values():
			for k, v in state.items():
				if isinstance(v, torch.Tensor):
					state[k] = v.to(self.device)

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


