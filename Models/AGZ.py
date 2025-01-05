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

class AGZ(nn.Module):

	def __init__(self, name, identifier):
		super().__init__()
		self.name = name
		self.identifier = identifier
		self.set_seed(42)
		self.batch_num = 0
		self.learning_rate = 0.001
		dropout_prob = 0.2
		checkpoint = False
		self.from_saved = False
		self.temperature = 1
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		print(self.device)
		
		'''Convolutional layers'''
		self.conv_3x3x128_0 = nn.Conv2d(4, 128, kernel_size=3, stride=1, padding=1)
		self.projection_4_128 = nn.Conv2d(4, 128, kernel_size=1, stride=1, padding=0)
		self.conv_5x5x128_1 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2)
		self.conv_5x5x128_2 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2	)
		self.conv_5x5x128_3 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2)
		self.layer_norm_3x3x128_0 = nn.BatchNorm2d([128])
		self.layer_norm_5x5x128_1 = nn.BatchNorm2d([128])
		self.layer_norm_5x5x128_2 = nn.BatchNorm2d([128])
		self.layer_norm_5x5x128_3 = nn.BatchNorm2d([128])
		self.projection_layer_norm_3x3x128 = nn.BatchNorm2d([128])
		self.residual_weight_0 = nn.Parameter(torch.tensor(0.1))
		self.residual_weight_1 = nn.Parameter(torch.tensor(0.1))
		self.residual_weight_2 = nn.Parameter(torch.tensor(0.1))
		self.residual_weight_3 = nn.Parameter(torch.tensor(0.1))

		'''Policy network'''
		self.policy_layers_dims = [4505, 2048, 1024, 512, 256, 96] #  6-layer model
		policy_layers = []
		for n in range(len(self.policy_layers_dims) - 2):
			policy_layers.append(nn.Linear(self.policy_layers_dims[n], self.policy_layers_dims[n+1]))
			policy_layers.append(nn.LayerNorm(self.policy_layers_dims[n+1]))
			policy_layers.append(nn.ReLU())
			policy_layers.append(nn.Dropout(dropout_prob))
		policy_layers.append(nn.Linear(self.policy_layers_dims[n+1], self.policy_layers_dims[n+2]))
		self.policy_output = nn.Sequential(*policy_layers)

		'''Value network'''
		self.value_layers_dims = [4096, 2048, 1024, 512, 256, 96, 1] #  7-layer model
		value_layers = []
		for n in range(len(self.value_layers_dims) - 2):
			value_layers.append(nn.Linear(self.value_layers_dims[n], self.value_layers_dims[n+1]))
			value_layers.append(nn.LayerNorm(self.value_layers_dims[n+1]))
			value_layers.append(nn.ReLU())
			value_layers.append(nn.Dropout(dropout_prob))
		value_layers.append(nn.Linear(self.value_layers_dims[n+1], self.value_layers_dims[n+2]))
		self.value_output = nn.Sequential(*value_layers)

		
		self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
		self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=1)
		available = self.check_for_params()
		if available:
			if input("Start from saved?") == "Y":
				# Restore variables from disk.
				self.load_checkpoint(self.name, self.identifier)
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
		self.to(self.device)


	def forward(self, board, pieces, mask, y=None):
		projection_128_0 = self.projection_layer_norm_3x3x128(self.projection_4_128(board))
		out_3x3x128_0 = (F.relu(self.layer_norm_3x3x128_0(self.conv_3x3x128_0(board))) * self.residual_weight_0) + projection_128_0# Shape: (batch_size, 16, 8, 4)
		out_5x5x128_1 = (F.relu(self.layer_norm_5x5x128_1(self.conv_5x5x128_1(out_3x3x128_0))) * self.residual_weight_1) + projection_128_0 # Shape: (batch_size, 32, 8, 4)
		out_5x5x128_2 = (F.relu(self.layer_norm_5x5x128_2(self.conv_5x5x128_2(out_5x5x128_1))) * self.residual_weight_2) + projection_128_0 # Shape: (batch_size, 64, 8, 4)
		out_5x5x128_3 = (F.relu(self.layer_norm_5x5x128_3(self.conv_5x5x128_3(out_5x5x128_2))) * self.residual_weight_3) + projection_128_0 # Shape: (batch_size, 128, 8, 4)

		
		# Flatten convolutional output for the fully connected layers
		out_conv_flat = out_5x5x128_3.view(out_5x5x128_3.size(0), -1)  # Shape: (batch_size, 32 * 8 * 4)
		
		# Concatenate with the direct input vector
		combined_input = torch.cat((out_conv_flat, pieces), dim=1)  # Shape: (batch_size, 32*8*4 + input_size)

		'''Feed into policy network'''
		policy_logits = self.policy_output(combined_input) / self.temperature
		masked_policy_logits = policy_logits.masked_fill(mask == 0, float('-inf'))  # Mask illegal moves
		policy_output = F.softmax(masked_policy_logits, dim=1)  # Apply softmax only on legal moves

		'''Feed into value network'''
		value_logits = self.value_output(out_conv_flat)
		value_output = torch.tanh(value_logits)

		return policy_output, value_output

	def update_learning_rate(self, m):
		self.learning_rate *= m
		for param_group in self.optimizer.param_groups:
			param_group['lr'] = self.learning_rate

	def new_learning_rate(self, m):
		self.learning_rate = m
		for param_group in self.optimizer.param_groups:
			param_group['lr'] = self.learning_rate


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

	def forward_pass(self, board, pieces, mask):
		board = torch.from_numpy(np.array(board, dtype=np.float32))
		board = board.to(self.device)
		pieces = self.convert(pieces)
		pieces = pieces.to(self.device)
		mask = self.convert(mask)
		mask = mask.to(self.device)
		self.eval()
		with torch.no_grad():
			policy_output, value_output = self(board, pieces, mask)
			
		policy_output = self.deconvert(policy_output)
		value_output = self.deconvert(value_output)
		return policy_output, value_output

	def generate_move(self, AL): # generate a move from a probabilities vector
		choice = np.squeeze(np.random.choice(96, 1, p=AL.flatten()/np.sum(AL.flatten()))) # roll the dice and p b
		one_hot_move = np.eye(96, dtype = 'int')[choice] #generate one-hot version
		piece_number = int(np.argmax(one_hot_move)/8) # get the piece number that the move applies to
		move = one_hot_move[(8 * piece_number):((8 * piece_number) + 8)] # generate the move for that piece

		return one_hot_move, piece_number, move

	def train_model(self, Y, V, board, pieces, mask):
		"""
		y: parallel set of unit-normalized legal move vectors to calculate cost.
		x: parallel set of input vectors.
		weights: parallel set of number of attempts at a move to weight the cost.
		mask: parallel set of non-normalized legal move vectors
		"""
		'''Timing'''
		total_start = time.time()
		train_init_start = time.time()

		'''Initialize variables'''
		epsilon = 1e-8
		training_stats = []
		params = {}

		'''Create filtered arrays'''

		'''Setup'''
		self.add_hooks()
		self.train()

		'''Convert to tensors with appropriate dimensions'''
		pieces = self.convert(pieces)
		pieces = pieces.to(self.device)
		board = torch.from_numpy(np.array(board, dtype=np.float32))
		board = board.to(self.device)
		Y = self.convert(Y)
		Y = Y.to(self.device)
		V = self.convert(V)
		V = V.to(self.device)
		mask_t = self.convert(mask)
		mask_t = mask_t.to(self.device)

		'''Timing'''
		train_init_end = time.time()
		train_init_time = train_init_end - train_init_start
		training_stats.append(train_init_time)

		'''Timing'''
		forward_prop_start = time.time()


		torch.cuda.reset_peak_memory_stats()
		policy_output, value_output = self(board, pieces, mask_t)


		forward_prop_end = time.time()
		forward_prop_time = forward_prop_end - forward_prop_start
		training_stats.append(forward_prop_time)
		backward_prop_start = time.time()
		policy_cost = -torch.sum(Y * torch.log(policy_output + epsilon), dim=-1).mean()
		value_cost = F.mse_loss(value_output, V)
		cost = policy_cost + value_cost
		cost.backward()
		max_mem = torch.cuda.max_memory_allocated()
		self.optimizer.step()
		backward_prop_end = time.time()
		backward_prop_time = backward_prop_end - backward_prop_start
		training_stats.append(backward_prop_time)

		'''Training done: update batch count.'''
		self.batch_num += 1
		bookkeeping_start = time.time()

		self.save_weights()
		mins = []
		maxes = []
		weights = [self.residual_weight_0, self.residual_weight_1, self.residual_weight_2, self.residual_weight_3]
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
		params["weights"] = weights

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
			if isinstance(output, tuple):
				# Convert each tensor in the tuple to CPU and detach
				self.activations[activation_name] = [out.detach().cpu() for out in output]
			else:
				# Output is a single tensor
				self.activations[activation_name] = output.detach().cpu()
		return hook

	def save_weights(self):
		self.weights = []
		for layer in self.modules():
			# Check if the layer is an instance of nn.Linear
			if isinstance(layer, nn.Linear):
				self.weights.append(layer.weight.detach().cpu().numpy())

	def convert(self, x):
		#try:
		x = torch.from_numpy(np.array(x, dtype=np.float32)).transpose(0,1)
		#except:
			#pdb.set_trace()
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

	def save_parameters(self, identifier):
		self.save_obj(self.name, identifier)

	def save_obj(self, name, identifier):
		checkpoint = {
			'model_state_dict': self.state_dict(),
			'optimizer_state_dict': self.optimizer.state_dict(),
			'scheduler_state_dict': self.scheduler.state_dict(),  # Save scheduler state
		}

		# Save the checkpoint
		torch.save(checkpoint, f"Torch_{identifier}/{name}.pth")
		print(f"Model parameters saved to Torch_{identifier}/{name}.pth")


	def check_for_params(self):
		available = os.path.isfile(f"Torch_{self.identifier}/{self.name}.pth")

		return available

	def load_checkpoint(self, name, identifier):
		checkpoint = torch.load(f"Torch_{identifier}/{name}.pth", map_location=torch.device(self.device))
		self.load_state_dict(checkpoint['model_state_dict'])
		self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
		for state in self.optimizer.state.values():
			for k, v in state.items():
				if isinstance(v, torch.Tensor):
					state[k] = v.to(self.device)

	def get_input_vector(self, board, color, jump_piece_number):
		if color == 'Red':
			v = board.red_home_view() # get the board state 
		else:
			v = board.black_home_view()
		pieces = board.get_piece_vector(color)
		if jump_piece_number != None:
			j_vector = np.eye(12)[jump_piece_number]
			jump = np.array([1])
		else:
			j_vector = np.zeros((12))
			jump = np.array([0])
		pieces = np.append(pieces, j_vector)
		pieces = np.append(pieces, jump)	
		return v, pieces


