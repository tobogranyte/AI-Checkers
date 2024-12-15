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

class PTC2(nn.Module):

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
		self.layers_dims = [4505, 2048, 1024, 512, 256, 96] #  6-layer model
		layers = []
		self.conv_3x3 = nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1)
		self.conv_5x5 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
		self.conv_7x7 = nn.Conv2d(32, 64, kernel_size=7, stride=1, padding=3)
		self.conv_9x9 = nn.Conv2d(64, 128, kernel_size=9, stride=1, padding=4)
		self.layer_norm_3x3 = nn.LayerNorm([16, 8, 4])
		self.layer_norm_5x5 = nn.LayerNorm([32, 8, 4])
		self.layer_norm_7x7 = nn.LayerNorm([64, 8, 4])
		self.layer_norm_9x9 = nn.LayerNorm([128, 8, 4])
		for n in range(len(self.layers_dims) - 2):
			layers.append(nn.Linear(self.layers_dims[n], self.layers_dims[n+1]))
			layers.append(nn.LayerNorm(self.layers_dims[n+1]))
			layers.append(nn.ReLU())
			layers.append(nn.Dropout(dropout_prob))
		layers.append(nn.Linear(self.layers_dims[n+1], self.layers_dims[n+2]))
		self.feed_forward = nn.Sequential(*layers)
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
		# Apply convolutions and ReLU
		out_3x3 = F.relu(self.layer_norm_3x3(self.conv_3x3(board)))  # Shape: (batch_size, 16, 8, 4)
		out_5x5 = F.relu(self.layer_norm_5x5(self.conv_5x5(out_3x3)))  # Shape: (batch_size, 32, 8, 4)
		out_7x7 = F.relu(self.layer_norm_7x7(self.conv_7x7(out_5x5)))  # Shape: (batch_size, 64, 8, 4)
		out_9x9 = F.relu(self.layer_norm_9x9(self.conv_9x9(out_7x7)))  # Shape: (batch_size, 128, 8, 4)

		
		# Flatten convolutional output for the fully connected layers
		out_conv_flat = out_9x9.view(out_9x9.size(0), -1)  # Shape: (batch_size, 32 * 8 * 4)
		
		# Concatenate with the direct input vector
		combined_input = torch.cat((out_conv_flat, pieces), dim=1)  # Shape: (batch_size, 32*8*4 + input_size)

		logits = self.feed_forward(combined_input) / self.temperature
		masked_logits = logits.masked_fill(mask == 0, float('-inf'))  # Mask illegal moves
		masked_probs = F.softmax(masked_logits, dim=1)  # Apply softmax only on legal moves
		if torch.isnan(masked_probs).any():
			pdb.set_trace()

		return masked_probs

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
			x = self(board, pieces, mask)
			
		x = self.deconvert(x)
		return x

	def generate_move(self, AL): # generate a move from a probabilities vector
		choice = np.squeeze(np.random.choice(96, 1, p=AL.flatten()/np.sum(AL.flatten()))) # roll the dice and p b
		one_hot_move = np.eye(96, dtype = 'int')[choice] #generate one-hot version
		piece_number = int(np.argmax(one_hot_move)/8) # get the piece number that the move applies to
		move = one_hot_move[(8 * piece_number):((8 * piece_number) + 8)] # generate the move for that piece

		return one_hot_move, piece_number, move

	def train_model(self, Y, board, pieces, mask, reward):
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
		reward_non_zero_mask = np.squeeze(reward, axis=0) != 0
		reward = reward[:, reward_non_zero_mask]
		Y = Y[:, reward_non_zero_mask]
		pieces = pieces[:, reward_non_zero_mask]
		mask = mask[:, reward_non_zero_mask]
		board = board[reward_non_zero_mask, :, :, :]

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
		mask_t = self.convert(mask)
		mask_t = mask_t.to(self.device)
		reward = self.convert(reward)
		reward = reward.to(self.device)

		'''Timing'''
		train_init_end = time.time()
		train_init_time = train_init_end - train_init_start
		training_stats.append(train_init_time)

		'''Timing'''
		forward_prop_start = time.time()


		torch.cuda.reset_peak_memory_stats()
		X = self(board, pieces, mask_t)
		forward_prop_end = time.time()
		forward_prop_time = forward_prop_end - forward_prop_start
		training_stats.append(forward_prop_time)
		backward_prop_start = time.time()
		log_prob = torch.log(X + epsilon)
		reward_win = torch.where(reward > 0, reward, 0)
		reward_loss = torch.where(reward < 0, reward, 0)
		log_prob_move = log_prob[Y == 1].unsqueeze(1)
		cost_win = - reward_win * log_prob_move
		cost_loss = - reward_loss * log_prob_move
		cost_win = cost_win.sum()
		cost_loss = cost_loss.sum()
		cost = cost_win.sum() + cost_loss.sum()
		with open(f"Torch_{self.identifier}/cost.csv", mode="a", newline="") as file:
			writer = csv.writer(file)
			writer.writerow([cost.item(), cost_win.item(), cost_loss.item(), reward_win.sum().item(), reward_loss.sum().item(), (reward == 0).sum().item()])  # Write the list as a new row		self.optimizer.zero_grad(set_to_none=True)
		cost.backward()
		max_mem = torch.cuda.max_memory_allocated()
		mem_stats = [self.trainings, max_mem, reward.shape]
		print(f"Peak memory used: {max_mem} bytes")
		with open(f"Torch_{self.identifier}/max_mem.csv", mode="a", newline="") as file:
			writer = csv.writer(file)
			writer.writerow(mem_stats)  # Write the list as a new row
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

		return cost.detach().cpu().numpy().astype(np.float64), cost_win.detach().cpu().numpy().astype(np.float64), cost_loss.detach().cpu().numpy().astype(np.float64), params


	def _save_activation(self, name):
		# Hook function to save activations
		def hook(module, input, output):
			layer_type = type(module).__name__
			activation_name = f"{name}_{layer_type}"
			self.activations[activation_name] = output.detach().cpu()
		return hook

	def save_weights(self):
		self.weights = []
		for layer in self.modules():
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


