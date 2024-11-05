import torch
import torch.nn as nn
import os

class Network(nn.Module):

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

	def forward(self, x, nograd = True):
		x = self.convert(x)
		x = x.to(self.device)
		if nograd:
			with torch.no_grad():
				x = self.model(x)
				x = self.deconvert(x)
				return x
		else:
			return self.model(x)

	def _save_activation(self, name):
		# Hook function to save activations
		def hook(model, input, output):
			self.activations[name] = output.detach().cpu()
		return hook

	def convert(self, x):
		x = torch.from_numpy(np.array(x, dtype=np.float32)).transpose(0,1)
		return x

	def deconvert(self, x):
		if x.requires_grad:
			x = x.transpose(0,1).detach().numpy().astype(np.float64)
		else:
			x = x.transpose(0,1).numpy().astype(np.float64)
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
			self.model.load_state_dict(torch.load('Torch/' + name + '.pth"))
		except (OSError, IOError) as e:
			checkpoint = False
			print("Can't find that checkpoint...")
		if checkpoint != False:
			print("Checkpoint " + name + ".pth loaded!")
		return checkpoint
