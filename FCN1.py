import numpy as np
from utility_functions import sigmoid, sigmoid_backward, relu, relu_backward, softmax, softmax_backward, dictionary_to_vector, vector_to_dictionary, gradients_to_vector
import pickle
import matplotlib.pyplot as plt
import os

class FCN1:

	def __init__(self):
		self.layers_dims = [537, 1024, 524, 262, 131, 48] #  6-layer model
		params, available = self.check_for_params()
		checkpoint = False
		name = ''
		self.cuda = ''

		try:
			cuda = os.environ['CUDA_PRESENT']
		except:
			cuda = ''

		if cuda == 'Yes':
			self.cuda = True
		else:
			self.cuda = False

		if self.cuda:
			print("Got CUDA!")
		else:
			print("No CUDA... *sniff*")

		if available:
			if input("Start from saved?") == "Y":
				self.parameters = params
				if input("Make checkpoint from saved?") == "Y":
					name = input("Checkpoint name:")
					self.save_obj(self.parameters, name)
			else:
				if input("Start from checkpoint?") == "Y":
					while checkpoint == False and name != 's':
						name = input("Checkpoint name [s = skip]:")
						if name != "s":
							checkpoint = self.load_checkpoint(name)
							self.parameters = checkpoint
						else:
							print("Initializing parameters...")
							self.parameters = self.initialize_parameters_deep(self.layers_dims)
							print("Parameters initialized!")
				else:
					print("Initializing parameters...")
					self.parameters = self.initialize_parameters_deep(self.layers_dims)
					print("Parameters initialized!")
		else:
			print("Initializing parameters...")
			self.parameters = self.initialize_parameters_deep(self.layers_dims)
			print("Parameters initialized!")
		if input("Gradient check?") == "Y":
			self.plot_activations()
			self.check_gradients()
			input("Paused.")
		self.initialize_training_batch()
		self.legal_means = []
		self.illegal_means = []
		self.trainings = 0

	def plot_activations(self):
		plt.figure(2)
		plt.ion()
		plt.show()
		g_norm = np.random.normal(scale = 1, size = self.layers_dims[0]).reshape(self.layers_dims[0],1)
		self.AL, caches = self.L_model_forward(g_norm, self.parameters)
		Y = np.zeros((48,1))
		L = len(caches)
		for l in reversed(range(L-1)):
			sp = 100 + (10 * (L - 1)) + (l + 1)
			plt.subplot(sp)
			linear_cache, activation_cache = caches[l]
			plt.hist(activation_cache)
		plt.draw()
		plt.pause(0.001)
		plt.figure(3)
		plt.ion()
		plt.show()
		plt.hist(self.AL)

	def check_gradients(self):
		g_norm = np.random.normal(scale = 1, size = self.layers_dims[0]).reshape(self.layers_dims[0],1)
		self.AL, caches = self.L_model_forward(g_norm, self.parameters)
		Y = np.zeros((48,1))
		grads = self.L_model_backward(self.AL, Y, caches)
		difference = self.gradient_check_n(self.parameters, grads, g_norm, Y)
		input("Paused...")

	def initialize_training_batch(self):
		self.moves = []
		self.illegal_masks = []
		self.probabilities = []
		self.X_batch = []
		self.attempts = [] # list with attempted (illegal) moves
		self.attempts_illegal_masks = [] # parallel list with illegal masks for those attempts
		self.attempts_probabilities = [] # parallel list with probability vectors for those attempts
		self.attempts_X_batch = [] # parallel list with board inputs for those attempts
		self.num_attempts = 0 # total number of attempts to get to a legal move
		self.num_attempts_batch = []

	def save_parameters(self):
		self.save_obj(self.parameters, 'parameters_temp')

	def save_obj(self, obj, name ):
		with open('FCN1/'+ name + '.pkl', 'wb') as f:
			pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
		print("File " + name + ".pkl saved!")

	def load_obj(self, name):
		with open('FCN1/' + name + '.pkl', 'rb') as f:
			return pickle.load(f)

	def load_checkpoint(self, name):
		try:
			checkpoint = pickle.load(open("FCN1/" + name + ".pkl", "rb"))
		except (OSError, IOError) as e:
			checkpoint = False
			print("Can't find that checkpoint...")
		if checkpoint != False:
			print("Checkpoint " + name + ".pkl loaded!")
		return checkpoint

	def check_for_params(self):
		try:
			params = pickle.load(open("FCN1/parameters_temp.pkl", "rb"))
			available = True
		except (OSError, IOError) as e:
			params = 0
			available = False

		return params, available

	def initialize_parameters_deep(self, layer_dims):
	    """
	    Arguments:
	    layer_dims -- python array (list) containing the dimensions of each layer in our network
	    
	    Returns:
	    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
	                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
	                    bl -- bias vector of shape (layer_dims[l], 1)
	    """
	    
	    parameters = {}
	    L = len(layer_dims)            # number of layers in the network

	    for l in range(1, L):
	        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1]/2)
	        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1)) 
	        
	        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
	        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

	        
	    return parameters

	def move(self, board, color, jump_piece_number = None, jump_rule = True, illegal = False):
		self.num_attempts += 1
		if illegal == False: # everything within this sets up the stuff that won't change until a legal move is executed
			self.board_legal_moves = board.legal_moves(color = color, jump_piece_number = jump_piece_number, jump_rule = jump_rule) # get legal moves for current board position
			self.illegal_mask = np.zeros((48)) # create a holder for the illegal mask (starting filled with zeros)
			self.illegal_mask[self.board_legal_moves != 0] = 1 # ones for anything that's legal
			self.illegal_mask = self.illegal_mask.reshape(self.illegal_mask.size, -1) # make into a column vector
			self.X = self.get_input_vector(board, self.board_legal_moves, color, jump_piece_number = jump_piece_number) # create the input vector
			self.AL, caches = self.L_model_forward(self.X, self.parameters) # run forward prop
		move = np.squeeze(np.random.choice(48, 1, p=self.AL.flatten()/np.sum(self.AL.flatten()))) # roll the dice and pick a move from the output probs
		one_hot_move = np.eye(48, dtype = 'int')[move] # turn it into a one-hot vector
		new_move = one_hot_move.reshape(one_hot_move.size, -1) # make it into a column vector

		"""
		This block adds a training example (moves, probs, inputs and masks) for every attempt, not just successful attempts.
		This is not what you'd want to use for training anything but learning to make legal moves.
		new_move: this is the move that was attempted
		"""
#		self.attempts.append(new_move) # append the attempted move to the list of attempts
#		self.attempts_probabilities.append(self.AL) # append the probabilities to the list of probs . It will append the same thing for every attempt (lots of repeats).
#		self.attempts_X_batch.append(self.X) # append the input vector. It will append the same thing for every attempt (lots of repeats).
#		self.attempts_illegal_masks.append(self.illegal_mask) # append the illegal mask. It will append the same thing for every attempt (lots of repeats).

		"""
		This block creates the same basic set of data, but only for each successful move. Not necessary if the above block is being used
		"""

		if illegal == False:
			self.moves.append(new_move)
			self.probabilities.append(self.AL)
			self.X_batch.append(self.X)
			self.illegal_masks.append(self.illegal_mask)
			self.num_attempts_batch.append(self.num_attempts)
			self.num_attempts = 0
		else:
			self.moves[-1] = new_move

		return one_hot_move, self.board_legal_moves

	def train(self):
		params = {}
		learning_rate = 0.075

		Y = self.make_Y(np.hstack(self.probabilities), np.hstack(self.illegal_masks)) # use if only training on legal moves, not all moves
#		Y = self.make_Y(np.hstack(self.attempts_probabilities), np.hstack(self.attempts_illegal_masks)) # use if training on all illegal attempts

		weights = np.hstack(self.num_attempts_batch)

		for i in range(0, 1):

			AL, caches = self.L_model_forward(np.hstack(self.X_batch), self.parameters) # use if only training on legal moves, not all moves
#			AL, caches = self.L_model_forward(np.hstack(self.attempts_X_batch), self.parameters) # use if training on all illegal attempts

			if i == 0:
				pre_cost = self.compute_cost_mean_square_error(AL, Y, weights)
				#pre_cost = self.compute_cost_cross_entropy(AL, Y)

			grads = self.L_model_backward(AL, Y, caches, weights)

			self.parameters = self.update_parameters(self.parameters, grads, learning_rate=learning_rate)

		cost = self.compute_cost_mean_square_error(AL, Y, weights)
		#cost = self.compute_cost_cross_entropy(AL, Y)
		self.trainings += 1
		self.plot_activations()
		legal_mean, illegal_mean = self.get_means(AL, np.hstack(self.illegal_masks)) # use if only training on legal moves, not all moves
#		legal_mean, illegal_mean = self.get_means(AL, np.hstack(self.attempts_illegal_masks)) # use if training on all illegal attempts

		self.legal_means.append(legal_mean)
		self.illegal_means.append(illegal_mean)

		params["illegal_means"] = self.illegal_means
		params["legal_means"] = self.legal_means
		params["trainings"] = self.trainings

		self.initialize_training_batch()

		return cost, params

	def get_means(self, AL, illegal_masks):
		legal_mean = np.sum(AL * illegal_masks)/np.count_nonzero(illegal_masks)
		illegal_mean = np.sum(AL * (1 - illegal_masks))/np.count_nonzero(1 - illegal_masks)

		return legal_mean, illegal_mean 


	def make_Y(self, probs, masks):
		Y = np.minimum(probs, masks)
		Y[Y > 0] = 1
		S = np.sum(Y, axis = 0)
		S[S == 0] = 1
		Y = Y / (S)

		return Y

	def get_input_vector(self, board, board_legal_moves, color, jump_piece_number):
		if color == 'Red':
			v = board.red_home_view().flatten()
		else:
			v = board.black_home_view().flatten()
		v = np.append(v, board.get_piece_vector(color))
		if jump_piece_number != None:
			j_vector = np.eye(12)[jump_piece_number]
			jump = np.array([1])
		else:
			j_vector = np.zeros((12))
			jump = np.array([0])
		v = np.append(v, j_vector)
		v = np.append(v, jump)	

		return v.reshape(v.size, -1)

	def L_model_forward(self, X, parameters):
	    """
	    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
	    
	    Arguments:
	    X -- data, numpy array of shape (input size, number of examples)
	    parameters -- output of initialize_parameters_deep()
	    
	    Returns:
	    AL -- last post-activation value
	    caches -- list of caches containing:
	                every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
	                the cache of linear_sigmoid_forward() (there is one, indexed L-1)
	    """

	    caches = []
	    A = X
	    L = len(parameters) // 2                  # number of layers in the neural network
	    
	    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
	    for l in range(1, L):
	        A_prev = A 
	        A, cache = self.linear_activation_forward(A_prev, 
	                                             parameters["W" + str(l)], 
	                                             parameters["b" + str(l)], 
	                                             activation='relu')
	        caches.append(cache)

	    
	    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
	    AL, cache = self.linear_activation_forward(A, 
                                            	 parameters["W" + str(L)], 
	                                             parameters["b" + str(L)], 
	                                             activation='softmax')
	    caches.append(cache)
	    
	    assert(AL.shape == (self.layers_dims[L],X.shape[1]))
	            
	    return AL, caches

	def linear_forward(self, A, W, b):
		"""
		Implement the linear part of a layer's forward propagation.

		Arguments:
		A -- activations from previous layer (or input data): (size of previous layer, number of examples)
		W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
		b -- bias vector, numpy array of shape (size of the current layer, 1)

		Returns:
		Z -- the input of the activation function, also called pre-activation parameter 
		cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
		"""

		Z = np.dot(W, A) + b
		

		assert(Z.shape == (W.shape[0], A.shape[1]))
		cache = (A, W, b)

		return Z, cache

	def linear_activation_forward(self, A_prev, W, b, activation):
	    """
	    Implement the forward propagation for the LINEAR->ACTIVATION layer

	    Arguments:
	    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
	    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
	    b -- bias vector, numpy array of shape (size of the current layer, 1)
	    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

	    Returns:
	    A -- the output of the activation function, also called the post-activation value 
	    cache -- a python dictionary containing "linear_cache" and "activation_cache";
	             stored for computing the backward pass efficiently
	    """
	    
	    if activation == "sigmoid":
	        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
	        Z, linear_cache = self.linear_forward(A_prev, W, b)
	        A, activation_cache = sigmoid(Z)
	    
	    elif activation == "relu":
	        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
	        Z, linear_cache = self.linear_forward(A_prev, W, b)
	        A, activation_cache = relu(Z)
	    
	    elif activation == "softmax":
	        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
	        Z, linear_cache = self.linear_forward(A_prev, W, b)
	        A, activation_cache = softmax(Z)
	    
	    assert (A.shape == (W.shape[0], A_prev.shape[1]))
	    cache = (linear_cache, activation_cache)

	    return A, cache

	def compute_cost_cross_entropy(self, AL, Y):
	    """
	    Implement the cross-entropy cost function.

	    Arguments:
	    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
	    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

	    Returns:
	    cost -- cross-entropy cost
	    """
	    
	    m = Y.shape[1]

	    # Compute loss from aL and y.
	    cost = (-1./ m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply((1-Y), np.log( 1-AL)))
	    
	    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
	    assert(cost.shape == ())
	    
	    return cost

	def compute_cost_mean_square_error(self, AL, Y, weights):
		"""
		Implement the mean square error cost function.

		Arguments:
		AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
		Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

		Returns:
		cost -- cross-entropy cost
		"""
		m = Y.shape[1]

		# Compute loss from aL and y.
		cost = (1./ m) * np.sum(weights * np.square(Y - AL))

		cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
		assert(cost.shape == ())

		return cost

	def linear_backward(self, dZ, cache):
		"""
		Implement the linear portion of backward propagation for a single layer (layer l)

		Arguments:
		dZ -- Gradient of the cost with respect to the linear output (of current layer l)
		cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

		Returns:
		dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
		dW -- Gradient of the cost with respect to W (current layer l), same shape as W
		db -- Gradient of the cost with respect to b (current layer l), same shape as b
		"""
		A_prev, W, b = cache
		m = A_prev.shape[1]

		dW = (1. / m) * np.dot(dZ, cache[0].T) 
		db = (1. / m) * np.sum(dZ, axis=1, keepdims=True)
		dA_prev = np.dot(cache[1].T, dZ)

		return dA_prev, dW, db

	def linear_activation_backward(self, dA, cache, activation):
	    """
	    Implement the backward propagation for the LINEAR->ACTIVATION layer.
	    
	    Arguments:
	    dA -- post-activation gradient for current layer l 
	    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
	    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
	    
	    Returns:
	    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
	    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
	    db -- Gradient of the cost with respect to b (current layer l), same shape as b
	    """
	    linear_cache, activation_cache = cache
	    
	    if activation == "relu":
	        dZ = relu_backward(dA, activation_cache)
	        dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
	        
	    elif activation == "sigmoid":
	        dZ = sigmoid_backward(dA, activation_cache)
	        dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
	    
	    elif activation == "softmax":
	        dZ = softmax_backward(dA, activation_cache)
	        dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
	    
	    return dA_prev, dW, db

	def L_model_backward(self, AL, Y, caches):
	    """
	    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
	    
	    Arguments:
	    AL -- probability vector, output of the forward propagation (L_model_forward())
	    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
	    caches -- list of caches containing:
	                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
	                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
	    
	    Returns:
	    grads -- A dictionary with the gradients
	             grads["dA" + str(l)] = ... 
	             grads["dW" + str(l)] = ...
	             grads["db" + str(l)] = ... 
	    """
	    grads = {}
	    L = len(caches) # the number of layers
	    m = AL.shape[1]
	    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
	    
	    # Initializing the backpropagation (derivative of cost)
	    #dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL)) # Cross-entropy derivative
	    dAL = 2 * weights * (AL - Y) # Mean Square Error derivative

	    
	    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
	    current_cache = caches[-1]
	    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = self.linear_activation_backward(dAL, current_cache, activation="softmax")
	    
	    for l in reversed(range(L-1)):
	        # lth layer: (RELU -> LINEAR) gradients.
	        # Inputs: "grads["dA" + str(l + 2)], caches". Outputs: "grads["dA" + str(l + 1)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
	        current_cache = caches[l]
	        
	        dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(grads["dA" + str(l + 2)], current_cache, activation="relu")
	        grads["dA" + str(l + 1)] = dA_prev_temp
	        grads["dW" + str(l + 1)] = dW_temp
	        grads["db" + str(l + 1)] = db_temp

	    return grads

	def update_parameters(self, parameters, grads, learning_rate):
		"""
		Update parameters using gradient descent

		Arguments:
		parameters -- python dictionary containing your parameters 
		grads -- python dictionary containing your gradients, output of L_model_backward

		Returns:
		parameters -- python dictionary containing your updated parameters 
		              parameters["W" + str(l)] = ... 
		              parameters["b" + str(l)] = ...
		"""

		L = len(parameters) // 2 # number of layers in the neural network

		# Update rule for each parameter. Use a for loop.
		for l in range(L):
			parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
			parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
		    
		return parameters

	def gradient_check_n(self, parameters, gradients, X, Y, epsilon = 1e-7):
		"""
		Checks if backward_propagation_n computes correctly the gradient of the cost output by forward_propagation_n

		Arguments:
		parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
		grad -- output of backward_propagation_n, contains gradients of the cost with respect to the parameters. 
		x -- input datapoint, of shape (input size, 1)
		y -- true "label"
		epsilon -- tiny shift to the input to compute approximated gradient with formula(1)

		Returns:
		difference -- difference (2) between the approximated gradient and the backward propagation gradient
		"""

		# Set-up variables
		parameters_values, _ = dictionary_to_vector(parameters)
		grad = gradients_to_vector(gradients)
		print(grad.shape)
		num_parameters = parameters_values.shape[0]
		print(parameters_values.shape)
		J_plus = np.zeros((num_parameters, 1))
		J_minus = np.zeros((num_parameters, 1))
		gradapprox = np.zeros((num_parameters, 1))
		maxdif = 0

		# Compute gradapprox
		for i in range(num_parameters):
			if i % 1000 == 0:
				print()
				print(i)
		    
			# Compute J_plus[i]. Inputs: "parameters_values, epsilon". Output = "J_plus[i]".
			# "_" is used because the function you have to outputs two parameters but we only care about the first one
			### START CODE HERE ### (approx. 3 lines)
			thetaplus = np.copy(parameters_values)                                      # Step 1
			thetaplus[i][0] += epsilon

			AL, _ = self.L_model_forward(X, vector_to_dictionary(thetaplus, self.parameters))
			J_plus[i] = self.compute_cost_cross_entropy(AL, Y)

			# Compute J_minus[i]. Inputs: "parameters_values, epsilon". Output = "J_minus[i]".
			thetaminus = np.copy(parameters_values)                                     # Step 1
			thetaminus[i][0] -= epsilon                               # Step 2    
			AL, _ = self.L_model_forward(X, vector_to_dictionary(thetaminus, self.parameters))    
			J_minus[i] = self.compute_cost_cross_entropy(AL, Y)                               # Step 3

			# Compute gradapprox[i]
			gradapprox[i] = (J_plus[i] - J_minus[i]) / (2. * epsilon)
			dif = np.squeeze(grad[i] - gradapprox[i])

			print(dif,"      \r", end='')
			if abs(dif) > abs(maxdif):
				maxdif = dif
				print(i, maxdif)

		# Compare gradapprox to backward propagation gradients by computing difference.
		### START CODE HERE ### (approx. 1 line)
		numerator = np.linalg.norm(grad - gradapprox)                               # Step 1'
		denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)             # Step 2'
		difference = numerator / denominator  

		if difference > 1e-7:
		    print ("\033[93m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
		else:
		    print ("\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")

		return difference
