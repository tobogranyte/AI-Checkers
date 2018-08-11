import numpy as np
from utility_functions import sigmoid, sigmoid_backward, relu, relu_backward, softmax, softmax_backward

class FCN1:

	def __init__(self):
		self.layers_dims = [524, 1024, 524, 262, 131, 48] #  6-layer model
		self.parameters = self.initialize_parameters_deep(self.layers_dims)

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
		X = self.get_input_vector(board, color, jump_piece_number = None)
		if illegal == False: # only run forward prop again if it's a brand new move. If the model is just trying again, just get another random choice from the same probs
			self.AL, self.caches = self.L_model_forward(X, self.parameters)
		move = np.squeeze(np.random.choice(48, 1, p=self.AL.flatten()))
		one_hot_move = np.eye(48, dtype = 'int')[move]

		return one_hot_move	

	def get_input_vector(self, board, color, jump_piece_number):
		if color == 'Red':
			v = board.red_home_view().flatten()
		else:
			v = board.black_home_view().flatten()
		v = np.append(v, board.get_piece_vector(color))
		if jump_piece_number != None:
			j_vector = np.eye(12)[jump_piece_number]
		else:
			j_vector = np.zeros((12))
		v = np.append(v, j_vector)

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

	def compute_cost_mean_square_error(self, AL, Y):
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
	    cost = (1./ m) * np.sum(np.square(Y - AL))
	    
	    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
	    assert(cost.shape == ())
	    
	    return cost

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
	        dA_prev, dW, db = linear_backward(dZ, linear_cache)
	        
	    elif activation == "sigmoid":
	        dZ = sigmoid_backward(dA, activation_cache)
	        dA_prev, dW, db = linear_backward(dZ, linear_cache)
	    
	    elif activation == "softmax":
	        dZ = softmax_backward(dA, activation_cache)
	        dA_prev, dW, db = linear_backward(dZ, linear_cache)
	    
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
	    
	    # Initializing the backpropagation (derivative of )
	    dAL = 2 * (AL - Y)
	    
	    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
	    current_cache = caches[-1]
	    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation="sigmoid")
	    
	    for l in reversed(range(L-1)):
	        # lth layer: (RELU -> LINEAR) gradients.
	        # Inputs: "grads["dA" + str(l + 2)], caches". Outputs: "grads["dA" + str(l + 1)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
	        current_cache = caches[l]
	        
	        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, activation="relu")
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