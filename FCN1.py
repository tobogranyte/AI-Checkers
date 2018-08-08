import numpy as np

class FCN1:

	def initialize_parameters_deep(layer_dims):
	    """
	    Arguments:
	    layer_dims -- python array (list) containing the dimensions of each layer in our network
	    
	    Returns:
	    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
	                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
	                    bl -- bias vector of shape (layer_dims[l], 1)
	    """
	    
	    np.random.seed(3)
	    parameters = {}
	    L = len(layer_dims)            # number of layers in the network

	    for l in range(1, L):
	        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
	        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1)) 
	        
	        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
	        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

	        
	    return parameters

    def linear_forward(A, W, b):
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

	def L_model_forward(X, parameters):
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
	        A, cache = linear_activation_forward(A_prev, 
	                                             parameters["W" + str(l)], 
	                                             parameters["b" + str(l)], 
	                                             activation='relu')
	        caches.append(cache)

	    
	    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
	    AL, cache = linear_activation_forward(A, 
	                                             parameters["W" + str(L)], 
	                                             parameters["b" + str(L)], 
	                                             activation='sigmoid')
	    caches.append(cache)
	    
	    
	    assert(AL.shape == (1,X.shape[1]))
	            
	    return AL, caches

	def compute_cost(AL, Y):
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

	def L_model_backward(AL, Y, caches):
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
	    
	    # Initializing the backpropagation
	    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
	    
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

	def update_parameters(parameters, grads, learning_rate):
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