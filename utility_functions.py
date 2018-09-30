import numpy as np

def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy
    
    Arguments:
    Z -- numpy array of any shape
    
    Returns:
    A -- output of sigmoid(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """
    
    A = 1/(1+np.exp(-Z))
    cache = Z
    
    return A, cache

def softmax(Z):
    """
    Implement the softmax function.

    Arguments:
    Z -- Output of the linear layer, of any shape. Assumes that training examples are in columns.

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache --a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """

    M = np.amax(Z, axis = 0, keepdims = True)
    Z = Z - M
    exps = np.exp(Z)
    A = exps / np.sum(exps, axis = 0)
    #print("Z, exps")
    #print(np.append(np.append(Z, exps, axis = 1), A, axis = 1))

    cache = Z

    return A, cache

def relu(Z):
    """
    Implement the RELU function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """
    
    A = np.maximum(0,Z)
        
    cache = Z 
    return A, cache


def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
        
    return dZ

def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
        
    return dZ

def softmax_backward(dA, cache):
    """
    Implement the backward propagation for a single SOFTMAX unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    
    exps = np.exp(Z)
    sm = exps / np.sum(exps, axis = 0, keepdims=True)

    dA_sm = np.sum(dA * sm, axis = 0)

    dZ = sm * (dA - dA_sm)
        
    return dZ

def dictionary_to_vector(parameters):
    """
    Roll all our parameters dictionary into a single vector satisfying our specific required shape.
    """
    keys = []
    L = len(parameters) // 2
    count = 0
    for l in range(1,L + 1):

        # flatten parameter
        new_vector = np.reshape(parameters["W" + str(l)], (-1,1))
        keys = keys + ["W" + str(l)]*new_vector.shape[0]

        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1

        new_vector = np.reshape(parameters["b" + str(l)], (-1,1))
        keys = keys + ["b" + str(l)]*new_vector.shape[0]

        theta = np.concatenate((theta, new_vector), axis=0)


    return theta, keys

def vector_to_dictionary(theta, p):
    """
    Unroll all our parameters dictionary from a single vector satisfying our specific required shape.
    """
    parameters = {}
    counter = 0
    for key in p:
        parameters[key] = theta[counter:(counter + p[key].size)].reshape(p[key].shape)
        counter += p[key].size

    return parameters

def gradients_to_vector(gradients):
    """
    Roll all our gradients dictionary into a single vector satisfying our specific required shape.
    """
    L = len(gradients) // 3
    count = 0
    for l in range(1,L + 1):
        # flatten parameter
        new_vector = np.reshape(gradients["dW" + str(l)], (-1,1))
        
        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1

        new_vector = np.reshape(gradients["db" + str(l)], (-1,1))

        theta = np.concatenate((theta, new_vector), axis=0)


    return theta

