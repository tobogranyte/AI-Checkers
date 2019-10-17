import numpy as np
from utility_functions import sigmoid, sigmoid_backward, relu, relu_backward, softmax, softmax_backward, dictionary_to_vector, vector_to_dictionary, gradients_to_vector
import pickle
import matplotlib.pyplot as plt
import os
import tensorflow as tf

class FCN_TF_p:

	def __init__(self):
		tf.reset_default_graph()
		self.batch_num = 0
		self.sess = tf.Session()
		self.layers_dims = [397, 1024, 512, 256, 128, 64, 48] #  6-layer model
		self.learning_rate = 0.03
		checkpoint = False
		print("Initializing parameters...")
		self.parameters = self.initialize_parameters_deep(self.layers_dims)
		print("Parameters initialized!")
		self.X_m, self.Y_m, self.weights = self.create_placeholders(397, 48, 1)
		self.AL_m, self.caches_m = self.L_model_forward(self.X_m, self.parameters)
		self.cost = self.compute_cost_mean_square_error(self.AL_m, self.Y_m, self.weights)
		self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
		self.init = tf.global_variables_initializer()
		self.saver = tf.train.Saver()
		available = self.check_for_params()
		self.sess.run(self.init)
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
				# Restore variables from disk.
				self.saver.restore(self.sess, "FCN_TF_p/model_temp.ckpt")
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
		if input("Gradient check?") == "Y":
			self.plot_activations()
			self.check_gradients()
			input("Paused.")
		self.initialize_training_batch()
		self.legal_means = []
		self.illegal_means = []
		self.trainings = 0
		#with tf.device('/cpu:0'):
		print(self.sess.run(self.parameters))

	def rs(self):
		inc_p = self.parameters["W1"].assign(self.parameters["W1"] + 1)
		self.sess.run(self.init)
		# Do some work with the model.
		print(self.sess.run(self.parameters["W1"]))
		self.sess.run(inc_p)
		print(self.sess.run(self.parameters["W1"]))
		self.save_obj('test')
		self.sess.run(self.init)
		print(self.sess.run(self.parameters["W1"]))
		self.saver.restore(self.sess, "FCN_TF_p/test.ckpt")
		print(self.sess.run(self.parameters["W1"]))

	def plot_activations(self):
		plt.figure(2)
		plt.ion()
		plt.show()
		g_norm = np.random.normal(scale = 1, size = self.layers_dims[0]).reshape(self.layers_dims[0],1)
		print(g_norm.shape)
		caches = self.sess.run(self.caches_m, feed_dict = {self.X_m: g_norm})
		Y = np.zeros((48,1))
		L = len(caches)
		for l in reversed(range(L)):
			sp = 100 + (10 * (L)) + (l + 1)
			plt.subplot(sp)
			linear_cache, activation_cache = caches[l]
			plt.hist(activation_cache)
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
		self.save_obj('model_temp')

	def save_obj(self, name):
		save_path = self.saver.save(self.sess, 'FCN_TF_p/'+ name + '.ckpt', )
		print("Model saved in path: %s" % save_path)

	def load_checkpoint(self, name):
		try:
			checkpoint = pickle.load(open("FCN_TF_p/" + name + ".pkl", "rb"))
			self.saver.restore(self.sess, "FCN_TF_p/" + name + ".ckpt")
		except (OSError, IOError) as e:
			checkpoint = False
			print("Can't find that checkpoint...")
		if checkpoint != False:
			print("Checkpoint " + name + ".ckpt loaded!")
		return checkpoint

	def check_for_params(self):
		available = os.path.isfile("FCN_TF_p/model_temp.ckpt.data-00000-of-00001")

		return available

	def initialize_network_architecture_deep(self, layer_dims):
		layers_dims = tf.constant(layer_dims, name='layers_dims')
		print(layers_dims)

		return layers_dims


	def initialize_parameters_deep(self, layer_dims):
		"""
		Arguments:
		layer_dims -- python array (list) containing the dimensions of each layer in the network

		Returns:
		parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
		                Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
		                bl -- bias vector of shape (layer_dims[l], 1)
		"""

		parameters = {}
		L = len(layer_dims)          # number of layers in the network

		for l in range(1, L):
			#parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1]/2)
			parameters['W' + str(l)] = tf.get_variable('W' + str(l), [layer_dims[l], layer_dims[l-1]], initializer=tf.contrib.layers.xavier_initializer())
			#parameters['b' + str(l)] = np.zeros((layer_dims[l], 1)) 
			parameters['b' + str(l)] = tf.get_variable('b' + str(l), [layer_dims[l], 1], initializer=tf.zeros_initializer())

			#assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
			#assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

	        
		return parameters

	def create_placeholders(self, n_x, n_y, n_w):
		"""
		Creates the placeholders for the tensorflow session.

		Arguments:
		n_x -- scalar, size of feature vector for the board/player
		n_y -- scalar, number of classes (12 pieces * maximum of 4 possible moves per piece = 48)

		Returns:
		X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
		Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"

		Tips:
		- You will use None because it let's us be flexible on the number of examples you will for the placeholders.
		  In fact, the number of examples during test/train is different.
		"""

		### START CODE HERE ### (approx. 2 lines)
		X = tf.placeholder(tf.float32, shape=[n_x, None], name='X')
		Y = tf.placeholder(tf.float32, shape=[n_y, None], name='Y')
		weights = tf.placeholder(tf.int32, shape=[n_w, None], name='weights')
		### END CODE HERE ###

		return X, Y, weights

	def move(self, board, color, jump_piece_number = None, jump_rule = True, illegal = False):
		self.num_attempts += 1
		if illegal == False: # everything within this sets up the stuff that won't change until a legal move is executed
			#print('legal')
			self.board_legal_moves = board.legal_moves(color = color, jump_piece_number = jump_piece_number, jump_rule = jump_rule) # get legal moves (48,) for current board position (0: illegal, 1:legal, 2:jump-legal)
			self.illegal_mask = np.zeros((48)) # create a holder (48,) for the illegal mask (starting filled with zeros)
			self.illegal_mask[self.board_legal_moves != 0] = 1 # ones for anything that's legal
			self.illegal_mask = self.illegal_mask.reshape(self.illegal_mask.size, -1) # make into a column vector
			self.X = self.get_input_vector(board, color, jump_piece_number = jump_piece_number) # create the input vector
			#self.AL, caches = self.L_model_forward(self.X, self.parameters) # run forward prop
			[self.AL, self.caches] = self.sess.run([self.AL_m, self.caches_m], feed_dict = {self.X_m: self.X})
			self.AL_forsort = np.append(self.AL, np.arange(self.AL.shape[0], dtype = 'int').reshape(self.AL.shape[0], 1), axis = 1)
		if self.move_type == "R":
			move = np.squeeze(np.random.choice(48, 1, p=self.AL.flatten()/np.sum(self.AL.flatten()))) # roll the dice and pick a move from the output probs
		elif self.num_attempts <= self.AL.size:
			move = int(self.AL_forsort[np.argsort(self.AL_forsort[:,0], axis = 0), :][(self.AL.size - self.num_attempts), 1])
			#print(self.num_attempts, self.AL_forsort[np.argsort(self.AL_forsort[:,0], axis = 0), :][(self.AL.size - self.num_attempts), :])
		else:
			move = 0
		one_hot_move = np.eye(48, dtype = 'int')[move] # turn it into a one-hot vector
		self.new_move = one_hot_move.reshape(one_hot_move.size, -1) # make it into a column vector

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
			self.moves.append(self.new_move)
			self.probabilities_batch.append(self.AL)
			self.X_batch.append(self.X)
			self.illegal_masks.append(self.illegal_mask)
			self.num_attempts_batch.append(self.num_attempts)
		else:
			self.moves[-1] = self.new_move

		return one_hot_move, self.board_legal_moves

	def parallel_predict(self, X, game_numbers):
		[self.AL, self.caches] = self.sess.run([self.AL_m, self.caches_m], feed_dict = {self.X_m: X})

		return self.AL

	def generate_move(self, AL): # generate a move from a probabilities vector
		choice = np.squeeze(np.random.choice(48, 1, p=AL.flatten()/np.sum(AL.flatten()))) # roll the dice and p b
		one_hot_move = np.eye(48, dtype = 'int')[choice] #generate one-hot version
		piece_number = int(np.argmax(one_hot_move)/4) # get the piece number that the move applies to
		move = one_hot_move[(4 * piece_number):((4 * piece_number) + 4)] # generate the move for that piece

		return one_hot_move, piece_number, move

	def complete_move(self):
		self.moves[-1] = self.new_move
		self.num_attempts_batch[-1] = self.num_attempts
		self.num_attempts = 0



	def train(self, Y, X, weights, illegal_masks):
		params = {}
		self.batch_num += 1
#		masks = np.hstack(self.illegal_masks)

#		Y = self.make_Y(np.hstack(self.probabilities_batch), masks) # use if only training on legal moves, not all moves
#		Y = self.make_Y(np.hstack(self.attempts_probabilities), np.hstack(self.attempts_illegal_masks)) # use if training on all illegal attempts
#		X_batch = np.hstack(self.X_batch)
		print(X.shape)

#		weights = np.hstack(self.num_attempts_batch)
#		weights = weights.reshape(1, weights.size)

		print(weights.shape)

		for i in range(0, 10):

			AL, caches, cost = self.sess.run([self.AL_m, self.caches_m, self.cost], feed_dict = {self.X_m: X, self.Y_m: Y, self.weights: weights})
#			AL, caches = self.L_model_forward(np.hstack(self.X_batch), self.parameters) # use if only training on legal moves, not all moves
#			AL, caches = self.L_model_forward(np.hstack(self.attempts_X_batch), self.parameters) # use if training on all illegal attempts

#			if i == 0:
#				r = np.random.randint(0,AL.shape[1])
#				R_AL = AL[:, r]
#				R_masks = masks[:, r]
#				R = [R_AL, R_masks]
#				with open('FCN_TF_p/random/random_move_'+ str(self.batch_num) + '.pkl', 'wb') as f:
#					pickle.dump(R, f, pickle.HIGHEST_PROTOCOL)

#				pre_cost = cost
				#pre_cost = self.compute_cost_mean_square_error(AL, Y, weights)
				#pre_cost = self.compute_cost_cross_entropy(AL, Y)

			_ , cost = self.sess.run([self.optimizer, self.cost], feed_dict={self.X_m: X, self.Y_m: Y, self.weights: weights})
			#grads = self.L_model_backward(AL, Y, caches, weights)

			#self.parameters = self.update_parameters(self.parameters, grads, learning_rate=learning_rate)

		L = len(caches)
		mins = []
		maxes = []
		for l in reversed(range(L)):
			_, activation_cache = caches[l]
			print(l)
			mins.append(np.amin(activation_cache))
			maxes.append(np.amax(activation_cache))

		#cost = self.compute_cost_mean_square_error(AL, Y, weights)
		#cost = self.compute_cost_cross_entropy(AL, Y)
		self.trainings += 1
		self.plot_activations()
		legal_mean, illegal_mean = self.get_means(AL, illegal_masks) # use if only training on legal moves, not all moves
#		legal_mean, illegal_mean = self.get_means(AL, np.hstack(self.attempts_illegal_masks)) # use if training on all illegal attempts

		self.legal_means.append(legal_mean)
		self.illegal_means.append(illegal_mean)

		params["illegal_means"] = self.illegal_means
		params["legal_means"] = self.legal_means
		params["trainings"] = self.trainings
		params["mins"] = mins
		params["maxes"] = maxes

		self.initialize_training_batch()

		return cost, params

	def get_means(self, AL, illegal_masks):
		legal_mean = np.sum(AL * illegal_masks)/np.count_nonzero(illegal_masks)
		illegal_mean = np.sum(AL * (1 - illegal_masks))/np.count_nonzero(1 - illegal_masks)

		return legal_mean, illegal_mean 


	def make_Y(self, probs, masks):
		Y = masks
		# looks like Y should end up being masks
		S = np.sum(Y, axis = 0) # sum of total legal moves
		S[S == 0] = 1
		# This seems to be forcing a 1 for the sum when there are no legal moves.
		# But in the fullness of time I'm not sure why a situation with no legal moves would
		# ever make it into the training set.
		Y = Y / (S)

		return Y

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

	def L_model_forward(self, X, parameters):
		"""
		Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SSOFTMAX computation

		Arguments:
		X -- data, tensor with shape (input size, number of examples)
		parameters -- output of initialize_parameters_deep()

		Returns:
		AL -- last post-activation value
		caches -- list of caches containing:
		            every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
		            the cache of linear_sigmoid_forward() (there is one, indexed L-1)
		"""
		#with tf.device('/cpu:0'):
		caches = []
		A = X
		L = len(parameters) // 2                  # number of layers in the neural network
		# Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
		for l in range(1, L):
			print(l)
			A_prev = A 
			# A, cache = self.linear_activation_forward(A_prev, 
			#                                     parameters["W" + str(l)], 
			#                                     parameters["b" + str(l)], 
			#                                     activation='relu')
			linear_cache = (A_prev, parameters["W" + str(l)], parameters["b" + str(l)])
			Z = tf.add(tf.matmul(parameters["W" + str(l)], A_prev), parameters["b" + str(l)])
			activation_cache = Z
			A = tf.nn.relu(Z)
			cache = (linear_cache, activation_cache)
			caches.append(cache)


		# Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
		#AL, cache = self.linear_activation_forward(A, 
		#                                    	 parameters["W" + str(L)], 
		#                                         parameters["b" + str(L)], 
		#                                         activation='softmax')
		linear_cache = (A, parameters["W" + str(L)], parameters["b" + str(L)])
		Z = tf.add(tf.matmul(parameters["W" + str(L)], A), parameters["b" + str(L)])
		activation_cache = Z
		AL = tf.nn.softmax(Z, 0)
		cache = (linear_cache, activation_cache)
		caches.append(cache)
		        
		return AL, caches


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
		Y -- true "label" vector 

		Returns:
		cost -- cross-entropy cost
		"""
		w = tf.cast(weights, tf.float32)
		# Compute loss from aL and y.
		#cost = tf.losses.mean_squared_error(Y, AL, weights=weights, reduction=Reduction.MEAN)
		cost = tf.reduce_mean(tf.reduce_sum(tf.multiply(w, tf.square(tf.subtract(Y, AL))), axis = 0))
		#cost = (1./ m) * np.sum(weights * np.square(Y - AL))

		return cost

