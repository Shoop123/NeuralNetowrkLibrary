from layer import Layer, OutputLayer
from neuron import Neuron
import random, math
import cPickle as pickle

# TODO update back prop so that it works with different activation function (linear and non linear)

class NeuralNetwork():
	def __init__(self, neurons_per_layer, function='logistic'):
		assert neurons_per_layer is not None and len(neurons_per_layer) >= 2, 'Must have at least one input layer and one output layer'
		assert neurons_per_layer[0] > 0, 'Must have at least one input neuron'
		assert neurons_per_layer[-1] > 0, 'Must have at least one output neuron'

		self._neurons_per_layer = neurons_per_layer
		self._layers = []

		input_layer = self._create_inputs(neurons_per_layer[0], neurons_per_layer[1], 'identity')

		self._layers.append(input_layer)

		hidden_layers = self._create_hidden(neurons_per_layer[1:-1], neurons_per_layer[-1], function)

		self._layers += hidden_layers

		output_layer = self._create_output(neurons_per_layer[-1], function)

		self._layers.append(output_layer)

	def _create_inputs(self, num_inputs, input_neuron_connections, neuron_activation_function):
		input_neurons = []

		for i in range(num_inputs):
			connection_weights = self._create_weights(input_neuron_connections)

			input_neuron = Neuron(connection_weights, neuron_activation_function, i + 1, 1)
			input_neurons.append(input_neuron)

		connection_weights = self._create_weights(input_neuron_connections)
		input_bias = Neuron(connection_weights, 'identity', num_inputs + 1, 1)

		return Layer(input_neurons, input_bias, 1)

	def _create_hidden(self, hidden_neurons_per_layer, num_outputs, neuron_activation_function):
		layers = []

		num_hidden_layers = len(hidden_neurons_per_layer)

		for i in range(num_hidden_layers):
			neurons = []

			if i < num_hidden_layers - 1:
				connections = hidden_neurons_per_layer[i + 1]
			else:
				connections = num_outputs

			for j in range(hidden_neurons_per_layer[i]):
				connection_weights = self._create_weights(connections)
				hidden_neuron = Neuron(connection_weights, neuron_activation_function, j + 1, i + 2)
				neurons.append(hidden_neuron)

			connection_weights = self._create_weights(connections)
			bias = Neuron(connection_weights, 'identity', hidden_neurons_per_layer[i] + 1, i + 2)

			layer = Layer(neurons, bias, i + 2)

			layers.append(layer)

		return layers

	def _create_output(self, num_outputs, function):
		neurons = []

		layer_index = len(self._neurons_per_layer)

		for i in range(num_outputs):
			output_neuron = Neuron([1], function, i + 1, layer_index)
			neurons.append(output_neuron)

		layer = OutputLayer(neurons)

		return layer

	def _create_weights(self, amount):
		weights = []

		for i in range(amount):
			weights.append(random.random())

		return weights

	def run(self, current_input):
		return self._forward_pass(current_input)

	def train(self, train, targets, learning_rate):
		for i in range(len(train)):
			result = self._forward_pass(train[i])
			self._back_prop(targets[i], learning_rate)

	# Stochastic gradient descent (update weights based on error of every training case)
	def train(self, train, targets, learning_rate):
		self._stochastic_gd(train, targets, learning_rate)

	# Batch gradient descent (Update weights based on error of all training cases)
	def train_batch(self, train, targets, learning_rate):
		self._batch_gd(train, targets, learning_rate)

	# Mini-batch gradient descent (Update weights based on error of batch_size training cases)
	def train_mini_batch(self, train, targets, learning_rate, batch_size):
		self._mini_batch_gd(train, targets, learning_rate, batch_size)

	def _stochastic_gd(self, train, targets, learning_rate):
		for i in range(len(train)):
			result = self._forward_pass(train[i])
			self._online_back_prop(targets[i], learning_rate)

	def _batch_gd(self, train, targets, learning_rate):
		layer_changes = []

		for i in range(len(train)):
			result = self._forward_pass(train[i])

			if layer_changes == []:
				layer_changes = self._offline_back_prop(targets[i], learning_rate)
			else:
				layer_updates = self._offline_back_prop(targets[i], learning_rate)
				
				for l in range(len(layer_changes)):
					for n in range(len(layer_changes[l])):
						for w in range(len(layer_changes[l][n])):
							layer_changes[l][n][w] += layer_updates[l][n][w]

		self._update_layers(layer_changes)

	def _mini_batch_gd(self, train, targets, learning_rate, batch_size):
		layer_changes = []

		iteration = 0

		for i in range(len(train)):
			result = self._forward_pass(train[i])

			if iteration >= batch_size:
				self._update_layers(layer_changes)
				layer_changes = []
				iteration = 0
			
			if layer_changes == []:
				layer_changes = self._offline_back_prop(targets[i], learning_rate)
			else:
				layer_updates = self._offline_back_prop(targets[i], learning_rate)
				
				for l in range(len(layer_changes)):
					for n in range(len(layer_changes[l])):
						for w in range(len(layer_changes[l][n])):
							layer_changes[l][n][w] += layer_updates[l][n][w]

			iteration += 1

		if layer_changes != []:
			self._update_layers(layer_changes)

	def mean_squared_error(self, result, target):
		errors = []

		for i in range(len(target)):
			error = (target[i] - result[i][0])**2 / 2
			errors.append(error)

		return sum(errors)

	def _forward_pass(self, inputs):
		current_output = inputs

		for layer in self._layers:
			current_output = layer.next(current_output)

		return current_output

	def _offline_back_prop(self, target, learning_rate):
		return self._back_prop(target, learning_rate)

	def _online_back_prop(self, target, learning_rate):
		layer_changes = self._back_prop(target, learning_rate)

		self._update_layers(layer_changes)

	def _back_prop(self, target, learning_rate):
		# derivatives of the error with respect to the net input to every neuron in the layer before
		previous_derivatives = []

		# a 3d list of every weight change for every neuron in every layer
		# 1st dimension - consists of a list of layers
		# 2nd dimension - consists of a list of neurons in each layer
		# 3rd dimension - consists of a list of weights in each neuron
		# e.g. if the list is: [[[[1]], [4]], [[]]] then there are 2 layers, 
		# the first layer has 2 neurons with 1 synapse (weight) each
		# the second layer is the ouput layer so it doesn't have any weights
		layer_changes = []

		# go through all of the output neurons
		for i in range(len(self._layers[-1].neurons())):
			# get the output of the current neuron after activation
			# this is just what value the neuron takes on after using the activation function on its input
			output = self._layers[-1].neurons()[i].value
			
			# error derivative with respect with the output of the neuron
			dEdout = output - target[i]
			# output derivative with respect to the input to the neuron
			doutdnet = self._layers[-1].neurons()[i].derivative()
			# error derivative with respect to the input
			dEdnet = dEdout * doutdnet

			previous_derivatives.append(dEdnet)

		# loop through the layers, starting with the layer before the output layer, and going backwards from there
		for i in range(len(self._layers) - 2, -1, -1):
			# all of the weight adjustments for each neuron
			neuron_changes = []

			# loop through all of the neurons in the current layer
			# doesn't handle biases
			for j in range(len(self._layers[i].neurons())):
				# all of the weight updates for the current neuron
				weight_updates = []

				# loop through all of the weights in the neuron
				for k in range(len(self._layers[i].neurons()[j].weights())):
					# derivative of the input to the previous neuron with respect to the weight
					# this is just what value the neuron takes on after using the activation function on its input
					dnetdw = self._layers[i].neurons()[j].value
					# error derivative with respect to the weight
					dEdw = previous_derivatives[k] * dnetdw

					# add the weight change scaled by the learning rate
					weight_updates.append(learning_rate * dEdw)

				neuron_changes.append(weight_updates)

			# this is where the bias weight updates are calculated
			# updates to the bias weights for the layer
			bias_updates = []

			# loop through all of the weights in the bias
			for j in range(len(self._layers[i].bias().weights())):
				# derivative of the input to the previous neuron with respect to the weight of the bias
				dnetdw = self._layers[i].bias().value
				# error derivative with respect to the weight of the bias
				dEdw = previous_derivatives[j] * dnetdw

				# add the weight change scaled by the learning rate
				bias_updates.append(learning_rate * dEdw)

			neuron_changes.append(bias_updates)

			# this is where the "previous_derivatives" list gets updated to prepare for the calculations in the next layer
			# new derivatives of the error with respect to the net input to every neuron in the layer before
			derivatives = []

			# loop through all of layers
			for j in range(len(self._layers[i].neurons())):
				# list of error derivatives with respect to the output of the current neuron
				dEdout_arr = []

				# loop through all of the "previous_derivatives" because each new value is based on the previous ones
				for k in range(len(previous_derivatives)):
					# previous_derivatives[k] is the derivate of the error with respect to the input to the previous neuron
					# self._layers[i].neurons()[j].weights()[k] is the derivative of the input to the previous neuron with respect to the output of the current one
					# multiplying them give us the derivative of the error with respect to the output of the current neuron
					dEdout_arr.append(previous_derivatives[k] * self._layers[i].neurons()[j].weights()[k])

				# sum of the derivatives calculated before
				dEdout = sum(dEdout_arr)
				# derivative of the output of the current neuron with respect to its input
				# so the derivative of the activation function
				doutdnet = self._layers[i].neurons()[j].derivative()
				# derivative of the error with respect to the net input of the current neuron
				dEdnet = dEdout * doutdnet

				# save it!
				derivatives.append(dEdnet)

			# update the previous derivatives to prepare for the calculations in the next layer
			previous_derivatives = derivatives

			# save the changes for this layer
			layer_changes.append(neuron_changes)

		# we have to reverse the list since we iterated through the layers backwards
		# so all of the changes are saved starting with the last
		layer_changes.reverse()

		# return the layer changes for updating depending on the type of learning (full-batch, mini-batch, stochastic)
		return layer_changes

	def _update_layers(self, layer_changes):
		for i in range(len(layer_changes)):
			self._layers[i].update_neurons(layer_changes[i])

	def save_verbose(self, filename='verbose_network.txt'):
		file = open(filename, 'w')

		file.write('Number of layers: ' + str(len(self._neurons_per_layer)) + '\n')
		file.write('Neurons per layer: ' + str(self._neurons_per_layer) + '\n')

		for l in range(len(self._layers)):
			if l == 0:
				file.write('\n-------------Input Layer-------------\n')
			elif l < len(self._layers) - 1:
				file.write('\n-------------Hidden Layer ' + str(l) + '-------------\n')
			else:
				file.write('\n-------------Output Layer-------------\n')

			for n in range(len(self._layers[l].neurons())):
				file.write('\tNeuron ' + str(n + 1) + '\n')

				for w in range(len(self._layers[l].neurons()[n].weights())):
					file.write('\t\tWeight ' + str(w + 1) + ': ' + str(self._layers[l].neurons()[n].weights()[w]) + '\n')

			if l < len(self._layers) - 1:
				file.write('\tBias\n')

				for w in range(len(self._layers[l].bias().weights())):
					file.write('\t\tWeight ' + str(w + 1) + ': ' + str(self._layers[l].bias().weights()[w]) + '\n')

		file.close()

	def save(self, filename='network.pkl'):
		pickle.dump(self, open(filename, 'wb'))

def create_saved_network(filename):
	return pickle.load(open(filename, 'rb'))