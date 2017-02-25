from layer import Layer, OutputLayer
from neuron import Neuron
import random, math
import cPickle as pickle

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
				hidden_neuron = Neuron(connection_weights, neuron_activation_function, j + 1, i + 1)
				neurons.append(hidden_neuron)

			connection_weights = self._create_weights(connections)
			bias = Neuron(connection_weights, 'identity', hidden_neurons_per_layer[i] + 1, i + 1)

			layer = Layer(neurons, bias, i + 1)

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
		return self._forward_pass(current_input)[0]

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
		previous_derivatives = []

		layer_changes = []

		for i in range(len(self._layers[-1].neurons())):
			output = self._layers[-1].neurons()[i].value

			dEdout = output - target[i]
			doutdnet = output * (1 - output)
			dEdnet = dEdout * doutdnet

			previous_derivatives.append(dEdnet)

		for i in range(len(self._layers) - 2, -1, -1):
			neuron_changes = []

			for j in range(len(self._layers[i].neurons())):
				weight_updates = []

				for k in range(len(self._layers[i].neurons()[j].weights())):
					dnetdw = self._layers[i].neurons()[j].value
					dEdw = previous_derivatives[k] * dnetdw

					weight_updates.append(learning_rate * dEdw)

				neuron_changes.append(weight_updates)

			bias_updates = []

			for j in range(len(self._layers[i].bias().weights())):
				dnetdw = self._layers[i].bias().value
				dEdw = previous_derivatives[j] * dnetdw

				bias_updates.append(learning_rate * dEdw)

			neuron_changes.append(bias_updates)

			derivatives = []

			for j in range(len(self._layers[i].neurons())):
				dEdout_arr = []

				for k in range(len(previous_derivatives)):
					dEdout_arr.append(previous_derivatives[k] * self._layers[i].neurons()[j].weights()[k])

				output = self._layers[i].neurons()[j].value

				dEdout = sum(dEdout_arr)
				doutdnet = output * (1 - output)
				dEdnet = dEdout * doutdnet

				derivatives.append(dEdnet)

			previous_derivatives = derivatives

			layer_changes.append(neuron_changes)

		layer_changes.reverse()

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
				file.write('\n-------------Layer ' + str(l) + '-------------\n')
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