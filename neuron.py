import math

class Neuron():
	_FUNCTIONS = (			#range
		'identity',			#(-inf, inf)
		'logistic',			#(0, 1)
		'tanh',				#(-1, 1)
		'arctan'			#(-pi/2, pi/2)
	)

	def __init__(self, weights, function, neuron_index, layer_index):
		assert weights is not None and len(weights) > 0, 'Must have at least one weight'
		assert function in self._FUNCTIONS, 'Function must be one of: ' + str(self._FUNCTIONS)
		assert neuron_index > 0, 'Neuron index must be greater than 0'
		assert layer_index > 0, 'Layer index must be greater than 0'

		self._weights = weights
		self._function = function
		self._name = 'Layer ' + str(layer_index) + ' Neuron ' + str(neuron_index)

	def activate(self, local_input):
		outputs = []

		if self._function == 'identity':
			value = local_input
		elif self._function == 'logistic':
			value = 1 / (1 + math.exp(-local_input))
		elif self._function == 'tanh':
			value = (2 / (1 + math.exp(-2 * local_input))) - 1
		elif self._function == 'arctan':
			value = math.atan(local_input)		

		self.value = value

		for weight in self._weights:
			outputs.append(value * weight)

		self.output = outputs

		return outputs

	def weights(self):
		return self._weights

	def update_weights(self, changes):
		for i in range(len(changes)):
			self._weights[i] -= changes[i]

	def __str__(self):
		return self._name