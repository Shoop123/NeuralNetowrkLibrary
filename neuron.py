import math

class Neuron():
	_FUNCTIONS = (			#range
		'identity',			#(-inf, inf)
		'logistic',			#(0, 1)
	)

	def __init__(self, weights, function, neuron_index, layer_index):
		assert weights is not None and len(weights) > 0, 'Must have at least one weight'
		assert function in self._FUNCTIONS, 'Function must be one of: ' + str(self._FUNCTIONS)
		assert neuron_index > 0, 'Neuron index must be greater than 0'
		assert layer_index > 0, 'Layer index must be greater than 0'

		self._weights = weights
		self._function = function
		self._name = 'Layer ' + str(layer_index) + ' Neuron ' + str(neuron_index)
		self._activated = False

	def activate(self, local_input):
		outputs = []

		self.value = self._run_function(local_input)

		for weight in self._weights:
			outputs.append(self.value * weight)

		self.output = outputs

		self._activated = True

		return outputs

	def derivative(self):
		if self._activated:
			derivative = self._run_function(self.value, derivative=True)
		else:
			derivative = 0

		return derivative

	def _run_function(self, local_input, derivative=False):
		if self._function == 'identity':
			value = self._identity(local_input, derivative)
		elif self._function == 'logistic':
			value = self._sigmoid(local_input, derivative)

		return value

	def _identity(self, local_input, derivative):
		if not derivative:
			value = local_input
		else:
			value = 1.

		return value

	def _sigmoid(self, local_input, derivative):
		if not derivative:
			try:
				exp = math.exp(-local_input)
			except OverflowError:
				exp = float('inf')

			value = 1 / (1 + exp)
		else:
			value = self._sigmoid(local_input, False) * (1 - self._sigmoid(local_input, False))

		return value

	def weights(self):
		return self._weights

	def update_weights(self, changes):
		for i in range(len(changes)):
			self._weights[i] -= changes[i]

	def __str__(self):
		return self._name