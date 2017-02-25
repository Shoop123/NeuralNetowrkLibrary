import math

class Neuron():
	def __init__(self, weights, function):
		self._weights = weights
		self._function = function

	def activate(self, local_input):
		outputs = []

		if self._function == 'logistic':
			value = 1 / (1 + math.exp(-local_input))
		elif self._function == 'linear':
			value = local_input

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