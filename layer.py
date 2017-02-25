class Layer():
	def __init__(self, neurons, bias, index):
		assert neurons is not None and len(neurons) > 0, 'Must have at least 1 neuron per layer'
		assert index > 0, 'Layer index must be greater than 0'

		self._neurons = neurons
		self._bias = bias
		self._name = 'Layer ' + str(index)

	def next(self, inputs):
		outputs = []

		neuron_inputs = [0] * len(self._neurons)

		for i in range(len(inputs)):
			local_inputs = inputs[i]

			for j in range(len(local_inputs)):
				neuron_inputs[j] += local_inputs[j]

		for i in range(len(neuron_inputs)):
			output = self._neurons[i].activate(neuron_inputs[i])

			outputs.append(output)

		if self._bias is not None:
			bias = self._bias.activate(1)
			outputs.append(bias)

		return outputs

	def neurons(self):
		return self._neurons

	def bias(self):
		return self._bias

	def update_neurons(self, changes):
		for i in range(len(changes) - 1):
			self._neurons[i].update_weights(changes[i])

		self._bias.update_weights(changes[-1])

	def __str__(self):
		return self._name

class OutputLayer(Layer):
	def __init__(self, neurons):
		self._neurons = neurons
		self._name = 'Output Layer'

	def next(self, inputs):
		outputs = []

		neuron_inputs = [0] * len(self._neurons)

		for i in range(len(inputs)):
			local_inputs = inputs[i]

			for j in range(len(local_inputs)):
				neuron_inputs[j] += local_inputs[j]

		for i in range(len(neuron_inputs)):
			output = self._neurons[i].activate(neuron_inputs[i])

			outputs.append(output)

		return outputs