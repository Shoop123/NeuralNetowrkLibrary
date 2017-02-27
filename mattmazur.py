# Import the class
from neural_net import NeuralNetwork

# Create a nerual net with 2 inputs, 2 hidden neurons, and 2 output neurons.
# The hidden and output neurons will use the logistic activation function by default
# Currenty no support for any other activation function
# Replicates Matt Mazur's network
net = NeuralNetwork([2, 2, 2])

# Hack the network weights to set them to the same weights an in Matt Mazur's example
net._layers[0].neurons()[0]._weights[0] = 0.15
net._layers[0].neurons()[0]._weights[1] = 0.25

net._layers[0].neurons()[1]._weights[0] = 0.2
net._layers[0].neurons()[1]._weights[1] = 0.3

net._layers[0].bias()._weights[0] = 0.35
net._layers[0].bias()._weights[1] = 0.35

net._layers[1].neurons()[0]._weights[0] = 0.4
net._layers[1].neurons()[0]._weights[1] = 0.5

net._layers[1].neurons()[1]._weights[0] = 0.45
net._layers[1].neurons()[1]._weights[1] = 0.55

net._layers[1].bias()._weights[0] = 0.6
net._layers[1].bias()._weights[1] = 0.6

# Training dataset
training_data = [[(0.05, 0.1)]]

# Targets
target_data = [[0.01, 0.99]]

# Train once
net.train(training_data, target_data, 0.5)

# Save the net in the file "verbose_network.txt"
# In the file the network will be outlined in decent detail including
# all of its weights and layers.
# You will see that the weights are the same as in the example Matt Mazur presented after 1 round of training
net.save_verbose()

# Train the network 9999 more times for a total of 10,000 rounds (including the time above)
for i in range(9999):
	net.train(training_data, target_data, 0.5)

# Use the now trained neural network to predict values
print 'Net Output:', net.run([(0.05, 0.1)])
# The values Matt Mazur got at the end of his 10,000 rounds
# I suspect the difference is based on the float precision difference
print 'Matt\'s Output:', [[0.015912196], [0.984065734]]