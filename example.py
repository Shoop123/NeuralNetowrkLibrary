# Train an OR gate

# Import the class
from neural_net import NeuralNetwork

# Create a nerual net with 2 inputs, 2 hidden neurons, and one output neuron. 
# The hidden and output neurons will use the logistic activation function
net = NeuralNetwork([2, 2, 1], function='logistic')

# Training dataset
train = [[(1, 0)], [(0, 1)], [(0, 0)], [(1, 1)]]

# Targets
targets = [[1], [1], [0], [1]]

# Train using full-batch gradient descent for 1000 epochs with a learning rate of 10
for i in range(1000):
	net.train_batch(train, targets, 10)

# Train using mini-batch gradient descent for 1000 epochs, with a batch size of 2 with a learning rate of 1
for i in range(1000):
	net.train_mini_batch(train, targets, 1, 2)

# Train using stochastic gradient descent for 1000 epochs with a learning rate of 0.1
for i in range(1000):
	net.train(train, targets, 0.1)

# Use the now trained neural network to predict values
print net.run([[1, 0]])
print net.run([[0, 1]])
print net.run([[0, 0]])
print net.run([[1, 1]])

# train on binary_step