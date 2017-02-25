from neural_net import NeuralNetwork

net = NeuralNetwork([2, 1], function='logistic')

train = [[(1, 0)], [(0, 1)], [(0, 0)], [(1, 1)]]
targets = [[1], [1], [0], [1]]

for i in range(1000):
	net.train(train, targets, 1)

for i in range(10000):
	net.train(train, targets, 0.1)

for i in range(100000):
	net.train(train, targets, 0.01)

print net.run([[1, 0]])
print net.run([[0, 1]])
print net.run([[0, 0]])
print net.run([[1, 1]])