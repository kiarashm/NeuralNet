import numpy as np
from mnist import MNIST
from NueralNet import Net
from sklearn.preprocessing import normalize
from csv import writer


NUM_CLASSES = 10


def one_hot(labels):
    return np.eye(NUM_CLASSES)[labels]


def split(data, labels):
	Xt = np.zeros((50000, 784))
	Yt = np.zeros((50000, 10))
	Xv = np.zeros((10000, 784))
	Yv = np.zeros((10000, 10))
	visited_indeces = []
	count = 0
	while count < 10000:
		rand = np.random.randint(60000)
		if rand not in visited_indeces:
			Xv[count] = data[rand]
			Yv[count] = labels[rand]
			visited_indeces.append(rand)
			count += 1
	not_visited = []
	for count in range(60000):
		if count not in visited_indeces:
			not_visited.append(count)
	count = 0
	while count < 50000:
		for rand in not_visited:
			Xt[count] = data[rand]
			Yt[count] = labels[rand]
			count += 1

	return Xt, Xv, Yt, Yv 
	 


mndata = MNIST('./data/')
X, Y = map(np.array, mndata.load_training())
Xt = normalize(X)
Yt = one_hot(Y)


Xt, Xv, Yt, Yv, = split(Xt, Yt)

net = Net(n_in=784, n_hid=200, n_out=10)

net.train(Xt, Yt, Xv, Yv)

Xp, _ = map(np.array, mndata.load_testing())
Xp = normalize(Xp)
predLabels = net.classify(net.predict(Xp))
write_fil(predLabels)

def write_fil(predLabels):
    wrt = writer(open('predictions.csv', 'wb'))
    wrt.writerow( ('Id', 'Category') )
    for i, label in enumerate(predLabels):
        wrt.writerow( (i+1, label) )
