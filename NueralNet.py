import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


def loss(y, z):
    e = 1e-50
    for i in range(len(z)):
        if z[i] < e:
            z[i] = e
    return -(y*np.log2(z)).sum()

def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))

def relu(z, derivative = False):
    if derivative:
        sol = np.array([x > 0 for x in z])
        return sol.astype(float)
    else:
        return np.maximum(0, z)

def plot_charts(train_acc, train_loss):
    numEpochs = range(1, (len(train_acc)+1) )
    plt.plot(numEpochs, train_acc, marker='o')
    plt.ylabel('Training Accuracy')
    plt.xlabel('10000 Iterations')
    plt.show()

    plt.plot(numEpochs, train_loss, marker='o')
    plt.ylabel('Training Loss')
    plt.xlabel('10000 Iterations')
    plt.show()


class Net():
    def __init__(self, n_in, n_hid, n_out):
        self.alpha = 0.1 
        self.epochs = 3
        self.train_accuracy = []
        self.val_accuracy = []
        self.train_loss = []
        self.hiddenLayer = Layer(activation=relu, n_in=n_in, n_out=n_hid)
        self.outputLayer = Layer(activation=softmax, n_in=n_hid, n_out=n_out)
        

    def train(self, data, labels, Vdata, Vlabels):
        i = 0
        while i < self.epochs:
            print("EPOCH #" + str(i+1))
            s_data, s_labels = shuffle(data, labels, random_state=0)
            j = 0
            while j < len(s_data):
                if j % 9999 == 0 and j != 0:
                    self.training_accuracy(data, labels)
                    self.validation_accuracy(Vdata, Vlabels)
                x = s_data[j]
                target = s_labels[j]
                output = self.forward(x)
                delta = output - target
                self.backprop(delta)
                j += 1

            self.alpha *= 0.7
            i += 1
        
        plot_charts(self.train_accuracy, self.train_loss)


    def predict(self, data):
        return np.array([self.forward(x) for x in data])

    def classify(self, predicts):
        return np.argmax(predicts, axis=1)

    def forward(self, inputs):
        return self.outputLayer.forward(self.hiddenLayer.forward(inputs))

    def backprop(self, deltaOut):
        delta2 = self.hiddenLayer.deriv() * (self.outputLayer.weights).dot(deltaOut)
        self.outputLayer.update_weights(deltaOut, self.alpha)
        self.hiddenLayer.update_weights(delta2, self.alpha)
    
    def training_accuracy(self, data, labels):
        o = labels
        labels = self.classify(labels)
        y_pred = self.predict(data)
        y_predi = self.classify(y_pred)
        error = 0
        losses = []
        for i in range(labels.shape[0]):
            losses.append(loss(o[i], y_pred[i]))
            if y_predi[i] != labels[i]:
                error += 1
        error = 1 - (error / (1.0 * y_predi.size))
        self.train_accuracy.append(error)
        print("Training accuracy: " + str( error ))
        t_loss = np.mean(losses)
        print("Training loss: " + str(t_loss))
        self.train_loss.append(t_loss)

    def validation_accuracy(self, data, labels):
        labels = self.classify(labels)
        y_pred = self.predict(data)
        y_pred = self.classify(y_pred)
        error = 0
        for i in range(labels.size):
            if y_pred[i] != labels[i]:
                error += 1
        error = 1 - (error / (1.0 * y_pred.size))
        self.val_accuracy.append(error)
        print("Validation accuracy: " + str(error ))

class Layer():
    def __init__(self, activation, n_in, n_out):
        self.activation = activation
        self.weights = .01*np.random.randn(n_in, n_out)
        self.bias = .01*np.random.randn(n_out)

    def forward(self, inputs):
        self.input = inputs
        self.z = inputs.dot(self.weights) + self.bias
        return self.activation(self.z)

    def deriv(self):
        return self.activation(self.z, derivative=True)

    def update_weights(self, delta, lr):
        self.weights -= lr * np.outer(self.input, delta)
        self.bias -= lr * delta





    