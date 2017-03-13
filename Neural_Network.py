import numpy as np


class Neural_Network:

        # to calculate forward propogation for our defined input

    def __init__(self):
        # will assume our input layer size = 2, op layer size =1 and hidden
        # layer size is 1
        self.inputLayerSize = 2
        self.hiddenLayerSize = 3
        self.outputLayerSize = 1
        # will initialize with random weights of input layer
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)

    def forward(self, X):
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        return self.sigmoid(self.z3)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
