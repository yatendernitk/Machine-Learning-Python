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
        yHat = self.sigmoid(self.z3)
        return yHat

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def getParams(self):
        # Get W1 and W2 unrolled into vector:
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params

    def setParams(self, params):
        # Set W1 and W2 using single paramater vector.
        W1_start = 0
        W1_end = self.hiddenLayerSize * self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end],
                             (self.inputLayerSize, self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize * self.outputLayerSize
        self.W2 = np.reshape(
            params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))

    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))

    def computeNumericalGradient(N, X, y):
        paramsInitial = N.getParams()
        numgrad = np.zeros(paramsInitial.shape)
        perturb=np.zeros(paramsInitial.shape)
        e=1e-4

        for p in range(len(paramsInitial)):
            # Set perturbation vector
            perturb[p]=e
            N.setParams(paramsInitial + perturb)
            loss2=N.costFunction(X, y)
            N.setParams(paramsInitial - perturb)
            loss1=N.costFunction(X, y)
            # Compute Numerical Gradient
            numgrad[p]=(loss2 - loss1) / (2 * e)
            # Return the value we changed to zero:
            perturb[p]=0

        N.setParams(paramsInitial)
        return numgrad


    def f(self, x):
        return x**2

    def sigmoidPrime(self, z):
        # derivative of sigmoid Function
        return np.exp(-z) / (np.square(1 + np.exp(-z)))

    def costFunctionPrime(self, X, y):
        # this is gradient descent
        self.yHat=self.forward(X)

        delta3=np.multiply(-(y - self.yHat), self.sigmoidPrime(self.z3))
        djdW2=np.dot(self.a2.T, delta3)

        delta2=np.dot(delta3, self.W2.T) * self.sigmoidPrime(self.z2)
        djW1=np.dot(X.T, delta2)
        return djW1, djdW2

    def costFunction(self, X, y):
        self.yHat=self.forward(X)
        # k = y - self.yHat
        # kSquared = np.square(k)
        # # row is the number of training examples
        # row, colum = np.shape(kSquared)
        # sum = np.matrix.sum(kSquared)
        # cost = (1 / (2 * row)) * sum
        # return cost
        J=0.5 * sum(np.square(y - self.yHat))
        return J

    def main():
        """Return the neural network initialization."""
        from Neural_Network import Neural_Network
        print("main code")
        # X = (hours sleeping, hours studying), y = Score on test
        X=np.array(([3, 5], [5, 1], [10, 2]))
        y=np.array(([75], [82], [93]))
        NN=Neural_Network()
        costList=[]
        scalar=3
        for num in range(0, 1000):
            cost=NN.costFunction(X, y)
            # print(cost)
            costList.append(cost)
            (w1, w2)=NN.costFunctionPrime(X, y)
            NN.W1=NN.W1 - scalar * w1
            NN.W2=NN.W2 - scalar * w2
            # print(NN.W1, NN.W2)

        # print(costList)

    if __name__ == "__main__":
        main()
