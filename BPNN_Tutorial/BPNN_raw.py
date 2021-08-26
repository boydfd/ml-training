import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def deriv_sigmoid(x):
    fx = sigmoid(x)
    return fx * (1 - fx)


def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()


class NeuralNetwork:
    def __init__(self, learn_rate=0.1, epochs=1000):
        # 权重，Weights
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()
        self.sum_h1 = np.random.normal()
        self.h1 = np.random.normal()
        self.sum_h2 = np.random.normal()
        self.h2 = np.random.normal()
        self.sum_o1 = np.random.normal()
        self.o1 = np.random.normal()
        self.learn_rate = learn_rate
        self.epochs = epochs

    def feedforward(self, x):
        # x is a numpy array with 2 elements.
        self.sum_h1 = self.w1 * x[0] + self.w2 * x[1]
        self.h1 = sigmoid(self.sum_h1)
        self.sum_h2 = self.w3 * x[0] + self.w4 * x[1]
        self.h2 = sigmoid(self.sum_h2)
        self.sum_o1 = self.w5 * self.h1 + self.w6 * self.h2
        return sigmoid(self.sum_o1)

    def backward(self, x, y):
        [y_pred, y_true] = y
        d_L_d_ypred = -2 * (y_true - y_pred)
        d_ypred_d_w5 = self.h1 * deriv_sigmoid(self.sum_o1)
        d_ypred_d_w6 = self.h2 * deriv_sigmoid(self.sum_o1)
        d_ypred_d_h1 = self.w5 * deriv_sigmoid(self.sum_o1)
        d_ypred_d_h2 = self.w6 * deriv_sigmoid(self.sum_o1)
        d_h1_d_w1 = x[0] * deriv_sigmoid(self.sum_h1)
        d_h1_d_w2 = x[1] * deriv_sigmoid(self.sum_h1)
        d_h2_d_w3 = x[0] * deriv_sigmoid(self.sum_h2)
        d_h2_d_w4 = x[1] * deriv_sigmoid(self.sum_h2)
        # --- Update weights and biases
        # Neuron h1
        self.w1 -= self.learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
        self.w2 -= self.learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
        self.w3 -= self.learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
        self.w4 -= self.learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
        self.w5 -= self.learn_rate * d_L_d_ypred * d_ypred_d_w5
        self.w6 -= self.learn_rate * d_L_d_ypred * d_ypred_d_w6

    def train(self, data, all_y_trues):
        for epoch in range(self.epochs):
            for x, y_true in zip(data, all_y_trues):
                y_pred = self.feedforward(x)
                self.backward(x, [y_pred, y_true])
            # --- Calculate total loss at the end of each epoch
            if epoch % 10 == 0:
                y_preds = np.apply_along_axis(self.feedforward, 1, data)
                print(y_preds)
                loss = mse_loss(all_y_trues, y_preds)
                print("Epoch %d loss: %.3f" % (epoch, loss))


if __name__ == '__main__':
    # Define dataset
    data = np.array([
        [-2, -1],  # Alice
        [25, 6],  # Bob
        [17, 4],  # Charlie
        [-15, -6],  # Diana
    ])
    all_y_trues = np.array([
        1,  # Alice
        0,  # Bob
        0,  # Charlie
        1,  # Diana
    ])

    # Train our neural network!
    network = NeuralNetwork()
    network.train(data, all_y_trues)
    emily = np.array([-7, -3])  # 128 pounds, 63 inches
    frank = np.array([20, 2])  # 155 pounds, 68 inches
    print("Emily: %.3f" % network.feedforward(emily))  # 0.951 - F
    print("Frank: %.3f" % network.feedforward(frank))  # 0.039 - M
