import numpy as np
import math

np.random.seed(1234)


# input matrix has each picture as a column vector (example: 2304 pixels, 5000 examples) 
def append_bias(matrix):
    bias = np.ones(matrix.shape[1]) # 5000
    return np.r_[matrix,[bias]]

# return list of tuples (start,end) for slicing each batch X
def get_indexes(n, batchSize):
    indexes = []  # list of (start,end) for slicing each batch X
    index = 0
    for round in range(math.ceil(n / batchSize)):
        index += batchSize
        if index > n:
            index -= batchSize
            indexes.append((index, n))
            break
        indexes.append((index - batchSize, index))
    return indexes


# # Given a vector of weights w, a design matrix Xtilde, and a vector of labels y, and a regularization strength
# # alpha (default value of 0), return the gradient of the (regularized) MSE loss.
# def grad_MSE(weight, Xtilde, y,alpha):
#     n = y.shape[0]
#     yhat = np.dot(Xtilde.T,weight)
#     distance = yhat - y #(5000)
#     coeff = 1 / n

#     # This version, simply modifies the last index of the weight array and coverts it into 0
#     # this is done to not penalize the bias. This code is an alternative to the identity matrix method performed bellow
#     # wReg = np.copy(weight)
#     # wReg[-1] = 0
#     # regularization = (alpha / n) * wReg #(2305,)

#     identity_matrix =np.diag( np.append(np.ones(Xtilde.shape[0]-1), 0)) ## <-- dimension is 2305 x 2305 where last element must be 0 which means the bias is not included 
#     regularization = alpha / n * (np.transpose(weight).dot(identity_matrix))

#     gradient =  (1/n) * np.dot(Xtilde,distance) + regularization

#     return gradient 

# def mse (w, X_tilde, y):
#     yhat = np.dot(X_tilde.T, w)
#     coeff = 1 / (2 * X_tilde.shape[1])
#     sum = np.sum((yhat - y)**2)
#     mse = coeff * sum
#     return mse

def SGD(X_tilde, ytr, batch_size, epochs, epsilon, alpha):

    # randomize training set     
    permute = np.random.permutation(X_tilde.shape[1]) # this should be 5000
    shuffled_X  = X_tilde.T[permute].T #(2305, 4000)
    shuffled_y = ytr[permute] #(4000)

    # Xtilde (2305,4000)
    sample_size = X_tilde.shape[1] # total batch size

    # get all indexes based on batch size
    rounds = get_indexes(sample_size, batch_size) # list of (start,end) for slicing each batch X

    # initialize weights to random values with an standard deviation of 0.01
    weights = np.random.rand(7840).reshape(784,10) * 0.01 # (784, 10)

    # start iteration loop
    for epoch in range(epochs):
        for indexes in rounds:
            start, finish = indexes
            # gradient =  grad_MSE(weights, shuffled_X[:,start:finish], shuffled_y[start:finish], alpha)
            gradient =  grad_CE(weights, shuffled_X[:,start:finish], shuffled_y[start:finish], alpha)
            weights = weights - epsilon * gradient
    return weights


# Calculate 
def softmax(weights, Xtilde):
    z = np.dot(Xtilde.T, weights) # (55000x10)
    yhat = np.exp(z) / np.sum(np.exp(z), axis=1) # axis=1 means sum columns (55000)
    print(f"Yhat shape is ({yhat.shape}) and its supposed to be (55000)") #  8888888888888888888888888888***************************************
    return yhat # (55000)

# Calculate the gradient of cross entropy loss
def grad_CE(weights, Xtilde, y, alpha):
    yhat = softmax(Xtilde,weights)
    distance = yhat - y
    n = y.shape[0] # 55000

    # This version, simply modifies the last index of the weight array and coverts it into 0
    # this is done to not penalize the bias. This code is an alternative to the identity matrix method performed bellow
    # TODO: MAKE WREG LAST INDEX ALL 0s ***************************************************************************************************
    wReg = np.copy(weight)
    # wReg[-1] = 0
    regularization = (alpha / n) * wReg #(2305,)
    # identity_matrix =np.diag( np.append(np.ones(Xtilde.shape[0]-1), 0)) ## <-- dimension is 2305 x 2305 where last element must be 0 which means the bias is not included 
    # regularization = alpha / n * (np.transpose(weights).dot(identity_matrix))

    gradient = 1/n * np.dot(Xtilde,distance) + regularization
    return gradient

# Calculate Cross Entropy without regularization 
def CE():

    pass

# Percent of correctly classified images
def PC (w, X_tilde, y):
    yhat = np.dot(X_tilde.T, w)
    coeff = 1 / (2 * X_tilde.shape[1])
    sum = np.sum((yhat - y)**2)
    mse = coeff * sum
    return mse



def train_number_classifier ():
    # Load data
    X_tr = np.load("mnist_train_images.npy").T  # (784, 55000)
    y_tr = np.load("mnist_train_labels.npy") # (55000, 10)

    X_val = np.load("mnist_validation_images.npy") # (5000, 784)
    y_val = np.load("mnist_validation_labels.npy") # (5000, 10)

    X_te = np.load("mnist_test_images.npy")
    y_te = np.load("mnist_test_labels.npy")

    print(X_te.shape) # (10000, 784)
    print(y_te.shape) # (10000, 10)

    # TODO APPENDS BIAAAAAAAAAAAAAAAAAAAS ************************************************************************************************
    # # append bias
    # Xtilde = append_bias(X_tr)
    # X_te = append_bias(X_te)


    # Hyper parameters 
    mini_batch_sizes = [100, 500, 1000, 2000] # mini batch sizes
    epochs = [1, 10, 50, 100] # number of epochs
    epochs = [1, 2,3,4] # number of epochs
    epsilons = [0.1, 3e-3, 1e-3, 3e-5] # learning rates
    alphas = [0.1, 0.01, 0.05, 0.001] # regularization alpha

    # key: [int] mse
    # value: tuple of hyperparameters (nTilde, epoch, epsilon, alpha, weights)
    # Dictionary to store our all the different hyperparameter sets, their weights and their MSE
    hyper_param_grid = {}

    count = 0

    # train weights based on all the different sets of hyperparameters
    for mini_batch_size in mini_batch_sizes:
        for epoch in epochs:
            for epsilon in epsilons:
                for alpha in alphas:
                        weights = SGD(Xtr_tilde, y_tr, mini_batch_size, epoch, epsilon, alpha)
                        
                        # calculate the mse with the validation set
                        # ceVal = CE(yhat) ************************************
                        # mseVal = CE(weights, X_val, Y_val)
                        count += 1
                        print("The MSE for [" + str(count) + "] training set is " + str(mseVal))

                        # add to dictionary
                        hyper_param_grid[mseVal] = (mini_batch_size, epoch, epsilon, alpha, np.copy(weights))
                        print("miniBatch: {}, epoch: {}, epsilon: {}, alpha: {}".format(mini_batch_size,epoch,epsilon,alpha))


    # # get key of dictionary with smallest MSE
    # smallMSE = min(hyper_param_grid.keys())

    # # Report fMSE cost on the training
    # # print("--------------------------------------------------------")
    # print("The MSE for training set is " + str(smallMSE))
    # print("--------------------------------------------------------")

    # # show the best hyperparameters
    # print("My best hyperparameters were: ")
    # print("Mini Batch Size: {}, epoch: {}, epsilon: {}, alpha: {}".format(hyper_param_grid[smallMSE][0], hyper_param_grid[smallMSE][1], hyper_param_grid[smallMSE][2], hyper_param_grid[smallMSE][3]))
    # print("--------------------------------------------------------")

    # # Report fMSE cost on the training
    # fmse_te = mse(hyper_param_grid[smallMSE][4], X_te, yte)
    # print("The MSE for testing set is " + str(fmse_te))


def main():
    train_number_classifier()

if __name__ == '__main__':
    main()
