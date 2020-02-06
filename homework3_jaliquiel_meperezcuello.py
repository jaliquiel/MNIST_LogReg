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


def SGD(X_tilde, ytr, batch_size, epochs, epsilon, alpha):

    # randomize training set     
    permute = np.random.permutation(X_tilde.shape[1])
    shuffled_X  = X_tilde.T[permute].T #(784, 55000)
    shuffled_y = ytr.T[permute].T #(10,55000)

    sample_size = X_tilde.shape[1] # total batch size

    # get all indexes based on batch size
    rounds = get_indexes(sample_size, batch_size) # list of (start,end) for slicing each batch X

    # initialize weights to random values with an standard deviation of 0.01
    weights = np.random.rand(785,10) * 0.01 

    # start iteration loop
    for epoch in range(epochs):
        for indexes in rounds:
            start, finish = indexes
            gradient =  grad_CE(weights, shuffled_X[:,start:finish], shuffled_y[:,start:finish], alpha)
            weights = weights - epsilon * gradient
    return weights


# Calculate 
def softmax(weights, Xtilde):
    z = np.dot(weights.T, Xtilde) # (55000x10)
    yhat = np.exp(z) / np.sum(np.exp(z), axis=0) # axis=0 means sum rows (55000)
    return yhat # (10,55000)

# Calculate the gradient of cross entropy loss
def grad_CE(weights, Xtilde, y, alpha):
    yhat = softmax(weights,Xtilde)
    distance = yhat - y
    n = y.shape[0] # 55000

    # This version, simply modifies the last index of the weight array and coverts it into 0
    # this is done to not penalize the bias. This code is an alternative to the identity matrix method performed bellow
    # wReg = np.copy(weights)
    # wReg[-1] = 0
    # regularization = (alpha / n) * wReg #(2305,)
    identity_matrix =np.diag( np.append(np.ones(Xtilde.shape[0]-1), 0)) ## <-- dimension is 2305 x 2305 where last element must be 0 which means the bias is not included 
    regularization = alpha / n * (np.transpose(weights).dot(identity_matrix))

    gradient = 1/n * np.dot(Xtilde,distance.T) + regularization.T
    return gradient

# Calculate Cross Entropy without regularization 
def CE(yhat, y):
    # vectorize formula
    ce = y * np.log(yhat) # (10,5000)
    verticalSum = np.sum(ce, axis=0) # (5000)
    celoss = np.mean(verticalSum) * -1
    return celoss

# Percent of correctly classified images
def PC (yhat, y):
    # https://stackoverflow.com/questions/20295046/numpy-change-max-in-each-row-to-1-all-other-numbers-to-0
    yhat_bool = (yhat.T == yhat.T.max(axis=1)[:, None]).astype(int).T # make probabilities into 1 and 0s
    num_of_classes = y.shape[0]
    similar = np.equal(y,yhat_bool)
    sum = np.sum(similar, axis=0)
    ones = np.ones(sum.shape[0])
    divide_by_classes = sum / num_of_classes
    correctness = np.equal(divide_by_classes, ones)
    accuracy = np.mean(correctness)
    return accuracy


def train_number_classifier ():
    # Load data and append bias
    X_tr = append_bias(np.load("mnist_train_images.npy").T)  # (784, 55000)
    y_tr = np.load("mnist_train_labels.npy").T # (10, 55000)
    X_val = append_bias(np.load("mnist_validation_images.npy").T) # (784, 5000)
    y_val = np.load("mnist_validation_labels.npy").T # (10, 5000)   
    X_te = append_bias(np.load("mnist_test_images.npy").T)
    y_te = np.load("mnist_test_labels.npy").T
    

    # Hyper parameters 
    mini_batch_sizes = [100, 500, 1000, 2000] # mini batch sizes
    epochs = [1, 2,3,4] # number of epochs
    epsilons = [0.1, 3e-3, 1e-3, 3e-5] # learning rates
    alphas = [0.1, 0.01, 0.05, 0.001] # regularization alpha

    # key: [int] mse
    # value: tuple of hyperparameters (nTilde, epoch, epsilon, alpha, weights, pcVal)
    # Dictionary to store our all the different hyperparameter sets, their weights and their MSE
    hyper_param_grid = {}
    count = 0

    # train weights based on all the different sets of hyperparameters
    for mini_batch_size in mini_batch_sizes:
        for epoch in epochs:
            for epsilon in epsilons:
                for alpha in alphas:
                        weights = SGD(X_tr, y_tr, mini_batch_size, epoch, epsilon, alpha)
                        yhat = softmax(weights, X_val)

                        # calculate the CE and PC with the validation set
                        ceVal = CE(yhat, y_val)
                        pcVal = PC(yhat, y_val)

                        count += 1
                        print("The CE for [" + str(count) + "] validation set is " + str(ceVal))
                        print("The PC for [" + str(count) + "] validation set is " + str(pcVal) + "correct")

                        # add to dictionary
                        hyper_param_grid[ceVal] = (mini_batch_size, epoch, epsilon, alpha, np.copy(weights), pcVal) 
                        print("miniBatch: {}, epoch: {}, epsilon: {}, alpha: {}".format(mini_batch_size,epoch,epsilon,alpha))


    # get key of dictionary with smallest MSE
    smallCE = min(hyper_param_grid.keys())

    # # Report CE cost on the training
    # # print("--------------------------------------------------------")
    print("The CE for training set is " + str(smallCE))
    print("The PC for training set is " + str(hyper_param_grid[smallCE][5]) + " correct")
    # print("--------------------------------------------------------")

    # show the best hyperparameters
    print("My best hyperparameters were: ")
    print("Mini Batch Size: {}, epoch: {}, epsilon: {}, alpha: {}".format(hyper_param_grid[smallCE][0], hyper_param_grid[smallCE][1], hyper_param_grid[smallCE][2], hyper_param_grid[smallCE][3]))
    # print("--------------------------------------------------------")

    # # Report CE cost on the training
    best_yhat = softmax(hyper_param_grid[smallCE][4], X_te)
    ce_te = CE(best_yhat, y_te)
    pc_te = PC(best_yhat, y_te)
    print("The CE for test set is " + str(ce_te))
    print("The PC for test set is " + str(pc_te) + "% correct")


def main():
    train_number_classifier()

if __name__ == '__main__':
    main()
