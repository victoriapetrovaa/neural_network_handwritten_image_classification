#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSE 5052 Final Project

@author: vpetrova
"""

#%% Stage setting: Import libraries, seed the random number generator, etc
import scipy.io
import numpy as np
from matplotlib import pyplot as plt

np.random.seed(2)


#%%  Define a function that "knows" about the format of the input data file:

## Function import_data() takes a string argument and returns a pair of 
## arrays RAW_DATA_X and RAW_DATA_Y:
def import_data(filename):
    """Import input patterns and their corresponding labels from a .MAT file."""
    data = scipy.io.loadmat(filename)
    
    X = np.array(data["x"])
    Y = np.array(data["y"][0])     
    Y[Y==-1] = 0    # recode the negative labels as zeroes
    
    # Reshape Y from (3000,) to (1, 3000)
    Y = Y.reshape((1, len(Y)))
    
    return X, Y


#%% Perform the import itself.

# Import the data assigned for this project:
RAW_DATA_X, RAW_DATA_Y = import_data("DATA_mnist_49_3000")

# Extract size-related information:
LEN_INPUT_VECT, OVERALL_N_SAMPLES = RAW_DATA_X.shape  # values extracted from tuple

assert RAW_DATA_Y.shape[1] == OVERALL_N_SAMPLES

# Each row of RAW_DATA_X is a linearized image. The images are square-shaped.
# Therefore, LEN_INPUT_VECT should be an exact square.
IMAGE_SIZE = round(np.sqrt(LEN_INPUT_VECT))     # IMAGE_SIZE is of type int

assert LEN_INPUT_VECT == IMAGE_SIZE**2


#%%  Define a function that converts a vector to image:
def vector_to_image(vect):
    '''Reshape a vector to a square matrix.'''
    return vect.reshape(IMAGE_SIZE,IMAGE_SIZE)


#%% Plot the first training image
sample_image = vector_to_image(RAW_DATA_X[:,0])
plt.imshow(sample_image, interpolation="nearest")
plt.show()


#%% Divide the dataset into training and test sets
TRAINING_SAMPLE_SIZE = 2000

assert TRAINING_SAMPLE_SIZE <= OVERALL_N_SAMPLES

TEST_SAMPLE_SIZE = OVERALL_N_SAMPLES - TRAINING_SAMPLE_SIZE    # == 1000

trainX = RAW_DATA_X[:, 0:TRAINING_SAMPLE_SIZE]
trainY = RAW_DATA_Y[:, 0:TRAINING_SAMPLE_SIZE]

assert trainX.shape[0] == LEN_INPUT_VECT
assert trainX.shape[1] == TRAINING_SAMPLE_SIZE
assert trainY.shape[1] == TRAINING_SAMPLE_SIZE

testX = RAW_DATA_X[:, TRAINING_SAMPLE_SIZE: OVERALL_N_SAMPLES] #starts at index 2000
testY = RAW_DATA_Y[:, TRAINING_SAMPLE_SIZE: OVERALL_N_SAMPLES]

assert testX.shape[0] == LEN_INPUT_VECT
assert testX.shape[1] == TEST_SAMPLE_SIZE
assert testY.shape[1] == TEST_SAMPLE_SIZE


#%% Make sure there is a near-equal distribution of 4s and 9s in either set
## Straightforward way of doing this is calculating the average proportion of 
## the labels (because they are 0s and 1s).
assert abs(np.average(trainY) - 0.5) <= 0.10    # 10 % tolerance
assert abs(np.average(testY) - 0.5) <= 0.10


#%% Define the network architecture
N_HIDDEN = 200   # number of neurons on hidden layer; Their activation values are denoted Z[]
N_INPUT = LEN_INPUT_VECT
N_OUTPUT = 1     # by default
LEARNING_RATE = 0.05

## There are two classes of variables -- weights and activations.
## We use global variables for the weights so that they can be used each iteration.
## By contrast, the activation variables will be local variables inside functions.
##
## There are four weight variables. All of them are initialized to small 
## random values drawn from a normal distribution like so:
global W, bH, U, bO
    
RANDN_SIGMA = 0.01

## The input-to-hidden connections have weights W. W is a matrix.   
W = np.random.randn(N_HIDDEN, N_INPUT) * RANDN_SIGMA

## The bias of the hidden units is bH (short for "bias_hidden")
## The array dimension (axis==1) will be "broadcast" automatically as needed.
bH = np.random.randn(N_HIDDEN, 1) * RANDN_SIGMA

## The hidden-to-ouput connections are denoted U. U is a matrix.
U = np.random.randn(N_OUTPUT, N_HIDDEN) * RANDN_SIGMA

## The bias of the single output unit is bO (short for "bias_output")
## The array dimension (axis==1) will be "broadcast" automatically as needed.
bO = np.random.randn(N_OUTPUT, 1) * RANDN_SIGMA


#%% Define a function that re-initializes the weights. A useful tool for
## interactive experimentation with the model (eg. to optimize n_iterations).
def reinitialize_weights():
    '''Set the global weight matrices to small random values.'''
    global W, bH, U, bO
    W = np.random.randn(N_HIDDEN, N_INPUT) * RANDN_SIGMA
    bH = np.random.randn(N_HIDDEN, 1) * RANDN_SIGMA
    U = np.random.randn(N_OUTPUT, N_HIDDEN) * RANDN_SIGMA
    bO = np.random.randn(N_OUTPUT, 1) * RANDN_SIGMA
    

#%% Define a function that updates the global weights variables using the negative gradient.
def update_weights(dW = 0.0, dbH = 0.0, dU = 0.0, dbO = 0.0, learning_rate = LEARNING_RATE):
    '''Decrement the global variables W, bH, U, and bO.'''
    global W, bH, U, bO
    W  -= dW  * learning_rate
    bH -= dbH * learning_rate
    U  -= dU  * learning_rate
    bO -= dbO * learning_rate


#%% Activation functions and their derivatives
def act_func_hidden(WX_bH):
    """Activation function of the hidden-layer units."""
    return np.tanh(WX_bH)

def deriv_act_func_hidden(arg):
    """The derivative of the activation function of the hidden-layer units."""
    return 1 - arg**2     # works SPECIFICALLY for tanh

def act_func_output(UZ_bO): 
    """Activation function of the output-layer units."""
    return 1/(1+np.exp(-UZ_bO))


#%% Define a function that does one forward and backward pass across the network,
## updating the weight variables, and returning Y_hat.
## Notational conventions:
    # The activations across the input layer are denoted X.
    #
    # The linear part at the hidden layer is WX_bH = W.dot(X) + bH
    # The activations of the hidden units are denoted Z. 
    #
    # The linear part at the output is UZ_bO = U.dot(Z) + bO
    # The activation of the output unit is denoted Y.

def run_network_once(X, Y, training_mode = True):
    '''Do the forward pass. Do backward pass only if training_mode == True.'''
    # The activations across the input layer are denoted X.
    # The correct classification labels for the output layer are denoted Y.

    # The weight matrices are stored in these global variables:
    global W, bH, U, bO
    
    # The number of input patterns processed in parallel (batch mode):
    sample_size = X.shape[1]
    assert Y.shape[1] == sample_size
    
    #### FORWARD PASS: ####
    # Propagate activation from input to hidden: 
    # In other words, we are processing ALL input patterns simultaneously in parallel.
    # The linear part at the hidden layer is WX_bH = W.dot(X) + bH
    # WX_bH.shape == (N_HIDDEN, sample_size)
    WX_bH = W.dot(X) + bH
        
    # Apply the activation function at the hidden layer:
    # The activations of the hidden units are denoted Z. 
    # Z.shape == WX_bH.shape == (N_HIDDEN, sample_size)
    Z = act_func_hidden(WX_bH)
    
    # Propagate activation from hidden to output:
    # The linear part at the output is UZ_bO = U.dot(Z) + bO
    # UZ_bO.shape == (N_OUTPUT, sample_size)
    UZ_bO = U.dot(Z) + bO

    # Apply the activation function at the output:
    # The activation of the output unit is denoted Y_hat.
    # Y_hat.shape == UZ_bO.shape == (N_OUTPUT, sample_size)
    Y_hat = act_func_output(UZ_bO)
    
    #### BACKWARD PASS: ####
    if (training_mode):
        # Calculate the error at the output layer.
        # delta_output.shape == (N_OUTPUT, sample_size)
        delta_output = Y_hat - Y 
        
        # The changes in the bias parameters at the output units are the 
        # average delta_output across all training samples.
        # dbO.shape == bO.shape == (N_OUTPUT, 1)
        # The sum operator forces the second dimension to become 1.
        dbO = (np.sum(delta_output, axis = 1, keepdims = True)) / sample_size
        
        # The changes in the weights from hidden to output.
        # dU.shape == U.shape == (N_OUTPUT, N_HIDDEN)
        dU = (np.dot(delta_output, Z.T)) / sample_size
        
        # Calculate the training signal for the hidden units. This back 
        # error propogation is based on the chain rule of derivation.
        # delta_hidden.shape == Z.shape == (N_HIDDEN, sample_size)
        delta_hidden = np.dot(U.T, delta_output) * deriv_act_func_hidden(Z)
        
        # The changes in the bias parameters at the hidden units are the 
        # average delta_hidden across all training samples.
        # dbH.shape == bH.shape == (N_HIDDEN, 1)
        dbH = (np.sum(delta_hidden, axis = 1, keepdims = True)) / sample_size
        
        # The changes in the weights from input to hidden.
        # dW.shape == W.shape == (N_HIDDEN, N_INPUT)
        dW = (np.dot(delta_hidden, X.T)) / sample_size
        
        # Update the weights -- side effects on certain global variables:
        update_weights(dW = dW, dbH = dbH, dU = dU, dbO = dbO)
    
    #else training_mode == False
    # do nothing -- skip the backward pass altogether
    
    # Return network's own activations. They are useful for calculating the 
    # loss function.
    return Y_hat


    
#%% Define the cross entropy cost function.
## Let's use the cross-entropy loss function (rather than summed-squared-error). This is the
## appropriate loss for *binary* outputs. We train the network to predict the *probability* of
## having classification label==1.
def cost_func(Y_correct, Y_hat):
    '''Cross entropy cost function. Arguments must have the same number of elements.'''
    Yc = np.reshape(Y_correct, -1)   # Flatten to a 1D vector of shape == (n,)
    Yh = np.reshape(Y_hat, -1)       # Ditto

    n = len(Yc)
    assert len(Yh) == n
    
    cost = (-1.0/n) * (np.dot(Yc.T, np.log(Yh)) + np.dot((1-Yc).T, np.log(1-Yh)))
    return cost  # / -np.log(0.5) for max possible cost for small random weights




#%%  Main training loop:
def train_network(trainX, trainY, n_iterations = 1500, print_interval = 250):
    '''Call run_network_once n_iterations number of times. Return a vector of cost values.'''
    # Allocate memory for the return value of cost as a function of time, as
    # calculated by cost_func().
    learning_curve = np.zeros(n_iterations)
    
    # Run the network multiple times in a loop.
    # The changes in the weights accumulate incrementally.
    for i in range(n_iterations):
        Y_hat = run_network_once(trainX, trainY, training_mode = True)
        cost = cost_func(trainY, Y_hat)
        learning_curve[i] = cost
        
        # Print on console once-in-a-while in case loop duration is long.
        if (i % print_interval == 0):
            print('Cost at time {} is {}'.format(i, cost))
            
    print('Cost at end of training (time {}) is {}'.format(n_iterations, cost))  

    return learning_curve


    
#%% Call train_network() for the first time only

#reinitialize_weights()  # uncomment this line to start over

# This is the time-consuming part of the job:
learning_curve = train_network(trainX, trainY, n_iterations = 1000)


#%% To train some more, keep running this cell
learning_curve2 = train_network(trainX, trainY, n_iterations = 500)
learning_curve = np.concatenate((learning_curve, learning_curve2))


#%% Plot the learning_curve:
plt.plot(learning_curve)    
print('\nCost at end of training is {}'.format(learning_curve[-1]))
    

#%% Testing the network after the weights (global variables) have been trained:
Y_hat_test = run_network_once(testX, testY, training_mode = False)

test_cost = cost_func(testY, Y_hat_test)   
print('\nTest cost is {}'.format(test_cost)) 


#%% ########### ########## ########### ########### ##########
## 
## Until now the code was generic in the sense that N_OUTPUT could be >1, etc.
## From this point onward, we will depend on the particular details of the
## given data set:
    
assert N_OUTPUT == 1
assert all(np.unique(RAW_DATA_Y) == [0, 1])   # all labels are 0s or 1s


#%% Convert the non-binary network output (Y_hat_test) to binary networkY
## Examples:  0.012307 --> 0 ; 0.99643 --> 1
networkY = (np.sign(Y_hat_test - 0.5) + 1) / 2
networkY = networkY.astype(type(testY[0,0]))         # int16 for both


#%% Divide the test examples into 4 classes:
    
# Both testY and networkY are 0s --> the network is correct:
matches = (testY == 0) & (networkY == 0)
ind_t0_n0 = np.flatnonzero(matches)
print('There are {} cases of correct classification of a 0 as a 0.'.format(len(ind_t0_n0)))

# Error: the true label is 0, but the network classified it as 1:
matches = (testY == 0) & (networkY == 1)
ind_t0_n1 = np.flatnonzero(matches)
print('There are {} cases of incorrect classification of a 0 as a 1.'.format(len(ind_t0_n1)))

# Error: the true label is 1, but the network classified it as 0:
matches = (testY == 1) & (networkY == 0)
ind_t1_n0 = np.flatnonzero(matches)
print('There are {} cases of incorrect classification of a 1 as a 0.'.format(len(ind_t1_n0)))

# Both testY and networkY are 1s --> the network is correct:
matches = (testY == 1) & (networkY == 1)
ind_t1_n1 = np.flatnonzero(matches)
print('There are {} cases of correct classification of a 1 as a 1.'.format(len(ind_t1_n1)))

del matches   # clean up the workspace


#%% Calculate overall error rate of the model.
error_rate = (len(ind_t0_n1) + len(ind_t1_n0)) / TEST_SAMPLE_SIZE
print('\nThe overall error rate is {}.'.format(error_rate))


#%% Tools for plotting some images...
def plot_chosen_image(image_index, X = testX):
    """Plot the square image stored under X[:, image_index]."""
    chosen_image = vector_to_image(X[:,image_index])
    plt.imshow(chosen_image, interpolation="nearest")
    plt.show()


def plot_multiple_images(index_vector, how_many = 10, X = testX):
    """Plot the first few images with indices from the given vector."""
    how_many = min(how_many, len(index_vector))
    for i in range(how_many):
        plot_chosen_image(index_vector[i], X)


#%% Plot the first 10 instances of the first kind of mistake:
# Images that are supposed to be digit 4s, but were misclassified as 9s:
plot_multiple_images(ind_t0_n1, how_many = 10)


#%% Plot the first 20 instances of the second kind of mistake:
# Images that are supposed to be digit 9s, but were misclassified as 4s:
plot_multiple_images(ind_t1_n0, how_many = 20)


