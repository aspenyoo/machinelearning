import pandas as pd
import logging
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split


###################################
#### setting feature range [0 1]
def feature_normalization(train, test):
    """Rescale the data so that each feature in the training set is in
    the interval [0,1], and apply the same transformations to the test
    set, using the statistics computed on the training set.

    Args:
        train - training set, a 2D numpy array of size (num_instances, num_features)
        test  - test set, a 2D numpy array of size (num_instances, num_features)
    Returns:
        train_normalized - training set after normalization
        test_normalized  - test set after normalization

    """
    minn = np.min(train, axis=0)
    maxx = np.max(train, axis=0)
    
    # discard constant feature dimensions
    idx = np.equal(minn,maxx)
    train_normalized = train[:,~idx]
    test_normalized = test[:,~idx]
    minn = minn[~idx]
    maxx = maxx[~idx]
    
    # normalize
    test_normalized = (test - minn) / (maxx - minn)
    train_normalized = (train - minn) / (maxx - minn)
    
    return train_normalized, test_normalized


########################################
#### The square loss function

def compute_square_loss(X, y, theta):
    """
    Given a set of X, y, theta, compute the square loss for predicting y with X*theta

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D array of size (num_features)

    Returns:
        loss - the square loss, scalar
    """
    
    loss = 0 #initialize the square_loss
    h = np.dot(X,theta) # prediction
    diff = h - y # error
    
    try:
        loss = np.dot(diff,diff) / len(diff) # average squared error
    except TypeError:
        loss = np.dot(diff,diff)

    return loss

########################################
### compute the gradient of square loss function
def compute_square_loss_gradient(X, y, theta):
    """
    Compute gradient of the square loss (as defined in compute_square_loss), at the point theta.

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)

    Returns:
        grad - gradient vector, 1D numpy array of size (num_features)
    """
    try:
        grad = 2*np.dot(np.transpose(X), np.dot(X,theta) - y)/len(y)
    except TypeError:
        grad = 2*np.dot(np.transpose(X), np.dot(X,theta) - y)

    return grad



###########################################
### Gradient Checker
#Getting the gradient calculation correct is often the trickiest part
#of any gradient-based optimization algorithm.  Fortunately, it's very
#easy to check that the gradient calculation is correct using the
#definition of gradient.
#See http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization
def grad_checker(X, y, theta, epsilon=0.01, tolerance=1e-4):
    """Implement Gradient Checker
    Check that the function compute_square_loss_gradient returns the
    correct gradient for the given X, y, and theta.

    Let d be the number of features. Here we numerically estimate the
    gradient by approximating the directional derivative in each of
    the d coordinate directions:
    (e_1 = (1,0,0,...,0), e_2 = (0,1,0,...,0), ..., e_d = (0,...,0,1)

    The approximation for the directional derivative of J at the point
    theta in the direction e_i is given by:
    ( J(theta + epsilon * e_i) - J(theta - epsilon * e_i) ) / (2*epsilon).

    We then look at the Euclidean distance between the gradient
    computed using this approximation and the gradient computed by
    compute_square_loss_gradient(X, y, theta).  If the Euclidean
    distance exceeds tolerance, we say the gradient is incorrect.

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)
        epsilon - the epsilon used in approximation
        tolerance - the tolerance error

    Return:
        A boolean value indicate whether the gradient is correct or not

    """
    true_gradient = compute_square_loss_gradient(X, y, theta) #the true gradient
    num_features = theta.shape[0]
    approx_grad = np.zeros(num_features) #Initialize the gradient we approximate
    #TODO

#################################################
### Generic Gradient Checker
def generic_gradient_checker(X, y, theta, objective_func, gradient_func, epsilon=0.01, tolerance=1e-4):
    """
    The functions takes objective_func and gradient_func as parameters. And check whether gradient_func(X, y, theta) returned
    the true gradient for objective_func(X, y, theta).
    Eg: In LSR, the objective_func = compute_square_loss, and gradient_func = compute_square_loss_gradient
    """
    #TODO


####################################
#### Batch Gradient Descent
def batch_grad_descent(X, y, alpha=0.1, num_iter=1000, check_gradient=False):
    """
    In this question you will implement batch gradient descent to
    minimize the square loss objective

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - step size in gradient descent
        num_iter - number of iterations to run
        check_gradient - a boolean value indicating whether checking the gradient when updating

    Returns:
        theta_hist - store the the history of parameter vector in iteration, 2D numpy array of size (num_iter+1, num_features)
                    for instance, theta in iteration 0 should be theta_hist[0], theta in ieration (num_iter) is theta_hist[-1]
        loss_hist - the history of objective function vector, 1D numpy array of size (num_iter+1)
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta_hist = np.zeros((num_iter+1, num_features))  # initialize theta_hist

    loss_hist = np.zeros(num_iter+1) # initialize loss_hist
    theta = np.ones(num_features) # initialize theta

    theta_hist[0] = theta
    loss_hist[0] = compute_square_loss(X,y,theta)
    for iterr in range(1,num_iter+1):
        
        grad = compute_square_loss_gradient(X,y,theta)
        theta = theta - alpha*grad

        theta_hist[iterr] = theta
        loss_hist[iterr] = compute_square_loss(X,y,theta)

    return theta_hist, loss_hist

####################################
###Q2.4b: Implement backtracking line search in batch_gradient_descent
###Check http://en.wikipedia.org/wiki/Backtracking_line_search for details
#TODO



###################################################
### Compute the gradient of Regularized Batch Gradient Descent
def compute_regularized_square_loss_gradient(X, y, theta, lambda_reg):
    """
    Compute the gradient of L2-regularized square loss function given X, y and theta

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)
        lambda_reg - the regularization coefficient

    Returns:
        grad - gradient vector, 1D numpy array of size (num_features)
    """
    try: 
        grad = 2*np.dot(np.transpose(X), np.dot(X,theta) - y)/len(y) + 2*lambda_reg*np.transpose(theta)
    except TypeError:
        grad = 2*np.dot(np.transpose(X), np.dot(X,theta) - y) + 2*lambda_reg*np.transpose(theta)
        
    return grad

###################################################
### Batch Gradient Descent with regularization term
def regularized_grad_descent(X, y, alpha=0.1, lambda_reg=1, num_iter=1000):
    """
    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - step size in gradient descent
        lambda_reg - the regularization coefficient
        numIter - number of iterations to run

    Returns:
        theta_hist - the history of parameter vector, 2D numpy array of size (num_iter+1, num_features)
        loss_hist - the history of loss function without the regularization term, 1D numpy array.
    """
    (num_instances, num_features) = X.shape
    theta = np.zeros(num_features) #Initialize theta
    theta_hist = np.zeros((num_iter+1, num_features))  #Initialize theta_hist
    loss_hist = np.zeros(num_iter+1) #Initialize loss_hist
    

    theta_hist[0] = theta
    loss_hist[0] = compute_square_loss(X,y,theta) + lambda_reg*np.dot(theta,theta)
    for iterr in range(1,num_iter+1):
        
        grad = compute_regularized_square_loss_gradient(X,y,theta,lambda_reg)
        theta = theta - alpha*grad

        theta_hist[iterr] = theta
        loss_hist[iterr] = compute_square_loss(X,y,theta)
        
    return theta_hist, loss_hist

#############################################
### Stochastic Gradient Descent
def stochastic_grad_descent(X, y, alpha=0.1, lambda_reg=1, num_iter=1000, t=1):
    """
    Stochastic gradient descent with a regularization term

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - string or float. step size in gradient descent
                NOTE: In SGD, it's not always a good idea to use a fixed step size. Usually it's set to 1/sqrt(t) or 1/t
                if alpha is a float, then the step size in every iteration is alpha.
                if alpha == "1/sqrt(t)", alpha = 1/sqrt(t)
                if alpha == "1/t", alpha = 1/t
        lambda_reg - the regularization coefficient
        num_iter - number of epochs (i.e number of times) to go through the whole training set

    Returns:
        theta_hist - the history of parameter vector, 3D numpy array of size (num_iter, num_instances, num_features)
        loss hist - the history of regularized loss function vector, 2D numpy array of size(num_iter, num_instances)
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta = np.ones(num_features) #Initialize theta
    theta_hist = np.zeros((num_iter, num_instances, num_features))  #Initialize theta_hist
    loss_hist = np.zeros((num_iter, num_instances)) #Initialize loss_hist
    
    concatdata = np.hstack((X,y.reshape(len(y),1))) # making matrix of X and y so rows are linked
    Xshuff = concatdata[:,:-1] # Xshuff is all but last column
    yshuff = concatdata[:,-1] # yshuf is last column
    
    for i_iter in range(num_iter):
        
        np.random.shuffle(concatdata)
        
        for i_inst in range(num_instances):

            theta_hist[i_iter,i_inst] = theta
            loss_hist[i_iter,i_inst] = compute_square_loss(Xshuff[i_inst],yshuff[i_inst],theta) + lambda_reg*np.dot(theta,theta)

            grad = compute_regularized_square_loss_gradient(Xshuff[i_inst],yshuff[i_inst],theta,lambda_reg)
            if isinstance(alpha, float):
                aa = alpha
            elif alpha == '1/sqrt(t)':
                aa = 1/np.sqrt(i_iter*num_iter+i_inst+t)
            elif alpha == '1/t':
                aa = 1/(i_iter*num_iter+i_inst+t)
                    
            theta = theta - aa*grad
            
    return theta_hist, loss_hist

###########################
#### load and tranform dataset
def main():
    #Loading the dataset
    print('loading the dataset')

    df = pd.read_csv('data.csv', delimiter=',')
    X = df.values[:,:-1]
    y = df.values[:,-1]

    print('Split into Train and Test')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =100, random_state=10)

    print("Scaling all to [0, 1]")
    X_train, X_test = feature_normalization(X_train, X_test)
    X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))  # Add bias term
    X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1)))) # Add bias term
          
    return X_train, X_test, y_train, y_test

#if __name__ == "__main__":
#    main()
