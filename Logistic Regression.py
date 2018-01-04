"""
@author: Shubham Shantaram Pawar

"""

import numpy as np
import matplotlib .pyplot as plt
from sklearn.datasets import load_iris

# function to plot the training data
def plotTrainingData(X, y):
    versicolor = np.where(y==0)
    verginica = np.where(y==1)
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('Scatter Plot of Training Data')
    ax.scatter(X[versicolor][:,0], X[versicolor][:,1], color='blue', label='versicolor', marker='o')
    ax.scatter(X[verginica][:,0], X[verginica][:,1], color='red', label='verginica', marker='+')
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('petal length (cm)')
    ax.set_ylabel('petal width (cm)')
    ax.legend()
    fig.set_size_inches(10, 6)
    fig.show()

# function to plot total cost vs iterations
def plotCostVsIterations(J_history, iterations):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('cost vs iterations')
    ax.set_xlabel(r'iterations')
    ax.set_ylabel(r'$J(\theta)$')
    ax.scatter(range(iterations), J_history, color='blue', s=10)
    fig.set_size_inches(8, 5)
    fig.show()
    
# function to calculate sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# function to campute cost and gradient
def computeCost(X, y, theta):
    m = len(y)
    
    z = np.dot(X, theta)
    h = sigmoid(z)
    
    # cost function
    J = (1 / m) * np.sum((-1 * np.multiply(y, np.log10(h))) - np.multiply((1 - y), np.log10(1 - h)))
    
    grad = (1 / m) * np.dot(X.T, (h - y))
    
    return J, grad

# function to perform gradient descent
def gradientDescent(X, y, theta, alpha, num_iters):
    
    J_history = np.zeros((num_iters,1))
    
    for iter in range(0, num_iters):
        
        J, grad = computeCost(X, y, theta)
        
        theta = np.subtract(theta, (alpha * grad))

        J_history[iter] = J
        
    return theta, J_history

# function to make a 100 folds of the data for leave-one-out analysis 
def leaveOneOut_split(X, y):
    k_folds = 100
    data_splits = []
    
    for i in range(k_folds):
        temp = []
        train_data = {}
        index = list(range(k_folds))
        index.pop(i)
        train_data['X'] = X[index]
        train_data['y'] = y[index]
        test_data = {}
        test_data['X'] = X[i]
        test_data['y'] = y[i]
        temp.append(train_data)
        temp.append(test_data)
        data_splits.append(temp)
    
    return data_splits

# function to perform leave-one-out analysis
def leaveOneOutAnalysis(X, y, alpha, iterations):
    total_error = 0
    data_splits = leaveOneOut_split(X, y)
    
    for i, data_split in enumerate(data_splits):
        
        print('\nTraining with fold ' + str(i+1) + '...')
        
        X_train = data_split[0]['X']
        y_train = data_split[0]['y']
        
        X_test = data_split[1]['X']
        y_test = data_split[1]['y'][0, 0]
        
        m, n = X_train.shape
        theta = np.zeros((n,1))
        
        theta, J_history = gradientDescent(X_train, y_train, theta, alpha, iterations)
        
        predict_probability = sigmoid(np.matmul(np.matrix(X_test), theta))
        
        # predicting class label for the test data
        if predict_probability >= 0.5:
            y_predict = 1.0
        else:
            y_predict = 0.0
        
        # comparing predicted class label with the test/actual class label
        # if not equal, increase total error by 1
        if y_predict != y_test:
            total_error += 1
            
    return total_error/100
        
def main():
    # loading iris dataset
    iris = load_iris()
    
    # selecting indices for samples corresponding to versicolor and virginica classes respectively
    versicolor_target = np.where(iris.target==1)
    virginica_target = np.where(iris.target==2)
    
    # extracting training dataset corresponding to versicolor and virginica classes
    X_train = iris.data[np.concatenate((versicolor_target[0], virginica_target[0]), axis = 0)][:, [2, 3]]
    y_train = iris.target[0:100]
    
    # ploting training data
    plotTrainingData(X_train, y_train)
    
    # number of iterations
    iterations = 10000
    
    # learning rate
    alpha = 0.1
    
    # min-max normalization/scaling
    X_train[:, 0] = (X_train[:, 0] - np.min(X_train[:, 0])) / (np.max(X_train[:, 0]) - np.min(X_train[:, 0]))
    X_train[:, 1] = (X_train[:, 1] - np.min(X_train[:, 1])) / (np.max(X_train[:, 1]) - np.min(X_train[:, 1]))
    
    m, n = X_train.shape
    
    # initializing parameters/thetas to zero
    theta = np.zeros((n+1,1))
    
    # adding one's for the bias term
    X = np.concatenate((np.ones((m,1)), X_train), axis=1)
    
    y = np.matrix(y_train).reshape(100,1)
    
    print('\nPerforming logistic regression on the entire dataset...')
    
    theta, J_history = gradientDescent(X, y, theta, alpha, iterations)
        
    print("\ntheta 0:", theta[0][0])
    print("theta 1:", theta[1][0])
    print("theta 2:", theta[2][0])
    
    # plotting cost vs iterations
    plotCostVsIterations(J_history, iterations)
    
    print('\nPerforming leave-one-out analysis...')

    # computing average error rate for the model using leave-one-out analysis
    avg_error = leaveOneOutAnalysis(X, y, alpha, iterations)

    print('\nThe average error rate for the logistic regression model after performing leave-one-out analysis is ' + str(avg_error) +'.')
    
if __name__ == '__main__':
    main()

