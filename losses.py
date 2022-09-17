import numpy as np

def mean_squared_error(predicted, labels):
    '''
    The simplest loss function. Easy to calculate, easy to take its derivative. There's also a modified version which uses the square root of the mse
    '''
    mse = np.sum((predicted-labels)**2) # From everything I've seen, this is more for visualization and see how our model is performing, so we can sum it.
    # Also, use keep_dims=True if you want to plot it.
    
    derivative = 2.0 * (output-labels) # Remember: f(x) = (x-k)² ---> f(x) = x² - 2kx + k² ==> df(x)/dx = 2x - 2k + 0 ---> f'(x) = 2(x-k)
    # Also, don't sum anything in the derivative.
    
    return mse, derivative
  

def binary_cross_entropy(predicted, labels):
    '''
    Used for binary classification models(0, 1 outputs). This one is the variation just like the one used by tensorflow, since it's more stable.
    '''
    bce = np.mean(np.sum(np.maximum(predicted, 0) - predicted*labels + np.log(1+ np.exp(- np.abs(predicted)))))
    
    derivative = np.mean(((1/(1+np.exp(- predicted))) - labels)) # Can't say anything anymore...I always hated log and exponential derivatives...
    
    return bce, derivative

def categorical_cross_entropy(predicted, labels):
    '''
    Hello again! This one is for...uh...categorical problems...when you have more than 2 labels.
    '''
    ce = -np.sum(labels * np.log(predicted + 10**-100))
    # Remember to perform one-hot encoding, so each sample in both labels and predicted is an array where the sum of its elements is equal to 1.0(100%)
    
    derivative = -y_true/(y_pred + 10**-100) # Well...this one is a bit easier...though I had to review the product rule.
    
    return ce, derivative
  
# Protip: be careful with means and sums in derivatives...or anything that can change its shape. Shapes are quite a pain when working with matrices...
