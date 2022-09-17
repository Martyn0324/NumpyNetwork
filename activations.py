import numpy as np

def ReLU(input):
    '''
    The simplest one, stills quite popular, even among its variations(LeakyReLU).
    Returns 0 for everything equal or below 0. Else, returns the same number.
    Its derivative is even more simple: Returns 0 for everything equal or below 0, else, returns 1.
    '''
    relu = np.maximum(input, 0)
    
    drelu = np.ones(input.shape) * (x > 0)
    
    return relu, drelu
  
def Sigmoid(input):
    '''
    This one is used specially to generate outputs for binary classification problems. It will return a value between 0 and 1, a probability, but each output
    probability is independent from each other.
    However, I really believe that one could use Sigmoid instead of Softmax without big annoyances, just don't use one-hot encoding and maybe it'll go fine.
    '''
    
    sig = 1/(1+np.exp(-input))
    dsig = sig * (1 - sig)
    
    return sig, dsig
  
def Softmax(input):
    '''
    This one is used specially to generate outputs for multi classification. It will generate an array where the sum of its elements will be equal to 1, so
    remember to use one-hot encoding. Each element probability is dependent from each other(if an element has higher probability, other elements will have lower ones)
    Also appreciated in Reinforcement Learning.
    '''
    
    input = input - np.max(input)
    soft = (np.exp(input)/np.sum(np.exp(input), axis=0))
    dsoft = soft * (1-soft) # As far as I know and researched about derivatives, softmax's derivative should be equal itself, but...ok.
    
    return soft, dsoft
  
def Tanh(input):
    '''
    Classic and appreciated in GANs.
    '''
    tanh = np.tanh(input)
    dtanh = 1-(input**2)
    
    return tanh, dtanh
    
