import numpy as np
from PIL import Image
from activations.py import ReLU, Tanh
from losses.py import mean_squared_error

PATH = PATH
EPOCHS = EPOCHS # Number of Iterations
BATCH_SIZE = BATCH_SIZE # Number of samples that are gonna be passed through the Neural Network on each epoch
LR = LEARNING_RATE # Constant that will multiply the weights derivative. Preferably a number below 1 in order to avoid exploding gradients...be careful with vanishing ones
LAYER1 = LAYER1 # Size of first layer, AKA number of neurons
LAYER2 = LAYER2 # idem for second layer
LAYER3 = LAYER3 # and so on...
OUTPUT = OUTPUT # This one can vary. Do you want a binary classifier? Use 1. Multi-class classifier? Then use the number of labels. Image? (BATCHxHEIGHTxWIDTHxCHANNELS)

image = Image.open(PATH)
image = image.resize((100, 100))
image = image.convert("RGB")
image = np.array(image)
image = image/127.5 - 1.0 # Remember to normalize your data. Your neural network might not work if you don't. Here, the pixel values are in the interval [-1, 1]

# In order to divide the data into batches, we're gonna need a function for this. If you don't want to use batches, then consider batch=1 and skip this function.
def DataLoader(data, batch_size):
    for batch in range(0, len(data), batch_size):
        yield data[batch:min(batch+batch_size, len(data))]

# With Linear(Dense) layers, we have to flatten the data, in order to properly make its operations(aka: matrix multiplication)
INPUT = BATCH_SIZE * image.shape[1] * image.shape[2] * image.shape[3] # Since we're using an image, INPUT = BATCH x HEIGHT x WIDTH x CHANNELS
# If you get IndexError: tuple index out of range, then your input isn't in the shape (N_SAMPLES, HEIGHT, WIDTH, CHANNELS). I suggest preprocessing it.

# Generating the weights and bias. The weights will be multiplying the layer's input, and this multiplication will be summed to the bias. The bias can be deleted
# in some cases, sticking to the weights only.
# Also, initiating the weights through normal distribution around 0, since this method seems more common to GANs.
# For the bias we can just initiate with zeros.
w1 = np.random.normal(loc=0, scale=0.01, size=(INPUT, LAYER1)) # Remember that, the greater the numbers, the more computation power will be needed.
b1 = np.zeros(LAYER1)

w2 = np.random.normal(0, 0.01, (LAYER1, LAYER2))
b2 = np.zeros(LAYER2)

w3 = np.random.normal(0, 0.01, (LAYER2, LAYER3))
b3 = np.zeros(LAYER3)

w_out = np.random.normal(0, 0.01, (LAYER3, OUTPUT))
b_out = np.zeros(OUTPUT)

# Let's begin the training loop. You can create a function with this, but I think it's more didactical to make it like this

for epoch in range(EPOCHS):
    input = next(DataLoader(image, BATCH_SIZE)) # This is the actual input
    input = input.flatten() # Remember that when we're dealing with Linear layers we have to flatten our data.
    
    # And this is where the fun begins.
    # Remember: the output of a Linear Layer is out = (input * weights) + bias.
    
    l1 = np.matmul(input, w1) + b1 # Many tutorials use np.dot(), but, sincerely, this is pratically the same thing(if not more effective) in this case.
    l1_out, dact1 = ReLU(l1) # The output of an activation layer is out = act(input). In this case, out = ReLU(input)
    
    l2 = np.matmul(l1_out, w2) + b2 # The input of the second layer is the output of the first one, and so forth.
    l2_out, dact2 = ReLU(l2) # Feel free to test other activation functions...I would avoid using softmax before the actual output, though.
    
    l3 = np.matmul(l2_out, w3) + b3
    l3_out, dact3 = ReLU(l3)
    
    out = np.matmul(l3_out, w_out) + b_out
    output, dactout = Tanh(out) # Tanh is a good function to get values between -1 and 1, a normalized image.
    
    # The loss function and its derivative.
    # My idea here is to simply decompose and then recompose my image, so my labels are my inputs. Maybe this could get me to a SuperResolution Model...
    loss, dloss = mean_squared_error(output, input) # Remember: The loss here is just for us. Our dear NN will actually be using the derivative.
    
    # Now this is where things get messy. Beginning backpropagation + optimization through Stochastic Gradient Descent.
    '''
    Using the chain rule
    dloss/dw_out = dloss/doutput * doutput/dout * dout/dw_out

    loss = (output-labels)**2 ---> dloss/doutput = 2(output-labels)
    output = act(out) ----> doutput/dout = act'(out) = dtanh(out)
    out = w_out * l3_out + b_out ----> dout/dwout = l3_out
    
    dloss/dwtout = dloss(output, input) * dtanh(out) * l3_out
    dloss/dwout = dloss * dactout * l3_out
    '''
    A = dloss * dactout # Applying derivative of activation to the loss derivative. This part MUST be done this way. Otherwise you'll have problems with shapes.
    
    dw_out = np.matmul(A.T, l3_out) # A.shape is (BATCH_SIZE, OUTPUT), while l3_out.shape is (100, OUTPUT), so transposition is required.
    
    # We're gonna use the w_out later, so let's apply the optimization last.
    '''
    dloss/dbout = dloss/doutput * doutput/dout * dout/dbout

    out = wout * l3_out + bout ---> dout/dbout = 1
    '''
    db_out = A * 1
    
    # Optimizing bias later also for claryfication
    
    '''
    Chain rule on the chain rule
    dloss/dw3 = dloss/doutput * doutput/dout * dout/dl3_out * dl3_out/dl3 * dl3/dw3

    out = wout * l3_out + bout ---> dout/dl3_out = wout
    l3_out = act(l3) ---> dl3_out/dl3 = act'(l3) = dReLU(l3)
    l3 = w3 * l2_out + b3 ---> dl3/dw3 = l2_out

    dloss/dw3 = [dloss(output, input) * dtanh(out)] * w_out * dReLU(l3) * l2_out
    dloss/dw3 = [dloss * dactout] * w_out * dact3 * l2_out
    dloss/dw3 = A * wout * dact3 * l2_out
    '''
    B = np.matmul(A, w_out.T) * dact3 # Same case as before. Also attention for A.shape and w_out.shape.
    
    dw3 = np.matmul(B.T, l2_out)
    
    '''
    dloss/db3 = dloss/doutput * doutput/dout * dout/dl3_out * dl3_out/dl3 * dl3/db3

    l3 = w3 * l2_out + b3 ---> dl3/db3 = 1
    '''
    db3 = B * 1
    
    '''
    dloss/dw2 = [dloss/doutput * doutput/dout * dout/dl3_out * dl3_out/dl3] * dl3/dl2_out * dl2_out/dl2 * dl2/dw2
    dloss/dw2 = [dloss * dactout * w_out * dact3] * w3 * dReLU(l2) * l1_out

    dloss/dw2 = B * w3 * dact2 * l1_out
    '''
    C = np.matmul(B, w3.T) * dact2
    
    dw2 = np.matmul(C.T, l1_out)
    
    db2 = C
    
    '''
    dloss/dw1 = [B * dl3/dl2_out * dl2_out/dl2] * dl2/dl1_out * dl1_out/dl1 * dl1/dw1
    dloss/dw1 = [B * w3 * dact2] * w2 * dact1 * input
    dloss/dw1 = C * w2 * dact1 * input
    '''
    D = np.matmul(C, w2.T) * dact1
    
    dw1 = np.matmul(D.T, input)
    
    db1 = D
    
    # Finally, the optimization itself. Remember: dw > 0 ---> we're on the right side of the optimized weight(Graphic Weights(x) x Loss(y)),
    # So we have to subtract the dw from the weight. That way, if dw > 0, the new weight will be lower than the old one.
    
    w_out = w_out - LR * dw_out.T # Transpose is required
    b_out = b_out - LR * db_out # But not here. Don't use transpose here. We need bias.shape = (BATCH, layer_output)
    
    w3 = w3 - LR * dw3.T
    b3 = b3 - LR * db3
    
    w2 = w2 - LR * dw2.T
    b2 = b2 - LR * db2
    
    w1 = w1 - LR * dw1.T
    b1 = b1 - LR * db1

    if epoch+1 % 100 == 0:
        print(f"Epoch: {epoch}\nLoss: {loss}")
        
        
        
# Since we're working with an image, let's see the result

output = np.reshape(output, (BATCH_SIZE, 100, 100, 3))
print(output[0])
output[0] = (output[0] + 1.0)*127.5
output = Image.fromarray(output[0].astype(np.uint8))
print(output.show())


    
# Simply sticking to the linear layer is too meh. Any mediocre tutorial does this. How about some Conv2D?

kernel = np.random.normal(0, 0.01, (3,3))

# Each kernel value is a weight...we already know how to perform Stochastic Gradient Descent with a single value, and we can easily grab one
# of those weights from the kernel, so...

def Conv2D(data, kernel, padding=0, strides=1, out_channels=1, activation=None):
    input = next(DataLoader(data, BATCH_SIZE))
    kernel = np.flipud(np.fliplr(kernel)) # Cross-correlation
    xi, yi = input.shape[1], input.shape[2]
    xk, yk = kernel.shape[0], kernel.shape[1]
    
    xout = (xi - xk + 2*padding)/strides + 1.0
    xout = int(xout)

    yout = (yi - yk + 2*padding)/strides + 1.0
    yout = int(yout)

    output = np.zeros((xout, yout))

    # Remember: A TransposedConv is simply a very padded input + normal Conv

    if padding != 0:
        for sample in range(input.shape[0]):
            input[sample] = np.pad(input[sample], (padding, padding))
            xi, yi = input.shape[1], input.shape[2]

    for y in range(yi):
        if y > yi-yk:
            break
        if y % strides == 0:
            for x in range(xi):
                if x > xi-xk:
                    break

                try:
                    if x % strides == 0:
                        output[x,y] = (kernel * input[x:x+xk, y:y+yk]).sum()

                    if activation:
                        output[x,y], dact = activation(output[x,y])

                except:
                    break

    return output, dact # Remember to repeat this conv over each channel of your input

# Backprop

output, dact = Conv2D(image, kernel, padding=1, activation=ReLU())

# Need some help with the calculations? Here, take this:

def conv2out(input_shape, kernel, stride, padding):
    x = 2*padding
    y = 1*(kernel-1)
    z = (input + x - y - 1)/stride

    output = z + 1
    return output

dcost = dloss(output, image)

def Conv2Dbackward():
    for y in range(kernel.shape[1]):
        for x in range(kernel.shape[0]):
            weight = kernel[x,y]

            K = dcost * dact
            dwout = np.matmul(K.T, image)

            weight = weight - lr * dwout.T
