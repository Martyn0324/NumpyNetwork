import numpy as np
from PIL import Image
from activations.py import ReLU, Tanh
from losses.py import mean_squared_error

PATH = PATH
EPOCHS = 100 # Number of Iterations
BATCH_SIZE = 1 # Number of samples that are gonna be passed through the Neural Network on each epoch
LR = 0.01 # Constant that will multiply the weights derivative. Preferably a number below 1 in order to avoid exploding gradients...be careful with vanishing ones

image = Image.open(PATH)
image = image.resize((100, 100))
image = image.convert("RGB")
image = np.array(image)
image = image/127.5 - 1.0 # Remember to normalize your data. Your neural network might not work if you don't. Here, the pixel values are in the interval [-1, 1]

# In order to divide the data into batches, we're gonna need a function for this. If you don't want to use batches, then consider batch=1 and skip this function.
def DataLoader(data, batch_size):
    for batch in range(0, len(data), batch_size):
        yield data[batch:min(batch+batch_size, len(data))]

        
# There's no need for flatten here


# Generating the weights and bias. Here, the weights are gonna compose the kernels. Remember that each kernel is a matrix that will be multiplying
# the input in order to generate a feature map(hidden layer) or an output(output layer)
# I just don't know if it's better to pass 1 kernel for each channel, or 1 kernel for all channels per convolution.
# I suppose the first one might be more precise, but also computationally expensive and more prone to overfitting.

w11 = np.random.normal(loc=0, scale=0.01, size=(3,3)) # Remember that, the greater the numbers, the more computation power will be needed.
b11 = np.zeros((100,100)) # Careful with the bias. It must have the same shape as your Conv output since it'll be summed to the output.

w12 = np.random.normal(0, 0.01, (3, 3))
b12 = np.zeros((100, 100))
w13 = np.random.normal(0, 0.01, (3, 3))
b13 = np.zeros((100, 100))


# The convolution itself

def Conv2D(input, kernel, bias, padding=0, strides=1):
    kernel = np.flipud(np.fliplr(kernel)) # Cross-correlation
    xi, yi = input.shape[1], input.shape[2] # Keep in mind that input.shape[0] = BATCH_SIZE
    xk, yk = kernel.shape[0], kernel.shape[1]
    
    xout = (xi - xk + 2*padding)/strides + 1.0
    xout = int(xout)

    yout = (yi - yk + 2*padding)/strides + 1.0
    yout = int(yout)

    output = np.zeros((xout, yout))

    # Remember: A TransposedConv is simply a very padded input + normal Conv

    if padding != 0:
        input = np.pad(input, [(0,0), (padding, padding), (padding, padding)]) # Applying padding only to Height and Width, not batch neither channels.
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
                        output[x,y] = (kernel * input[x:x+xk, y:y+yk]).sum() + bias

                except:
                    break
    
    return output
  
# Need some help with the calculations? Here, take this:

def conv2out(input_shape, kernel, stride, padding):
    x = 2*padding
    y = 1*(kernel-1)
    z = (input_shape + x - y - 1)/stride

    output = z + 1
    return output
  

for epoch in range(EPOCHS):
    input = next(DataLoader(image, BATCH_SIZE))
    
    # And this is where the fun begins.
    
    out1 = Conv2D(input=input[:, :, :, 0], kernel=w11, bias=b11, padding=1, strides=1) # Conv2D in the Red channel of my RGB images batch
    output1, dact1 = Tanh(out1)
    out2 = Conv2D(input[:, :, :, 1], w12, b12, 1, 1) # Green Channel
    output2, dact2 = Tanh(out2)
    out3 = Conv2D(input[:, :, :, 2], w13, b13, 1, 1) # Blue Channel
    output3, dact3 = Tanh(out3)

    output = np.stack((out1, out2, out3), axis=-1) # RGB output
    
    loss, dloss = mean_squared_error(output, input)
    
    # Beginning backpropagation + optimization through Stochastic Gradient Descent.
    R = dloss[:, :, :, 0] * dact1 # (BATCH, HEIGHT, WIDTH, RED) * (BATCH, HEIGHT, WIDTH)

    print(R.shape)
    
    # The derivative of the Conv2D weights is the Conv2D of the same input, using a derivative as kernel. If your dw11 doesn't have the same
    # shape as your kernel, then you did something wrong.
    dw11 = Conv2D(input=input[:, :, :, 0], kernel=R, bias=np.zeros_like(R), padding=1, strides=1)

    G = dloss[:, :, :, 1] * dact2

    dw12 = Conv2D(input=input[:, :, :, 1], kernel=G, bias=np.zeros_like(G), padding=1, strides=1)
    db12 = G * 1

    B = dloss[:, :, :, 2] * dact3

    dw13 = Conv2D(input=input[:, :, :, 2], kernel=B, bias=np.zeros_like(B), padding=1, strides=1)
    db13 = B * 1
    '''
    Using the chain rule
    dloss/dw_out = dloss/doutput * doutput/dout * dout/dw_out

    loss = (output-labels)**2 ---> dloss/doutput = 2(output-labels)
    output = act(out) ----> doutput/dout = act'(out) = dtanh(out)
    out = w_out * l3_out + b_out ----> dout/dwout = l3_out
    
    dloss/dwtout = dloss(output, input) * dtanh(out) * l3_out
    dloss/dwout = dloss * dactout * l3_out
    '''
    
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

output[0] = (output[0] + 1.0)*127.5
output = Image.fromarray(output[0].astype(np.uint8))
print(output.show())


'''def Conv2Dbackward():
    for y in range(kernel.shape[1]):
        for x in range(kernel.shape[0]):
            weight = kernel[x,y]

            K = dcost * dact
            dwout = np.matmul(K.T, image)

            weight = weight - lr * dwout.T'''
