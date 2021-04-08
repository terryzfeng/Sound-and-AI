import math

def sigmoid(x):
    y = 1.0/(1 + math.exp(-x))
    return y

def activate(inputs, weights):
    # perform net input
    # essentially inputs dotted with weights
    sum = 0
    for x,y, in zip(inputs,weights):
        sum += x*y

    # activation
    return sigmoid(sum)

if __name__ == "__main__":
    inputs = [.5,.3,.2]
    weights = [.1,.2,.3]
    output = activate(inputs, weights)
    print(output);
