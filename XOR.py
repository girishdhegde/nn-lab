import nn.nn as nn
import numpy as np
# XOR
###############################################################
net  = nn([2, 3, 2], ['relu', 'softmax'], 'CrossEntropyLoss')
###############################################################


# Data generation
x  = np.linspace(-40, 40, 40)
y  = np.linspace(-40, 40, 40)
x  = np.column_stack([[xi, yi] for xi in x for yi in y])
x  = np.array(list(zip(x[0], x[1])))

y  = []
for X in x:
    xi, yi = X
    if (xi > 0 and yi > 0) or (xi < 0 and yi <0):
        y.append(1)
    else:
        y.append(0)
y = np.array(y)


index = np.arange(len(y))
np.random.shuffle(index)
x = x[index]
y = y[index]

# Params
bs = 10
lr = 1e-3
iterations = len(y) // bs


# training
for e in range(10000):
    Loss = 0
    for i in range(iterations):
        start = i * bs
        net.zero_grad()
        out, loss = net(x[start: start+bs], y[start: start+bs])
        Loss += loss
        net.adam(10*lr)
    if e%5 == 0:
        net.viz(epoch=e, loss=Loss/iterations)
        # net.viz(rows=2, cols=2)

    print(f'epoch: {e}, loss: {Loss/iterations}')

