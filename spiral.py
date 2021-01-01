

import nn.nn as nn
from sklearn import datasets


# SPIRAL

x, y = datasets.make_moons(n_samples=1000, noise=0.1, random_state=0)

# Netwotk Structure
net  = nn([2, 5, 5, 3, 2], ['relu', 'relu', 'relu', 'softmax'], 'CrossEntropyLoss', inp=x, target=y)
# net.load()

bs = 10
lr = 1e-2
iterations = len(y) // bs

# Training
for e in range(5000):
    Loss = 0    
    for i in range(iterations):
        start = i * bs
        net.zero_grad()
        out, loss = net(x[start: start+bs], y[start: start+bs])
        Loss += loss
        net.adam(10*lr)
    if e%5 == 0:
        net.viz(epoch=e, loss=Loss/iterations)
        net.save()
        # net.viz(rows=2, cols=2)

    print(f'epoch: {e}, loss: {Loss/iterations}')

