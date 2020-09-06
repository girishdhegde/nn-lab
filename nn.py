import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import pickle 


class neuron:
    def __init__(self, W=None, b=0, act='relu', features=1):
            if W:
                self.weight = W
            else:
                self.weight = np.random.random(features)
            if b:
                self.bias   = b
            else:
                self.bias   = np.random.random(1)[0]
            self.z          = 0 
            self.a          = 0
            self.gradB      = 0 
            self.gradW      = np.zeros_like(self.weight)
            self.act        = act
            if not act:
                self.activation = lambda x: x
                self.activation_grad = self.no_act_grad
            else:
                self.activation = getattr(self, act)
                self.activation_grad = getattr(self, act+'_grad')
        
    def relu(self, x):
        return max(x, 0)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def relu_grad(self):
        return 1 if self.z > 0 else 0
    
    def sigmoid_grad(self):
        return self.a * (1 - self.a)

    def no_act_grad(self):
        return 1

    def __call__(self, x):
        self.x = x
        self.z = np.dot(x, self.weight) + self.bias
        self.a = self.activation(self.z)
        return self.a

    def backward(self, grad):
        gradZ = grad * self.activation_grad()
        self.gradB += gradZ
        self.gradW  = self.gradW + gradZ * self.x
        return gradZ * self.weight

    def zero_grad(self):
        self.gradB = 0
        self.gradW = np.zeros_like(self.weight)



    def optimize(self, lr=0.1, lmda=0.0):
        self.bias   -= lr * self.gradB
        self.weight  = self.weight - lr * self.gradW - lmda * self.weight

    def __str__(self):
        return f'''type<neuron>: weight:{self.weight}, bias:{self.bias}
                   gradB:{self.gradB}, gradW:{self.gradW}, activation:{self.act}'''



class layer:
    def __init__(self, in_features, out_features, activation='relu'):
        self.in_features  = in_features
        self.out_features = out_features
        self.neurons = [neuron(act=activation, features=in_features) for i in range(out_features)]

    def __call__(self, x):
        out = [N(x) for N in self.neurons]
        return np.array(out)

    def backward(self, in_grad):
        out_grad = np.zeros(self.in_features)
        for N, grad in zip(self.neurons, in_grad):
            out_grad = out_grad + N.backward(grad)
        return out_grad

    def zero_grad(self):
        for N in self.neurons:
            N.zero_grad()

    def optimize(self, lr=1e-4, lmda=0.0):
        for N in self.neurons:
            N.optimize(lr, lmda)


class nn:
    def __init__(self, shape, activations=None, criterion='MSELoss', inp=np.array([]), target=np.array([]), viz=True):
        '''shape: [in_features, hidden1, hidden2, ..., hiddenN, out_features]
           activations: [act_fn for layer in shape[1:]]'''
        if activations == None:
            activations = [None for _ in range(len(shape)-1)]
        self.shape = shape
        self.activations = activations
        self.criterion = getattr(self, criterion)
        if activations[-1] == 'softmax':
            self.layers = [layer(shape[i-1], shape[i], activations[i-1]) for i in range(1, len(shape)-1)]
            self.layers.append(layer(shape[-2], shape[-1], None))
        else:
            self.layers = [layer(shape[i-1], shape[i], activations[i-1]) for i in range(1, len(shape))]
        

        # For nn-visualization
        if viz:
            self.given = False
            if len(inp) > 0:
                self.given = True
                self.ginp  = inp
                xmin, xmax = inp[:, 0].min() - 1, inp[:, 0].max() + 1
                ymin, ymax = inp[:, 1].min() - 1, inp[:, 1].max() + 1
                self.vizx  = np.linspace(xmin, xmax, 40)
                self.vizy  = np.linspace(ymin, ymax, 40)
                if not target.all() == None:
                    self.inpclr = ['blue' if label == 1 else 'red' for label in target]
            else:
                self.vizx = np.linspace(-40, 40, 40)
                self.vizy = np.linspace(-40, 40, 40)

            self.xy   = np.column_stack([[xi, yi] for xi in self.vizx for yi in self.vizy])
            self.X, self.Y = np.meshgrid(self.vizx, self.vizy)
            self.inp = np.c_[self.X.ravel(), self.Y.ravel()]
            self.colors = list(map(self.assignColor, self.inp[:, 0], self.inp[:, 1], 
                                   np.ones_like(self.inp[:, 0])*np.max(self.inp[:, 0]),
                                   np.ones_like(self.inp[:, 1])*np.max(self.inp[:, 1])))
            
            self.fig  = plt.figure()
            self.cmap = plt.get_cmap('hsv')
            plt.ion()



    def __call__(self, x, target):
        self.b_size = len(x)
        pred = []
        loss = 0
        for xi, yi in zip(x, target):
            out = xi.copy()
            for L in self.layers:
                out = L(out)
            if self.activations[-1] == 'softmax':
                out = self.softmax(np.array(out))
            loss += self.criterion(out, yi)
            self.backward()
            pred.append(out)
        return np.array(pred), loss / self.b_size

    def softmax(self, x):
        x  -= np.max(x)
        exp = np.exp(x)
        return  exp / np.sum(exp)

    def MSELoss(self, pred, target):
        ''' For regression/prediction with Linear output layer'''
        self.loss_grad = 2 * (pred - target) / self.shape[-1]
        return np.mean((pred - target)**2)

    def BCELoss(self, pred, target):
        eps = 1e-9
        ''' For Binary Classification with Sigmoid output layer'''
        self.loss_grad = (pred - target) / (self.shape[-1] * np.array([N.sigmoid_grad() for N in self.layers[-1].neurons]))
        loss = np.sum(target * np.log(pred + eps)) + np.sum((1 - target) * np.log(1 - pred + eps))
        return -np.mean(loss)



    def CrossEntropyLoss(self, pred, target):
        ''' For Multi-Class Classification with Softmax output layer
            target: index of correct class in one-hot encoding'''
        one_hot_target = np.zeros(len(pred))
        one_hot_target[target] = 1.0
        self.loss_grad = pred - one_hot_target
        return -np.log(pred[target])

    def backward(self, in_grad=None):
        if in_grad:
            self.loss_grad = in_grad
        grad = self.loss_grad.copy()
        for L in self.layers[::-1]:
            grad = L.backward(grad)
        return grad

    def zero_grad(self):
        for L in self.layers:
            L.zero_grad()

    def optimize(self, lr=1e-4, lmda=0.0):
        # mini-batch gradient descent
        lr /= self.b_size
        for L in self.layers:
            L.optimize(lr, lmda)
    
    def predict(self, x, classification=True):
        pred = []
        for xi in x:
            out = xi
            for L in self.layers:
                out = L(out)
            pred.append(out)
        pred = np.array(pred)
        if self.activations[-1] == 'softmax':
            pred = self.softmax(pred)
        if classification:
            if self.activations[-1] == 'softmax':
                return np.argmax(pred, axis=1)
            return np.uint8(pred > 0.5)
        return pred

    def save(self, path='model.pkl'):
        weight = []
        for layer in self.layers:
            weight.append([])
            for n in layer.neurons:
                weight[-1].append([n.weight, n.bias])
        with open(path, 'wb') as f:
            pickle.dump(weight, f)

    def load(self, path='model.pkl'):
        with open(path, 'rb') as f:
            weight = pickle.load(f)

        for wlayer, layer in zip(weight, self.layers):
            for wn, n in zip(wlayer, layer.neurons):
                n.weight = wn[0]
                n.bias   = wn[1]


    @staticmethod
    def assignColor(x, y, xmax, ymax):
        r = 127 + 127 * x / xmax
        g = 127 + 127 * y / ymax
        b = 0 
        return (r/255, g/255, b/255)    

    def viz(self, rows=0, cols=0, save=True, path='nnViz.png', epoch=None, loss=None):
        pred = [[] for _ in self.layers]
        for xi in self.inp:
            out = xi
            for i, L in enumerate(self.layers):
                out = L(out)
                pred[i].append(out)

        if not rows:
            rows = 1
        if not cols:
            cols = 1
            for n in self.shape[1:]:
                if n < 4:
                    cols += 1

        self.fig.suptitle(f'epoch: {epoch}, loss: {loss}')           
        self.ax = self.fig.add_subplot(rows, cols, 1)

        label = np.reshape(self.predict(self.inp), (self.vizx.shape[0], self.vizy.shape[0])) 
        self.ax.contourf(self.X, self.Y, label, cmap='gray', alpha=0.3, )
        
        self.ax.scatter(self.inp[:, 0], self.inp[:, 1], s=40, c=self.colors, alpha=1)
        if self.given:
            self.ax.scatter(self.ginp[:, 0], self.ginp[:, 1], s=10, alpha=0.5, c=self.inpclr)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.title.set_text(f'Input(2D)+Output Decision Boundary')

        pos = 2
        for i, layer in enumerate(pred, 2): 
            layer = np.array(layer)   
            if len(layer[0]) == 3:
                self.ax = self.fig.add_subplot(rows, cols, pos, projection='3d')
                pos += 1
                # self.ax.grid(b=False)
                # self.ax.view_init(elev=30, azim=30)
                self.ax.scatter3D(layer[:, 0], layer[:, 1], layer[:, 2], c=self.colors, s=40, alpha=0.6)
                self.ax.set_xlabel('x')
                self.ax.set_ylabel('y')
                self.ax.set_zlabel('z')
                if i == len(pred) + 1:
                    self.ax.title.set_text(f'Output Layer Representations(3D)')
                    self.ax.scatter3D(layer[:, 0], layer[:, 1], layer[:, 2], s=40, c=self.colors, alpha=0.6)
                else:
                    self.ax.title.set_text(f'Hidden Layer-{i-1} Representations(3D)')


            if len(layer[0]) == 2:
                self.ax = self.fig.add_subplot(rows, cols, pos)
                pos += 1
                if i == len(pred) + 1:
                    self.ax.scatter(layer[:, 0], layer[:, 1], s=40, c=self.colors, alpha=0.6)
                    self.ax.title.set_text(f'Output Layer Representations(2D)')
                    minx = np.min(layer[:, 0])
                    maxx = np.max(layer[:, 1])
                    self.ax.plot([minx, maxx], [minx, maxx])
                else:
                    self.ax.scatter(layer[:, 0], layer[:, 1], s=40, c=self.colors, alpha=0.6)
                    self.ax.title.set_text(f'Hidden Layer-{i-1} Representations(2D)')
                self.ax.set_xlabel('x')
                self.ax.set_ylabel('y')


        plt.draw()
        plt.pause(1e-12)
        plt.clf()


# XOR

# net  = nn([2, 3, 2], ['sigmoid', 'softmax'], 'CrossEntropyLoss')

# x  = np.linspace(-40, 40, 40)
# y  = np.linspace(-40, 40, 40)
# x  = np.column_stack([[xi, yi] for xi in x for yi in y])
# x  = np.array(list(zip(x[0], x[1])))

# y  = []
# for X in x:
#     xi, yi = X
#     if (xi > 0 and yi > 0) or (xi < 0 and yi <0):
#         y.append(1)
#     else:
#         y.append(0)
# y = np.array(y)


# index = np.arange(len(y))
# np.random.shuffle(index)
# x = x[index]
# y = y[index]

# bs = 10
# lr = 1e-3
# iterations = len(y) // bs

# for e in range(10000):
#     Loss = 0
#     for i in range(iterations):
#         start = i * bs
#         net.zero_grad()
#         out, loss = net(x[start: start+bs], y[start: start+bs])
#         Loss += loss
#         net.optimize(lr, lmda=0.0)
#     if e%5 == 0:
#         net.viz(epoch=e, loss=Loss/iterations)
#         # net.viz(rows=2, cols=2)

#     print(f'epoch: {e}, loss: {Loss/iterations}')



# SPIRAL

from sklearn import datasets

x, y = datasets.make_moons(n_samples=1000, noise=0.1, random_state=0)

net  = nn([2, 5, 5, 3, 2], ['relu', 'relu', 'relu', 'softmax'], 'CrossEntropyLoss', inp=x, target=y)
net.load()

bs = 10
lr = 1e-2
iterations = len(y) // bs

for e in range(5000):
    Loss = 0    
    for i in range(iterations):
        start = i * bs
        net.zero_grad()
        out, loss = net(x[start: start+bs], y[start: start+bs])
        Loss += loss
        net.optimize(lr, lmda=0.0)
    if e%20 == 0:
        net.viz(epoch=e, loss=Loss/iterations)
        net.save()
        # net.viz(rows=2, cols=2)

    print(f'epoch: {e}, loss: {Loss/iterations}')

