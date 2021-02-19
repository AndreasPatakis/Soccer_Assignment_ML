import numpy as np
import random as rand
import soccer_ml_package.functions as smp


class Neural_Network:
    def __init__(self,X,Layers,a):
        self.X = np.array(X)
        self.Layers = Layers
        self.a = a
        self.Weights = []
        self.Biases = []
        self.H = []
        self.do_prework()

    def activation_net(self, W,X,b):
        return W@X+b

    def activation_out(self,net):
        return self.sigmoid(net)

    def sigmoid(self,x):
        return 1/(1 + np.exp(-x))

    #Methods of parent class
    def do_prework(self):
        num_of_layers = len(self.Layers)
        weights = [[]for x in range(num_of_layers)]
        biases = [[]for x in range(num_of_layers)]
        for layer,nodes in enumerate(self.Layers):
            b = []
            if(layer == 0): #The first layer so we have to take as inputs the Xi's
                w = [[]for x in range(nodes)]
                for node in range(nodes):
                    b.append(0.0)
                    for inputs in range(len(self.X)):
                        w[node].append(rand.uniform(0.0,1.0))
            else:
                w = [[]for x in range(nodes)]
                for node in range(nodes):
                    b.append(0.0)
                    for inputs in range(self.Layers[layer-1]):
                        w[node].append(rand.uniform(0.0,1.0))
            biases[layer] = np.array(b)
            w_i = np.array(w).T
            weights[layer] = w_i
        self.Weights = weights
        self.Biases = biases
        input(self.Weights)
        self.H = [[]for x in range(num_of_layers)]

    def forward_feed(self):
        for layer,nodes in enumerate(self.Layers):
            print("\nLAYER :",layer)
            h = []
            if(layer == 0): #In this case we have to take Xi's as inputs
                X = self.X
            else:
                X = self.H[layer-1][:,1]    #0 for the net value and 1 for the out value
            print(self.Weights[layer])
            for node in range(nodes):
                W = self.Weights[layer][:,node]
                b = self.Biases[layer][node]
                net = self.activation_net(W,X,b)
                out = self.activation_out(net)
                h.append([net,out])
            self.H[layer] = np.array(h)
            print("\nNodes:")
            input(self.H[layer][:,1])








net1 = Neural_Network([1,2,3],[2,3],0)
net1.forward_feed()
