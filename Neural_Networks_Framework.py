import numpy as np
import random as rand



class Neural_Network:
    def __init__(self,X,Layers,a):
        self.X = np.array(X)
        self.Layers = Layers
        self.a = a
        self.Weights = []
        self.Biases = []
        self.H = []
        self.Results = []
        self.NewWeights = []
        self.__do_prework()

    def __activation_net(self, W,X,b):
        return W@X+b

    def __activation_out(self,net):
        return self.__sigmoid(net)

    def __sigmoid(self,x):
        return 1/(1 + np.exp(-x))

    def __d_sigmoid(self,z):
        f = 1/(1+np.exp(-z))
        return f * (1 - f)

    #Calculates the derivative dErrorTotal/da(L-1)out for the node curr_node
    def __d_SSR(self,observed,curr_node):
        sum = 0
        for node in range(self.Layers[-1]):
            predicted = self.H[-1][node][1]
            z = self.H[-1][node][0]
            w = self.Weights[-1][curr_node][node]
            sum += -2*(observed-predicted)*self.__d_sigmoid(z)*w
        return sum

    #Methods of parent class
    def __do_prework(self):
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
        #print("WEIGHTS LAST:\n ",self.Weights[-1])
        self.H = [[]for x in range(num_of_layers)]

    def forward_feed(self,inputs):
        for layer,nodes in enumerate(self.Layers):
            #print("\nLAYER :",layer)
            h = []
            if(layer == 0): #In this case we have to take Xi's as inputs
                X = inputs
            else:
                X = self.H[layer-1][:,1]    #0 for the net value and 1 for the out value
            #print(self.Weights[layer])
            for node in range(nodes):
                W = self.Weights[layer][:,node]
                b = self.Biases[layer][node]
                net = self.__activation_net(W,X,b)
                out = self.__activation_out(net)
                h.append([net,out])
            self.H[layer] = np.array(h)
            #print("\nNodes:")
            #print(self.H[layer][:,1])
        self.Results.append(self.H[len(self.Layers)-1][:,1])
        #print("Last Layer Nodes:\n",self.H[-1])
        #print("Previous Layer Nodes:\n",self.H[-2])
        #input()
        #print("LAST NODES:\n",self.H[-1])
        self.backprop_Last(1)
        self.backprop_Previous(1)

    def backprop_Last(self,observed):
        layers = len(self.Layers)
        x = self.Layers[-2]
        y = self.Layers[-1]
        w_reverse = np.zeros(shape=(x,y))
        for curr_node in range(self.Layers[-1]):
            w_temp = []
            predicted_curr = self.H[-1][curr_node][1]
            z = self.H[-1][curr_node][0]
            for prev_node in range(self.Layers[-2]):
                predicted_prev = self.H[-2][prev_node][1]
                r = -2*(observed-predicted_curr)*self.__d_sigmoid(z)*predicted_prev
                #print("aL_out: ",predicted_curr,"\naL_net: ",z,"\nSigmoid(aL_net): ",d_sigmoid(z),"\naL-1out: ",predicted_prev,"\nW: ",r)
                #input()
                w_temp.append(r)
            w_reverse[:,curr_node] += np.array(w_temp).T
        if(len(self.NewWeights) == 0):
            self.NewWeights.append(w_reverse)
        else:
            self.NewWeights[0]+=w_reverse

    def backprop_Previous(self,observed):
        x = len(self.X)
        y = self.Layers[-2]
        w_reverse = np.zeros(shape=(x,y))
        #for each node in the layer -2
        for node in range(self.Layers[-2]):
            print("\nNODE ",node)
            w_temp = []
            SSR_sum = self.__d_SSR(1,node)
            z = self.__d_sigmoid(self.H[-2][node][0])
            for input in self.X:
                r = SSR_sum*z*input
                w_temp.append(r)
                print("SSR: ",SSR_sum,"\nZ: ",z,"\nInput: ",input,"\nResult: ",r,"\n")
            w_reverse[:,node] += np.array(w_temp).T
        print(w_reverse)









net1 = Neural_Network([1,2],[2,2],0)
net1.forward_feed([1,2])
