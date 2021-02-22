import numpy as np
import random as rand

class Neural_Network:
    def __init__(self,X,Layers,a):
        self.X = X
        self.Layers = Layers
        self.Learning_Rate = a
        self.Weights = []
        self.Biases = []
        self.H = []
        self.Evaluation = 0
        self.d_Weights = []
        self.d_Biases = []
        self.__do_prework()

    def __activation_net(self, W,X,b):
        return W@X+b

    def __activation_out(self,net):
        return self.__sigmoid(net)

    def __sigmoid(self,x):
        return 1/(1 + np.exp(-x))

    def __d_sigmoid(self,z):
        f = 1/(1+np.exp(-z))
        r = f * (1 - f)
        return r

    #Calculates the derivative dErrorTotal/da(L-1)out for the node curr_node
    def __d_SSR(self,observed,curr_node):
        sum = 0
        for node in range(self.Layers[-1]):
            predicted = self.H[-1][node][1]
            z = self.H[-1][node][0]
            w = self.Weights[-1][curr_node][node]
            sum += -2*(observed[node]-predicted)*self.__d_sigmoid(z)*w
        return sum

    def __batches(self,inputs):
        batch = 45
        remaining = inputs%batch
        while(remaining!=0):
            batch += 1
            remaining = inputs%batch
        return batch

    #Methods of parent class
    def __do_prework(self):
        num_of_layers = len(self.Layers)
        weights = [[]for x in range(num_of_layers)]
        biases = [[]for x in range(num_of_layers)]
        for layer,nodes in enumerate(self.Layers):
            b = []
            if(layer == 0): #The first layer so we have to take as inputs the Xi's
                w = [[]for x in range(nodes)]
                features = self.X
                for node in range(nodes):
                    b.append(0.0)
                    for inputs in range(features):
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
        self.H = [[]for x in range(num_of_layers)]

    def __forward_feed(self,set):
        for layer,nodes in enumerate(self.Layers):
            h = []
            if(layer == 0): #In this case we have to take Xi's as inputs
                X = set
            else:
                X = self.H[layer-1][:,1]    #0 for the net value and 1 for the out value
            for node in range(nodes):
                W = self.Weights[layer][:,node]
                b = self.Biases[layer][node]
                net = self.__activation_net(W,X,b)
                out = self.__activation_out(net)
                h.append([net,out])
            self.H[layer] = np.array(h)


    def __backprop_Last(self,observed): #Observed is a vector of the real outputs for current set
        layers = len(self.Layers)
        x = self.Layers[-2]
        y = self.Layers[-1]
        w_reverse = np.zeros(shape=(x,y))
        b_reverse = np.zeros(y)
        for curr_node in range(self.Layers[-1]):    #For each node of the last layer
            w_temp = []
            predicted_curr = self.H[-1][curr_node][1]
            z = self.H[-1][curr_node][0]
            b = -2*(observed[curr_node]-predicted_curr)*self.__d_sigmoid(z)
            for prev_node in range(self.Layers[-2]):
                predicted_prev = self.H[-2][prev_node][1]
                r = -2*(observed[curr_node]-predicted_curr)*self.__d_sigmoid(z)*predicted_prev
                w_temp.append(r)
            w_reverse[:,curr_node] += np.array(w_temp).T
            b_reverse[curr_node] += b
        if(len(self.d_Weights) == 0):
            self.d_Weights.append(w_reverse)
            self.d_Biases.append(b_reverse)
        else:
            self.d_Weights[0]+=w_reverse
            self.d_Biases[0]+=b_reverse


    def __backprop_Previous(self,observed,dataset):
        x = self.X
        y = self.Layers[-2]
        w_reverse = np.zeros(shape=(x,y))
        b_reverse = np.zeros(y)
        #for each node in the layer -2
        for node in range(self.Layers[-2]):
            w_temp = []
            SSR_sum = self.__d_SSR(observed,node)
            z = self.__d_sigmoid(self.H[-2][node][0])
            b = SSR_sum*z
            for input in dataset:
                r = SSR_sum*z*input
                w_temp.append(r)
            w_reverse[:,node] += np.array(w_temp).T
            b_reverse[node] += b
        if(len(self.d_Weights) == 1):
            self.d_Weights.append(w_reverse)
            self.d_Biases.append(b_reverse)
        else:
            self.d_Weights[1]+=w_reverse
            self.d_Biases[1]+=b_reverse



    #X is a matrix, each row represents each input and each column the current value of a set
    #Y is a matrix containing the observed results for each set of inputs
    def __GradientDescent(self,X,Y):
        #input(self.d_Weights)
        total_sets = len(X[0])
        batches = self.__batches(total_sets)
        mini_batch = int(total_sets/batches)
        for j in range(batches):
            for i in range(j*mini_batch,(j+1)*mini_batch):
                set = X[:,i]
                observed = Y[:,i]
                self.__forward_feed(set)
                self.__backprop_Last(observed)
                self.__backprop_Previous(observed,set)
            self.d_Weights.reverse()                     #d_Weights contains the Sum of the derivatives of the weights. Reverse(because [0] == last layer weights)
            self.d_Biases.reverse()                      #Same thing but for the biases
            layers = len(self.Layers)
            for layer in range(layers):
                inputs = len(self.Weights[layer])
                #Calculating new weights
                for feature in range(inputs):
                    #input(self.Weights[layer][feature])
                    nodes =  len(self.Weights[layer][feature])
                    for node in range(nodes):
                        old_weight = self.Weights[layer][feature][node]
                        derivative = self.d_Weights[layer][feature][node]
                        step_size = derivative*self.Learning_Rate
                        new_weight = old_weight - step_size
                    #    print(new_weight," = ",old_weight," - ",step_size)
                        self.Weights[layer][feature][node] = new_weight
                #Calculating new biases
                for node in range(len(self.Biases[layer])):
                    old_bias = self.Biases[layer][node]
                    derivative = self.d_Biases[layer][node]
                    step_size = derivative*self.Learning_Rate
                    new_bias = old_bias - step_size
                    self.Biases[layer][node] = new_bias
            self.d_Weights = []
            self.d_Biases = []


    def get_Eval(self):
        return self.Evaluation

    def train(self,X,Y,repeats):
        X = np.array(X).T
        Y = np.array(Y).T
        for repeat in range(repeats):
            print("\tTraining: ",repeat+1,"/",repeats," completed.")
            self.__GradientDescent(X,Y)



    def test(self,X,Y):
        X = np.array(X).T
        Y = np.array(Y).T
        total_sets = len(X[0])
        correct = 0
        for i in range(total_sets):               #For each pair of data(one set of inputs, one set of outputs)
            set = X[:,i]                          #Pair of inputs
            observed = Y[:,i]
            correct += self.__evaluate(set,observed)
        score = (correct/total_sets)*100
        self.Evaluation = score
        print("\nClassifier's Accuracy: ",score,"%")


    def __evaluate(self,set,observed):
        self.__forward_feed(set)
        prediction = []
        result = observed.tolist()
        last_layer = self.H[-1]
        for node in last_layer:
            prediction.append(node[1])               #0 for net value(before generilization(sigmoid)), 1 is for output value
        pos_prediction = prediction.index(max(prediction))
        pos_result = result.index(max(result))
        if(pos_prediction == pos_result):
            return 1
        else:
            return 0
