import activation_functions as af

class Network():

    def __init__(self, num_inputs : int):
        self.neurons = []
        self.layers = 1
        self.add_layer()

    def forward_pass(self):
        for layer in range(self.layers):
            self.forward_step()

    def forward_step(self):
        pass

    def add_layer(self, num_nodes : int, activation : af.ActivationFunction, is_output : bool):

        pass


class Neuron():

    def __init__(self, activation : af.ActivationFunction):
        # self.id = 
        self.activation = activation
        self.neighbors = {}

    def forward(self):
        pass
        # for neighbor in self.neighbors:


    def recieve_output(self, inp):
        pass

    def add_neighbor(self, n):
        pass


