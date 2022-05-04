from typing import Tuple
import activation_functions as af
import random as rand

class Network():

    def __init__(self, num_inputs : int, init_weights=[-1.0, 1.0]):
        self.neurons = []
        self.weights = [] # e.g., weights[layer][n][m], gets weight between node n in layer and node m in layer + 1
        self.layers = 0
        self.init_weights = init_weights
        # add input layer
        self.add_layer(num_inputs, af.Linear, False)

    def forward_pass(self, inputs : Tuple) -> Tuple:
        assert len(self.neurons[0]) == len(inputs), "Input size must match size of input layer"
        # set input nodes
        for idx, inpt in enumerate(inputs):
            self.neurons[0][idx].output = inpt

        # compute forward pass
        for layer_idx in range(len(self.neurons)-1):
            self.forward_step(layer_idx)

        output = []
        for neuron in self.neurons[self.layers-1]:
            output.append(neuron.output)
        return tuple(output)

    def forward_step(self, layer_idx):
        # reset outputs of next layer
        for neuron in self.neurons[layer_idx+1]:
            neuron.output = 0
        
        # compute outputs for each layer
        for neuron_idx, neuron in enumerate(self.neurons[layer_idx]):
            for next_neuron_idx, next_neuron in enumerate(self.neurons[layer_idx+1]):
                weight = self.weights[layer_idx][neuron_idx][next_neuron_idx]
                next_neuron.output += weight * neuron.output

        # apply activation function to output
        for next_neuron in self.neurons[layer_idx+1]:
            next_neuron.output = next_neuron.activation(next_neuron.output)

    def add_layer(self, num_nodes : int, activation : af.ActivationFunction, is_output : bool):
        # create neurons
        layer = []
        for idx in range(num_nodes):
            layer.append(Neuron(self.layers, idx, activation()))
        self.neurons.append(layer)
        # create weights
        if self.layers >= 1:
            num_nodes_prev_layer = len(self.neurons[self.layers-1])
            weights = [[0 for _ in range(len(layer))] for _ in range(num_nodes_prev_layer)]
            # random initialization of weights
            for prev_node_idx in range(len(weights)):
                for node_idx in range(len(weights[0])):
                    weights[prev_node_idx][node_idx] = rand.uniform(self.init_weights[0], self.init_weights[1])  

            self.weights.append(weights)
        self.layers += 1

class Neuron():

    def __init__(self, layer_idx : int, node_idx : int, activation : af.ActivationFunction):
        self.layer = layer_idx
        self.index = node_idx
        self.activation = activation
        self.output = 0


