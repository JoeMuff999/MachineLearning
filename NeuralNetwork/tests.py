import activation_functions as af
import neural_network as nn

def callable_tests():
    x = af.ActivationFunction()
    print(type(x))
    n = x
    m = x
    print(id(n))
    print(id(m))

def simple_network():
    N = nn.Network(1, init_weights=[1.0, 1.0])
    N.add_layer(1, af.ReLU, True)
    inpt = (-1,)
    output = N.forward_pass(inpt)
    assert output == (0,), output
    # print(output)

def two_input_network():
    N = nn.Network(2, init_weights=[1.0, 1.0])
    N.add_layer(2, af.Linear, True)
    # print(N.weights)
    inpt = (2,-1)
    output = N.forward_pass(inpt)
    assert output == (1,1), output


simple_network()
two_input_network()
# callable_tests()