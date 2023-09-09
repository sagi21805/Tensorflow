from tensorflow import sigmoid, nn, Tensor, math
class Activation_Function(object):
    
    def function(z: Tensor):
        "`the activation func`"
        raise Exception("create activation function")
    
    def derivative(z: Tensor):
        "`the activation derivative`"
        raise Exception("create activation derivative")
    
class Sigmoid(Activation_Function):

    def function(z: Tensor):
        """`the sigmoid func`"""
        return sigmoid(z)
    
    def derivative(z: Tensor):
        """`sigmoid derivative`"""
        return sigmoid(z)*(1-sigmoid(z))

class ReLU(Activation_Function):

    def function(z: Tensor):
        """`the ReLU func`"""
        return nn.relu(z)
    
    def derivative(z: Tensor):
        """`sigmoid derivative`"""
        return max()

    