from yra_grad.tensor import Tensor


class Optimizer:
    def __init__(self, parameters):
        for param in parameters:
            if not isinstance(param, Tensor):
                raise ValueError("all parameters should be Tensor instances")

        self.parameters = parameters

    def zero_grad(self):
        for param in self.parameters:
            param.zero_grad()


class SGD(Optimizer):
    def __init__(self, parameters, lr):
        super(SGD, self).__init__(parameters)
        self.lr = lr

    def step(self):
        for param in self.parameters:
            param.step(param.data * self.lr)
