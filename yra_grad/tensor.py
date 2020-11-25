# inspired by https://github.com/evcu/numpy_autograd
import numpy as np


def check(*others):
    for other in others:
        if not isinstance(other, Tensor):
            raise ValueError("other needs to be a Tensor instance")


class Tensor:
    def __init__(self, data, is_leaf=True, backward_fn=None):
        if not is_leaf and backward_fn is None:
            raise ValueError("non leaf nodes requires backward_fn")

        self.is_leaf = is_leaf
        self.prev = []
        self.backward_fn = backward_fn
        self.data = data
        self.grad = np.zeros(self.data.shape)

    def __repr__(self):
        return f'Tensor(data:{self.data}, grad:{self.grad})\n'

    # grad stuff

    def zero_grad(self):
        self.grad = np.zeros(self.data.shape)

    def calculate_grad(self):
        self.backward_fn(dy=self.grad)

    def backward(self):
        topological_sorted = self._topological_sort()
        self.grad = np.ones(self.data.shape)
        for tensor in reversed(topological_sorted):
            tensor.calculate_grad()

    def _topological_sort(self):
        tensors_seen = set()
        topological_sorted = []

        def helper(tensor):
            if tensor in tensors_seen or tensor.is_leaf:
                pass
            else:
                tensors_seen.add(tensor)
                for prev_tensor in tensor.prev:
                    helper(prev_tensor)
                topological_sorted.append(tensor)

        helper(self)
        return topological_sorted

    # ops

    def assign(self, other):
        self.data = other.data

    def div(self, other):
        check(other)

        def b_fn(dy):
            self.grad += dy / other.data
            other.grad -= dy * self.data / np.power(other.data, 2)

        res = Tensor(self.data / other.data, is_leaf=False, backward_fn=b_fn)
        res.prev.extend([self, other])
        return res

    def dot(self, other):
        check(other)

        def b_fn(dy):
            if np.isscalar(dy):
                dy = np.ones(1) * dy
            self.grad += np.dot(dy, other.data.T)
            other.grad += np.dot(self.data.T, dy)

        res = Tensor(np.dot(self.data, other.data), is_leaf=False, backward_fn=b_fn)
        res.prev.extend([self, other])
        return res

    def minus(self, other):
        check(other)

        def b_fn(dy):
            self.grad -= dy
            other.grad -= dy

        res = Tensor(self.data - other.data, is_leaf=False, backward_fn=b_fn)
        res.prev.extend([self, other])
        return res

    def multiply(self, other):
        check(other)

        def b_fn(dy):
            if np.isscalar(dy):
                dy = np.ones(1) * dy
            self.grad += np.multiply(dy, other.data)
            other.grad += np.multiply(dy, self.data)

        res = Tensor(np.multiply(self.data, other.data), is_leaf=False, backward_fn=b_fn)
        res.prev.extend([self, other])
        return res

    def plus(self, other):
        check(other)

        def b_fn(dy):
            self.grad += dy
            other.grad += dy

        res = Tensor(self.data + other.data, is_leaf=False, backward_fn=b_fn)
        res.prev.extend([self, other])
        return res

    def plus_bias(self, bias):
        check(bias)

        def b_fn(dy):
            bias.grad += dy.sum(axis=0)
            self.grad += dy

        res = Tensor(self.data + bias.data, is_leaf=False, backward_fn=b_fn)
        res.prev.extend([self, bias])
        return res

    def pow(self, powers):
        check(powers)

        def b_fn(dy):
            self.grad += dy * powers.data * np.power(self.data, powers.data - 1)
            powers.grad += dy * np.power(self.data, powers.data) * np.log(self.data)

        res = Tensor(self.data ** powers.data, is_leaf=False, backward_fn=b_fn)
        res.prev.extend([self, powers])
        return res

    def pow_scalar(self, power):
        def b_fn(dy):
            self.grad += dy * power * np.power(self.data, power - 1)

        res = Tensor(self.data ** power, is_leaf=False, backward_fn=b_fn)
        res.prev.extend([self])
        return res

    def relu(self):
        def b_fn(dy=1):
            self.grad[self.data > 0] += dy[self.data > 0]

        res = Tensor(np.maximum(self.data, 0), is_leaf=False, backward_fn=b_fn)
        res.prev.append(self)
        return res

    def scalar_mul(self, scalar):
        if not isinstance(scalar, (int, float)):
            raise ValueError('c needs to be one of (int, float)')

        def b_fn(dy=1):
            self.grad += dy * scalar

        res = Tensor(self.data * scalar, is_leaf=False, backward_fn=b_fn)
        res.prev.append(self)
        return res

    def sum(self):
        def b_fn(dy=1):
            self.grad += np.ones(self.data.shape) * dy

        res = Tensor(np.sum(self.data), is_leaf=False, backward_fn=b_fn)
        res.prev.append(self)
        return res

    def transpose(self):
        def b_fn(dy):
            self.grad += dy.T

        res = Tensor(self.data.T, is_leaf=False, backward_fn=b_fn)
        res.prev.append(self)
        return res

    # helper stuff

    def shape(self):
        return self.data.shape

    @staticmethod
    def zeros(*shape):
        return Tensor(np.zeros(shape, dtype=np.float32))

    @staticmethod
    def ones(*shape):
        return Tensor(np.ones(shape, dtype=np.float32))

    @staticmethod
    def randn(*shape):
        return Tensor(np.random.randn(*shape).astype(np.float32))

    @staticmethod
    def eye(dim):
        return Tensor(np.eye(dim).astype(np.float32))

    # data model

    def __add__(self, other):
        check(other)
        if len(other.shape()) == 2 and other.shape()[-1] == 1:
            return self.plus_bias(other)
        return self.plus(other)

    def __sub__(self, other):
        check(other)
        if len(other.shape()) == 2 and other.shape()[-1] == 1:
            return self.plus_bias(other.scalar_mul(-1))
        return self.minus(other)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return self.scalar_mul(other)
        return self.multiply(other)

    def __matmul__(self, other):
        return self.dot(other)

    def __pow__(self, power):
        if isinstance(power, (int, float)):
            return self.pow_scalar(power)
        return self.pow(power)
