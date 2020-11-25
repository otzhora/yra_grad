import unittest

from yra_grad.tensor import Tensor
from yra_grad.optim import SGD


class TestOptim(unittest.TestCase):
    def test_SGD(self):
        x = Tensor.randn(3, 3)
        W = Tensor.randn(3, 3)

        optim = SGD([W], 0.5)

        for _ in range(100):
            out = ((x @ W) ** 2).sum()
            out.backward()

            optim.step()
            optim.zero_grad()

        assert ((x @ W) ** 2).sum().data < 1e-5


if __name__ == "__main__":
    unittest.main()
