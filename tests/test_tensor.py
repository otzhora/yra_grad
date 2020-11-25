import unittest

import numpy as np
import torch

from yra_grad.tensor import Tensor


x_init = np.random.randn(1,3).astype(np.float32)
W_init = np.random.randn(3,3).astype(np.float32)
m_init = np.random.randn(1,3).astype(np.float32)


class TestTensor(unittest.TestCase):
    def test_forward(self):
        a_init = np.random.rand(3, 3)
        b_init = np.random.rand(3, 3)
        a = Tensor(a_init)
        b = Tensor(b_init)

        a_torch = torch.tensor(a_init)
        b_torch = torch.tensor(b_init)

        np.testing.assert_allclose((a_torch @ b_torch).numpy(), a.dot(b).data)
        np.testing.assert_allclose((a_torch * b_torch).numpy(), a.multiply(b).data)
        np.testing.assert_allclose((a_torch - b_torch).numpy(), a.minus(b).data)
        np.testing.assert_allclose((a_torch + b_torch).numpy(), a.plus(b).data)

        scalar = 4.6

        np.testing.assert_allclose((a_torch * scalar).numpy(), a.scalar_mul(scalar).data)

        bias_init = np.random.rand(3)
        bias = Tensor(bias_init)
        bias_torch = torch.tensor(bias_init)

        np.testing.assert_allclose((a_torch + bias_torch).numpy(), a.plus_bias(bias).data)

        np.testing.assert_allclose(torch.relu(a_torch).numpy(), a.relu().data)

        np.testing.assert_allclose(a_torch.sum().numpy(), a.sum().data)

        np.testing.assert_allclose(a_torch.T.numpy(), a.transpose().data)

    def test_backward(self):
        def test_Tensor():
            x = Tensor(x_init)
            W = Tensor(W_init)
            m = Tensor(m_init)
            out = x.dot(W).relu()
            out = out.multiply(m).plus(m).sum()
            out.backward()
            return out.data, x.grad.data, W.grad.data

        def test_pytorch():
            x = torch.tensor(x_init, requires_grad=True)
            W = torch.tensor(W_init, requires_grad=True)
            m = torch.tensor(m_init)
            out = x.matmul(W).relu()
            out = out.mul(m).add(m).sum()
            out.backward()
            return out.detach().numpy(), x.grad, W.grad

        for x, y in zip(test_Tensor(), test_pytorch()):
            np.testing.assert_allclose(x, y, atol=1e-5)

    def test_to_fail(self):
        assert False


if __name__ == "__main__":
    unittest.main()
