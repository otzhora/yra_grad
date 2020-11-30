![tests](https://github.com/otzhora/yra_grad/workflows/tests/badge.svg)
# yra_grad

Simple implementation of Automatic differentiation. 

For more info see [Automatic differentiation in machine learning: a survey](https://arxiv.org/abs/1502.05767)

# Usage

```python
import numpy as np
from yra_grad import Tensor

l1 = Tensor(np.arange(-4,4).reshape(2,4))
l2 = Tensor(np.arange(-2,2).reshape(4,1))
n1 = l1 @ l2
n2 = n1.relu()
n3 = n2.sum()
n3.backward()
print(l1.grad)
# [[-2. -1.  0.  1.]
#  [-2. -1.  0.  1.]]
print(l2.grad)
# [[-4.]
#  [-2.]
#  [ 0.]
#  [ 2.]]
```