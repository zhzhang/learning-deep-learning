import abc
from typing import Any, Tuple, Union

import numpy as np


class Tensor:
    """
    Like PyTorch's Tensor class, with implementation inspired by micrograd.
    However, takes a batch-first approach, using numpy.arrays for the value and grad.
    This helps indicate where blocks of device memory need to be allocated together,
    to build intuition about what is happening on the GPU in real frameworks.
    """

    def __init__(
        self,
        value: Union[np.array, int, float],
        prev: Tuple["Tensor"] = (),
        name: str = None,
    ):
        self.value = value
        self.zero_grad()
        self.name = name
        self._backward = lambda: None
        self._prev = set(prev)

    def zero_grad(self):
        if type(self.value) == np.ndarray:
            self.grad = np.zeros_like(self.value)

    def accumulate_grad(self, grad):
        if type(self.value) == np.ndarray:
            # Assume that the values were broadcastable here, since a
            # forward pass is typically run before the backward pass.
            broadcast_axes = []
            print(self.name)
            print(grad.shape, self.value.shape)
            for i in range(len(grad.shape)):
                if grad.shape[i] > 1 and self.value.shape[i] == 1:
                    broadcast_axes.append(i)
            for axis in broadcast_axes:
                grad = np.sum(grad, axis=axis, keepdims=True)
            self.grad += grad

    def backward(self):
        self._backward()
        seen = set()
        fringe = list(self._prev)
        for node in fringe:
            # Backprop must be breadth first.
            if node not in seen:
                seen.add(node)
                fringe.extend(node._prev)
                node._backward()

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.value + other.value, (self, other))

        def _backward():
            self.accumulate_grad(out.grad)
            other.accumulate_grad(out.grad)

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.value * other.value, (self, other))

        def _backward():
            self.accumulate_grad(out.grad * other.value)
            other.accumulate_grad(out.grad * self.value)

        out._backward = _backward
        return out

    def __pow__(self, pow):
        out = Tensor(self.value**pow, (self,))

        def _backward():
            self.accumulate_grad(out.grad * pow * self.value ** (pow - 1))

        out._backward = _backward
        return out

    def exp(self):
        out = Tensor(np.exp(self.value), (self,))

        def _backward():
            self.accumulate_grad(out.grad * out.value)

        out._backward = _backward

        return out

    def log(self):
        out = Tensor(np.log(self.value), (self,))

        def _backward():
            self.accumulate_grad(out.grad / self.value)

        out._backward = _backward
        return out

    def dot(self, other):
        # Self A x B, other B x C, out A x C
        out = Tensor(np.dot(self.value, other.value), (self, other))

        def _backward():
            # self.grad A x B = (out) A x C * (other.T) C x B
            self.accumulate_grad(np.dot(out.grad, other.value.T))
            # other.grad B x C = (self.T) B x A * (out) A x C
            other.accumulate_grad(np.dot(self.value.T, out.grad))

        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.value), (self,))

        def _backward():
            self.accumulate_grad(out.grad * (self.value > 0))

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):  # self / other
        return self * other**-1

    def __rtruediv__(self, other):  # other / self
        return other * self**-1

    def __repr__(self):
        return f"Tensor({self.name}: {self.value})"


class Module(abc.ABC):
    def __init__(self):
        self._parameters = []

    def __setattr__(self, name: str, value: Any) -> None:
        if type(value) == Tensor:
            self._parameters.append(value)
            value.name = name
        object.__setattr__(self, name, value)

    def parameters(self):
        return self._parameters

    def init_weights(self):
        raise NotImplementedError()

    def forward(self):
        raise NotImplementedError()


# Layers
class Linear(Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W = Tensor(np.random.rand(d_in, d_out))
        self.b = Tensor(np.random.rand(d_out))

    def forward(self, x):
        return x.dot(self.W) + self.b


class MLP(Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.linear1 = Linear(d_in, 128)
        self.linear2 = Linear(128, d_out)

    def forward(self, x):
        x = self.linear1.forward(x).relu()
        x = self.linear2.forward(x).relu()
        return softmax(x)


def softmax(x):
    exps = (x - np.max(x.value, axis=1, keepdims=True)).exp()
    return exps / np.sum(exps.value)


# Loss
def cross_entropy(y_hat, y):
    return -(y_hat.log() * y)


class DataLoader:
    def __init__(self, data, batch_size=64):
        self.data = data
        self.batch_size = batch_size

    def __iter__(self):
        for i in range(0, len(self.data), self.batch_size):
            batch = self.data[i : i + self.batch_size]
            yield (np.array([row[i] for row in batch]) for i in range(len(batch[0])))


class Optimizer(abc.ABC):
    def __init__(self):
        raise NotImplementedError()

    def step(self):
        raise NotImplementedError()

    def zero_grad(self):
        for param in self.parameters:
            param.zero_grad()


class SGD(Optimizer):
    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for param in self.parameters:
            param.value -= param.grad * self.lr