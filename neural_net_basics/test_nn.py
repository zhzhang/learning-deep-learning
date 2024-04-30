import nn
import numpy as np
import torch


def test_tensor_init():
    # numpy input
    x = nn.Tensor(np.ones([2, 3, 4]))
    assert np.array_equal(x.grad, np.zeros([2, 3, 4]))
    assert x._backward() is None
    assert len(x._prev) == 0
    # list init
    x = nn.Tensor([[1, 2, 3], [4, 5, 6]])
    assert np.array_equal(x.grad, np.zeros([2, 3]))


def helper_test_binary_op(op_name, a=None, b=None):
    a = a if a is not None else np.random.random([2, 3, 4])
    b = b if b is not None else np.random.random([2, 3, 4])
    op_func_np = getattr(a, op_name)
    at = nn.Tensor(a)
    bt = nn.Tensor(b) if isinstance(b, np.ndarray) else b
    op_func = getattr(at, op_name)
    st = op_func(bt)
    st.backward()
    assert np.allclose(st.value, op_func_np(b), rtol=1e-15, atol=1e-20)
    apt = torch.tensor(a, requires_grad=True)
    bpt = torch.tensor(b, requires_grad=True) if isinstance(b, np.ndarray) else b
    op_func_pt = getattr(apt, op_name)
    spt = op_func_pt(bpt)
    spt.sum().backward()
    assert np.allclose(apt.grad.numpy(), at.grad, rtol=1e-15, atol=1e-20)
    if isinstance(b, np.ndarray):
        assert np.allclose(bpt.grad.numpy(), bt.grad, rtol=1e-15, atol=1e-20)


def test_add():
    # Two tensors with the same shape.
    helper_test_binary_op("__add__")

    # # Two tensors with different shape that are broadcastable.
    a = np.random.random([2, 3, 4])
    b = np.random.random([2, 3, 1])
    helper_test_binary_op("__add__", a, b)


def test_mul():
    # Two tensors with the same shape.
    helper_test_binary_op("__mul__")

    # # Two tensors with different shape that are broadcastable.
    a = np.random.random([2, 3, 4])
    b = np.random.random([2, 3, 1])
    helper_test_binary_op("__mul__", a, b)


def test_pow():
    helper_test_binary_op("__pow__", b=3)
    helper_test_binary_op("__pow__", b=3.0)


def test_neg():
    a = np.random.random([2, 3, 4])
    at = nn.Tensor(a)
    st = -at
    st.backward()
    assert np.array_equal(st.value, -a)
    apt = torch.tensor(a, requires_grad=True)
    spt = -apt
    spt.sum().backward()
    assert np.array_equal(apt.grad.numpy(), at.grad)


def test_radd():
    helper_test_binary_op("__radd__")


def test_sub():
    helper_test_binary_op("__sub__")


def test_rsub():
    helper_test_binary_op("__rsub__")


def test_rmul():
    helper_test_binary_op("__rmul__")


def test_log():
    helper_test_unary_op("log")


def test_truediv():
    helper_test_binary_op("__truediv__")


def test_rtruediv():
    helper_test_binary_op("__rtruediv__")


def test_exp():
    a = np.random.random([2, 3, 4])
    at = nn.Tensor(a)
    st = at.exp()
    st.backward()
    assert np.allclose(st.value, np.exp(a), rtol=1e-15, atol=1e-20)
    apt = torch.tensor(a, requires_grad=True)
    spt = apt.exp()
    spt.sum().backward()
    assert np.allclose(apt.grad.numpy(), at.grad, rtol=1e-15, atol=1e-20)


def test_log():
    a = np.random.random([2, 3, 4])
    at = nn.Tensor(a)
    st = at.log()
    st.backward()
    assert np.allclose(st.value, np.log(a), rtol=1e-15, atol=1e-20)
    apt = torch.tensor(a, requires_grad=True)
    spt = apt.log()
    spt.sum().backward()
    assert np.allclose(apt.grad.numpy(), at.grad, rtol=1e-15, atol=1e-20)


def test_matmul():
    a = np.random.random([2, 12])
    b = np.random.random([12, 2])
    at = nn.Tensor(a)
    bt = nn.Tensor(b) if isinstance(b, np.ndarray) else b
    st = at.matmul(bt)
    st.backward()
    assert np.array_equal(st.value, np.matmul(a, b))
    assert st.value.shape == (2, 2)
    apt = torch.tensor(a, requires_grad=True)
    bpt = torch.tensor(b, requires_grad=True) if isinstance(b, np.ndarray) else b
    spt = torch.matmul(apt, bpt)
    spt.sum().backward()
    assert np.array_equal(apt.grad.numpy(), at.grad)
    assert np.array_equal(bpt.grad.numpy(), bt.grad)


def test_relu():
    a = np.random.random([2, 3, 4])
    at = nn.Tensor(a)
    st = at.relu()
    st.backward()
    assert np.array_equal(st.value, np.maximum(0, a))
    apt = torch.tensor(a, requires_grad=True)
    spt = torch.relu(apt)
    spt.sum().backward()
    assert np.array_equal(apt.grad.numpy(), at.grad)
