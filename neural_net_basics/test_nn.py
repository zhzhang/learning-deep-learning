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
    assert np.array_equal(st.value, op_func_np(b))
    apt = torch.tensor(a, requires_grad=True)
    bpt = torch.tensor(b, requires_grad=True) if isinstance(b, np.ndarray) else b
    op_func_pt = getattr(apt, op_name)
    spt = op_func_pt(bpt)
    spt.sum().backward()
    assert np.array_equal(apt.grad.numpy(), at.grad)
    if isinstance(b, np.ndarray):
        assert np.array_equal(bpt.grad.numpy(), bt.grad)


def helper_test_unary_op(op_name, a=None):
    a = a if a is not None else np.random.random([2, 3, 4])
    op_func_np = getattr(np, op_name)
    at = nn.Tensor(a)
    op_func = getattr(at, op_name)
    st = op_func()
    st.backward()
    assert np.array_equal(st.value, op_func_np(a))
    apt = torch.tensor(a, requires_grad=True)
    op_func_pt = getattr(apt, op_name)
    spt = op_func_pt()
    spt.sum().backward()
    assert np.array_equal(apt.grad.numpy(), at.grad)


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


def test_matmul():
    a = np.random.random([2, 3, 4])
    b = np.random.random([2, 3, 4])
    op_func_np = getattr(a, op_name)
    at = nn.Tensor(a)
    bt = nn.Tensor(b) if isinstance(b, np.ndarray) else b
    op_func = getattr(at, op_name)
    st = op_func(bt)
    st.backward()
    assert np.array_equal(st.value, op_func_np(b))
    apt = torch.tensor(a, requires_grad=True)
    bpt = torch.tensor(b, requires_grad=True) if isinstance(b, np.ndarray) else b
    op_func_pt = getattr(apt, op_name)
    spt = op_func_pt(bpt)
    spt.sum().backward()
    assert np.array_equal(apt.grad.numpy(), at.grad)
    if isinstance(b, np.ndarray):
        assert np.array_equal(bpt.grad.numpy(), bt.grad)


def test_log():
    helper_test_unary_op("log")


def test_exp():
    helper_test_unary_op("exp")
