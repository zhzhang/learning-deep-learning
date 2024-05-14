import nn
import numpy as np
import scipy as sp
import torch


def test_tensor_init():
    # numpy input
    x = nn.Tensor(np.ones([2, 3, 4]))
    assert np.array_equal(x.grad, np.zeros([2, 3, 4]))
    assert x._backprop() is None
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


def test_sum():
    a = np.random.random([2, 3, 4])
    at = nn.Tensor(a)
    st = at.sum(axis=0)
    st.backward()
    assert np.allclose(
        st.value, np.sum(a, axis=0, keepdims=True), rtol=1e-15, atol=1e-20
    )
    apt = torch.tensor(a, requires_grad=True)
    spt = apt.sum(axis=0, keepdim=True)
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


def test_backwards():
    # Create a forked dag.
    a = nn.Tensor.random(2, 3, 4)
    b = nn.Tensor.random(1, 3, 4)
    c = a * b
    d = nn.Tensor.random(2, 3, 4)
    e = (a + d) * b
    f = c + e + a
    f.backward()
    apt = torch.tensor(a.value, requires_grad=True)
    bpt = torch.tensor(b.value, requires_grad=True)
    cpt = apt * bpt
    dpt = torch.tensor(d.value, requires_grad=True)
    ept = (apt + dpt) * bpt
    fpt = cpt + ept + apt
    fpt.sum().backward()
    assert np.allclose(apt.grad.numpy(), a.grad, rtol=1e-15, atol=1e-20)
    assert np.allclose(bpt.grad.numpy(), b.grad, rtol=1e-15, atol=1e-20)
    assert np.allclose(dpt.grad.numpy(), d.grad, rtol=1e-15, atol=1e-20)


def test_softmax():
    a = np.random.random([4, 8])
    at = nn.Tensor(a)
    st = nn.softmax(at)
    st.backward()
    apt = torch.tensor(a, requires_grad=True)
    spt = torch.softmax(apt, dim=1)
    spt.sum().backward()
    assert np.allclose(st.value, spt.detach().numpy(), rtol=1e-15, atol=1e-15)
    assert np.allclose(apt.grad.numpy(), at.grad, rtol=1e-15, atol=1e-15)


def test_cross_entropy():
    a = np.random.random([4, 8])
    y = np.random.randint(0, 8, 4).tolist()
    at = nn.Tensor(a)
    st = nn.cross_entropy(nn.softmax(at), y)
    st.backward()
    apt = torch.tensor(a, requires_grad=True)
    ypt = torch.tensor(y, requires_grad=False)
    spt = torch.nn.functional.cross_entropy(apt, ypt)
    spt.backward()
    assert np.allclose(st.value.sum(), spt.detach().numpy(), rtol=1e-15, atol=1e-10)
    assert np.allclose(apt.grad.numpy(), at.grad, rtol=1e-15, atol=1e-10)


def test_kaiming_uniform_init():
    # Determine that the initialization is correct by comparing the distribution of the weights
    # with a critical value of 0.01.
    d_in, d_out = 300, 4000
    l = nn.Linear(d_in, d_out)
    ltw = torch.zeros((d_out, d_in))
    torch.nn.init.kaiming_uniform_(ltw, nonlinearity="relu")
    result = sp.stats.kstest(l.W.value.flatten(), ltw.detach().numpy().flatten())
    assert result.pvalue > 0.01
