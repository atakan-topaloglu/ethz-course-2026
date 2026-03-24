import sys
import os
import inspect
import pytest
import torch
import importlib.util

# ==========================================
# CONFIGURATION
# ==========================================
# The name of your python file (without .py)
MODULE_NAME = "ex1" 

# ==========================================
# DYNAMIC IMPORT UTILITY
# ==========================================
def load_submission():
    """Dynamically imports the student's python file."""
    if not os.path.exists(f"{MODULE_NAME}.py"):
        pytest.fail(f"File '{MODULE_NAME}.py' not found. Did you convert your notebook?")
    
    try:
        spec = importlib.util.spec_from_file_location(MODULE_NAME, f"{MODULE_NAME}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[MODULE_NAME] = mod
        spec.loader.exec_module(mod)
        return mod
    except Exception as e:
        pytest.fail(f"Failed to import {MODULE_NAME}.py: {e}\nHint: Make sure you don't have top-level code running outside of functions that might crash.")

submission = load_submission()

# ==========================================
# HELPER: SIGNATURE & CONSTRAINT CHECKER
# ==========================================
def assert_signature(func_name, expected_arg_names):
    """Checks if function exists and has correct argument names."""
    if not hasattr(submission, func_name):
        pytest.fail(f"Function '{func_name}' not implemented/found.")
    
    func = getattr(submission, func_name)
    sig = inspect.signature(func)
    params = list(sig.parameters.keys())
    
    assert params == expected_arg_names, \
        f"Signature mismatch for {func_name}. Expected {expected_arg_names}, got {params}"

def assert_source_contains(func_name, snippet):
    """Checks if source code contains a specific snippet (e.g. 'einsum')."""
    func = getattr(submission, func_name)
    source = inspect.getsource(func)
    if snippet not in source:
        pytest.fail(f"Function {func_name} must contain '{snippet}' in implementation.")

def assert_no_loops(func_name):
    """Heuristic check to ensure no for/while loops are used."""
    func = getattr(submission, func_name)
    source = inspect.getsource(func)
    # Remove comments to avoid false positives
    lines = [l.split('#')[0] for l in source.split('\n')]
    clean_source = '\n'.join(lines)
    
    if "for " in clean_source or "while " in clean_source:
        pytest.fail(f"Function {func_name} should not use explicit Python loops (for/while).")

# ==========================================
# 1. TENSOR CREATION TESTS
# ==========================================
def test_make_tensor():
    assert_signature("make_tensor", ["data", "dtype", "device"])
    data = [[1, 2], [3, 4]]
    t = submission.make_tensor(data, dtype=torch.float32)
    assert isinstance(t, torch.Tensor)
    assert t.dtype == torch.float32
    assert t.shape == (2, 2)
    assert torch.equal(t, torch.tensor(data, dtype=torch.float32))

def test_make_zeros():
    assert_signature("make_zeros", ["shape", "dtype", "device"])
    t = submission.make_zeros((2, 3), dtype=torch.int)
    assert torch.all(t == 0)
    assert t.shape == (2, 3)
    assert t.dtype == torch.int

def test_make_ones_like():
    assert_signature("make_ones_like", ["x"])
    base = torch.zeros((2, 2), dtype=torch.double)
    t = submission.make_ones_like(base)
    assert torch.all(t == 1)
    assert t.shape == base.shape
    assert t.dtype == base.dtype

def test_make_arange():
    assert_signature("make_arange", ["start", "end", "step", "dtype", "device"])
    t = submission.make_arange(0, 5, 2)
    expected = torch.tensor([0, 2, 4])
    assert torch.equal(t, expected)
    t2 = submission.make_arange(0, 4, 1)
    assert t2[-1] == 3

def test_make_linspace():
    assert_signature("make_linspace", ["start", "end", "steps", "dtype", "device"])
    t = submission.make_linspace(0, 10, steps=3) 
    expected = torch.tensor([0., 5., 10.])
    assert torch.allclose(t, expected)
    assert t[-1] == 10

def test_make_randn():
    assert_signature("make_randn", ["shape", "seed", "dtype", "device"])
    t1 = submission.make_randn((10, 10), seed=42)
    assert t1.shape == (10, 10)
    t2 = submission.make_randn((10, 10), seed=42)
    assert torch.equal(t1, t2), "Seeding did not produce identical results."
    t3 = submission.make_randn((10, 10), seed=43)
    assert not torch.equal(t1, t3)

def test_cast_dtype_and_move():
    assert_signature("cast_dtype_and_move", ["x", "device", "dtype"])
    x = torch.tensor([1, 2])
    t = submission.cast_dtype_and_move(x, torch.device("cpu"), torch.float32)
    assert t.dtype == torch.float32

# ==========================================
# 2. SHAPE MANIPULATION TESTS
# ==========================================
def test_reshape_tensor():
    x = torch.arange(6)
    t = submission.reshape_tensor(x, (2, 3))
    assert t.shape == (2, 3)
    assert torch.equal(t, x.view(2, 3))

def test_view_tensor():
    x = torch.arange(6)
    t = submission.view_tensor(x, (2, 3))
    assert t.shape == (2, 3)
    
    x_nc = torch.randn(4, 6)[:, ::2] # non-contiguous
    with pytest.raises(RuntimeError):
        # Must fail if implemented correctly using .view() on non-contiguous input
        submission.view_tensor(x_nc, (12,))

def test_flatten_from_dim():
    x = torch.randn(2, 3, 4)
    t = submission.flatten_from_dim(x, start_dim=1)
    assert t.shape == (2, 12)
    assert torch.equal(t.view(2, 3, 4), x)

def test_add_singleton_dim():
    x = torch.randn(5, 7)
    t = submission.add_singleton_dim(x, dim=1)
    assert t.shape == (5, 1, 7)

def test_remove_singleton_dims():
    x = torch.randn(2, 1, 3)
    t = submission.remove_singleton_dims(x)
    assert t.shape == (2, 3)

def test_transpose_last_two():
    x = torch.randn(2, 3, 4)
    t = submission.transpose_last_two(x)
    assert t.shape == (2, 4, 3)
    assert x[0, 1, 2] == t[0, 2, 1]

def test_permute_bhwc_to_bchw():
    x = torch.randn(2, 5, 5, 3) # B, H, W, C
    t = submission.permute_bhwc_to_bchw(x)
    assert t.shape == (2, 3, 5, 5) # B, C, H, W
    assert x[0, 1, 2, 0] == t[0, 0, 1, 2]

def test_make_contiguous():
    x = torch.randn(4, 6)[:, ::2]
    assert not x.is_contiguous()
    t = submission.make_contiguous(x)
    assert t.is_contiguous()
    assert torch.equal(t, x)

# ==========================================
# 3. INDEXING TESTS
# ==========================================
def test_slice_rows():
    x = torch.arange(12).reshape(4, 3)
    t = submission.slice_rows(x, 1, 3)
    expected = x[1:3, :]
    assert torch.equal(t, expected)

def test_select_columns():
    x = torch.arange(9).reshape(3, 3)
    t = submission.select_columns(x, [0, 2])
    expected = x[:, [0, 2]]
    assert torch.equal(t, expected)

def test_get_diagonal():
    x = torch.tensor([[1, 2], [3, 4]])
    t = submission.get_diagonal(x)
    assert torch.equal(t, torch.tensor([1, 4]))

def test_set_subtensor():
    base = torch.zeros(2, 2)
    t = submission.set_subtensor(base, 0, 1, 5.0)
    assert t[0, 1] == 5.0
    assert base[0, 1] == 0.0, "Original tensor was modified in-place!"

def test_gather_rows():
    x = torch.tensor([[10, 11], [20, 21], [30, 31]])
    idx = torch.tensor([2, 0])
    t = submission.gather_rows(x, idx)
    expected = torch.tensor([[30, 31], [10, 11]])
    assert torch.equal(t, expected)

# ==========================================
# 4. BROADCASTING TESTS
# ==========================================
def test_sum_over_dim():
    x = torch.ones(2, 3)
    t = submission.sum_over_dim(x, 1, False)
    assert t.shape == (2,)
    assert torch.all(t == 3)
    t2 = submission.sum_over_dim(x, 1, True)
    assert t2.shape == (2, 1)

def test_mean_over_dim():
    x = torch.tensor([[1., 2.], [3., 4.]])
    t = submission.mean_over_dim(x, 0)
    assert torch.equal(t, torch.tensor([2., 3.]))

def test_max_over_dim():
    x = torch.tensor([[10., 5.], [3., 20.]])
    val, idx = submission.max_over_dim(x, 1)
    assert torch.equal(val, torch.tensor([10., 20.]))
    assert torch.equal(idx, torch.tensor([0, 1]))

def test_broadcast_add_vector():
    x = torch.zeros(3, 2)
    v = torch.tensor([10.0, 20.0])
    t = submission.broadcast_add_vector(x, v)
    assert torch.all(t[:, 0] == 10.0)
    assert torch.all(t[:, 1] == 20.0)

# ==========================================
# 5. VECTORIZATION TESTS
# ==========================================
def test_concat_tensors():
    t1 = torch.zeros(2, 2)
    t2 = torch.ones(2, 2)
    res = submission.concat_tensors([t1, t2], dim=0)
    assert res.shape == (4, 2)
    assert res[2, 0] == 1

def test_stack_tensors():
    t1 = torch.zeros(2)
    t2 = torch.zeros(2)
    res = submission.stack_tensors([t1, t2], dim=0)
    assert res.shape == (2, 2) 

def test_repeat_tensor():
    x = torch.tensor([[1]])
    t = submission.repeat_tensor(x, (2, 2))
    assert t.shape == (2, 2)
    assert t.storage().nbytes() > x.storage().nbytes()

def test_expand_tensor():
    x = torch.tensor([[1]])
    t = submission.expand_tensor(x, 2, 2)
    assert t.shape == (2, 2)

def test_cumsum_over_dim():
    x = torch.ones(5)
    t = submission.cumsum_over_dim(x)
    assert t[-1] == 5

def test_where_select():
    mask = torch.tensor([True, False])
    a = torch.tensor([1, 1])
    b = torch.tensor([2, 2])
    t = submission.where_select(mask, a, b)
    assert torch.equal(t, torch.tensor([1, 2]))

def test_one_hot():
    assert_no_loops("one_hot")
    indices = torch.tensor([0, 2])
    t = submission.one_hot(indices, 4)
    assert t.shape == (2, 4)
    assert t[0, 0] == 1 and t[0, 1] == 0
    assert t[1, 2] == 1

def test_scatter_add_1d():
    assert_no_loops("scatter_add_1d")
    vals = torch.ones(4)
    idx = torch.tensor([0, 0, 1, 2])
    t = submission.scatter_add_1d(vals, idx, size=3)
    expected = torch.tensor([2., 1., 1.])
    assert torch.equal(t, expected)

def test_batched_token_histogram():
    assert_no_loops("batched_token_histogram")
    tokens = torch.tensor([[0, 1, 1], [2, 2, 0]])
    t = submission.batched_token_histogram(tokens, 3)
    expected = torch.tensor([[1, 2, 0], [1, 0, 2]])
    assert torch.equal(t, expected)

def test_masked_mean():
    x = torch.tensor([1.0, 2.0, 100.0])
    mask = torch.tensor([True, True, False])
    t = submission.masked_mean(x, mask, dim=0)
    assert t.item() == 1.5
    
    mask_all_false = torch.tensor([False, False, False])
    t_zero = submission.masked_mean(x, mask_all_false, dim=0)
    assert t_zero.item() == 0.0

# ==========================================
# 6. EINSUM TESTS
# ==========================================
def test_einsum_linear():
    assert_source_contains("einsum_linear_btd_dh_to_bth", "einsum")
    B, T, D, H = 2, 3, 4, 5
    x = torch.randn(B, T, D)
    W = torch.randn(D, H)
    t = submission.einsum_linear_btd_dh_to_bth(x, W)
    expected = x @ W
    assert torch.allclose(t, expected, atol=1e-5)

def test_einsum_pairwise_dot():
    assert_source_contains("einsum_pairwise_dot", "einsum")
    x = torch.randn(2, 3, 4)
    y = torch.randn(2, 3, 4)
    t = submission.einsum_pairwise_dot(x, y)
    expected = (x * y).sum(dim=-1)
    assert torch.allclose(t, expected, atol=1e-5)

def test_einsum_qk_scores():
    assert_source_contains("einsum_qk_scores", "einsum")
    q = torch.randn(2, 2, 4, 8)
    k = torch.randn(2, 2, 4, 8)
    t = submission.einsum_qk_scores(q, k)
    expected_slice = torch.dot(q[0,0,0,:], k[0,0,0,:])
    assert torch.isclose(t[0,0,0,0], expected_slice)
    assert t.shape == (2, 2, 4, 4)

def test_einsum_apply_attention():
    assert_source_contains("einsum_apply_attention", "einsum")
    w = torch.randn(2, 2, 4, 4)
    v = torch.randn(2, 2, 4, 8)
    t = submission.einsum_apply_attention(w, v)
    assert t.shape == (2, 2, 4, 8)
    expected_first = (w[0,0,0,:,None] * v[0,0,:,:]).sum(dim=0)
    assert torch.allclose(t[0,0,0], expected_first, atol=1e-5)

# ==========================================
# 7. ATTENTION FUNDAMENTALS TESTS
# ==========================================
def test_stable_softmax():
    x = torch.tensor([1000.0, 1000.0])
    t = submission.stable_softmax(x, dim=0)
    assert torch.allclose(t, torch.tensor([0.5, 0.5]))
    assert not torch.isnan(t).any()

def test_masked_fill_tensor():
    x = torch.zeros(2, 2)
    mask = torch.tensor([[True, False], [False, True]])
    t = submission.masked_fill_tensor(x, mask, 99.0)
    assert t[0, 0] == 99.0
    assert t[0, 1] == 0.0
    assert x[0, 0] == 0.0

def test_masked_softmax():
    x = torch.randn(2, 3)
    mask = torch.tensor([[True, False, False], [False, True, True]])
    t = submission.masked_softmax(x, mask, dim=1)
    assert t[0, 0] == 0.0
    assert torch.isclose(t[0, 1:].sum(), torch.tensor(1.0))
    
    # Stability Check (all masked)
    mask_all = torch.tensor([True, True, True])
    t_all = submission.masked_softmax(x[0], mask_all, dim=0)
    assert torch.all(t_all == 0.0)

def test_make_causal_mask():
    T = 3
    t = submission.make_causal_mask(T)
    expected = torch.tensor([
        [False, True, True],
        [False, False, True],
        [False, False, False]
    ])
    assert torch.equal(t, expected)

def test_apply_causal_mask():
    logits = torch.randn(1, 1, 3, 3)
    t = submission.apply_causal_mask(logits, value=-999)
    assert t[0,0,0,1] == -999
    assert t[0,0,0,0] == logits[0,0,0,0]