"""Numerical and state-lifetime tests; CUDA tests require a real compiled kernel."""
from copy import deepcopy
import os
from unittest.mock import patch

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from src.adapters.bnpa_conv_bn_act_2d import BNPAConvBNAct2D
from src.adapters.frozen_conv_bn_act_minimal_2d import FrozenConvBNActMinimal2D
from src.runtime import configure_runtime_optimizations
from src.utils.bitpack import masked_scaled_grad, pack_bool_mask


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture(params=['cpu', 'cuda'])
def device(request):
    if request.param == 'cuda':
        if not torch.cuda.is_available():
            pytest.skip('CUDA unavailable')
        from src.utils._bitpack_cuda import cuda_extension
        with patch.dict(os.environ, MEMFLORA_BITPACK_BACKEND='cuda'):
            assert cuda_extension() is not None  # never silently test the fallback
            yield torch.device('cuda')
    else:
        yield torch.device('cpu')


def block(device='cpu', activation=2):
    torch.manual_seed(11)
    result = BNPAConvBNAct2D(
        nn.Conv2d(3, 5, 3, padding=1), nn.BatchNorm2d(5),
        [nn.Identity(), nn.ReLU(), nn.ReLU6()][activation], rank=2,
        bnpa_bottleneck_bn='off', fa_port_layout=True,
        post_bn_adapter_scale_by_source_bn=True,
        optimized_post_bn_bnr_off_scaled=True,
    ).to(device)
    with torch.no_grad():
        result.U.weight.normal_(std=.1)
        result.batch_norm.weight.copy_(torch.tensor([-1., .5, 2., -2., 1.], device=device))
        result.batch_norm.running_var.uniform_(.5, 2.)
    return result


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64, torch.float16, torch.bfloat16])
def test_masked_scaled_grad_matches_torch_and_preserves_inputs(device, dtype):
    for shape in [(2, 3, 5, 7), (1, 1, 1, 1), (0, 3, 5, 7)]:
        grad = torch.randn(shape, device=device, dtype=dtype)
        mask = torch.rand(shape, device=device) > .5
        scale = torch.linspace(-2, 2, shape[1], device=device, dtype=dtype)
        packed, _ = pack_bool_mask(mask)
        before = grad.clone()
        with torch.no_grad():
            actual = masked_scaled_grad(grad, packed, scale)
        torch.testing.assert_close(actual, torch.where(mask, grad, 0) * scale[None, :, None, None], rtol=0, atol=0)
        torch.testing.assert_close(grad, before, rtol=0, atol=0)
        if grad.numel():
            assert actual.data_ptr() != grad.data_ptr()


def test_mask_nan_inf_and_noncontiguous_fallback(device):
    grad = torch.tensor([float('nan'), 2., float('inf'), -3., 0., 1.], device=device).reshape(1, 2, 1, 3)
    mask = torch.tensor([False, True, False, True, False, True], device=device).reshape_as(grad)
    packed, _ = pack_bool_mask(mask)
    scale = torch.tensor([float('inf'), -2.], device=device)
    with torch.no_grad():
        actual = masked_scaled_grad(grad, packed, scale)
    torch.testing.assert_close(actual, torch.where(mask, grad, 0) * scale[None, :, None, None], equal_nan=True)
    grad = torch.randn(2, 3, 7, 5, device=device).transpose(2, 3)
    mask = grad > 0
    packed, _ = pack_bool_mask(mask)
    scale = torch.randn(3, device=device)
    with patch('src.utils.bitpack.cuda_extension', side_effect=AssertionError('noncontiguous must fall back')):
        with torch.no_grad():
            actual = masked_scaled_grad(grad, packed, scale)
    torch.testing.assert_close(actual, torch.where(mask, grad, 0) * scale[None, :, None, None])


def test_mask_differentiable_fallback():
    grad = torch.randn(1, 2, 2, 3, dtype=torch.double, requires_grad=True)
    scale = torch.randn(2, dtype=torch.double, requires_grad=True)
    packed, _ = pack_bool_mask(grad.detach() > 0)
    fn = lambda g, s: masked_scaled_grad(g, packed, s)
    assert torch.autograd.gradcheck(fn, (grad, scale))
    assert torch.autograd.gradgradcheck(fn, (grad, scale))
    with pytest.raises(ValueError):
        masked_scaled_grad(grad, packed[:0], scale)


@pytest.mark.parametrize('activation', [0, 1, 2])
@pytest.mark.parametrize('input_grad', [False, True])
def test_adapter_matches_reference_and_native_autograd(device, activation, input_grad):
    reference = block(device, activation)
    fast, native = deepcopy(reference), deepcopy(reference)
    configure_runtime_optimizations(fast)
    xs = [torch.randn(2, 3, 5, 7, device=device)]
    xs += [xs[0].clone(), xs[0].clone()]
    xs = [x.requires_grad_(input_grad) for x in xs]
    # Independent native-autograd equation, not just two copies of custom code.
    scale, shift = native._bn_scale_shift()
    z = native.base_conv(xs[2]) * scale[None, :, None, None] + shift[None, :, None, None]
    z = z + native.scale * native.U(native.P(xs[2])) * scale[None, :, None, None]
    yn = [lambda x: x, F.relu, F.relu6][activation](z)
    ys = [reference(xs[0]), fast(xs[1]), yn]
    gradient = torch.randn_like(yn)
    for y in ys:
        torch.testing.assert_close(y, yn, rtol=3e-5, atol=3e-6)
        y.backward(gradient)
    for candidate in (reference, fast):
        torch.testing.assert_close(candidate.U.weight.grad, native.U.weight.grad, rtol=5e-5, atol=2e-5)
        assert candidate.P.weight.grad is None
        assert candidate.base_conv.weight.grad is None
    if input_grad:
        for x in xs[:2]:
            torch.testing.assert_close(x.grad, xs[2].grad, rtol=5e-5, atol=2e-5)


@pytest.mark.parametrize('train_bias', [False, True])
def test_frozen_depthwise_backward(device, train_bias):
    ref = FrozenConvBNActMinimal2D(
        nn.Conv2d(3, 3, 3, padding=1, groups=3, bias=True),
        nn.BatchNorm2d(3), nn.ReLU6(), activation_mask_mode='bitpack',
        keep_bn_bias_trainable_in_eval=train_bias,
        keep_base_bias_trainable=train_bias,
    ).to(device)
    ref.train()
    fast = configure_runtime_optimizations(deepcopy(ref))
    x = torch.randn(2, 3, 5, 7, device=device, requires_grad=True)
    other = x.detach().clone().requires_grad_()
    y, z = ref(x), fast(other)
    torch.testing.assert_close(y, z)
    y.square().sum().backward(); z.square().sum().backward()
    torch.testing.assert_close(x.grad, other.grad)
    for name, p in ref.named_parameters():
        q = dict(fast.named_parameters())[name]
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad)


def test_cache_reuses_and_invalidates_coefficients():
    layer = configure_runtime_optimizations(block())
    a, _ = layer._bn_scale_shift()
    assert layer._bn_scale_shift()[0] is a
    for field in ['running_mean', 'running_var', 'weight', 'bias']:
        with torch.no_grad():
            getattr(layer.batch_norm, field).add_(.25)
        b, shift = layer._bn_scale_shift()
        assert b is not a
        expected = layer._compute_bn_scale_shift()
        torch.testing.assert_close(b, expected[0]); torch.testing.assert_close(shift, expected[1])
        a = b
    layer.batch_norm.eps *= 2
    assert layer._bn_scale_shift()[0] is not a
    a, _ = layer._bn_scale_shift()
    layer.load_state_dict(deepcopy(layer.state_dict()))
    assert layer._bn_scale_shift()[0] is not a
    layer.double()
    assert layer._bn_scale_shift()[0].dtype == torch.double
    with torch.inference_mode():
        layer._bn_scale_shift()
    assert layer._runtime_bn_scale is None
    assert not layer._bn_scale_shift()[0].is_inference()
    layer.batch_norm.bias.requires_grad_(True)
    assert layer._bn_scale_shift()[1].requires_grad
    assert layer._runtime_bn_scale is None


def test_cache_counted_as_buffers_not_checkpoint_and_no_extra_saved_activations():
    ref = block()
    fast = configure_runtime_optimizations(deepcopy(ref))
    fast._bn_scale_shift()
    assert set(ref.state_dict()) == set(fast.state_dict())
    names = dict(fast.named_buffers())
    assert names['_runtime_bn_scale'].numel() == 5
    assert names['_runtime_bn_shift'].numel() == 5
    seen = []
    for layer in (ref, fast):
        shapes = []
        def pack(tensor):
            shapes.append((tuple(tensor.shape), tensor.dtype))
            return tensor
        x = torch.randn(2, 3, 5, 7, requires_grad=True)
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            y = layer(x)
        y.sum().backward()
        seen.append(shapes)
    assert seen[0] == seen[1]
    configure_runtime_optimizations(fast, False)
    assert not any(n.startswith('_runtime_bn') for n, _ in fast.named_buffers())


def test_optimized_training_restores_mode_after_evaluation():
    from experiments.benchmark_training import fit_steps_best
    for optimized, expected_train_calls in [(False, 7), (True, 3)]:
        model = nn.Sequential(nn.BatchNorm1d(3), nn.Linear(3, 2))
        configure_runtime_optimizations(model, optimized)
        loader = [(torch.randn(4, 3), torch.tensor([0, 1, 0, 1]))]
        def evaluate(m, *args):
            m.eval()
            return {'loss': 1., 'accuracy': 0., 'macro_f1': 0.}
        states = []
        handle = model.register_forward_pre_hook(lambda m, args: states.append((m.training, m[0].training)))
        with patch.object(model, 'train', wraps=model.train) as train:
            with patch('experiments.benchmark_training.prepare_batch_inputs', side_effect=lambda x, _: x):
                with patch('experiments.benchmark_training.evaluate', side_effect=evaluate):
                    fit_steps_best(model, loader, loader, 7, .001, 0., 3, torch.device('cpu'), 'test', force_bn_eval=True)
            assert sum(not c.args or c.args[0] is True for c in train.call_args_list) == expected_train_calls
        handle.remove()
        assert states == [(True, False)] * 7


@pytest.mark.parametrize('backbone,layers', [('t_resnet_official', 'all'), ('mobilenet_v2', 'pointwise_only')])
def test_complete_backbones_match_reference(device, backbone, layers):
    from experiments.benchmark_cli import parse_args, apply_dataset_defaults
    from experiments.benchmark_common import prepare_batch_inputs
    from experiments.minimal_methods_benchmark import build_backbone, configure_method
    args = parse_args(['--backbone', backbone, '--adapter-layers', layers, '--mobilenet-v2-pretrained', 'false'])
    apply_dataset_defaults(args)
    args.method, args.rank, args.bnpa_bottleneck_bn = 'bnpa_fa_postbn_bnr_off_scaled', 2, 'on'
    ref = build_backbone(args).to(device)
    configure_method(ref, args)
    ref.to(device)
    fast = configure_runtime_optimizations(deepcopy(ref))
    x = prepare_batch_inputs(torch.randn(2, 97, 60, device=device), backbone)
    ref.train(); fast.train()
    # Both method wrappers enforce their source-BN policy.
    from src.train import freeze_bn_eval
    freeze_bn_eval(ref); freeze_bn_eval(fast)
    torch.manual_seed(1)
    a = ref(x)
    torch.manual_seed(1)
    b = fast(x)
    torch.testing.assert_close(a, b, rtol=3e-5, atol=3e-6)
    a.square().mean().backward(); b.square().mean().backward()
    for name, p in ref.named_parameters():
        q = dict(fast.named_parameters())[name]
        assert (p.grad is None) == (q.grad is None)
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, rtol=5e-5, atol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_fused_mask_reduces_temporary_allocation_and_uses_current_stream():
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), torch.no_grad():
        grad = torch.randn(2, 16, 33, 35, device='cuda')
        scale = torch.randn(16, device='cuda')
        packed, _ = pack_bool_mask(grad > 0)
        # Warm up the torch fallback's tiny bit-weight cache first.
        with patch.dict(os.environ, MEMFLORA_BITPACK_BACKEND='torch'):
            warm = masked_scaled_grad(grad, packed, scale)
        del warm
        torch.cuda.synchronize()
        peaks = []
        values = []
        for backend in ['torch', 'cuda']:
            baseline = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            with patch.dict(os.environ, MEMFLORA_BITPACK_BACKEND=backend):
                result = masked_scaled_grad(grad, packed, scale)
            torch.cuda.synchronize()
            peaks.append(torch.cuda.max_memory_allocated() - baseline)
            values.append(result.cpu())
            del result
        torch.testing.assert_close(values[0], values[1], rtol=0, atol=0)
        assert peaks[1] < peaks[0]
