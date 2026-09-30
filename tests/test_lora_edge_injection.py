"""Tiny CPU-only models verify factory selection and gradient equivalence."""
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("jetson_profile_adapter_test", ROOT / "tools/jetson_profile.py")
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)

try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    torch = None

if torch is not None:
    from torch import nn
    from src.models.adapter_injection import inject_adapters
    from src.adapters.lora_edge import (
        LoRAEdgeConv2d, LoRAEdgeConv2dOptimized, LoRAEdgeConv2dOptimizedV2,
    )


@unittest.skipIf(torch is None, "CPU PyTorch required")
class LoRAEdgeRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_factory_selects_requested_class_for_both_backbones(self):
        for backbone, layers in [("t_resnet", "all"), ("mobilenet_v2", "pointwise_only")]:
            for method, expected in [("lora_edge_optimized", LoRAEdgeConv2dOptimized),
                                     ("lora_edge_optimized_v2", LoRAEdgeConv2dOptimizedV2)]:
                with self.subTest(backbone=backbone, method=method):
                    model = nn.Sequential(nn.Conv2d(4, 6, 3), nn.Conv2d(6, 6, 1))
                    replaced = inject_adapters(model, method=method, rank=2,
                                               backbone=backbone, adapter_layers=layers)
                    self.assertEqual(len(replaced), 2 if backbone == "t_resnet" else 1)
                    for name in replaced:
                        self.assertIs(type(model.get_submodule(name)), expected)
                    if method == "lora_edge_optimized":
                        counts = profile.audit_method_implementation(model, "lora_edge")
                        self.assertEqual(counts["LoRAEdgeConv2dOptimized"], len(replaced))

    def test_profiler_rejects_plain_v2_missing_and_mixed_adapters(self):
        for kind in ["plain", "v2", "missing", "mixed"]:
            with self.subTest(kind=kind):
                base = nn.Conv2d(4, 6, 3)
                if kind == "missing":
                    model = nn.Sequential(base)
                elif kind == "v2":
                    model = nn.Sequential(LoRAEdgeConv2dOptimizedV2(base, 2))
                elif kind == "mixed":
                    model = nn.Sequential(LoRAEdgeConv2dOptimized(base, 2),
                                          LoRAEdgeConv2d(base, 2))
                else:
                    model = nn.Sequential(LoRAEdgeConv2d(base, 2))
                with self.assertRaisesRegex(RuntimeError, "implementation mismatch"):
                    profile.audit_method_implementation(model, "lora_edge")

    def test_forward_and_gradients_match_unoptimized_math(self):
        for groups in [1, 2]:
            for input_grad in [False, True]:
                with self.subTest(groups=groups, input_grad=input_grad):
                    torch.manual_seed(5)
                    base = nn.Conv2d(4, 6, 3, padding=1, groups=groups)
                    plain = LoRAEdgeConv2d(base, 2)
                    optimized = LoRAEdgeConv2dOptimized(base, 2)
                    with torch.no_grad():
                        plain.g1.normal_(std=.1)
                    optimized.load_state_dict(plain.state_dict())
                    x1 = torch.randn(2, 4, 5, 5, requires_grad=input_grad)
                    x2 = x1.detach().clone().requires_grad_(input_grad)
                    y1, y2 = plain(x1), optimized(x2)
                    torch.testing.assert_close(y1, y2, atol=2e-6, rtol=2e-5)
                    y1.square().mean().backward()
                    y2.square().mean().backward()
                    torch.testing.assert_close(plain.g1.grad, optimized.g1.grad, atol=2e-6, rtol=2e-5)
                    if input_grad:
                        torch.testing.assert_close(x1.grad, x2.grad, atol=2e-6, rtol=2e-5)

    def test_optimized_backward_does_not_save_full_input(self):
        x = torch.randn(2, 4, 5, 5, requires_grad=True)
        for cls in [LoRAEdgeConv2d, LoRAEdgeConv2dOptimized]:
            with self.subTest(cls=cls.__name__):
                layer = cls(nn.Conv2d(4, 6, 3), 2)
                addresses = []
                def pack(tensor):
                    addresses.append(tensor.untyped_storage().data_ptr())
                    return tensor.detach()
                with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
                    layer(x).sum().backward()
                if cls is LoRAEdgeConv2dOptimized:
                    self.assertNotIn(x.untyped_storage().data_ptr(), addresses)
                else:
                    self.assertIn(x.untyped_storage().data_ptr(), addresses)


if __name__ == "__main__":
    unittest.main()
