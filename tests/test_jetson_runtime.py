"""No GPU or profiler required for trace-coverage metadata tests."""
import unittest

from tools.jetson_runtime import trace_coverage


class TraceCoverageTests(unittest.TestCase):
    def test_cpu_only_trace_is_not_zero_gpu_cost(self):
        result = trace_coverage({'traceEvents': [
            {'cat': 'cpu_op', 'name': 'aten::conv2d'},
            {'cat': 'cuda_runtime', 'name': 'cudaLaunchKernel'},
            {'name': 'metadata'},
        ]})
        self.assertFalse(result['cuda_trace_available'])
        self.assertEqual(result['cuda_kernel_events'], 0)
        self.assertIn('CPU-only', result['warning'])

    def test_real_cuda_kernel_events_are_required(self):
        result = trace_coverage({'traceEvents': [
            {'cat': 'kernel', 'name': 'convolution'},
            {'cat': 'kernel', 'name': 'masked_scaled_grad'},
        ]})
        self.assertTrue(result['cuda_trace_available'])
        self.assertEqual(result['cuda_kernel_events'], 2)
        self.assertIsNone(result['warning'])

    def test_empty_trace_is_unavailable(self):
        self.assertFalse(trace_coverage({})['cuda_trace_available'])


if __name__ == '__main__':
    unittest.main()
