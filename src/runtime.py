"""Opt-in runtime improvements; historical reference execution stays default."""

from src.adapters._common import AdapterBlockCommon


def configure_runtime_optimizations(model, enabled=True):
    """Enable fused packed backward and mutation-aware frozen-BN caching.

    Derived BN tensors are registered nonpersistent buffers, so memory profilers
    count them. Call after adapter injection. Disabling discards the caches.
    Do not mutate BN tensors via .data (which bypasses mutation tracking); after
    untracked writes call this function again to invalidate cached coefficients.
    """
    model._memflora_runtime_optimized = bool(enabled)
    for module in model.modules():
        if isinstance(module, AdapterBlockCommon):
            module.set_runtime_optimizations(enabled)
    return model
