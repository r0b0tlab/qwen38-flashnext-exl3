import types
from serve_openai import apply_offload_config


def test_flags_map():
    cfg = types.SimpleNamespace(infer_params = types.SimpleNamespace(
        moe_cpu_split = 0, moe_cpu_threads = None))
    args = types.SimpleNamespace(moe_cpu_split = 320, moe_cpu_threads = 6)
    out = apply_offload_config(cfg, args)
    assert (out.infer_params.moe_cpu_split, out.infer_params.moe_cpu_threads) == (320, 6)


def test_defaults_untouched():
    cfg = types.SimpleNamespace(infer_params = types.SimpleNamespace(
        moe_cpu_split = 0, moe_cpu_threads = None))
    args = types.SimpleNamespace(moe_cpu_split = 0, moe_cpu_threads = None)
    out = apply_offload_config(cfg, args)
    assert out.infer_params.moe_cpu_split == 0
    assert out.infer_params.moe_cpu_threads is None
