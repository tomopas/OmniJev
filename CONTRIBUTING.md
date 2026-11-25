# Contributing to OmniJev

Thank you for your interest in **OmniJev**!

## Development Setup

```bash
pip install -e ".[dev]"
omnijev info
omnijev summary
```

If you plan to run the GPU benchmark scripts in `experiments/` (Qwen2.5, SmolVLM-256M, Meta EnCodec-24kHz, MuJoCo `Reacher-v5` / `Pusher-v5`, HuggingFace `datasets`, 4-bit NF4 `bitsandbytes`), install the optional benchmark extras:

```bash
pip install -e ".[dev,benchmarks]"
```

## Running Verification Tests

The CPU-runnable unit, mathematical-invariant, API, and packaging suite runs in ~10 seconds without downloading external weights:

```bash
pytest tests/test_omnijev_suite.py -v
```

Please ensure:
1. All tests in `tests/test_omnijev_suite.py` pass (`18/18`), including export completeness (`omnijev.__all__`), local link integrity across `README.md` and `docs/*.md`, and zero hardcoded local paths.
2. Any new mathematical claim or benchmark table row in `docs/` is backed by an executable script in `experiments/` and its corresponding saved `.json` log.
3. All code preserves backwards compatibility across `src/omnijev/`.
