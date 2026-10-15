"""arange must not inherit another call's length through the compilation cache.

The dispatch cache key used the default ostream format for a float scalar, which
keeps 6 significant digits. arange(0.0, -8.000001, -4.0) and arange(0.0, -8.0, -4.0)
therefore wrote the same key. Their lengths differ (3 vs 2), so whichever compiled
first returned its length for both, with no error raised.

The key-level checks live in tests/cache/test_cache_key_generator.py. These run the
op, because a key test alone does not prove the caller gets the right data.

These tests are deliberately NOT in tests/cache/: they compile NEFFs, and
tests/cache/test_persistent_caching_e2e.py asserts on the number of entries in the
shared cache directory.
"""

import pytest
import torch


def _assert_matches_cpu(start, end, step, dtype):
    expected = torch.arange(start, end, step, dtype=dtype)
    actual = torch.arange(start, end, step, dtype=dtype, device="neuron")
    assert actual.shape == expected.shape, (
        f"arange({start}, {end}, {step}, {dtype}) gave shape {tuple(actual.shape)}, "
        f"cpu gave {tuple(expected.shape)}: the cache key lost float precision"
    )
    assert torch.equal(actual.cpu(), expected)


# Both orders matter: the cache is shared inside one process, so whichever call
# compiles first installs the entry. Each order uses its own value pair so it
# compiles a fresh entry instead of hitting the other test's.
@pytest.mark.parametrize("dtype", [torch.int32, torch.float32])
def test_arange_longer_result_compiled_first(dtype):
    # ceil(8.000001 / 4) == 3, ceil(8.0 / 4) == 2
    _assert_matches_cpu(0.0, -8.000001, -4.0, dtype)
    _assert_matches_cpu(0.0, -8.0, -4.0, dtype)


@pytest.mark.parametrize("dtype", [torch.int32, torch.float32])
def test_arange_shorter_result_compiled_first(dtype):
    # ceil(4.0 / 2) == 2, ceil(4.000001 / 2) == 3
    _assert_matches_cpu(0.0, 4.0, 2.0, dtype)
    _assert_matches_cpu(0.0, 4.000001, 2.0, dtype)
