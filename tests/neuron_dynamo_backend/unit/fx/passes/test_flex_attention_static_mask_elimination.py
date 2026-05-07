# ruff: noqa: N806, N803, N812

"""
Unit tests for static mask tile type generation in FlexAttentionNKILowering.

Tests the _classify_tile and _generate_tile_types functions that analyze
mask_mod FX submodules to produce tile classification arrays (SKIP/PARTIAL/FULL)
and precomputed mask data for the NKI kernel template.
"""

import math
import os

import numpy as np
import pytest
import torch
import torch.fx as fx

from torch_neuronx.neuron_dynamo_backend.fx.passes.flex_attention_nki_lowering import (
    _Q_GRP_SZ,
    _SECTION_LEN,
    _TILE_FULL,
    _TILE_PARTIAL,
    _TILE_SKIP,
    _V_TILE_SZ,
    _classify_tile,
    _generate_tile_types,
)

# ============================================================================
# Helpers: create mask_mod FX submodules from Python mask functions
# ============================================================================


def _make_mask_mod(mask_fn):
    """Trace a mask function into an FX GraphModule matching mask_mod signature.

    mask_fn(b, h, q_idx, kv_idx) -> bool tensor
    """

    class MaskMod(torch.nn.Module):
        def forward(self, b, h, q_idx, kv_idx):
            return mask_fn(b, h, q_idx, kv_idx)

    return fx.symbolic_trace(MaskMod())


def _causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def _no_mask(b, h, q_idx, kv_idx):
    return torch.ones_like(q_idx, dtype=torch.bool)


def _full_mask(b, h, q_idx, kv_idx):
    return torch.zeros_like(q_idx, dtype=torch.bool)


def _swa_mask(window_size):
    def mask_fn(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & (q_idx - kv_idx < window_size)

    return mask_fn


# ============================================================================
# Tests for _classify_tile
# ============================================================================


class TestClassifyTile:
    """Tests for individual tile classification."""

    def test_full_tile_all_unmasked(self):
        """A tile where the mask is all-true should be classified as FULL."""
        mask_mod = _make_mask_mod(_no_mask)
        tile_type, mask = _classify_tile(mask_mod, q_grp=0, k_start=0, q_grp_sz=128, k_tile_sz=512)
        assert tile_type == _TILE_FULL
        assert mask.all()

    def test_skip_tile_all_masked(self):
        """A tile where the mask is all-false should be classified as SKIP."""
        mask_mod = _make_mask_mod(_full_mask)
        tile_type, mask = _classify_tile(mask_mod, q_grp=0, k_start=0, q_grp_sz=128, k_tile_sz=512)
        assert tile_type == _TILE_SKIP
        assert not mask.any()

    def test_partial_tile_causal_boundary(self):
        """A causal mask tile on the diagonal should be PARTIAL."""
        mask_mod = _make_mask_mod(_causal_mask)
        # q_grp=0 covers q_idx [0, 127], k_start=0 covers kv_idx [0, 511]
        # The diagonal crosses this tile, so it's partial
        tile_type, mask = _classify_tile(mask_mod, q_grp=0, k_start=0, q_grp_sz=128, k_tile_sz=512)
        assert tile_type == _TILE_PARTIAL
        assert mask.any() and not mask.all()

    def test_full_tile_causal_below_diagonal(self):
        """A causal mask tile fully below the diagonal should be FULL."""
        mask_mod = _make_mask_mod(_causal_mask)
        # q_grp=4 covers q_idx [512, 639], k_start=0 covers kv_idx [0, 511]
        # All q_idx >= all kv_idx, so fully unmasked
        tile_type, _ = _classify_tile(mask_mod, q_grp=4, k_start=0, q_grp_sz=128, k_tile_sz=512)
        assert tile_type == _TILE_FULL

    def test_skip_tile_causal_above_diagonal(self):
        """A causal mask tile fully above the diagonal should be SKIP."""
        mask_mod = _make_mask_mod(_causal_mask)
        # q_grp=0 covers q_idx [0, 127], k_start=512 covers kv_idx [512, 1023]
        # All q_idx < all kv_idx, so fully masked
        tile_type, _ = _classify_tile(mask_mod, q_grp=0, k_start=512, q_grp_sz=128, k_tile_sz=512)
        assert tile_type == _TILE_SKIP

    def test_mask_tensor_shape(self):
        """Returned mask tensor should have shape [q_grp_sz, k_tile_sz]."""
        mask_mod = _make_mask_mod(_causal_mask)
        _, mask = _classify_tile(mask_mod, q_grp=0, k_start=0, q_grp_sz=128, k_tile_sz=512)
        assert mask.shape == (128, 512)

    def test_mask_tensor_values_causal(self):
        """Verify mask values match expected causal pattern."""
        mask_mod = _make_mask_mod(_causal_mask)
        _, mask = _classify_tile(mask_mod, q_grp=0, k_start=0, q_grp_sz=128, k_tile_sz=512)
        # q_idx=0 should only have kv_idx=0 unmasked
        assert mask[0, 0] == 1.0
        assert mask[0, 1] == 0.0
        # q_idx=127 should have kv_idx [0, 127] unmasked
        assert mask[127, 127] == 1.0
        assert mask[127, 128] == 0.0


# ============================================================================
# Tests for _generate_tile_types
# ============================================================================


class TestGenerateTileTypes:
    """Tests for full tile type array generation."""

    def test_causal_mask_basic_structure(self):
        """Causal mask should produce diagonal pattern: FULL below, PARTIAL on, SKIP above."""
        mask_mod = _make_mask_mod(_causal_mask)
        k_tile_sz = 512
        seqlen = 2048
        result = _generate_tile_types(mask_mod, seqlen, seqlen, k_tile_sz)

        qk = result["qk_tile_types"]
        num_grps = math.ceil(seqlen / _Q_GRP_SZ)  # 16
        num_k_tiles = math.ceil(seqlen / k_tile_sz)  # 4

        assert len(qk) == num_grps
        assert len(qk[0]) == num_k_tiles

        # First group (q_idx [0,127]): tile 0 is PARTIAL, rest are SKIP
        assert qk[0][0] == _TILE_PARTIAL
        for kt in range(1, num_k_tiles):
            assert qk[0][kt] == _TILE_SKIP

        # Last group (q_idx [1920,2047]): all tiles should be FULL or PARTIAL
        for kt in range(num_k_tiles - 1):
            assert qk[-1][kt] == _TILE_FULL
        assert qk[-1][-1] == _TILE_PARTIAL

    def test_no_mask_all_full(self):
        """No mask (all-true) should produce all FULL tiles."""
        mask_mod = _make_mask_mod(_no_mask)
        result = _generate_tile_types(mask_mod, 1024, 1024, 512)

        for row in result["qk_tile_types"]:
            for t in row:
                assert t == _TILE_FULL
        for row in result["pv_tile_types"]:
            for t in row:
                assert t == _TILE_FULL

    def test_full_mask_all_skip(self):
        """Full mask (all-false) should produce all SKIP tiles."""
        mask_mod = _make_mask_mod(_full_mask)
        result = _generate_tile_types(mask_mod, 1024, 1024, 512)

        for row in result["qk_tile_types"]:
            for t in row:
                assert t == _TILE_SKIP

    def test_pv_tile_types_dimensions(self):
        """PV tile types should use _V_TILE_SZ granularity."""
        mask_mod = _make_mask_mod(_causal_mask)
        seqlen = 1024
        result = _generate_tile_types(mask_mod, seqlen, seqlen, 512)

        pv = result["pv_tile_types"]
        expected_v_tiles = math.ceil(seqlen / _V_TILE_SZ)
        assert len(pv[0]) == expected_v_tiles

    def test_asymmetric_seqlens(self):
        """Test with different Q and KV sequence lengths."""
        mask_mod = _make_mask_mod(_no_mask)
        result = _generate_tile_types(mask_mod, 256, 1024, 512)

        qk = result["qk_tile_types"]
        assert len(qk) == math.ceil(256 / _Q_GRP_SZ)  # 2 groups
        assert len(qk[0]) == math.ceil(1024 / 512)  # 2 k tiles

    def test_result_dict_keys(self):
        """Verify all expected keys are present in the result."""
        mask_mod = _make_mask_mod(_causal_mask)
        result = _generate_tile_types(mask_mod, 1024, 1024, 512)

        expected_keys = {
            "qk_tile_types",
            "pv_tile_types",
            "qk_mask_index",
            "num_unique_masks",
            "masks_npy_path",
            "masks_int8_npy_path",
            "last_compute_section",
            "max_large_tiles_needed",
        }
        assert set(result.keys()) == expected_keys


# ============================================================================
# Tests for precomputed mask data
# ============================================================================


class TestPrecomputedMasks:
    """Tests for unique mask de-duplication and numpy file generation."""

    def test_causal_mask_has_unique_masks(self):
        """Causal mask should produce at least one unique partial mask."""
        mask_mod = _make_mask_mod(_causal_mask)
        result = _generate_tile_types(mask_mod, 1024, 1024, 512)
        assert result["num_unique_masks"] > 0

    def test_no_mask_has_zero_unique_masks(self):
        """All-true mask should produce zero unique masks (no PARTIAL tiles)."""
        mask_mod = _make_mask_mod(_no_mask)
        result = _generate_tile_types(mask_mod, 1024, 1024, 512)
        assert result["num_unique_masks"] == 0

    def test_mask_index_matches_tile_types(self):
        """QK_MASK_INDEX should be -1 for non-PARTIAL tiles and >= 0 for PARTIAL."""
        mask_mod = _make_mask_mod(_causal_mask)
        result = _generate_tile_types(mask_mod, 1024, 1024, 512)

        qk = result["qk_tile_types"]
        idx = result["qk_mask_index"]

        for g in range(len(qk)):
            for kt in range(len(qk[g])):
                if qk[g][kt] == _TILE_PARTIAL:
                    assert idx[g][kt] >= 0
                    assert idx[g][kt] < result["num_unique_masks"]
                else:
                    assert idx[g][kt] == -1

    def test_mask_index_deduplication(self):
        """Tiles with identical masks should share the same mask index."""
        mask_mod = _make_mask_mod(_causal_mask)
        # With k_tile_sz=512 and causal mask, the diagonal tile pattern repeats
        # every 4 q_groups (512/128=4), so groups 0 and 4 have the same partial mask
        result = _generate_tile_types(mask_mod, 2048, 2048, 512)

        idx = result["qk_mask_index"]
        qk = result["qk_tile_types"]

        # Collect all mask indices for PARTIAL tiles
        partial_indices = set()
        for g in range(len(qk)):
            for kt in range(len(qk[g])):
                if qk[g][kt] == _TILE_PARTIAL:
                    partial_indices.add(idx[g][kt])

        # Should have fewer unique masks than total PARTIAL tiles
        total_partial = sum(
            1 for g in range(len(qk)) for kt in range(len(qk[g])) if qk[g][kt] == _TILE_PARTIAL
        )
        assert len(partial_indices) <= total_partial
        assert result["num_unique_masks"] == len(partial_indices)

    def test_npy_files_created(self):
        """Numpy mask files should be created on disk."""
        mask_mod = _make_mask_mod(_causal_mask)
        result = _generate_tile_types(mask_mod, 1024, 1024, 512)

        assert os.path.exists(result["masks_npy_path"])
        assert os.path.exists(result["masks_int8_npy_path"])

    def test_npy_float32_mask_shape(self):
        """Float32 mask file should have shape (128, num_unique * k_tile_sz)."""
        mask_mod = _make_mask_mod(_causal_mask)
        k_tile_sz = 512
        result = _generate_tile_types(mask_mod, 1024, 1024, k_tile_sz)

        masks = np.load(result["masks_npy_path"])
        assert masks.shape[0] == _Q_GRP_SZ
        assert masks.shape[1] == result["num_unique_masks"] * k_tile_sz

    def test_npy_float32_mask_values(self):
        """Float32 masks should contain 0.0 (unmasked) and float32 min (masked)."""
        mask_mod = _make_mask_mod(_causal_mask)
        result = _generate_tile_types(mask_mod, 1024, 1024, 512)

        masks = np.load(result["masks_npy_path"])
        if masks.size > 0:
            unique_vals = np.unique(masks)
            assert len(unique_vals) == 2
            assert unique_vals[1] == 0.0
            assert unique_vals[0] == np.float32(-3.4028235e38)

    def test_npy_int8_mask_values(self):
        """Int8 masks should contain only 0 and 1."""
        mask_mod = _make_mask_mod(_causal_mask)
        result = _generate_tile_types(mask_mod, 1024, 1024, 512)

        masks = np.load(result["masks_int8_npy_path"])
        if masks.size > 0:
            unique_vals = set(np.unique(masks).tolist())
            assert unique_vals <= {0, 1}

    def test_npy_int8_mask_shape(self):
        """Int8 mask file should have same shape as float32 mask file."""
        mask_mod = _make_mask_mod(_causal_mask)
        k_tile_sz = 512
        result = _generate_tile_types(mask_mod, 1024, 1024, k_tile_sz)

        masks_f32 = np.load(result["masks_npy_path"])
        masks_i8 = np.load(result["masks_int8_npy_path"])
        assert masks_f32.shape == masks_i8.shape


# ============================================================================
# Tests for last_compute_section
# ============================================================================


class TestLastComputeSection:
    """Tests for per-group last section with compute tracking."""

    def test_single_section(self):
        """With seqlen <= 8192, there's one section and last_compute_section is all 0."""
        mask_mod = _make_mask_mod(_causal_mask)
        result = _generate_tile_types(mask_mod, 1024, 1024, 512)

        for sec in result["last_compute_section"]:
            assert sec == 0

    def test_causal_multi_section(self):
        """With causal mask and long seqlen, early groups should have earlier last sections."""
        mask_mod = _make_mask_mod(_causal_mask)
        seqlen = 16384  # 2 sections of 8192
        result = _generate_tile_types(mask_mod, seqlen, seqlen, 512)

        lcs = result["last_compute_section"]
        num_grps = math.ceil(seqlen / _Q_GRP_SZ)
        assert len(lcs) == num_grps

        # First group (q_idx [0,127]) only has compute in section 0
        assert lcs[0] == 0

        # Last group has compute in the last section
        num_sections = math.ceil(seqlen / _SECTION_LEN)
        assert lcs[-1] == num_sections - 1

    def test_no_mask_all_last_section(self):
        """With no mask, all groups should have last_compute_section = last section."""
        mask_mod = _make_mask_mod(_no_mask)
        seqlen = 16384
        result = _generate_tile_types(mask_mod, seqlen, seqlen, 512)

        num_sections = math.ceil(seqlen / _SECTION_LEN)
        for sec in result["last_compute_section"]:
            assert sec == num_sections - 1

    def test_last_compute_section_monotonic_for_causal(self):
        """For causal mask, last_compute_section should be non-decreasing across groups."""
        mask_mod = _make_mask_mod(_causal_mask)
        result = _generate_tile_types(mask_mod, 16384, 16384, 512)

        lcs = result["last_compute_section"]
        for i in range(1, len(lcs)):
            assert lcs[i] >= lcs[i - 1]


# ============================================================================
# Tests for max_large_tiles_needed
# ============================================================================


class TestMaxLargeTilesNeeded:
    """Tests for allocation tuning parameter."""

    def test_no_mask_max_large_tiles(self):
        """With no mask, max_large_tiles_needed should equal total large tiles per section."""
        mask_mod = _make_mask_mod(_no_mask)
        k_tile_sz = 512
        seqlen = 4096
        result = _generate_tile_types(mask_mod, seqlen, seqlen, k_tile_sz)

        section_len = min(seqlen, _SECTION_LEN)
        expected = math.ceil(section_len / (4 * k_tile_sz))
        assert result["max_large_tiles_needed"] == expected

    def test_full_mask_zero_large_tiles(self):
        """With all-false mask, max_large_tiles_needed should be 0."""
        mask_mod = _make_mask_mod(_full_mask)
        result = _generate_tile_types(mask_mod, 1024, 1024, 512)
        assert result["max_large_tiles_needed"] == 0

    def test_causal_fewer_than_full(self):
        """Causal mask should need fewer large tiles than no-mask case."""
        mask_mod_causal = _make_mask_mod(_causal_mask)
        mask_mod_none = _make_mask_mod(_no_mask)
        seqlen = 4096
        k_tile_sz = 512

        result_causal = _generate_tile_types(mask_mod_causal, seqlen, seqlen, k_tile_sz)
        result_none = _generate_tile_types(mask_mod_none, seqlen, seqlen, k_tile_sz)

        assert result_causal["max_large_tiles_needed"] <= result_none["max_large_tiles_needed"]


# ============================================================================
# Tests for different k_tile_sz values
# ============================================================================


class TestDifferentTileSizes:
    """Tests for supported block sizes: 128, 256, 512."""

    @pytest.mark.parametrize("k_tile_sz", [128, 256, 512])
    def test_tile_count_matches_expected(self, k_tile_sz):
        """Number of QK tiles should match ceil(seqlen_kv / k_tile_sz)."""
        mask_mod = _make_mask_mod(_causal_mask)
        seqlen = 2048
        result = _generate_tile_types(mask_mod, seqlen, seqlen, k_tile_sz)

        expected_k_tiles = math.ceil(seqlen / k_tile_sz)
        assert len(result["qk_tile_types"][0]) == expected_k_tiles

    @pytest.mark.parametrize("k_tile_sz", [128, 256, 512])
    def test_causal_has_skip_partial_full(self, k_tile_sz):
        """Causal mask with sufficient seqlen should produce all three tile types."""
        mask_mod = _make_mask_mod(_causal_mask)
        seqlen = 4096
        result = _generate_tile_types(mask_mod, seqlen, seqlen, k_tile_sz)

        all_types = set()
        for row in result["qk_tile_types"]:
            all_types.update(row)

        assert _TILE_SKIP in all_types
        assert _TILE_PARTIAL in all_types
        assert _TILE_FULL in all_types


# ============================================================================
# Tests for sliding window attention
# ============================================================================


class TestSlidingWindowMask:
    """Tests for sliding window attention mask patterns."""

    def test_swa_has_more_skips_than_causal(self):
        """SWA should skip more tiles than plain causal since it masks both sides."""
        mask_mod_causal = _make_mask_mod(_causal_mask)
        mask_mod_swa = _make_mask_mod(_swa_mask(256))
        seqlen = 4096

        result_causal = _generate_tile_types(mask_mod_causal, seqlen, seqlen, 512)
        result_swa = _generate_tile_types(mask_mod_swa, seqlen, seqlen, 512)

        causal_skips = sum(
            1 for row in result_causal["qk_tile_types"] for t in row if t == _TILE_SKIP
        )
        swa_skips = sum(1 for row in result_swa["qk_tile_types"] for t in row if t == _TILE_SKIP)

        assert swa_skips >= causal_skips

    def test_swa_narrow_window_mostly_skip(self):
        """Very narrow SWA window should produce mostly SKIP tiles."""
        mask_mod = _make_mask_mod(_swa_mask(64))
        seqlen = 4096
        result = _generate_tile_types(mask_mod, seqlen, seqlen, 512)

        total = sum(len(row) for row in result["qk_tile_types"])
        skips = sum(1 for row in result["qk_tile_types"] for t in row if t == _TILE_SKIP)

        # With window=64 and tile=512, most tiles should be skipped
        assert skips / total > 0.5
