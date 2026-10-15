import pytest
import torch

from torch_neuronx.neuron_dynamo_backend import decompositions


@pytest.mark.parametrize("upper", [False, True])
def test_wide_rhs_uses_one_kernel_call(monkeypatch, upper):
    batch, n, columns = 2, 2, 4097
    tile_columns = decompositions._TRIANGULAR_SOLVE_NKI_RHS_TILE_SIZE
    tiles = (columns + tile_columns - 1) // tile_columns

    matrix = torch.eye(n).expand(batch, n, n).clone()
    matrix[:, 0, 1] = 0.25 if upper else 0.0
    matrix[:, 1, 0] = 0.0 if upper else 0.25
    rhs = torch.arange(batch * n * columns, dtype=torch.float32).reshape(batch, n, columns)

    calls = []

    def fake_kernel(matrix_tiles, rhs_tiles, upper_arg, lnc):
        calls.append((matrix_tiles.shape, rhs_tiles.shape, upper_arg, lnc))
        width = rhs_tiles.shape[-1]
        result = torch.linalg.solve_triangular(
            matrix_tiles.reshape(-1, n, n),
            rhs_tiles.reshape(-1, n, width),
            upper=bool(upper_arg),
            unitriangular=True,
        )
        return result.reshape(-1, width)

    monkeypatch.setattr(decompositions, "get_logical_neuron_cores", lambda: 1)
    monkeypatch.setattr(decompositions, "wrapped_triangular_solve_batched_nki", {(1,): fake_kernel})

    actual = decompositions._solve_triangular_nki_block(matrix, rhs, upper=upper)
    expected = torch.linalg.solve_triangular(matrix, rhs, upper=upper, unitriangular=True)

    assert calls == [((batch * tiles * n, n), (batch * tiles * n, tile_columns), int(upper), 1)]
    torch.testing.assert_close(actual, expected)
