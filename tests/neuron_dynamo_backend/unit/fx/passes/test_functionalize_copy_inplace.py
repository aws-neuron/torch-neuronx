from unittest.mock import MagicMock

import pytest
import torch
import torch.fx as fx
import torch.nn as nn
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx import GraphModule, symbolic_trace
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from torch_neuronx.neuron_dynamo_backend.fx.io_mutation import (
    IOMutation,
    MutationKind,
    PassMutations,
    apply_output_mutations,
    get_alias_output_buffer_map,
    get_mutation_copy_back_inputs,
)
from torch_neuronx.neuron_dynamo_backend.fx.passes.functionalize_copy_inplace_result import (
    FunctionalizeCopyInplacePass,
)
from torch_neuronx.neuron_dynamo_backend.fx.passes.neuron_fx_pass import NeuronFXPassState

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def pass_instance():
    """Fresh pass instance for each test."""
    return FunctionalizeCopyInplacePass()


class TestNoMutationDetected:
    """Tests where no mutation aliasing should be detected."""

    def test_transpose_chain_no_mutation(self, pass_instance):
        """
        IR:
        ```
        def forward(self, arg0_1):
            t = torch.ops.aten.t.default(arg0_1);  arg0_1 = None
            t_1 = torch.ops.aten.t.default(t);  t = None
            return (t_1,)
        ```

        t → t → return should NOT be detected as mutation.
        """
        graph = fx.Graph()
        arg0 = graph.placeholder("arg0_1")
        t = graph.call_function(torch.ops.aten.t.default, (arg0,))
        t_1 = graph.call_function(torch.ops.aten.t.default, (t,))
        graph.output((t_1,))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        result = pass_instance(gm)
        mutation_info = pass_instance.get_mutation_info()

        assert len(mutation_info) == 0, f"Expected no mutation, got {mutation_info}"
        assert result.modified is False

    def test_view_operation_no_mutation(self, pass_instance):
        """
        IR:
        ```
        def forward(self, arg0_1):
            view = torch.ops.aten.view.default(arg0_1, [4, 4]);  arg0_1 = None
            return (view,)
        ```

        view → return should NOT be detected as mutation.
        """
        graph = fx.Graph()
        arg0 = graph.placeholder("arg0_1")
        view = graph.call_function(torch.ops.aten.view.default, (arg0, [4, 4]))
        graph.output((view,))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        _ = pass_instance(gm)
        mutation_info = pass_instance.get_mutation_info()

        assert len(mutation_info) == 0

    def test_reshape_operation_no_mutation(self, pass_instance):
        """
        IR:
        ```
        def forward(self, arg0_1):
            reshape = torch.ops.aten.reshape.default(arg0_1, [2, 8]);  arg0_1 = None
            return (reshape,)
        ```

        reshape → return should NOT be detected as mutation.
        """
        graph = fx.Graph()
        arg0 = graph.placeholder("arg0_1")
        reshape = graph.call_function(torch.ops.aten.reshape.default, (arg0, [2, 8]))
        graph.output((reshape,))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        _ = pass_instance(gm)
        mutation_info = pass_instance.get_mutation_info()

        assert len(mutation_info) == 0

    def test_unsqueeze_squeeze_chain_no_mutation(self, pass_instance):
        """
        IR:
        ```
        def forward(self, arg0_1):
            unsqueeze = torch.ops.aten.unsqueeze.default(arg0_1, 0);  arg0_1 = None
            squeeze = torch.ops.aten.squeeze.default(unsqueeze);  unsqueeze = None
            return (squeeze,)
        ```

        unsqueeze → squeeze → return should NOT be detected as mutation.
        """
        graph = fx.Graph()
        arg0 = graph.placeholder("arg0_1")
        unsqueeze = graph.call_function(torch.ops.aten.unsqueeze.default, (arg0, 0))
        squeeze = graph.call_function(torch.ops.aten.squeeze.default, (unsqueeze,))
        graph.output((squeeze,))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        _ = pass_instance(gm)
        mutation_info = pass_instance.get_mutation_info()

        assert len(mutation_info) == 0

    def test_permute_no_mutation(self, pass_instance):
        """
        IR:
        ```
        def forward(self, arg0_1):
            permute = torch.ops.aten.permute.default(arg0_1, [1, 0]);  arg0_1 = None
            return (permute,)
        ```

        permute → return should NOT be detected as mutation.
        """
        graph = fx.Graph()
        arg0 = graph.placeholder("arg0_1")
        permute = graph.call_function(torch.ops.aten.permute.default, (arg0, [1, 0]))
        graph.output((permute,))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        _ = pass_instance(gm)
        mutation_info = pass_instance.get_mutation_info()

        assert len(mutation_info) == 0

    def test_clone_only_no_mutation(self, pass_instance):
        """
        IR:
        ```
        def forward(self, arg0_1):
            clone = torch.ops.aten.clone.default(arg0_1);  arg0_1 = None
            return (clone,)
        ```

        clone → return (without scatter) should NOT be detected as mutation.
        """
        graph = fx.Graph()
        arg0 = graph.placeholder("arg0_1")
        clone = graph.call_function(torch.ops.aten.clone.default, (arg0,))
        graph.output((clone,))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        _ = pass_instance(gm)
        mutation_info = pass_instance.get_mutation_info()

        assert len(mutation_info) == 0

    def test_contiguous_no_mutation(self, pass_instance):
        """
        IR:
        ```
        def forward(self, arg0_1):
            contiguous = torch.ops.aten.contiguous.default(arg0_1);  arg0_1 = None
            return (contiguous,)
        ```

        contiguous → return should NOT be detected as mutation.
        """
        graph = fx.Graph()
        arg0 = graph.placeholder("arg0_1")
        contiguous = graph.call_function(torch.ops.aten.contiguous.default, (arg0,))
        graph.output((contiguous,))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        _ = pass_instance(gm)
        mutation_info = pass_instance.get_mutation_info()

        assert len(mutation_info) == 0

    def test_expand_no_mutation(self, pass_instance):
        """
        IR:
        ```
        def forward(self, arg0_1):
            expand = torch.ops.aten.expand.default(arg0_1, [2, 4, 4]);  arg0_1 = None
            return (expand,)
        ```

        expand → return should NOT be detected as mutation.
        """
        graph = fx.Graph()
        arg0 = graph.placeholder("arg0_1")
        expand = graph.call_function(torch.ops.aten.expand.default, (arg0, [2, 4, 4]))
        graph.output((expand,))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        _ = pass_instance(gm)
        mutation_info = pass_instance.get_mutation_info()

        assert len(mutation_info) == 0

    def test_direct_input_return_no_mutation(self, pass_instance):
        """
        IR:
        ```
        def forward(self, arg0_1):
            return (arg0_1,)
        ```

        Directly returning an input is detected as a pass-through (not mutation).
        Expected: 0 mutation entries, 1 passthrough entry, graph NOT modified.
        """
        graph = fx.Graph()
        arg0 = graph.placeholder("arg0_1")
        graph.output((arg0,))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        result = pass_instance(gm)
        mutation_info = pass_instance.get_mutation_info()
        passthrough_info = pass_instance.get_passthrough_info()

        assert len(mutation_info) == 0
        assert len(passthrough_info) == 1
        assert passthrough_info.aliases[0].parameter_number == 0
        assert passthrough_info.aliases[0].output_index == 0
        assert result.modified is False

    def test_select_slice_no_mutation(self, pass_instance):
        """
        IR:
        ```
        def forward(self, arg0_1):
            select = torch.ops.aten.select.int(arg0_1, 0, 0);  arg0_1 = None
            slice_1 = torch.ops.aten.slice.Tensor(select, 0, 0, 10);  select = None
            return (slice_1,)
        ```

        select → slice → return should NOT be detected as mutation.
        """
        graph = fx.Graph()
        arg0 = graph.placeholder("arg0_1")
        select = graph.call_function(torch.ops.aten.select.int, (arg0, 0, 0))
        slice_1 = graph.call_function(torch.ops.aten.slice.Tensor, (select, 0, 0, 10))
        graph.output((slice_1,))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        _ = pass_instance(gm)
        mutation_info = pass_instance.get_mutation_info()

        assert len(mutation_info) == 0

    def test_chained_views_no_mutation(self, pass_instance):
        """
        IR:
        ```
        def forward(self, arg0_1):
            t = torch.ops.aten.t.default(arg0_1);  arg0_1 = None
            view = torch.ops.aten.view.default(t, [-1]);  t = None
            unsqueeze = torch.ops.aten.unsqueeze.default(view, 0);  view = None
            squeeze = torch.ops.aten.squeeze.default(unsqueeze);  unsqueeze = None
            contiguous = torch.ops.aten.contiguous.default(squeeze);  squeeze = None
            return (contiguous,)
        ```

        Long chain of views should NOT be detected as mutation.
        """
        graph = fx.Graph()
        arg0 = graph.placeholder("arg0_1")
        t = graph.call_function(torch.ops.aten.t.default, (arg0,))
        view = graph.call_function(torch.ops.aten.view.default, (t, [-1]))
        unsqueeze = graph.call_function(torch.ops.aten.unsqueeze.default, (view, 0))
        squeeze = graph.call_function(torch.ops.aten.squeeze.default, (unsqueeze,))
        contiguous = graph.call_function(torch.ops.aten.contiguous.default, (squeeze,))
        graph.output((contiguous,))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        _ = pass_instance(gm)
        mutation_info = pass_instance.get_mutation_info()

        assert len(mutation_info) == 0

    def test_computation_no_mutation(self, pass_instance):
        """
        IR:
        ```
        def forward(self, arg0_1, arg1_1):
            mm = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg1_1 = None
            add = torch.ops.aten.add.Tensor(mm, arg0_1);  mm = arg0_1 = None
            return (add,)
        ```

        Pure computation (mm, add) should NOT be detected as mutation.
        """
        graph = fx.Graph()
        arg0 = graph.placeholder("arg0_1")
        arg1 = graph.placeholder("arg1_1")
        mm = graph.call_function(torch.ops.aten.mm.default, (arg0, arg1))
        add = graph.call_function(torch.ops.aten.add.Tensor, (mm, arg0))
        graph.output((add,))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        _ = pass_instance(gm)
        mutation_info = pass_instance.get_mutation_info()

        assert len(mutation_info) == 0

    def test_transpose_int_no_mutation(self, pass_instance):
        """
        IR:
        ```
        def forward(self, arg0_1):
            transpose = torch.ops.aten.transpose.int(arg0_1, 0, 1);  arg0_1 = None
            return (transpose,)
        ```

        transpose.int → return should NOT be detected as mutation.
        """
        graph = fx.Graph()
        arg0 = graph.placeholder("arg0_1")
        transpose = graph.call_function(torch.ops.aten.transpose.int, (arg0, 0, 1))
        graph.output((transpose,))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        _ = pass_instance(gm)
        mutation_info = pass_instance.get_mutation_info()

        assert len(mutation_info) == 0

    def test_unsafe_view_no_mutation(self, pass_instance):
        """
        IR:
        ```
        def forward(self, arg0_1):
            _unsafe_view = torch.ops.aten._unsafe_view.default(arg0_1, [16]);  arg0_1 = None
            return (_unsafe_view,)
        ```

        _unsafe_view → return should NOT be detected as mutation.
        """
        graph = fx.Graph()
        arg0 = graph.placeholder("arg0_1")
        unsafe_view = graph.call_function(torch.ops.aten._unsafe_view.default, (arg0, [16]))
        graph.output((unsafe_view,))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        _ = pass_instance(gm)
        mutation_info = pass_instance.get_mutation_info()

        assert len(mutation_info) == 0

    def test_squeeze_dim_no_mutation(self, pass_instance):
        """
        IR:
        ```
        def forward(self, arg0_1):
            squeeze = torch.ops.aten.squeeze.dim(arg0_1, 0);  arg0_1 = None
            return (squeeze,)
        ```

        squeeze.dim → return should NOT be detected as mutation.
        """
        graph = fx.Graph()
        arg0 = graph.placeholder("arg0_1")
        squeeze = graph.call_function(torch.ops.aten.squeeze.dim, (arg0, 0))
        graph.output((squeeze,))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        _ = pass_instance(gm)
        mutation_info = pass_instance.get_mutation_info()

        assert len(mutation_info) == 0


class TestMutationDetected:
    """Tests where mutation SHOULD be detected."""

    @pytest.mark.xfail
    def test_clone_scatter_src_mutation(self, pass_instance):
        """
        IR:
        ```
        def forward(self, arg0_1, arg1_1, arg2_1):
            clone = torch.ops.aten.clone.default(arg0_1);  arg0_1 = None
            scatter = torch.ops.aten.scatter.src(
                clone, 0, arg1_1, arg2_1
            );  clone = arg1_1 = arg2_1 = None
            return (scatter,)
        ```

        clone → scatter.src → return SHOULD be detected as mutation.
        Expected: MutationInfo(parameter_number=0, output_index=0)
        """
        graph = fx.Graph()
        arg0 = graph.placeholder("arg0_1")  # tensor to clone/mutate
        arg1 = graph.placeholder("arg1_1")  # index
        arg2 = graph.placeholder("arg2_1")  # src
        clone = graph.call_function(torch.ops.aten.clone.default, (arg0,))
        scatter = graph.call_function(torch.ops.aten.scatter.src, (clone, 0, arg1, arg2))
        graph.output((scatter,))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        _ = pass_instance(gm)
        mutation_info = pass_instance.get_mutation_info()

        assert len(mutation_info) == 1
        assert mutation_info.aliases[0].parameter_number == 0
        assert mutation_info.aliases[0].output_index == 0

    @pytest.mark.xfail
    def test_clone_select_scatter_mutation(self, pass_instance):
        """
        IR:
        ```
        def forward(self, arg0_1, arg1_1):
            clone = torch.ops.aten.clone.default(arg0_1);  arg0_1 = None
            select_scatter = torch.ops.aten.select_scatter.default(
                clone, arg1_1, 0, 0
            );  clone = arg1_1 = None
            return (select_scatter,)
        ```

        clone → select_scatter → return SHOULD be detected as mutation.
        Expected: MutationInfo(parameter_number=0, output_index=0)
        """
        graph = fx.Graph()
        arg0 = graph.placeholder("arg0_1")
        arg1 = graph.placeholder("arg1_1")  # source tensor
        clone = graph.call_function(torch.ops.aten.clone.default, (arg0,))
        select_scatter = graph.call_function(
            torch.ops.aten.select_scatter.default, (clone, arg1, 0, 0)
        )
        graph.output((select_scatter,))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        _ = pass_instance(gm)
        mutation_info = pass_instance.get_mutation_info()

        assert len(mutation_info) == 1
        assert mutation_info.aliases[0].parameter_number == 0
        assert mutation_info.aliases[0].output_index == 0

    @pytest.mark.xfail
    def test_clone_slice_scatter_mutation(self, pass_instance):
        """
        IR:
        ```
        def forward(self, arg0_1, arg1_1):
            clone = torch.ops.aten.clone.default(arg0_1);  arg0_1 = None
            slice_scatter = torch.ops.aten.slice_scatter.default(
                clone, arg1_1, 0, 0, 1
            );  clone = arg1_1 = None
            return (slice_scatter,)
        ```

        clone → slice_scatter → return SHOULD be detected as mutation.
        Expected: MutationInfo(parameter_number=0, output_index=0)
        """
        graph = fx.Graph()
        arg0 = graph.placeholder("arg0_1")
        arg1 = graph.placeholder("arg1_1")  # source tensor
        clone = graph.call_function(torch.ops.aten.clone.default, (arg0,))
        slice_scatter = graph.call_function(
            torch.ops.aten.slice_scatter.default, (clone, arg1, 0, 0, 1)
        )
        graph.output((slice_scatter,))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        _ = pass_instance(gm)
        mutation_info = pass_instance.get_mutation_info()

        assert len(mutation_info) == 1
        assert mutation_info.aliases[0].parameter_number == 0

    @pytest.mark.xfail
    def test_copy_default_mutation(self, pass_instance):
        """
        IR:
        ```
        def forward(self, arg0_1, arg1_1):
            clone = torch.ops.aten.clone.default(arg0_1);  arg0_1 = None
            copy = torch.ops.aten.copy.default(clone, arg1_1);  clone = arg1_1 = None
            return (copy,)
        ```

        clone → copy.default → return SHOULD be detected as mutation.
        Expected: MutationInfo(parameter_number=0, output_index=0)
        """
        graph = fx.Graph()
        arg0 = graph.placeholder("arg0_1")
        arg1 = graph.placeholder("arg1_1")
        clone = graph.call_function(torch.ops.aten.clone.default, (arg0,))
        copy = graph.call_function(torch.ops.aten.copy.default, (clone, arg1))
        graph.output((copy,))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        _ = pass_instance(gm)
        mutation_info = pass_instance.get_mutation_info()

        assert len(mutation_info) == 1
        assert mutation_info.aliases[0].parameter_number == 0

    @pytest.mark.xfail
    def test_index_put_mutation(self, pass_instance):
        """
        IR:
        ```
        def forward(self, arg0_1, arg1_1, arg2_1):
            clone = torch.ops.aten.clone.default(arg0_1);  arg0_1 = None
            index_put = torch.ops.aten.index_put.default(
                clone, [arg1_1], arg2_1
            );  clone = arg1_1 = arg2_1 = None
            return (index_put,)
        ```

        clone → index_put → return SHOULD be detected as mutation.
        Expected: MutationInfo(parameter_number=0, output_index=0)
        """
        graph = fx.Graph()
        arg0 = graph.placeholder("arg0_1")  # tensor
        arg1 = graph.placeholder("arg1_1")  # indices
        arg2 = graph.placeholder("arg2_1")  # values
        clone = graph.call_function(torch.ops.aten.clone.default, (arg0,))
        index_put = graph.call_function(torch.ops.aten.index_put.default, (clone, [arg1], arg2))
        graph.output((index_put,))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        _ = pass_instance(gm)
        mutation_info = pass_instance.get_mutation_info()

        assert len(mutation_info) == 1
        assert mutation_info.aliases[0].parameter_number == 0

    @pytest.mark.xfail
    def test_index_copy_mutation(self, pass_instance):
        """
        IR:
        ```
        def forward(self, arg0_1, arg1_1, arg2_1):
            clone = torch.ops.aten.clone.default(arg0_1);  arg0_1 = None
            index_copy = torch.ops.aten.index_copy.default(
                clone, 0, arg1_1, arg2_1
            );  clone = arg1_1 = arg2_1 = None
            return (index_copy,)
        ```

        clone → index_copy → return SHOULD be detected as mutation.
        Expected: MutationInfo(parameter_number=0, output_index=0)
        """
        graph = fx.Graph()
        arg0 = graph.placeholder("arg0_1")  # tensor
        arg1 = graph.placeholder("arg1_1")  # index
        arg2 = graph.placeholder("arg2_1")  # source
        clone = graph.call_function(torch.ops.aten.clone.default, (arg0,))
        index_copy = graph.call_function(torch.ops.aten.index_copy.default, (clone, 0, arg1, arg2))
        graph.output((index_copy,))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        _ = pass_instance(gm)
        mutation_info = pass_instance.get_mutation_info()  # .get('mutation_info')

        assert len(mutation_info) == 1
        assert mutation_info.aliases[0].parameter_number == 0

    @pytest.mark.xfail
    def test_scatter_value_mutation(self, pass_instance):
        """
        IR:
        ```
        def forward(self, arg0_1, arg1_1):
            clone = torch.ops.aten.clone.default(arg0_1);  arg0_1 = None
            scatter = torch.ops.aten.scatter.value(clone, 0, arg1_1, 1.0);  clone = arg1_1 = None
            return (scatter,)
        ```

        clone → scatter.value → return SHOULD be detected as mutation.
        Expected: MutationInfo(parameter_number=0, output_index=0)
        """
        graph = fx.Graph()
        arg0 = graph.placeholder("arg0_1")
        arg1 = graph.placeholder("arg1_1")  # index
        clone = graph.call_function(torch.ops.aten.clone.default, (arg0,))
        scatter = graph.call_function(
            torch.ops.aten.scatter.value,
            (clone, 0, arg1, 1.0),  # value=1.0
        )
        graph.output((scatter,))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        _ = pass_instance(gm)
        mutation_info = pass_instance.get_mutation_info()

        assert len(mutation_info) == 1
        assert mutation_info.aliases[0].parameter_number == 0

    @pytest.mark.xfail
    def test_scatter_add_mutation(self, pass_instance):
        """
        IR:
        ```
        def forward(self, arg0_1, arg1_1, arg2_1):
            clone = torch.ops.aten.clone.default(arg0_1);  arg0_1 = None
            scatter_add = torch.ops.aten.scatter_add.default(
                clone, 0, arg1_1, arg2_1
            );  clone = arg1_1 = arg2_1 = None
            return (scatter_add,)
        ```

        clone → scatter_add → return SHOULD be detected as mutation.
        Expected: MutationInfo(parameter_number=0, output_index=0)
        """
        graph = fx.Graph()
        arg0 = graph.placeholder("arg0_1")
        arg1 = graph.placeholder("arg1_1")
        arg2 = graph.placeholder("arg2_1")
        clone = graph.call_function(torch.ops.aten.clone.default, (arg0,))
        scatter_add = graph.call_function(
            torch.ops.aten.scatter_add.default, (clone, 0, arg1, arg2)
        )
        graph.output((scatter_add,))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        _ = pass_instance(gm)
        mutation_info = pass_instance.get_mutation_info()

        assert len(mutation_info) == 1

    @pytest.mark.xfail
    def test_scatter_reduce_mutation(self, pass_instance):
        """
        IR:
        ```
        def forward(self, arg0_1, arg1_1, arg2_1):
            clone = torch.ops.aten.clone.default(arg0_1);  arg0_1 = None
            scatter_reduce = torch.ops.aten.scatter_reduce.two(
                clone, 0, arg1_1, arg2_1, 'sum'
            );  clone = arg1_1 = arg2_1 = None
            return (scatter_reduce,)
        ```

        clone → scatter_reduce → return SHOULD be detected as mutation.
        Expected: MutationInfo(parameter_number=0, output_index=0)
        """
        graph = fx.Graph()
        arg0 = graph.placeholder("arg0_1")
        arg1 = graph.placeholder("arg1_1")
        arg2 = graph.placeholder("arg2_1")
        clone = graph.call_function(torch.ops.aten.clone.default, (arg0,))
        scatter_reduce = graph.call_function(
            torch.ops.aten.scatter_reduce.two, (clone, 0, arg1, arg2, "sum")
        )
        graph.output((scatter_reduce,))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        _ = pass_instance(gm)
        mutation_info = pass_instance.get_mutation_info()

        assert len(mutation_info) == 1

    @pytest.mark.xfail
    def test_scatter_followed_by_view_mutation(self, pass_instance):
        """
        IR:
        ```
        def forward(self, arg0_1, arg1_1, arg2_1):
            clone = torch.ops.aten.clone.default(arg0_1);  arg0_1 = None
            scatter = torch.ops.aten.scatter.src(
                clone, 0, arg1_1, arg2_1
            );  clone = arg1_1 = arg2_1 = None
            view = torch.ops.aten.view.default(scatter, [-1]);  scatter = None
            return (view,)
        ```

        clone → scatter → view → return SHOULD still be detected as mutation.
        Expected: MutationInfo(parameter_number=0, output_index=0)
        """
        graph = fx.Graph()
        arg0 = graph.placeholder("arg0_1")
        arg1 = graph.placeholder("arg1_1")
        arg2 = graph.placeholder("arg2_1")
        clone = graph.call_function(torch.ops.aten.clone.default, (arg0,))
        scatter = graph.call_function(torch.ops.aten.scatter.src, (clone, 0, arg1, arg2))
        view = graph.call_function(torch.ops.aten.view.default, (scatter, [-1]))
        graph.output((view,))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        _ = pass_instance(gm)
        mutation_info = pass_instance.get_mutation_info()

        assert len(mutation_info) == 1
        assert mutation_info.aliases[0].parameter_number == 0

    @pytest.mark.xfail
    def test_multiple_mutations_different_inputs(self, pass_instance):
        """
        IR:
        ```
        def forward(self, arg0_1, arg1_1, arg2_1, arg3_1):
            clone = torch.ops.aten.clone.default(arg0_1);  arg0_1 = None
            clone_1 = torch.ops.aten.clone.default(arg1_1);  arg1_1 = None
            scatter = torch.ops.aten.scatter.src(clone, 0, arg2_1, arg3_1);  clone = None
            scatter_1 = torch.ops.aten.scatter.src(
                clone_1, 0, arg2_1, arg3_1
            );  clone_1 = arg2_1 = arg3_1 = None
            return (scatter, scatter_1)
        ```

        Multiple scatter operations on different inputs SHOULD all be detected.
        Expected: MutationInfo(parameter_number=0, output_index=0)
                  MutationInfo(parameter_number=1, output_index=1)
        """
        graph = fx.Graph()
        arg0 = graph.placeholder("arg0_1")  # first cache
        arg1 = graph.placeholder("arg1_1")  # second cache
        arg2 = graph.placeholder("arg2_1")  # index
        arg3 = graph.placeholder("arg3_1")  # src

        clone0 = graph.call_function(torch.ops.aten.clone.default, (arg0,))
        clone1 = graph.call_function(torch.ops.aten.clone.default, (arg1,))

        scatter0 = graph.call_function(torch.ops.aten.scatter.src, (clone0, 0, arg2, arg3))
        scatter1 = graph.call_function(torch.ops.aten.scatter.src, (clone1, 0, arg2, arg3))

        graph.output((scatter0, scatter1))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        _ = pass_instance(gm)
        mutation_info = pass_instance.get_mutation_info()

        assert len(mutation_info) == 2
        param_numbers = {a.parameter_number for a in mutation_info.aliases}
        assert param_numbers == {0, 1}

    @pytest.mark.xfail
    def test_direct_scatter_on_input_mutation(self, pass_instance):
        """
        IR:
        ```
        def forward(self, arg0_1, arg1_1, arg2_1):
            scatter = torch.ops.aten.scatter.src(
                arg0_1, 0, arg1_1, arg2_1
            );  arg0_1 = arg1_1 = arg2_1 = None
            return (scatter,)
        ```

        Direct scatter on input (no clone) SHOULD be detected as mutation.
        Expected: MutationInfo(parameter_number=0, output_index=0)
        """
        graph = fx.Graph()
        arg0 = graph.placeholder("arg0_1")
        arg1 = graph.placeholder("arg1_1")
        arg2 = graph.placeholder("arg2_1")
        scatter = graph.call_function(torch.ops.aten.scatter.src, (arg0, 0, arg1, arg2))
        graph.output((scatter,))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        _ = pass_instance(gm)
        mutation_info = pass_instance.get_mutation_info()

        assert len(mutation_info) == 1
        assert mutation_info.aliases[0].parameter_number == 0

    @pytest.mark.xfail
    def test_chained_scatter_operations_mutation(self, pass_instance):
        """
        IR:
        ```
        def forward(self, arg0_1, arg1_1, arg2_1):
            clone = torch.ops.aten.clone.default(arg0_1);  arg0_1 = None
            scatter = torch.ops.aten.scatter.src(clone, 0, arg1_1, arg2_1);  clone = None
            scatter_1 = torch.ops.aten.scatter.src(
                scatter, 0, arg1_1, arg2_1
            );  scatter = arg1_1 = arg2_1 = None
            return (scatter_1,)
        ```

        clone → scatter → scatter → return (chained scatters) SHOULD be detected.
        Expected: MutationInfo(parameter_number=0, output_index=0)
        """
        graph = fx.Graph()
        arg0 = graph.placeholder("arg0_1")
        arg1 = graph.placeholder("arg1_1")
        arg2 = graph.placeholder("arg2_1")

        clone = graph.call_function(torch.ops.aten.clone.default, (arg0,))
        scatter1 = graph.call_function(torch.ops.aten.scatter.src, (clone, 0, arg1, arg2))
        scatter2 = graph.call_function(torch.ops.aten.scatter.src, (scatter1, 0, arg1, arg2))

        graph.output((scatter2,))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        _ = pass_instance(gm)
        mutation_info = pass_instance.get_mutation_info()

        # Should only record one mutation (even though there are two scatters)
        assert len(mutation_info) == 1
        assert mutation_info.aliases[0].parameter_number == 0


# ============================================================================
# Graph shapes shared by the copy_ and IO-mutation tests
#
# Each returns ``(gm, *nodes)``: the module first, then the nodes a caller
# asserts on.
# ============================================================================


def _aot_functional_copy_graph(dst_shape=(4, 4), dst_dtype=torch.float32):
    """AOT Autograd's spelling of ``buf.copy_(v); return v``.

    IR:
    ```
    def forward(self, arg0_1, arg1_1):
        mul = torch.ops.aten.mul.Tensor(arg0_1, 2)
        copy = torch.ops.aten.copy.default(arg1_1, mul)
        copy_ = torch.ops.aten.copy_.default(arg1_1, copy)
        return (mul,)
    ```

    Functionalizing an in-place ``copy_`` leaves a functional ``copy`` between the
    value and the ``copy_`` epilogue, so the returned node is neither the ``copy_``
    node nor its ``src``. ``dst_shape`` and ``dst_dtype`` set the destination's
    ``meta['val']``, which is what decides whether the functional copy broadcasts
    or casts. ``mul`` is always ``float32[4, 4]``.
    """
    graph = fx.Graph()
    arg0 = graph.placeholder("arg0_1")
    arg1 = graph.placeholder("arg1_1")
    mul = graph.call_function(torch.ops.aten.mul.Tensor, (arg0, 2))
    mul.meta["val"] = torch.zeros(4, 4, dtype=torch.float32)
    functional_copy = graph.call_function(torch.ops.aten.copy.default, (arg1, mul))
    functional_copy.meta["val"] = torch.zeros(dst_shape, dtype=dst_dtype)
    graph.call_function(torch.ops.aten.copy_.default, (arg1, functional_copy))
    graph.output((mul,))
    return fx.GraphModule(torch.nn.Module(), graph), mul, functional_copy


def _graph_returned_copy():
    """The mutation result is the returned value (NF-255 KV-cache shape)."""
    graph = fx.Graph()
    cache = graph.placeholder("cache")
    idx = graph.placeholder("idx")
    row = graph.placeholder("row")
    index_put = graph.call_function(torch.ops.aten.index_put.default, (cache, [idx], row))
    copy_inplace = graph.call_function(torch.ops.aten.copy_.default, (cache, index_put))
    graph.output((copy_inplace,))
    return (fx.GraphModule(torch.nn.Module(), graph),)


def _graph_src_returned():
    """The copy_ src is returned because AOT Autograd saved it for backward."""
    graph = fx.Graph()
    primals_1 = graph.placeholder("primals_1")
    primals_2 = graph.placeholder("primals_2")
    primals_3 = graph.placeholder("primals_3")
    add = graph.call_function(torch.ops.aten.add.Tensor, (primals_1, 1))
    mm = graph.call_function(torch.ops.aten.mm.default, (primals_2, primals_3))
    mul = graph.call_function(torch.ops.aten.mul.Tensor, (mm, add))
    graph.call_function(torch.ops.aten.copy_.default, (primals_1, add))
    graph.output((mul, primals_2, add))
    return fx.GraphModule(torch.nn.Module(), graph), mul, primals_2, add


def _graph_hidden_copy():
    """The mutation result is not returned, so the pass has to add an output."""
    graph = fx.Graph()
    cache = graph.placeholder("cache")
    row = graph.placeholder("row")
    other = graph.placeholder("other")
    graph.call_function(torch.ops.aten.copy_.default, (cache, row))
    add = graph.call_function(torch.ops.aten.add.Tensor, (row, other))
    graph.output((add,))
    return (fx.GraphModule(torch.nn.Module(), graph),)


def _graph_mixed_copies():
    """One returned mutation and one hidden mutation in the same graph."""
    graph = fx.Graph()
    returned_dst = graph.placeholder("returned_dst")
    returned_src = graph.placeholder("returned_src")
    hidden_dst = graph.placeholder("hidden_dst")
    hidden_src = graph.placeholder("hidden_src")
    x = graph.placeholder("x")
    returned_copy = graph.call_function(torch.ops.aten.copy_.default, (returned_dst, returned_src))
    graph.call_function(torch.ops.aten.copy_.default, (hidden_dst, hidden_src))
    add = graph.call_function(torch.ops.aten.add.Tensor, (x, 1))
    graph.output((returned_copy, add))
    return fx.GraphModule(torch.nn.Module(), graph), hidden_src, returned_src, add


def _graph_shared_src():
    """Two inputs mutated with the same returned value."""
    graph = fx.Graph()
    arg0 = graph.placeholder("arg0_1")
    arg1 = graph.placeholder("arg1_1")
    x = graph.placeholder("x")
    add = graph.call_function(torch.ops.aten.add.Tensor, (x, 1))
    graph.call_function(torch.ops.aten.copy_.default, (arg0, add))
    graph.call_function(torch.ops.aten.copy_.default, (arg1, add))
    graph.output((add,))
    return fx.GraphModule(torch.nn.Module(), graph), add


def _graph_returned_copy_with_none():
    """A returned mutation alongside a None output that RemoveNoneOutputs drops later."""
    graph = fx.Graph()
    cache = graph.placeholder("cache")
    idx = graph.placeholder("idx")
    row = graph.placeholder("row")
    index_put = graph.call_function(torch.ops.aten.index_put.default, (cache, [idx], row))
    copy_inplace = graph.call_function(torch.ops.aten.copy_.default, (cache, index_put))
    graph.output((None, copy_inplace))
    return (fx.GraphModule(torch.nn.Module(), graph),)


def _graph_copy_returned_twice():
    """The copy_ result is returned twice, so the second slot is a pass-through."""
    graph = fx.Graph()
    arg0 = graph.placeholder("arg0_1")
    arg1 = graph.placeholder("arg1_1")
    copy_inplace = graph.call_function(torch.ops.aten.copy_.default, (arg0, arg1))
    graph.output((copy_inplace, copy_inplace))
    return fx.GraphModule(torch.nn.Module(), graph), arg1


def _graph_no_mutation():
    """No copy_ at all, so the pass must emit nothing."""
    graph = fx.Graph()
    x = graph.placeholder("x")
    y = graph.placeholder("y")
    graph.output((graph.call_function(torch.ops.aten.add.Tensor, (x, y)),))
    return (fx.GraphModule(torch.nn.Module(), graph),)


def _data_dependent_shape_pair():
    """``meta['val']`` for a functional copy whose shapes cannot be compared.

    The destination's leading dim is an unbacked SymInt, so comparing it against the
    source's concrete 1 raises ``GuardOnDataDependentSymNode``. This is the shape
    ``t = row.new_zeros(n.item(), 4); t.copy_(row)`` traces to.
    """
    shape_env = ShapeEnv()
    with FakeTensorMode(shape_env=shape_env):
        return torch.empty(shape_env.create_unbacked_symint(), 4), torch.empty(1, 4)


class TestCopyInplace:
    """Tests for copy_ in-place operation handling."""

    def test_copy_inplace_detected_and_removed(self, pass_instance):
        """
        IR Before:
        ```
        def forward(self, arg0_1, arg1_1):
            copy_ = torch.ops.aten.copy_.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
            return (copy_,)
        ```

        IR After:
        ```
        def forward(self, arg0_1, arg1_1):
            return (arg1_1,)  # copy_ removed, existing output slot reused
        ```

        The graph already returned the copy_ result, so no output is added and the
        mutation aliases the slot that already carries it.
        Expected: Graph modified, MutationInfo(parameter_number=0, output_index=0)
        """
        graph = fx.Graph()
        arg0 = graph.placeholder("arg0_1")  # destination
        arg1 = graph.placeholder("arg1_1")  # source
        copy_inplace = graph.call_function(torch.ops.aten.copy_.default, (arg0, arg1))
        graph.output((copy_inplace,))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        result = pass_instance(gm)
        mutation_info = pass_instance.get_mutation_info()
        output_node = next(n for n in gm.graph.nodes if n.op == "output")

        assert result.modified is True
        assert output_node.args[0] == (arg1,)
        assert pass_instance.added_mutation_output_indices == []
        assert len(mutation_info) == 1
        assert mutation_info.aliases[0].parameter_number == 0
        assert mutation_info.aliases[0].output_index == 0

        # Verify copy_ was removed
        copy_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target == torch.ops.aten.copy_.default
        ]
        assert len(copy_nodes) == 0

    def test_copy_inplace_src_already_returned(self, pass_instance):
        """
        IR Before:
        ```
        def forward(self, arg0_1, arg1_1):
            add = torch.ops.aten.add.Tensor(arg0_1, 1)
            copy_ = torch.ops.aten.copy_.default(arg0_1, add)
            return (add,)
        ```

        IR After:
        ```
        def forward(self, arg0_1, arg1_1):
            add = torch.ops.aten.add.Tensor(arg0_1, 1)
            return (add,)
        ```

        The copy_ node itself is not returned, but its src already is, and
        replace_all_uses_with puts src in the copy_ node's place, so that slot
        carries the mutation result and no output is added.
        """
        graph = fx.Graph()
        arg0 = graph.placeholder("arg0_1")
        graph.placeholder("arg1_1")
        add = graph.call_function(torch.ops.aten.add.Tensor, (arg0, 1))
        graph.call_function(torch.ops.aten.copy_.default, (arg0, add))
        graph.output((add,))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        result = pass_instance(gm)
        mutation_info = pass_instance.get_mutation_info()
        output_node = next(n for n in gm.graph.nodes if n.op == "output")

        assert result.modified is True
        assert output_node.args[0] == (add,)
        assert pass_instance.added_mutation_output_indices == []
        assert len(mutation_info) == 1
        assert mutation_info.aliases[0].parameter_number == 0
        assert mutation_info.aliases[0].output_index == 0
        assert not [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target == torch.ops.aten.copy_.default
        ]

    def test_returned_value_behind_a_functional_copy_reuses_slot(self, pass_instance):
        """``buf.copy_(v); return v``, the plainest returned in-place write.

        IR After:
        ```
        def forward(self, arg0_1, arg1_1):
            mul = torch.ops.aten.mul.Tensor(arg0_1, 2)
            return (mul,)
        ```

        The functional copy neither broadcasts nor casts, so it reproduces ``mul``
        exactly and the slot holding ``mul`` already carries the mutation result.
        Only the ``copy_`` epilogue is removed; the functional copy is left in place,
        userless, for the compiler to drop.
        """
        gm, mul, functional_copy = _aot_functional_copy_graph()

        result = pass_instance(gm)
        mutation_info = pass_instance.get_mutation_info()
        output_node = next(n for n in gm.graph.nodes if n.op == "output")

        assert result.modified is True
        assert output_node.args[0] == (mul,)
        assert pass_instance.added_mutation_output_indices == []
        assert [(a.output_index, a.parameter_number) for a in mutation_info.aliases] == [(0, 1)]
        assert not [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target == torch.ops.aten.copy_.default
        ]
        assert functional_copy in gm.graph.nodes
        assert not functional_copy.users

    def test_functional_copy_itself_returned_reuses_slot(self, pass_instance):
        """``return buf.copy_(v)``, where AOT returns the functional copy itself.

        IR:
        ```
        def forward(self, arg0_1, arg1_1):
            mul = torch.ops.aten.mul.Tensor(arg0_1, 2)
            copy = torch.ops.aten.copy.default(arg1_1, mul)
            copy_ = torch.ops.aten.copy_.default(arg1_1, copy)
            return (copy,)
        ```

        Eager's ``copy_`` returns the destination, so the returned value here is the
        functional copy that ``src`` also holds, not the value behind it. That slot
        already carries the mutation result, so matching only the copy_ node or the
        unwrapped value would prepend a duplicate full-size output.
        """
        gm, _, functional_copy = _aot_functional_copy_graph()
        output_node = next(n for n in gm.graph.nodes if n.op == "output")
        output_node.args = ((functional_copy,),)

        result = pass_instance(gm)
        mutation_info = pass_instance.get_mutation_info()

        assert result.modified is True
        assert output_node.args[0] == (functional_copy,)
        assert pass_instance.added_mutation_output_indices == []
        assert [(a.output_index, a.parameter_number) for a in mutation_info.aliases] == [(0, 1)]

    @pytest.mark.parametrize(
        ("dst_shape", "dst_dtype"),
        [
            pytest.param((4, 4), torch.bfloat16, id="casts"),
            pytest.param((8, 4), torch.float32, id="broadcasts"),
        ],
    )
    def test_functional_copy_that_changes_the_value_is_not_unwrapped(
        self, pass_instance, dst_shape, dst_dtype
    ):
        """A functional copy that casts or broadcasts does not reproduce its source.

        ``copy.default`` takes the destination's shape and dtype, so the returned
        value and the mutation result are then two different values needing two
        buffers. The pass must add a slot rather than donate the destination's
        buffer to the slot holding the source.
        """
        gm, mul, functional_copy = _aot_functional_copy_graph(dst_shape, dst_dtype)

        result = pass_instance(gm)
        mutation_info = pass_instance.get_mutation_info()
        output_node = next(n for n in gm.graph.nodes if n.op == "output")

        assert result.modified is True
        assert output_node.args[0] == (functional_copy, mul)
        assert pass_instance.added_mutation_output_indices == [0]
        assert [(a.output_index, a.parameter_number) for a in mutation_info.aliases] == [(0, 1)]

    def test_functional_copy_without_metadata_is_not_unwrapped(self, pass_instance):
        """Without ``meta['val']`` the copy cannot be shown to be a no-op.

        Donating the destination's buffer to the slot holding the source would then
        be a guess, so the pass adds a slot instead.
        """
        gm, mul, functional_copy = _aot_functional_copy_graph()
        functional_copy.meta.pop("val")

        result = pass_instance(gm)
        output_node = next(n for n in gm.graph.nodes if n.op == "output")

        assert result.modified is True
        assert output_node.args[0] == (functional_copy, mul)
        assert pass_instance.added_mutation_output_indices == [0]

    def test_data_dependent_shapes_are_not_unwrapped(self, pass_instance):
        """A symbolic dim is never compared, so a data-dependent shape is left alone.

        Comparing an unbacked dim raises ``GuardOnDataDependentSymNode``, which would
        fail the whole compilation for a graph the pass would otherwise handle. A
        shape that is not concrete is not provably a no-op, so the pass adds a slot.
        """
        gm, mul, functional_copy = _aot_functional_copy_graph()
        functional_copy.meta["val"], mul.meta["val"] = _data_dependent_shape_pair()

        result = pass_instance(gm)
        output_node = next(n for n in gm.graph.nodes if n.op == "output")

        assert result.modified is True
        assert output_node.args[0] == (functional_copy, mul)
        assert pass_instance.added_mutation_output_indices == [0]

    def test_aot_saved_for_backward_mutation_reuses_slot(self, pass_instance):
        """
        Real AOT Autograd forward graph shape under keep_inference_input_mutations:
        the mutation src is returned because backward saved it.

        IR Before:
        ```
        def forward(self, primals_1, primals_2, primals_3):
            add = torch.ops.aten.add.Tensor(primals_1, 1)
            mm = torch.ops.aten.mm.default(primals_2, primals_3)
            mul = torch.ops.aten.mul.Tensor(mm, add)
            copy_ = torch.ops.aten.copy_.default(primals_1, add)
            return (mul, primals_2, add)
        ```

        IR After:
        ```
        def forward(self, primals_1, primals_2, primals_3):
            add = torch.ops.aten.add.Tensor(primals_1, 1)
            mm = torch.ops.aten.mm.default(primals_2, primals_3)
            mul = torch.ops.aten.mul.Tensor(mm, add)
            return (mul, primals_2, add)
        ```

        `add` is full-size, so prepending it would cost a second device buffer
        for a value already in the tuple.
        """
        gm, mul, primals_2, add = _graph_src_returned()

        result = pass_instance(gm)
        mutation_info = pass_instance.get_mutation_info()
        passthrough_info = pass_instance.get_passthrough_info()
        output_node = next(n for n in gm.graph.nodes if n.op == "output")

        assert result.modified is True
        assert output_node.args[0] == (mul, primals_2, add)
        assert pass_instance.added_mutation_output_indices == []
        assert [(a.output_index, a.parameter_number) for a in mutation_info.aliases] == [(2, 0)]
        assert [(a.output_index, a.parameter_number) for a in passthrough_info.aliases] == [(1, 1)]
        assert not [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target == torch.ops.aten.copy_.default
        ]

    def test_copy_inplace_prepends_to_output(self, pass_instance):
        """
        IR Before:
        ```
        def forward(self, arg0_1, arg1_1, arg2_1):
            copy_ = torch.ops.aten.copy_.default(arg0_1, arg1_1)
            add = torch.ops.aten.add.Tensor(arg1_1, arg2_1);  arg1_1 = arg2_1 = None
            return (add,)
        ```

        IR After:
        ```
        def forward(self, arg0_1, arg1_1, arg2_1):
            add = torch.ops.aten.add.Tensor(arg1_1, arg2_1);  arg2_1 = None
            return (arg1_1, add)  # mutation prepended
        ```

        copy_ mutation value should be prepended to outputs.
        Expected: 2 outputs, MutationInfo(parameter_number=0, output_index=0)
        """
        graph = fx.Graph()
        arg0 = graph.placeholder("arg0_1")
        arg1 = graph.placeholder("arg1_1")
        arg2 = graph.placeholder("arg2_1")

        _ = graph.call_function(torch.ops.aten.copy_.default, (arg0, arg1))
        add = graph.call_function(torch.ops.aten.add.Tensor, (arg1, arg2))

        graph.output((add,))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        _ = pass_instance(gm)
        mutation_info = pass_instance.get_mutation_info()

        # Check output now has 2 values (mutation + original)
        output_node = next(n for n in gm.graph.nodes if n.op == "output")
        outputs = output_node.args[0]
        assert len(outputs) == 2
        assert mutation_info.aliases[0].output_index == 0  # Prepended at index 0

    def test_multiple_copy_inplace(self, pass_instance):
        """
        IR Before:
        ```
        def forward(self, arg0_1, arg1_1, arg2_1, arg3_1):
            copy_ = torch.ops.aten.copy_.default(arg0_1, arg2_1)
            copy__1 = torch.ops.aten.copy_.default(arg1_1, arg3_1)
            return ()
        ```

        IR After:
        ```
        def forward(self, arg0_1, arg1_1, arg2_1, arg3_1):
            return (arg2_1, arg3_1)  # both mutations prepended
        ```

        Multiple copy_ operations should all be handled.
        Expected: 2 outputs, 2 MutationInfo entries
        """
        graph = fx.Graph()
        arg0 = graph.placeholder("arg0_1")
        arg1 = graph.placeholder("arg1_1")
        arg2 = graph.placeholder("arg2_1")
        arg3 = graph.placeholder("arg3_1")

        _ = graph.call_function(torch.ops.aten.copy_.default, (arg0, arg2))
        _ = graph.call_function(torch.ops.aten.copy_.default, (arg1, arg3))

        graph.output(())
        gm = fx.GraphModule(torch.nn.Module(), graph)

        result = pass_instance(gm)
        mutation_info = pass_instance.get_mutation_info()

        assert result.modified is True
        assert len(mutation_info) == 2
        assert pass_instance.added_mutation_output_indices == [0, 1]

    def test_mixed_returned_and_hidden_copy_inplace(self, pass_instance):
        """
        IR Before:
        ```
        def forward(self, returned_dst, returned_src, hidden_dst, hidden_src, x):
            returned_copy = torch.ops.aten.copy_.default(returned_dst, returned_src)
            copy__1 = torch.ops.aten.copy_.default(hidden_dst, hidden_src)
            add = torch.ops.aten.add.Tensor(x, 1)
            return (returned_copy, add)
        ```

        IR After:
        ```
        def forward(self, returned_dst, returned_src, hidden_dst, hidden_src, x):
            add = torch.ops.aten.add.Tensor(x, 1)
            return (hidden_src, returned_src, add)
        ```

        Only the hidden mutation is prepended; the returned one keeps its slot,
        shifted by the one prepended output.
        """
        gm, hidden_src, returned_src, add = _graph_mixed_copies()

        result = pass_instance(gm)
        output_node = next(n for n in gm.graph.nodes if n.op == "output")

        assert result.modified is True
        assert output_node.args[0] == (hidden_src, returned_src, add)
        assert pass_instance.added_mutation_output_indices == [0]
        assert [
            (a.output_index, a.parameter_number) for a in pass_instance.get_mutation_info().aliases
        ] == [(0, 2), (1, 0)]
        assert not [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target == torch.ops.aten.copy_.default
        ]

    def test_shared_src_second_mutation_gets_its_own_slot(self, pass_instance):
        """
        IR Before:
        ```
        def forward(self, arg0_1, arg1_1, x):
            add = torch.ops.aten.add.Tensor(x, 1)
            copy_ = torch.ops.aten.copy_.default(arg0_1, add)
            copy__1 = torch.ops.aten.copy_.default(arg1_1, add)
            return (add,)
        ```

        IR After:
        ```
        def forward(self, arg0_1, arg1_1, x):
            add = torch.ops.aten.add.Tensor(x, 1)
            return (add, add)
        ```

        Two inputs are mutated with the same value, so two buffers must be
        donated. One output can only be donated to one input, so the second
        mutation cannot reuse the slot the first claimed and gets its own.
        """
        gm, add = _graph_shared_src()

        result = pass_instance(gm)
        output_node = next(n for n in gm.graph.nodes if n.op == "output")

        assert result.modified is True
        assert output_node.args[0] == (add, add)
        assert pass_instance.added_mutation_output_indices == [0]
        assert [
            (a.output_index, a.parameter_number) for a in pass_instance.get_mutation_info().aliases
        ] == [(0, 1), (1, 0)]

    def test_copy_returned_twice_aliases_first_slot_only(self, pass_instance):
        """
        IR Before:
        ```
        def forward(self, arg0_1, arg1_1):
            copy_ = torch.ops.aten.copy_.default(arg0_1, arg1_1)
            return (copy_, copy_)
        ```

        IR After:
        ```
        def forward(self, arg0_1, arg1_1):
            return (arg1_1, arg1_1)
        ```

        One mutation cannot donate arg0_1 to both slots, so only the first is a
        mutation alias. The second is a plain pass-through of arg1_1.
        """
        gm, arg1 = _graph_copy_returned_twice()

        result = pass_instance(gm)
        output_node = next(n for n in gm.graph.nodes if n.op == "output")

        assert result.modified is True
        assert output_node.args[0] == (arg1, arg1)
        assert pass_instance.added_mutation_output_indices == []
        assert [
            (a.output_index, a.parameter_number) for a in pass_instance.get_mutation_info().aliases
        ] == [(0, 0)]
        assert [
            (a.output_index, a.parameter_number)
            for a in pass_instance.get_passthrough_info().aliases
        ] == [(1, 1)]

    def test_bare_returned_copy_is_not_duplicated(self, pass_instance):
        """
        IR Before:
        ```
        def forward(self, arg0_1, arg1_1):
            copy_ = torch.ops.aten.copy_.default(arg0_1, arg1_1)
            return copy_
        ```

        IR After:
        ```
        def forward(self, arg0_1, arg1_1):
            return (arg1_1,)
        ```

        A non-tuple return holding the copy_ result is still recognized.
        """
        graph = fx.Graph()
        arg0 = graph.placeholder("arg0_1")
        arg1 = graph.placeholder("arg1_1")
        copy_inplace = graph.call_function(torch.ops.aten.copy_.default, (arg0, arg1))
        graph.output(copy_inplace)
        gm = fx.GraphModule(torch.nn.Module(), graph)

        result = pass_instance(gm)
        output_node = next(n for n in gm.graph.nodes if n.op == "output")

        assert result.modified is True
        assert output_node.args[0] == (arg1,)
        assert pass_instance.added_mutation_output_indices == []
        assert [
            (a.output_index, a.parameter_number) for a in pass_instance.get_mutation_info().aliases
        ] == [(0, 0)]

    def test_copy_inplace_non_placeholder_destination(self, pass_instance):
        """
        IR:
        ```
        def forward(self, arg0_1, arg1_1):
            clone = torch.ops.aten.clone.default(arg0_1);  arg0_1 = None
            copy_ = torch.ops.aten.copy_.default(clone, arg1_1);  clone = arg1_1 = None
            return (copy_,)
        ```

        copy_ to non-placeholder should NOT trigger mutation handling.
        (copy_ destination is clone, not a placeholder)
        """
        graph = fx.Graph()
        arg0 = graph.placeholder("arg0_1")
        arg1 = graph.placeholder("arg1_1")
        clone = graph.call_function(torch.ops.aten.clone.default, (arg0,))
        copy_inplace = graph.call_function(torch.ops.aten.copy_.default, (clone, arg1))
        graph.output((copy_inplace,))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        result = pass_instance(gm)

        # copy_ to non-placeholder is not handled by _handle_copy_inplace
        # But _detect_mutation_lineage might detect it through clone
        # This depends on implementation - adjust assertion accordingly
        assert result.modified is False  # No copy_ to placeholder

    def test_copy_inplace_with_scatter_src(self, pass_instance):
        """
        IR Before:
        ```
        def forward(self, arg0_1, arg1_1, arg2_1, arg3_1):
            copy_ = torch.ops.aten.copy_.default(arg0_1, arg1_1)
            scatter = torch.ops.aten.scatter.src(arg0_1, 0, arg2_1, arg3_1)
            return (scatter,)
        ```

        IR After:
        ```
        def forward(self, arg0_1, arg1_1, arg2_1, arg3_1):
            scatter = torch.ops.aten.scatter.src(arg1_1, 0, arg2_1, arg3_1)
            return (arg1_1, scatter)
        ```

        copy_ followed by scatter.src should be handled.
        Expected: Graph modified, MutationInfo(parameter_number=0, output_index=0)
        """
        graph = fx.Graph()
        arg0 = graph.placeholder("arg0_1")  # destination
        arg1 = graph.placeholder("arg1_1")  # copy source
        arg2 = graph.placeholder("arg2_1")  # scatter index
        arg3 = graph.placeholder("arg3_1")  # scatter source
        _ = graph.call_function(torch.ops.aten.copy_.default, (arg0, arg1))
        scatter = graph.call_function(torch.ops.aten.scatter.src, (arg0, 0, arg2, arg3))
        graph.output((scatter,))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        result = pass_instance(gm)
        mutation_info = pass_instance.get_mutation_info()

        assert result.modified is True
        assert len(mutation_info) == 1
        assert mutation_info.aliases[0].parameter_number == 0

    def test_copy_inplace_with_select_scatter(self, pass_instance):
        """
        IR Before:
        ```
        def forward(self, arg0_1, arg1_1, arg2_1):
            copy_ = torch.ops.aten.copy_.default(arg0_1, arg1_1)
            select_scatter = torch.ops.aten.select_scatter.default(arg0_1, arg2_1, 0, 0)
            return (select_scatter,)
        ```

        IR After:
        ```
        def forward(self, arg0_1, arg1_1, arg2_1):
            select_scatter = torch.ops.aten.select_scatter.default(arg1_1, arg2_1, 0, 0)
            return (arg1_1, select_scatter)
        ```

        copy_ followed by select_scatter should be handled.
        Expected: Graph modified, MutationInfo(parameter_number=0, output_index=0)
        """
        graph = fx.Graph()
        arg0 = graph.placeholder("arg0_1")  # destination
        arg1 = graph.placeholder("arg1_1")  # copy source
        arg2 = graph.placeholder("arg2_1")  # select_scatter source
        _ = graph.call_function(torch.ops.aten.copy_.default, (arg0, arg1))
        select_scatter = graph.call_function(
            torch.ops.aten.select_scatter.default, (arg0, arg2, 0, 0)
        )
        graph.output((select_scatter,))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        result = pass_instance(gm)
        mutation_info = pass_instance.get_mutation_info()

        assert result.modified is True
        assert len(mutation_info) == 1
        assert mutation_info.aliases[0].parameter_number == 0

    def test_copy_inplace_with_index_put(self, pass_instance):
        """
        IR Before:
        ```
        def forward(self, arg0_1, arg1_1, arg2_1, arg3_1):
            copy_ = torch.ops.aten.copy_.default(arg0_1, arg1_1)
            index_put = torch.ops.aten.index_put.default(arg0_1, [arg2_1], arg3_1)
            return (index_put,)
        ```

        IR After:
        ```
        def forward(self, arg0_1, arg1_1, arg2_1, arg3_1):
            index_put = torch.ops.aten.index_put.default(arg1_1, [arg2_1], arg3_1)
            return (arg1_1, index_put)
        ```

        copy_ followed by index_put should be handled.
        Expected: Graph modified, MutationInfo(parameter_number=0, output_index=0)
        """
        graph = fx.Graph()
        arg0 = graph.placeholder("arg0_1")  # destination
        arg1 = graph.placeholder("arg1_1")  # copy source
        arg2 = graph.placeholder("arg2_1")  # indices
        arg3 = graph.placeholder("arg3_1")  # values
        _ = graph.call_function(torch.ops.aten.copy_.default, (arg0, arg1))
        index_put = graph.call_function(torch.ops.aten.index_put.default, (arg0, [arg2], arg3))
        graph.output((index_put,))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        result = pass_instance(gm)
        mutation_info = pass_instance.get_mutation_info()

        assert result.modified is True
        assert len(mutation_info) == 1
        assert mutation_info.aliases[0].parameter_number == 0

    def test_copy_inplace_with_scatter_reduce_two(self, pass_instance):
        """
        IR Before:
        ```
        def forward(self, arg0_1, arg1_1, arg2_1, arg3_1):
            copy_ = torch.ops.aten.copy_.default(arg0_1, arg1_1)
            scatter_reduce = torch.ops.aten.scatter_reduce.two(arg0_1, 0, arg2_1, arg3_1, 'sum')
            return (scatter_reduce,)
        ```

        IR After:
        ```
        def forward(self, arg0_1, arg1_1, arg2_1, arg3_1):
            scatter_reduce = torch.ops.aten.scatter_reduce.two(arg1_1, 0, arg2_1, arg3_1, 'sum')
            return (arg1_1, scatter_reduce)
        ```

        copy_ followed by scatter_reduce.two should be handled.
        Expected: Graph modified, MutationInfo(parameter_number=0, output_index=0)
        """
        graph = fx.Graph()
        arg0 = graph.placeholder("arg0_1")  # destination
        arg1 = graph.placeholder("arg1_1")  # copy source
        arg2 = graph.placeholder("arg2_1")  # index
        arg3 = graph.placeholder("arg3_1")  # source
        _ = graph.call_function(torch.ops.aten.copy_.default, (arg0, arg1))
        scatter_reduce = graph.call_function(
            torch.ops.aten.scatter_reduce.two, (arg0, 0, arg2, arg3, "sum")
        )
        graph.output((scatter_reduce,))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        result = pass_instance(gm)
        mutation_info = pass_instance.get_mutation_info()

        assert result.modified is True
        assert len(mutation_info) == 1
        assert mutation_info.aliases[0].parameter_number == 0


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_output(self, pass_instance):
        """
        IR:
        ```
        def forward(self, arg0_1):
            return ()
        ```

        Graph with empty output should not crash.
        """
        graph = fx.Graph()
        _ = graph.placeholder("arg0_1")
        graph.output(())
        gm = fx.GraphModule(torch.nn.Module(), graph)

        result = pass_instance(gm)
        mutation_info = pass_instance.get_mutation_info()

        assert len(mutation_info) == 0
        assert result.modified is False

    def test_none_output(self, pass_instance):
        """
        IR:
        ```
        def forward(self, arg0_1):
            return None
        ```

        Graph with None output should not crash.
        """
        graph = fx.Graph()
        _ = graph.placeholder("arg0_1")
        graph.output(None)
        gm = fx.GraphModule(torch.nn.Module(), graph)

        _ = pass_instance(gm)
        mutation_info = pass_instance.get_mutation_info()

        assert len(mutation_info) == 0

    @pytest.mark.xfail
    def test_single_output_not_tuple(self, pass_instance):
        """
        IR:
        ```
        def forward(self, arg0_1):
            clone = torch.ops.aten.clone.default(arg0_1);  arg0_1 = None
            scatter = torch.ops.aten.scatter.src(clone, 0, ...);  clone = None
            return scatter  # Not wrapped in tuple
        ```

        Single output (not tuple) should be handled correctly.
        """
        graph = fx.Graph()
        arg0 = graph.placeholder("arg0_1")
        arg1 = graph.placeholder("arg1_1")
        arg2 = graph.placeholder("arg2_1")
        clone = graph.call_function(torch.ops.aten.clone.default, (arg0,))
        scatter = graph.call_function(torch.ops.aten.scatter.src, (clone, 0, arg1, arg2))
        graph.output(scatter)  # Not a tuple
        gm = fx.GraphModule(torch.nn.Module(), graph)

        _ = pass_instance(gm)
        mutation_info = pass_instance.get_mutation_info()

        assert len(mutation_info) == 1

    @pytest.mark.xfail
    def test_mixed_mutation_and_non_mutation_outputs(self, pass_instance):
        """
        IR:
        ```
        def forward(self, arg0_1, arg1_1, arg2_1, arg3_1):
            clone = torch.ops.aten.clone.default(arg0_1);  arg0_1 = None
            scatter = torch.ops.aten.scatter.src(clone, 0, arg2_1, arg3_1);  clone = None
            view = torch.ops.aten.view.default(arg1_1, [-1]);  arg1_1 = None
            add = torch.ops.aten.add.Tensor(arg2_1, arg3_1);  arg2_1 = arg3_1 = None
            return (scatter, view, add)
        ```

        Mix of mutated and non-mutated outputs should only detect mutations.
        Expected: Only scatter (output 0) detected as mutation
        """
        graph = fx.Graph()
        arg0 = graph.placeholder("arg0_1")  # will be mutated
        arg1 = graph.placeholder("arg1_1")  # just viewed
        arg2 = graph.placeholder("arg2_1")  # index
        arg3 = graph.placeholder("arg3_1")  # src

        clone = graph.call_function(torch.ops.aten.clone.default, (arg0,))
        scatter = graph.call_function(torch.ops.aten.scatter.src, (clone, 0, arg2, arg3))
        view = graph.call_function(torch.ops.aten.view.default, (arg1, [-1]))
        add = graph.call_function(torch.ops.aten.add.Tensor, (arg2, arg3))

        graph.output((scatter, view, add))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        _ = pass_instance(gm)
        mutation_info = pass_instance.get_mutation_info()

        # Only scatter should be detected as mutation
        assert len(mutation_info) == 1
        assert mutation_info.aliases[0].output_index == 0
        assert mutation_info.aliases[0].parameter_number == 0

    def test_no_placeholders(self, pass_instance):
        """
        IR:
        ```
        def forward(self):
            ones = torch.ones([4, 4])
            return (ones,)
        ```

        Graph with no placeholders should not crash.
        """
        graph = fx.Graph()
        const = graph.call_function(torch.ones, ([4, 4],))
        graph.output((const,))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        _ = pass_instance(gm)
        mutation_info = pass_instance.get_mutation_info()

        assert len(mutation_info) == 0

    def test_non_node_in_output(self, pass_instance):
        """
        IR:
        ```
        def forward(self, arg0_1):
            return (None, 42, arg0_1)
        ```

        Graph with constant output should not crash.
        Pass-through detection should find arg0_1 at output index 2.
        """
        graph = fx.Graph()
        arg0 = graph.placeholder("arg0_1")
        graph.output((None, 42, arg0))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        _ = pass_instance(gm)
        mutation_info = pass_instance.get_mutation_info()
        passthrough_info = pass_instance.get_passthrough_info()

        # None and 42 are skipped, no mutations
        assert len(mutation_info) == 0
        # arg0 at index 2 is a pass-through
        assert len(passthrough_info) == 1
        assert passthrough_info.aliases[0].parameter_number == 0
        assert passthrough_info.aliases[0].output_index == 2

    def test_getitem_node_in_output(self, pass_instance):
        """
        IR:
        ```
        def forward(self, arg0_1):
            split = torch.ops.aten.split.Tensor(arg0_1, 2)
            getitem = split[0]
            return (getitem,)
        ```

        getitem node (not a mutation) should not be detected.
        """
        graph = fx.Graph()
        arg0 = graph.placeholder("arg0_1")
        split = graph.call_function(torch.ops.aten.split.Tensor, (arg0, 2))
        getitem = graph.call_function(lambda x: x[0], (split,))
        graph.output((getitem,))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        _ = pass_instance(gm)
        mutation_info = pass_instance.get_mutation_info()

        # split + getitem is not a mutation pattern
        assert len(mutation_info) == 0


class TestRealWorldPatterns:
    """Tests with realistic model patterns."""

    @pytest.mark.xfail
    def test_kv_cache_pattern_full(self, pass_instance):
        """
        IR (simplified KV cache update):
        ```
        def forward(self, primals_1, primals_2, primals_3, primals_4):
            # primals_1 = weight, primals_2 = x, primals_3 = k_cache, primals_4 = v_cache
            clone = torch.ops.aten.clone.default(primals_3);  primals_3 = None
            clone_1 = torch.ops.aten.clone.default(primals_4);  primals_4 = None
            t = torch.ops.aten.t.default(primals_1);  primals_1 = None
            mm = torch.ops.aten.mm.default(primals_2, t);  t = None
            select = torch.ops.aten.select.int(clone, 0, 0)
            slice_1 = torch.ops.aten.slice.Tensor(select, 0, 0, 1);  select = None
            copy = torch.ops.aten.copy.default(slice_1, mm);  slice_1 = None
            select_1 = torch.ops.aten.select.int(clone, 0, 0)
            slice_scatter = torch.ops.aten.slice_scatter.default(
                select_1, copy, 0, 0, 1
            );  select_1 = copy = None
            select_scatter = torch.ops.aten.select_scatter.default(
                clone, slice_scatter, 0, 0
            );  clone = slice_scatter = None
            select_4 = torch.ops.aten.select.int(clone_1, 0, 0)
            slice_4 = torch.ops.aten.slice.Tensor(select_4, 0, 0, 1);  select_4 = None
            copy_1 = torch.ops.aten.copy.default(slice_4, mm);  slice_4 = mm = None
            select_5 = torch.ops.aten.select.int(clone_1, 0, 0)
            slice_scatter_1 = torch.ops.aten.slice_scatter.default(
                select_5, copy_1, 0, 0, 1
            );  select_5 = copy_1 = None
            select_scatter_1 = torch.ops.aten.select_scatter.default(
                clone_1, slice_scatter_1, 0, 0
            );  clone_1 = slice_scatter_1 = None
            sum_1 = torch.ops.aten.sum.default(select_scatter)
            mul = torch.ops.aten.mul.Tensor(primals_2, sum_1);  sum_1 = None
            add = torch.ops.aten.add.Tensor(mul, 0);  mul = None
            return (select_scatter, select_scatter_1, add, primals_2)
        ```

        Should detect mutations for k_cache (primals_3) and v_cache (primals_4).
        Expected: MutationInfo(parameter_number=2, output_index=0)  # k_cache
                  MutationInfo(parameter_number=3, output_index=1)  # v_cache
        """
        graph = fx.Graph()

        # Inputs
        primals_1 = graph.placeholder("primals_1")  # weight
        primals_2 = graph.placeholder("primals_2")  # input x
        primals_3 = graph.placeholder("primals_3")  # k_cache
        primals_4 = graph.placeholder("primals_4")  # v_cache

        # Clone caches
        clone = graph.call_function(torch.ops.aten.clone.default, (primals_3,))
        clone_1 = graph.call_function(torch.ops.aten.clone.default, (primals_4,))

        # Compute k
        t = graph.call_function(torch.ops.aten.t.default, (primals_1,))
        mm = graph.call_function(torch.ops.aten.mm.default, (primals_2, t))

        # Update k_cache
        select = graph.call_function(torch.ops.aten.select.int, (clone, 0, 0))
        slice_1 = graph.call_function(torch.ops.aten.slice.Tensor, (select, 0, 0, 1))
        copy = graph.call_function(torch.ops.aten.copy.default, (slice_1, mm))
        select_1 = graph.call_function(torch.ops.aten.select.int, (clone, 0, 0))
        slice_scatter = graph.call_function(
            torch.ops.aten.slice_scatter.default, (select_1, copy, 0, 0, 1)
        )
        select_scatter = graph.call_function(
            torch.ops.aten.select_scatter.default, (clone, slice_scatter, 0, 0)
        )

        # Update v_cache (similar)
        select_4 = graph.call_function(torch.ops.aten.select.int, (clone_1, 0, 0))
        slice_4 = graph.call_function(torch.ops.aten.slice.Tensor, (select_4, 0, 0, 1))
        copy_1 = graph.call_function(torch.ops.aten.copy.default, (slice_4, mm))
        select_5 = graph.call_function(torch.ops.aten.select.int, (clone_1, 0, 0))
        slice_scatter_1 = graph.call_function(
            torch.ops.aten.slice_scatter.default, (select_5, copy_1, 0, 0, 1)
        )
        select_scatter_1 = graph.call_function(
            torch.ops.aten.select_scatter.default, (clone_1, slice_scatter_1, 0, 0)
        )

        # Output computation
        sum_1 = graph.call_function(torch.ops.aten.sum.default, (select_scatter,))
        mul = graph.call_function(torch.ops.aten.mul.Tensor, (primals_2, sum_1))
        add = graph.call_function(torch.ops.aten.add.Tensor, (mul, 0))

        graph.output((select_scatter, select_scatter_1, add, primals_2))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        _ = pass_instance(gm)
        mutation_info = pass_instance.get_mutation_info()

        # Should detect mutations for k_cache (primals_3) and v_cache (primals_4)
        assert len(mutation_info) == 2
        param_numbers = {a.parameter_number for a in mutation_info.aliases}
        assert param_numbers == {2, 3}  # primals_3 and primals_4

    @pytest.mark.xfail
    def test_simple_scatter_pattern(self, pass_instance):
        """
        IR (from test_kv_cache_pattern1):
        ```
        def forward(self, arg0_1, arg1_1, arg2_1):
            # arg0_1 = pos, arg1_1 = cache, arg2_1 = new_kv
            unsqueeze = torch.ops.aten.unsqueeze.default(arg0_1, 0);  arg0_1 = None
            unsqueeze_1 = torch.ops.aten.unsqueeze.default(unsqueeze, 0);  unsqueeze = None
            unsqueeze_2 = torch.ops.aten.unsqueeze.default(unsqueeze_1, -1);  unsqueeze_1 = None
            # expand = torch.ops.aten.expand.default(unsqueeze_2, [1, 1, 1, 64]);
            # unsqueeze_2 = None
            # unsqueeze_3 = torch.ops.aten.unsqueeze.default(arg2_1, 2);
            # arg2_1 = None
            # scatter = torch.ops.aten.scatter.src(arg1_1, 2, expand, unsqueeze_3);
            # arg1_1 = expand = unsqueeze_3 = None
            return (scatter,)
        ```

        Should detect mutation on arg1_1 (cache).
        Expected: MutationInfo(parameter_number=1, output_index=0)
        """
        graph = fx.Graph()

        arg0_1 = graph.placeholder("arg0_1")  # pos
        arg1_1 = graph.placeholder("arg1_1")  # cache
        arg2_1 = graph.placeholder("arg2_1")  # new_kv

        unsqueeze = graph.call_function(torch.ops.aten.unsqueeze.default, (arg0_1, 0))
        unsqueeze_1 = graph.call_function(torch.ops.aten.unsqueeze.default, (unsqueeze, 0))
        unsqueeze_2 = graph.call_function(torch.ops.aten.unsqueeze.default, (unsqueeze_1, -1))
        expand = graph.call_function(torch.ops.aten.expand.default, (unsqueeze_2, [1, 1, 1, 64]))
        unsqueeze_3 = graph.call_function(torch.ops.aten.unsqueeze.default, (arg2_1, 2))
        scatter = graph.call_function(torch.ops.aten.scatter.src, (arg1_1, 2, expand, unsqueeze_3))

        graph.output((scatter,))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        _ = pass_instance(gm)
        mutation_info = pass_instance.get_mutation_info()

        # Should detect mutation on arg1_1 (cache)
        assert len(mutation_info) == 1
        assert mutation_info.aliases[0].parameter_number == 1
        assert mutation_info.aliases[0].output_index == 0

    @pytest.mark.xfail
    def test_attention_kv_cache_pattern(self, pass_instance):
        """
        IR (attention-style KV cache with multiple heads):
        ```
        def forward(self, arg0_1, arg1_1, arg2_1, arg3_1):
            # arg0_1 = k_cache [batch, heads, seq, dim]
            # arg1_1 = v_cache [batch, heads, seq, dim]
            # arg2_1 = new_k [batch, heads, 1, dim]
            # arg3_1 = position index tensor
            clone = torch.ops.aten.clone.default(arg0_1)
            clone_1 = torch.ops.aten.clone.default(arg1_1)
            select_scatter = torch.ops.aten.select_scatter.default(clone, arg2_1, 2, 0)
            select_scatter_1 = torch.ops.aten.select_scatter.default(clone_1, arg2_1, 2, 0)
            return (select_scatter, select_scatter_1)
        ```

        Expected: MutationInfo(parameter_number=0, output_index=0)  # k_cache
                  MutationInfo(parameter_number=1, output_index=1)  # v_cache
        """
        graph = fx.Graph()

        arg0_1 = graph.placeholder("arg0_1")  # k_cache
        arg1_1 = graph.placeholder("arg1_1")  # v_cache
        arg2_1 = graph.placeholder("arg2_1")  # new_k
        _ = graph.placeholder("arg3_1")  # position (unused)

        clone = graph.call_function(torch.ops.aten.clone.default, (arg0_1,))
        clone_1 = graph.call_function(torch.ops.aten.clone.default, (arg1_1,))

        select_scatter = graph.call_function(
            torch.ops.aten.select_scatter.default, (clone, arg2_1, 2, 0)
        )
        select_scatter_1 = graph.call_function(
            torch.ops.aten.select_scatter.default, (clone_1, arg2_1, 2, 0)
        )

        graph.output((select_scatter, select_scatter_1))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        _ = pass_instance(gm)
        mutation_info = pass_instance.get_mutation_info()

        assert len(mutation_info) == 2
        param_numbers = {a.parameter_number for a in mutation_info.aliases}
        output_indices = {a.output_index for a in mutation_info.aliases}
        assert param_numbers == {0, 1}
        assert output_indices == {0, 1}

    @pytest.mark.xfail
    def test_nested_select_slice_scatter_pattern(self, pass_instance):
        """
        IR (nested select/slice before scatter - common in cache updates):
        ```
        def forward(self, arg0_1, arg1_1):
            # arg0_1 = cache [batch, heads, seq, dim]
            # arg1_1 = new_value
            clone = torch.ops.aten.clone.default(arg0_1)
            select = torch.ops.aten.select.int(clone, 0, 0)  # select batch
            select_1 = torch.ops.aten.select.int(select, 0, 0)  # select head
            slice_scatter = torch.ops.aten.slice_scatter.default(select_1, arg1_1, 0, 0, 1)
            select_scatter = torch.ops.aten.select_scatter.default(select, slice_scatter, 0, 0)
            select_scatter_1 = torch.ops.aten.select_scatter.default(clone, select_scatter, 0, 0)
            return (select_scatter_1,)
        ```

        Expected: MutationInfo(parameter_number=0, output_index=0)
        """
        graph = fx.Graph()

        arg0_1 = graph.placeholder("arg0_1")  # cache
        arg1_1 = graph.placeholder("arg1_1")  # new_value

        clone = graph.call_function(torch.ops.aten.clone.default, (arg0_1,))
        select = graph.call_function(torch.ops.aten.select.int, (clone, 0, 0))
        select_1 = graph.call_function(torch.ops.aten.select.int, (select, 0, 0))
        slice_scatter = graph.call_function(
            torch.ops.aten.slice_scatter.default, (select_1, arg1_1, 0, 0, 1)
        )
        select_scatter = graph.call_function(
            torch.ops.aten.select_scatter.default, (select, slice_scatter, 0, 0)
        )
        select_scatter_1 = graph.call_function(
            torch.ops.aten.select_scatter.default, (clone, select_scatter, 0, 0)
        )

        graph.output((select_scatter_1,))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        _ = pass_instance(gm)
        mutation_info = pass_instance.get_mutation_info()

        assert len(mutation_info) == 1
        assert mutation_info.aliases[0].parameter_number == 0
        assert mutation_info.aliases[0].output_index == 0


# ============================================================================
# Pass-through Detection Tests
# ============================================================================


class TestPassthroughDetection:
    """Tests for _detect_passthrough_outputs.

    Pass-throughs are outputs that directly return an input placeholder
    unchanged. This happens with activation checkpointing + torch.compile,
    where AOT Autograd includes input primals (weights) in the forward
    graph's return so backward can recompute the forward.

    Pass-throughs are tracked in passthrough_info (separate from mutation_info)
    because the executor must NOT filter them from outputs — AOTAutograd expects
    them in the return tuple for backward recomputation.
    """

    def test_single_passthrough(self, pass_instance):
        """
        IR:
        ```
        def forward(self, primals_1):
            add = torch.ops.aten.add.Tensor(primals_1, primals_1)
            return (add, primals_1)
        ```

        output[1] is a direct pass-through of input[0].
        Expected: 1 passthrough entry (output_index=1, parameter_number=0)
        """
        graph = fx.Graph()
        primals_1 = graph.placeholder("primals_1")
        add = graph.call_function(torch.ops.aten.add.Tensor, (primals_1, primals_1))
        graph.output((add, primals_1))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        result = pass_instance(gm)
        passthrough_info = pass_instance.get_passthrough_info()

        assert result.modified is False
        assert len(passthrough_info) == 1
        assert passthrough_info.aliases[0].parameter_number == 0
        assert passthrough_info.aliases[0].output_index == 1
        assert passthrough_info.aliases[0].parameter_index == []

    def test_multiple_passthrough_weights(self, pass_instance):
        """
        IR (simplified activation checkpointing pattern):
        ```
        def forward(self, primals_1, primals_2, primals_3):
            add = torch.ops.aten.add.Tensor(primals_1, primals_2)
            return (add, primals_1, primals_2, primals_3)
        ```

        Outputs 1, 2, 3 are pass-throughs of inputs 0, 1, 2.
        """
        graph = fx.Graph()
        primals_1 = graph.placeholder("primals_1")
        primals_2 = graph.placeholder("primals_2")
        primals_3 = graph.placeholder("primals_3")
        add = graph.call_function(torch.ops.aten.add.Tensor, (primals_1, primals_2))
        graph.output((add, primals_1, primals_2, primals_3))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        result = pass_instance(gm)
        passthrough_info = pass_instance.get_passthrough_info()

        assert result.modified is False
        assert len(passthrough_info) == 3
        aliases = sorted(passthrough_info.aliases, key=lambda a: a.output_index)
        assert aliases[0].output_index == 1 and aliases[0].parameter_number == 0
        assert aliases[1].output_index == 2 and aliases[1].parameter_number == 1
        assert aliases[2].output_index == 3 and aliases[2].parameter_number == 2

    def test_passthrough_not_first_input(self, pass_instance):
        """
        IR:
        ```
        def forward(self, primals_1, primals_2):
            add = torch.ops.aten.add.Tensor(primals_1, primals_2)
            return (add, primals_2)
        ```

        Only primals_2 (input[1]) is passed through at output[1].
        """
        graph = fx.Graph()
        primals_1 = graph.placeholder("primals_1")
        primals_2 = graph.placeholder("primals_2")
        add = graph.call_function(torch.ops.aten.add.Tensor, (primals_1, primals_2))
        graph.output((add, primals_2))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        _ = pass_instance(gm)
        passthrough_info = pass_instance.get_passthrough_info()

        assert len(passthrough_info) == 1
        assert passthrough_info.aliases[0].parameter_number == 1
        assert passthrough_info.aliases[0].output_index == 1

    def test_no_passthrough_computed_outputs(self, pass_instance):
        """
        IR:
        ```
        def forward(self, primals_1, primals_2):
            add = torch.ops.aten.add.Tensor(primals_1, primals_2)
            mul = torch.ops.aten.mul.Tensor(primals_1, primals_2)
            return (add, mul)
        ```

        Both outputs are computed, no pass-throughs.
        """
        graph = fx.Graph()
        primals_1 = graph.placeholder("primals_1")
        primals_2 = graph.placeholder("primals_2")
        add = graph.call_function(torch.ops.aten.add.Tensor, (primals_1, primals_2))
        mul = graph.call_function(torch.ops.aten.mul.Tensor, (primals_1, primals_2))
        graph.output((add, mul))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        _ = pass_instance(gm)
        passthrough_info = pass_instance.get_passthrough_info()

        assert len(passthrough_info) == 0

    def test_view_of_input_not_passthrough(self, pass_instance):
        """
        IR:
        ```
        def forward(self, primals_1):
            view = torch.ops.aten.view.default(primals_1, [4, 4])
            return (view, primals_1)
        ```

        view is NOT a pass-through (it's a computed node, not a placeholder).
        Only primals_1 at output[1] is a pass-through.
        """
        graph = fx.Graph()
        primals_1 = graph.placeholder("primals_1")
        view = graph.call_function(torch.ops.aten.view.default, (primals_1, [4, 4]))
        graph.output((view, primals_1))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        _ = pass_instance(gm)
        passthrough_info = pass_instance.get_passthrough_info()

        assert len(passthrough_info) == 1
        assert passthrough_info.aliases[0].output_index == 1
        assert passthrough_info.aliases[0].parameter_number == 0

    def test_passthrough_with_copy_inplace(self, pass_instance):
        """
        IR Before:
        ```
        def forward(self, arg0_1, arg1_1, arg2_1):
            copy_ = torch.ops.aten.copy_.default(arg0_1, arg1_1)
            return (copy_, arg2_1)
        ```

        IR After (copy_ removed, its existing output slot reused):
        ```
        def forward(self, arg0_1, arg1_1, arg2_1):
            return (arg1_1, arg2_1)
        ```

        output[0] holds arg1_1 but is the mutation's reused slot, so it aliases
        arg0_1, the destination, not arg1_1. A slot can only be donated to one
        input, so arg1_1 gets no pass-through of its own. arg2_1 at output[1] is an
        ordinary pass-through.
        """
        graph = fx.Graph()
        arg0 = graph.placeholder("arg0_1")
        arg1 = graph.placeholder("arg1_1")
        arg2 = graph.placeholder("arg2_1")
        copy_inplace = graph.call_function(torch.ops.aten.copy_.default, (arg0, arg1))
        graph.output((copy_inplace, arg2))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        result = pass_instance(gm)
        mutation_info = pass_instance.get_mutation_info()
        passthrough_info = pass_instance.get_passthrough_info()

        assert result.modified is True
        # copy_ mutation: output[0] aliases input[0]
        assert len(mutation_info) == 1
        assert mutation_info.aliases[0].output_index == 0
        assert mutation_info.aliases[0].parameter_number == 0
        # pass-through: arg2_1 (input[2]) at output[1]
        assert len(passthrough_info) == 1
        assert passthrough_info.aliases[0].output_index == 1
        assert passthrough_info.aliases[0].parameter_number == 2

    def test_passthrough_skips_already_aliased(self, pass_instance):
        """
        IR Before:
        ```
        def forward(self, arg0_1, arg1_1):
            copy_ = torch.ops.aten.copy_.default(arg0_1, arg1_1)
            return (copy_,)
        ```

        IR After:
        ```
        def forward(self, arg0_1, arg1_1):
            return (arg1_1,)
        ```

        output[0] holds arg1_1, a placeholder, but it is already a mutation alias
        of arg0_1. Recording it as a pass-through of arg1_1 too would claim two
        different donors for one slot, so pass-through detection skips it.
        """
        graph = fx.Graph()
        arg0 = graph.placeholder("arg0_1")
        arg1 = graph.placeholder("arg1_1")
        copy_inplace = graph.call_function(torch.ops.aten.copy_.default, (arg0, arg1))
        graph.output((copy_inplace,))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        result = pass_instance(gm)
        mutation_info = pass_instance.get_mutation_info()
        passthrough_info = pass_instance.get_passthrough_info()

        assert result.modified is True
        assert [(a.output_index, a.parameter_number) for a in mutation_info.aliases] == [(0, 0)]
        assert len(passthrough_info) == 0

    def test_reused_slot_may_hold_a_bare_placeholder(self, pass_instance):
        """
        IR:
        ```
        def forward(self, arg0_1, arg1_1):
            copy_ = torch.ops.aten.copy_.default(arg0_1, arg1_1)
            return (copy_,)
        ```

        The reused slot ends up holding arg1_1, a bare placeholder, while aliasing
        arg0_1. An output carrying no computation has no store in the compiled NEFF
        (see the note in ``torch_neuronx/python_ops/torch_mlir/kernel.py``), so both
        the mutation and this slot's value depend on the compiler honouring
        must_alias here. Pinned because it is the one alias shape in this pass where
        the aliased input differs from the value in the slot.
        """
        graph = fx.Graph()
        arg0 = graph.placeholder("arg0_1")
        arg1 = graph.placeholder("arg1_1")
        copy_inplace = graph.call_function(torch.ops.aten.copy_.default, (arg0, arg1))
        graph.output((copy_inplace,))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        pass_instance(gm)
        mutation_info = pass_instance.get_mutation_info()
        output_node = next(n for n in gm.graph.nodes if n.op == "output")

        assert output_node.args[0] == (arg1,)
        assert output_node.args[0][0].op == "placeholder"
        assert [(a.output_index, a.parameter_number) for a in mutation_info.aliases] == [(0, 0)]

    def test_activation_checkpoint_pattern(self, pass_instance):
        """
        Realistic activation checkpointing pattern from AOT Autograd.

        IR:
        ```
        def forward(self, primals_1, primals_2, primals_3, primals_4):
            # primals_1 = weight_1, primals_2 = weight_2
            # primals_3 = input, primals_4 = weight_3
            mm = torch.ops.aten.mm.default(primals_3, primals_1)
            add = torch.ops.aten.add.Tensor(mm, primals_2)
            mm_1 = torch.ops.aten.mm.default(add, primals_4)
            return (mm_1, primals_1, primals_2, primals_3, primals_4)
        ```

        output[0] = computed result
        outputs[1..4] = pass-through weights/inputs for backward recomputation
        """
        graph = fx.Graph()
        primals_1 = graph.placeholder("primals_1")
        primals_2 = graph.placeholder("primals_2")
        primals_3 = graph.placeholder("primals_3")
        primals_4 = graph.placeholder("primals_4")

        mm = graph.call_function(torch.ops.aten.mm.default, (primals_3, primals_1))
        add = graph.call_function(torch.ops.aten.add.Tensor, (mm, primals_2))
        mm_1 = graph.call_function(torch.ops.aten.mm.default, (add, primals_4))

        graph.output((mm_1, primals_1, primals_2, primals_3, primals_4))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        result = pass_instance(gm)
        passthrough_info = pass_instance.get_passthrough_info()

        assert result.modified is False
        assert len(passthrough_info) == 4

        aliases = sorted(passthrough_info.aliases, key=lambda a: a.output_index)
        for i, alias in enumerate(aliases):
            assert alias.output_index == i + 1  # outputs 1, 2, 3, 4
            assert alias.parameter_number == i  # inputs 0, 1, 2, 3
            assert alias.parameter_index == []

    def test_same_input_returned_twice(self, pass_instance):
        """
        IR:
        ```
        def forward(self, primals_1):
            add = torch.ops.aten.add.Tensor(primals_1, primals_1)
            return (add, primals_1, primals_1)
        ```

        Same input returned at two output positions.
        Both should be detected as pass-throughs.
        """
        graph = fx.Graph()
        primals_1 = graph.placeholder("primals_1")
        add = graph.call_function(torch.ops.aten.add.Tensor, (primals_1, primals_1))
        graph.output((add, primals_1, primals_1))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        _ = pass_instance(gm)
        passthrough_info = pass_instance.get_passthrough_info()

        assert len(passthrough_info) == 2
        aliases = sorted(passthrough_info.aliases, key=lambda a: a.output_index)
        assert aliases[0].output_index == 1 and aliases[0].parameter_number == 0
        assert aliases[1].output_index == 2 and aliases[1].parameter_number == 0

    def test_empty_graph_no_passthrough(self, pass_instance):
        """
        IR:
        ```
        def forward(self):
            return ()
        ```

        No inputs, no outputs — nothing to detect.
        """
        graph = fx.Graph()
        graph.output(())
        gm = fx.GraphModule(torch.nn.Module(), graph)

        result = pass_instance(gm)
        passthrough_info = pass_instance.get_passthrough_info()

        assert len(passthrough_info) == 0
        assert result.modified is False


def _run_with_state(build_graph):
    """Run the pass with a state attached so it emits PassMutations."""
    state = NeuronFXPassState(job=MagicMock(), model_name="test", dumping_enabled=False)
    pass_instance = FunctionalizeCopyInplacePass(state)
    gm, *_ = build_graph()
    pass_instance(gm)
    return gm, state


def _graph_outputs(gm):
    output_node = next(n for n in gm.graph.nodes if n.op == "output")
    outputs = output_node.args[0]
    return list(outputs) if isinstance(outputs, list | tuple) else [outputs]


def _add_indices(state):
    """Indices of the single MutationKind.ADD entry, or None when none was emitted."""
    added = [
        m.indices
        for pm in state.pass_mutations
        for m in pm.output_mutations
        if m.kind == MutationKind.ADD
    ]
    assert len(added) <= 1, f"expected at most one ADD entry, got {added}"
    return added[0] if added else None


def _caller_output_slots(gm, state, extra_pass_mutations=()):
    """NEFF output slot labels as the caller receives them after output mutations."""
    pass_mutations = [*state.pass_mutations, *extra_pass_mutations]
    n_graph_outputs = len(_graph_outputs(gm))
    n_neff_outputs = n_graph_outputs - sum(
        len(m.indices)
        for pm in extra_pass_mutations
        for m in pm.output_mutations
        if m.kind == MutationKind.REMOVE
    )
    return apply_output_mutations(
        [f"neff{i}" for i in range(n_neff_outputs)], pass_mutations, "cpu"
    )


# ============================================================================
# Emitted IO mutations, as the executor and the compiler consume them
# ============================================================================


class TestEmittedIOMutations:
    """Pins the PassMutations the pass emits, not the private bookkeeping.

    MutationKind.ADD makes the executor filter an output out, so it must name only
    the slots this pass actually added, not every slot that carries a mutation.
    """

    def test_returned_copy_emits_no_add(self):
        gm, state = _run_with_state(_graph_returned_copy)

        assert len(_graph_outputs(gm)) == 1
        assert _add_indices(state) is None
        assert get_alias_output_buffer_map(state.pass_mutations) == {0: 0}
        assert get_mutation_copy_back_inputs(state.pass_mutations) == {0: 0}
        assert _caller_output_slots(gm, state) == ["neff0"]

    def test_src_returned_emits_no_add(self):
        gm, state = _run_with_state(_graph_src_returned)

        assert len(_graph_outputs(gm)) == 3
        assert _add_indices(state) is None
        # output[2] is the mutation result, output[1] a plain weight pass-through.
        assert get_alias_output_buffer_map(state.pass_mutations) == {1: 1, 2: 0}
        assert get_mutation_copy_back_inputs(state.pass_mutations) == {2: 0}
        assert _caller_output_slots(gm, state) == ["neff0", "neff1", "neff2"]

    def test_hidden_copy_emits_add(self):
        gm, state = _run_with_state(_graph_hidden_copy)

        assert len(_graph_outputs(gm)) == 2
        assert _add_indices(state) == [0]
        assert get_alias_output_buffer_map(state.pass_mutations) == {0: 0}
        assert get_mutation_copy_back_inputs(state.pass_mutations) == {0: 0}
        # The added slot is filtered out, leaving only the model's own output.
        assert _caller_output_slots(gm, state) == ["neff1"]

    def test_mixed_copies_add_only_the_hidden_one(self):
        gm, state = _run_with_state(_graph_mixed_copies)

        assert len(_graph_outputs(gm)) == 3
        assert _add_indices(state) == [0]
        assert get_alias_output_buffer_map(state.pass_mutations) == {0: 2, 1: 0}
        assert get_mutation_copy_back_inputs(state.pass_mutations) == {0: 2, 1: 0}
        assert _caller_output_slots(gm, state) == ["neff1", "neff2"]

    def test_shared_src_donates_a_buffer_per_input(self):
        gm, state = _run_with_state(_graph_shared_src)

        assert len(_graph_outputs(gm)) == 2
        assert _add_indices(state) == [0]
        assert get_alias_output_buffer_map(state.pass_mutations) == {0: 1, 1: 0}
        assert get_mutation_copy_back_inputs(state.pass_mutations) == {0: 1, 1: 0}
        assert _caller_output_slots(gm, state) == ["neff1"]

    def test_reused_slot_survives_a_later_output_removal(self):
        """A reused slot is not at index 0, so a successor REMOVE has to shift it."""
        gm, state = _run_with_state(_graph_returned_copy_with_none)

        assert _graph_outputs(gm)[0] is None
        assert _add_indices(state) is None

        remove_none = PassMutations(
            pass_name="RemoveNoneOutputs",
            output_mutations=[
                IOMutation(
                    kind=MutationKind.REMOVE,
                    indices=[0],
                    transform_key="none",
                    pass_name="RemoveNoneOutputs",
                )
            ],
        )
        pass_mutations = [*state.pass_mutations, remove_none]

        assert get_alias_output_buffer_map(pass_mutations) == {0: 0}
        assert get_mutation_copy_back_inputs(pass_mutations) == {0: 0}
        assert _caller_output_slots(gm, state, [remove_none]) == [None, "neff0"]

    def test_reused_slot_copy_back_names_a_caller_visible_slot(self):
        """A reused slot's copy-back index is one the caller receives.

        ``Executor._cast_output_device`` yields ``original_input.copy_(...)`` for
        every copy-back index, i.e. the caller's own input tensor object, so under
        TORCH_NEURONX_RETAIN_DEVICE_MODE=1 the caller receives that object as this
        output. The value is the mutation result either way, and object identity
        matches eager, which returns the mutated tensor itself.
        """
        gm, state = _run_with_state(_graph_returned_copy)

        copy_back = get_mutation_copy_back_inputs(state.pass_mutations)
        caller_slots = _caller_output_slots(gm, state)

        assert copy_back == {0: 0}
        assert all(f"neff{i}" in caller_slots for i in copy_back)

    def test_added_slot_copy_back_names_a_filtered_slot(self):
        """A prepended slot's copy-back index never reaches the caller."""
        gm, state = _run_with_state(_graph_hidden_copy)

        copy_back = get_mutation_copy_back_inputs(state.pass_mutations)
        caller_slots = _caller_output_slots(gm, state)

        assert copy_back == {0: 0}
        assert all(f"neff{i}" not in caller_slots for i in copy_back)

    def test_no_mutations_emits_nothing(self):
        gm, state = _run_with_state(_graph_no_mutation)

        assert state.pass_mutations == []
        assert get_alias_output_buffer_map(state.pass_mutations) == {}
        assert _caller_output_slots(gm, state) == ["neff0"]


# ============================================================================
# Pre-AOT alias rebase hook
# ============================================================================


def _make_state_with_pre_aot(pre_aot_pm, preceding_pass_mutations=None):
    from unittest.mock import MagicMock

    from torch_neuronx.neuron_dynamo_backend.fx.passes.neuron_fx_pass import NeuronFXPassState

    state = NeuronFXPassState(
        job=MagicMock(),
        model_name="test",
        dumping_enabled=False,
        pre_aot_pass_mutations=pre_aot_pm,
    )
    for pm_entry in preceding_pass_mutations or []:
        state.pass_mutations.append(pm_entry)
    return state


def _make_alias_pm(out_indices, in_indices, is_mutation=False):
    from torch_neuronx.neuron_dynamo_backend.fx.io_mutation import (
        IOMutation,
        MutationKind,
        PassMutations,
    )

    return PassMutations(
        pass_name="AliasingAnalysis",
        output_mutations=[
            IOMutation(
                kind=MutationKind.ALIAS,
                indices=out_indices,
                alias_input_indices=in_indices,
                is_mutation=is_mutation,
                pass_name="AliasingAnalysis",
            )
        ],
    )


def _make_preceding_pm(pass_name, in_remove=None, in_add=None, out_add=None, out_remove=None):
    from torch_neuronx.neuron_dynamo_backend.fx.io_mutation import (
        IOMutation,
        MutationKind,
        PassMutations,
    )

    in_muts, out_muts = [], []
    if in_remove:
        in_muts.append(IOMutation(kind=MutationKind.REMOVE, indices=in_remove, pass_name=pass_name))
    if in_add:
        in_muts.append(
            IOMutation(
                kind=MutationKind.ADD,
                indices=in_add,
                transform_key="shard_index",
                transform_params={"shard_index": 0},
                pass_name=pass_name,
            )
        )
    if out_add:
        out_muts.append(IOMutation(kind=MutationKind.ADD, indices=out_add, pass_name=pass_name))
    if out_remove:
        out_muts.append(
            IOMutation(
                kind=MutationKind.REMOVE,
                indices=out_remove,
                transform_key="none",
                pass_name=pass_name,
            )
        )
    return PassMutations(pass_name=pass_name, input_mutations=in_muts, output_mutations=out_muts)


@pytest.mark.xfail(reason="Pre-AOT rebase temporarily disabled")
@pytest.mark.parametrize(
    "pre_aot_pm,preceding,exp_buffer_map",
    [
        # None pre_aot → nothing appended, buffer map empty
        (None, [], {}),
        # No preceding passes: indices unchanged
        (_make_alias_pm([1], [0]), [], {1: 0}),
        # Preceding input REMOVE [0]: alias input[1] → 0
        (_make_alias_pm([1], [1]), [_make_preceding_pm("Flex", in_remove=[0])], {1: 0}),
        # Preceding output ADD [0]: alias output[1] → 2
        (_make_alias_pm([1], [0]), [_make_preceding_pm("Flex", out_add=[0])], {2: 0}),
        # Preceding input ADD [0]: alias input[1] → 2
        (_make_alias_pm([0], [1]), [_make_preceding_pm("Flex", in_add=[0])], {0: 2}),
        # Preceding output REMOVE eliminates alias output → empty
        (_make_alias_pm([0], [0]), [_make_preceding_pm("Flex", out_remove=[0])], {}),
        # is_mutation=True also rebased correctly
        (
            _make_alias_pm([0], [1], is_mutation=True),
            [_make_preceding_pm("Flex", in_remove=[0])],
            {0: 0},
        ),
        # Multiple preceding passes chain
        (
            _make_alias_pm([1], [2]),
            [
                _make_preceding_pm("Flex1", in_remove=[0]),  # input[2] → 1
                _make_preceding_pm("Flex2", out_add=[0]),  # output[1] → 2
            ],
            {2: 1},
        ),
    ],
)
def test_pre_aot_rebase_hook(pre_aot_pm, preceding, exp_buffer_map):
    """FunctionalizeCopyInplacePass rebases pre-AOT ALIAS mutations through preceding passes."""
    from torch_neuronx.neuron_dynamo_backend.fx.io_mutation import (
        MutationKind,
        get_alias_output_buffer_map,
    )

    state = _make_state_with_pre_aot(pre_aot_pm, preceding)
    pass_instance = FunctionalizeCopyInplacePass(state)

    graph = fx.Graph()
    p = graph.placeholder("x")
    graph.output(p)
    gm = fx.GraphModule(torch.nn.Module(), graph)
    pass_instance(gm)

    # Verify buffer map from all pass_mutations matches expected
    assert get_alias_output_buffer_map(state.pass_mutations) == exp_buffer_map

    # If pre_aot_pm is not None, verify AliasingAnalysis entry exists with correct indices
    if pre_aot_pm is not None and exp_buffer_map:
        aa_entries = [p for p in state.pass_mutations if p.pass_name == "AliasingAnalysis"]
        assert len(aa_entries) == 1
        aa = aa_entries[0]
        alias_muts = [m for m in aa.output_mutations if m.kind == MutationKind.ALIAS]
        assert alias_muts
        actual_map = {
            o: i for m in alias_muts for o, i in zip(m.indices, m.alias_input_indices, strict=False)
        }
        assert actual_map == exp_buffer_map


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
