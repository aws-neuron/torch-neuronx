import pytest
import torch
import torch.fx as fx
import torch.nn as nn
from torch.fx import GraphModule, symbolic_trace

from torch_neuronx.neuron_dynamo_backend.fx.passes.functionalize_copy_inplace_result import (
    FunctionalizeCopyInplacePass,
)

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

        Directly returning an input should NOT be detected as mutation.
        """
        graph = fx.Graph()
        arg0 = graph.placeholder("arg0_1")
        graph.output((arg0,))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        _ = pass_instance(gm)
        mutation_info = pass_instance.get_mutation_info()

        assert len(mutation_info) == 0

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
            return (arg1_1, arg1_1)  # copy_ removed, src prepended
        ```

        copy_ operation should be removed and mutation recorded.
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

        assert result.modified is True
        assert len(mutation_info) == 1
        assert mutation_info.aliases[0].parameter_number == 0

        # Verify copy_ was removed
        copy_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target == torch.ops.aten.copy_.default
        ]
        assert len(copy_nodes) == 0

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
        """
        graph = fx.Graph()
        arg0 = graph.placeholder("arg0_1")
        graph.output((None, 42, arg0))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        _ = pass_instance(gm)
        mutation_info = pass_instance.get_mutation_info()

        # Should not crash, and should not detect None/42 as mutations
        assert len(mutation_info) == 0

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
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
