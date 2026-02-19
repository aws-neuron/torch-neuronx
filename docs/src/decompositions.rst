.. meta::
   :description: How decompositions transform complex PyTorch operations into simpler primitives
   :date-modified: 2026-01-26
   :keywords: neuron, pytorch, decompositions, aten ops

Decompositions
==============

Decompositions transform complex PyTorch operations into simpler, primitive operations. They serve as a bridge between high-level PyTorch APIs and lower-level representations that backends can handle.

**Key benefits:**

- **Simplification**: Complex ops become sequences of basic ops
- **Backend compatibility**: Backends only need to implement core operations
- **Optimization**: Simpler ops enable better analysis and optimization

How Decompositions Work
-----------------------

::

    PyTorch Op → Decomposition → Simpler Ops → Backend IR → Compiled Code

The ``@register_decomposition`` decorator registers a function as the decomposition rule for specific operations. It extends `PyTorch's decomposition system <https://github.com/pytorch/pytorch/blob/main/torch/_decomp/decompositions.py>`_.

Custom Neuron Decompositions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These are defined in ``torch_neuronx/neuron_dynamo_backend/decompositions.py``:

- ``aten.as_strided.default``
- ``aten.topk.default``
- ``aten._scaled_dot_product_fused_attention_overrideable.default``
- ``aten.scalar_tensor.default``
- ``aten.index_copy.default``
- ``aten.nonzero_static.default``
- ``aten.linalg_vector_norm.default``
- ``aten.sigmoid_backward.default``
- ``aten.linear_backward``

Inherited from PyTorch
~~~~~~~~~~~~~~~~~~~~~~

The base decomposition table is built using ``torch._decomp.get_decompositions()``:

.. code-block:: python

    from torch._decomp import get_decompositions

    neuron_decompositions = get_decompositions([
        aten._unsafe_index,
        aten.split,
        aten.unbind,
        aten.embedding,
        # ... more ops
    ])

This includes decompositions for:

- ``aten.split``, ``aten.unbind``, ``aten.embedding``
- ``aten.native_layer_norm_backward``, ``aten.nll_loss_backward``
- ``aten.gelu_backward``, ``aten.softplus_backward``
- And more

**To add additional PyTorch decompositions**, add the op to the list in ``get_decompositions()`` within ``decompositions.py``:

.. code-block:: python

    neuron_decompositions = get_decompositions([
        # Existing ops...
        aten.my_new_op,  # Add new op here
    ])

See `PyTorch decompositions <https://github.com/pytorch/pytorch/blob/main/torch/_decomp/decompositions.py>`_ for available decompositions.

Some decompositions may be candidates for upstreaming to PyTorch if they are generally useful across backends. Backend-specific decompositions (e.g., those optimized for Neuron hardware) will remain in this repository.

Adding a New Decomposition
--------------------------

Location
~~~~~~~~

Decompositions live in ``torch_neuronx/neuron_dynamo_backend/decompositions.py``.

Using the register_decomposition Decorator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Simply annotate your decomposition function with ``@register_decomposition``:

.. code-block:: python

    from torch_neuronx.neuron_dynamo_backend.decompositions import register_decomposition

    @register_decomposition([aten.my_op.default])
    def my_op_decomposition(x, y):
        """Decompose my_op into simpler operations."""
        return x + y

The decorator:

- Registers the function as the decomposition for the specified op(s)
- Accepts a single op or list of ops
- Raises ``ValueError`` if the op is already registered

Examples
--------

Example 1: Simple Scalar Tensor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    @register_decomposition([aten.scalar_tensor.default])
    def scalar_tensor(
        value,
        dtype=None,
        layout=None,
        device=None,
        pin_memory=None,
    ):
        """Decompose scalar_tensor to full() for better StableHLO support."""
        return torch.full((), value, dtype=dtype, device=device)

Example 2: Index Copy
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    @register_decomposition([aten.index_copy.default])
    def index_copy(
        input: torch.Tensor, dim: int, index: torch.Tensor, source: torch.Tensor
    ) -> torch.Tensor:
        """Decompose index_copy into functional scatter."""
        if dim < 0:
            dim = input.dim() + dim

        target_shape = [1] * source.dim()
        target_shape[dim] = index.numel()
        index_expanded = index.view(target_shape).expand(source.shape)
        return torch.scatter(input, dim, index_expanded, source)

Example 3: Sigmoid Backward
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    @register_decomposition([aten.sigmoid_backward.default])
    def sigmoid_backward_decomposition(grad_output, output):
        """Decompose sigmoid_backward into primitive ops."""
        return grad_output * (output * (1 - output))

Best Practices
--------------

When writing decompositions, follow these guidelines to ensure correctness and compatibility:

.. note::

   When implementing op decompositions, you must strictly follow the pattern used in
   `PyTorch meta registration <https://github.com/pytorch/pytorch/blob/main/torch/_meta_registrations.py>`_.
   This ensures consistency with PyTorch's internal decomposition system and prevents unexpected behavior.

   If the op you want to implement doesn't have a meta registration, you must also create and register one.
   Use ``register_fake`` (also known as ``register_meta``) to define how the op behaves on meta tensors
   (computing output shapes/dtypes without actual data):

   .. code-block:: python

       from torch.library import register_fake

       @register_fake("mylib::my_op")
       def my_op_fake(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
           """Meta implementation: compute output shape/dtype without real computation."""
           # Return a tensor with the correct shape and dtype
           return x.new_empty(x.shape)

   For more details, see `PyTorch custom_ops.py <https://github.com/pytorch/pytorch/blob/67bde519483a412360ef6e800b1e6a125dbc043b/torch/_library/custom_ops.py#L402>`_.

Normalize Negative Dimensions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    if dim < 0:
        dim = input.dim() + dim

Use Input Device for Intermediates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mixing CPU and neuron tensors can result in errors under Dynamo.

.. code-block:: python

    device = input_tensor.device
    idx = torch.zeros(size, dtype=torch.long, device=device)

Return NotImplemented for Unsupported Cases
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Let PyTorch handle cases we don't need to decompose
    if dim == x.ndim - 1 and largest and sorted:
        return NotImplemented

Testing
-------

See ``tests/neuron_dynamo_backend/unit/test_decompositions.py`` for existing test patterns.
