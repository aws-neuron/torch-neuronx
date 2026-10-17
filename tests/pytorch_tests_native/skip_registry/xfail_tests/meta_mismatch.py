META_MISMATCH_OUTPUT = [
    "test_meta.py::TestMetaNEURON::test_dispatch_meta_outplace__foreach_sqrt_neuron_float64",
    "test_meta.py::TestMetaNEURON::test_dispatch_meta_outplace_nn_functional_embedding_bag_neuron_bfloat16",
    "test_meta.py::TestMetaNEURON::test_dispatch_meta_outplace_nn_functional_embedding_bag_neuron_float16",
    "test_meta.py::TestMetaNEURON::test_dispatch_meta_outplace_nn_functional_embedding_bag_neuron_float32",
    "test_meta.py::TestMetaNEURON::test_dispatch_meta_outplace_nn_functional_embedding_bag_neuron_float64",
    "test_meta.py::TestMetaNEURON::test_dispatch_symbolic_meta_outplace__foreach_sqrt_neuron_float64",
    "test_meta.py::TestMetaNEURON::test_dispatch_symbolic_meta_outplace_nn_functional_embedding_bag_neuron_bfloat16",
    "test_meta.py::TestMetaNEURON::test_dispatch_symbolic_meta_outplace_nn_functional_embedding_bag_neuron_float16",
    "test_meta.py::TestMetaNEURON::test_dispatch_symbolic_meta_outplace_nn_functional_embedding_bag_neuron_float32",
    "test_meta.py::TestMetaNEURON::test_dispatch_symbolic_meta_outplace_nn_functional_embedding_bag_neuron_float64",
    "test_meta.py::TestMetaNEURON::test_dispatch_symbolic_meta_outplace_quantile_neuron_float64",
    "test_meta.py::TestMetaNEURON::test_meta_outplace__foreach_sqrt_neuron_float64",
    # _foreach_abs(bool): stablehlo.abs does not support bool dtype; the
    # compiler error surfaces as a meta kernel output mismatch.
    "test_meta.py::TestMetaNEURON::test_dispatch_meta_inplace__foreach_abs_neuron_bool",
    "test_meta.py::TestMetaNEURON::test_dispatch_meta_outplace__foreach_abs_neuron_bool",
    "test_meta.py::TestMetaNEURON::test_dispatch_symbolic_meta_inplace__foreach_abs_neuron_bool",
    "test_meta.py::TestMetaNEURON::test_dispatch_symbolic_meta_outplace__foreach_abs_neuron_bool",
    "test_meta.py::TestMetaNEURON::test_meta_inplace__foreach_abs_neuron_bool",
    "test_meta.py::TestMetaNEURON::test_meta_outplace__foreach_abs_neuron_bool",
    # _foreach_norm: meta kernel predicts wrong output dtype for these dtypes.
    "test_meta.py::TestMetaNEURON::test_dispatch_meta_outplace__foreach_norm_neuron_bool",
    "test_meta.py::TestMetaNEURON::test_dispatch_symbolic_meta_outplace__foreach_norm_neuron_bool",
    "test_meta.py::TestMetaNEURON::test_meta_outplace__foreach_norm_neuron_bool",
]
