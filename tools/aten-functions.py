#!/usr/bin/env python3
"""
Tool to analyze ATen functions from native_functions.yaml
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml


def load_native_functions(yaml_path):
    """Load and parse the native_functions.yaml file"""
    try:
        with open(yaml_path) as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Could not find native_functions.yaml at {yaml_path}", file=sys.stderr)
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}", file=sys.stderr)
        sys.exit(1)


def count_functions(functions):
    """Count the total number of ATen functions"""
    if not isinstance(functions, list):
        print("Error: Expected list of functions in YAML file", file=sys.stderr)
        sys.exit(1)

    count = 0
    for entry in functions:
        if isinstance(entry, dict) and "func" in entry:
            count += 1

    return count


# Define category patterns for classification
CATEGORY_PATTERNS = {
    "reduction": [
        r"^(sum|mean|max|min|all|any|norm|std|var|prod|amax|amin|argmax|argmin|median|mode|cumsum|cumprod|logsumexp|count_nonzero|unique)(_|$|\.|\.)",
        r"_(sum|mean|max|min|all|any|norm|std|var|prod)(_|$|\.|\.)",
        r"^(nansum|nanmean|nanmedian|nanquantile)(_|$|\.|\.)",
    ],
    "matmul": [
        r"^(mm|bmm|matmul|addmm|addmv|addr|baddbmm|dot|mv|ger|inner|outer|chain_matmul|multi_dot)(_|$|\.|\.)",
        r"_(mm|bmm|matmul)(_|$|\.|\.)",
        r"^(addbmm|spmm)(_|$|\.|\.)",
    ],
    "conv": [
        r"^(conv|convolution|conv_transpose|conv_tbc)(_|[0-9]d|$|\.|\.)",
        r"^(_convolution|_conv_depthwise2d|cudnn_convolution|mkldnn_convolution|miopen_convolution)",
    ],
    "pooling": [
        r"^(max_pool|avg_pool|adaptive_max_pool|adaptive_avg_pool|fractional_max_pool)(_|[0-9]d|$|\.|\.)",
        r"^(lp_pool|max_unpool)",
        r"^reflection_pad|replication_pad|constant_pad",
    ],
    "activation": [
        r"^(relu|sigmoid|tanh|softmax|log_softmax|gelu|silu|mish|hardswish|elu|selu|celu|leaky_relu|prelu|rrelu|hardshrink|hardsigmoid|hardtanh|softplus|softshrink|threshold|glu|logsigmoid|softmin|softsign|tanhshrink)(_|$|\.|\.)",
        r"^(swish|hinge_embedding_loss)",
    ],
    "normalization": [
        r"^(batch_norm|layer_norm|group_norm|instance_norm|local_response_norm|normalize|renorm|weight_norm)(_|$|\.|\.)",
        r"^(native_batch_norm|native_layer_norm|native_group_norm)",
        r"^(_batch_norm|_instance_norm)",
    ],
    "loss": [
        r"^(nll_loss|cross_entropy|binary_cross_entropy|mse_loss|l1_loss|smooth_l1_loss|huber_loss|cosine_embedding_loss|margin_ranking_loss|triplet_margin_loss|hinge_embedding_loss|multilabel_margin_loss|soft_margin_loss|multi_margin_loss|kl_div|poisson_nll_loss|ctc_loss)(_|$|\.|\.)",
        r"^(bce_with_logits|multilabel_soft_margin_loss)",
    ],
    "indexing": [
        r"^(index|gather|scatter|index_add|index_copy|index_fill|index_put|index_select|masked_fill|masked_scatter|masked_select|take|put|narrow|select|slice|split|chunk|unbind|nonzero|where|searchsorted|bucketize)(_|$|\.|\.)",
        r"^(_index|_gather|_scatter)",
    ],
    "creation": [
        r"^(zeros|ones|empty|rand|randn|randint|randperm|arange|linspace|logspace|eye|full|empty_like|zeros_like|ones_like|rand_like|randn_like|full_like|clone|tensor|as_tensor|from_numpy|scalar_tensor|asarray|sparse_coo_tensor)(_|$|\.|\.)",
        r"^(new_empty|new_full|new_ones|new_zeros)",
        r"^(_empty_affine_quantized|_empty_per_channel_affine_quantized)",
    ],
    "manipulation": [
        r"^(reshape|view|transpose|permute|squeeze|unsqueeze|flatten|unflatten|expand|repeat|tile|broadcast_to|roll|flip|fliplr|flipud|rot90|cat|stack|hstack|vstack|dstack|split|chunk|tensor_split|swapaxes|swapdims|movedim|moveaxis|narrow|unfold|contiguous|as_strided)(_|$|\.|\.)",
        r"^(t|T|H|mT|mH|adjoint)$",
        r"^(_reshape|_unsafe_view)",
        r"^(atleast_1d|atleast_2d|atleast_3d|column_stack|row_stack)(_|$|\.|\.)",
        r"^(concat|concatenate)(_|$|\.|\.)",
    ],
    "comparison": [
        r"^(eq|ne|lt|gt|le|ge|equal|not_equal|less|greater|less_equal|greater_equal|isclose|allclose|isfinite|isinf|isnan|isposinf|isneginf|signbit|isreal|iscomplex|is_same_size|is_nonzero|is_floating_point|is_complex|is_conj|is_neg)(_|$|\.|\.)",
        r"^(maximum|minimum|fmax|fmin|heaviside|logical_and|logical_or|logical_not|logical_xor)(_|$|\.|\.)",
    ],
    "arithmetic": [
        r"^(add|sub|mul|div|pow|remainder|fmod|floor_divide|true_divide|reciprocal|neg|abs|sign|sqrt|rsqrt|square|exp|exp2|expm1|log|log2|log10|log1p|sin|cos|tan|asin|acos|atan|atan2|sinh|cosh|tanh|asinh|acosh|atanh|erf|erfc|erfinv|erfcinv|lgamma|digamma|polygamma|ceil|floor|round|trunc|frac|lerp|clamp|clip|copysign|nextafter|ldexp|frexp|xlogy|hypot|i0|igamma|igammac|logit|sigmoid_backward)(_|$|\.|\.)",
        r"^(addcdiv|addcmul|multiply|divide|subtract|bitwise_and|bitwise_or|bitwise_xor|bitwise_not|bitwise_left_shift|bitwise_right_shift)(_|$|\.|\.)",
        r"^(__and__|__or__|__xor__|__lshift__|__rshift__|__iand__|__ior__|__ixor__|__ilshift__|__irshift__)$",
        r"^(arccos|arcsin|arctan|arccosh|arcsinh|arctanh)(_|$|\.|\.)",  # arc trig functions
        r"^(absolute|angle|conj|conj_physical|real|imag)(_|$|\.|\.)",  # complex number ops
        r"^(positive|negative)(_|$|\.|\.)",
        r"^(rad2deg|deg2rad)(_|$|\.|\.)",
    ],
    "linalg": [
        r"^(det|logdet|slogdet|trace|diagonal|tril|triu|tril_indices|triu_indices|diag|diag_embed|diagflat|cholesky|cholesky_solve|cholesky_inverse|lu|lu_solve|lu_unpack|qr|svd|svd_lowrank|pca_lowrank|eig|eigh|eigvals|eigvalsh|matrix_rank|pinverse|solve|lstsq|triangular_solve|ormqr|householder_product|orgqr|matrix_power|matrix_exp|norm|cond|vdot|cross|tensordot|cartesian_prod|cdist|pdist|symeig)(_|$|\.|\.)",
        r"^(linalg_)",
        r"^(_lu_with_info|_cholesky_helper|_triangular_solve_helper)",
        r"^_linalg_(det|slogdet|eigvals|eigh|svd)",  # Internal linalg functions
    ],
    "fft": [
        r"^(fft|ifft|rfft|irfft|hfft|ihfft|fftn|ifftn|rfftn|irfftn|fftfreq|rfftfreq|fftshift|ifftshift|stft|istft)(_|$|\.|\.)",
        r"^(_fft_c2c|_fft_r2c|_fft_c2r)",
    ],
    "sparse": [
        r"^(sparse_|_sparse_|to_sparse|to_dense|coalesce|is_coalesced|_coalesced|_indices|_values|_nnz|sparse_dim|dense_dim|sparse_mask|sparse_resize)",
        r"^(hspmm|sspaddmm|smm)",
    ],
    "quantization": [
        r"^(quantize_per_tensor|quantize_per_channel|dequantize|q_scale|q_zero_point|q_per_channel|fake_quantize|quantized_)",
        r"^(_quantize|_dequantize|_make_per_tensor_quantized_tensor|_make_per_channel_quantized_tensor)",
    ],
    "random": [
        r"^(bernoulli|multinomial|normal|poisson|uniform|cauchy|exponential|geometric|log_normal)(_|$|\.|\.)",
        r"^(random_|manual_seed|seed|get_rng_state|set_rng_state)",
    ],
    "special": [
        r"^(special_|bessel|beta|binom|erf|gamma|zeta|multigammaln|psi)",
    ],
    "distributed": [
        r"^(all_reduce|all_gather|reduce_scatter|broadcast|reduce|gather|scatter|barrier|send|recv|isend|irecv|alltoall|neighbor_alltoall|neighbor_alltoallv)(_|$|\.|\.)",
        r"^(_all_reduce|_all_gather|_reduce_scatter|_broadcast|_reduce|_gather|_scatter)",
    ],
    "nn_utils": [
        r"^(dropout|alpha_dropout|feature_alpha_dropout|dropout2d|dropout3d|pad|grid_sample|affine_grid|upsample|interpolate|pixel_shuffle|pixel_unshuffle|channel_shuffle|native_dropout|feature_dropout)(_|$|\.|\.)",
        r"^(rnn_tanh|rnn_relu|lstm|gru|rnn_tanh_cell|rnn_relu_cell|lstm_cell|gru_cell)",
        r"^(embedding|embedding_bag|one_hot|_embedding_bag)",
        r"^(bilinear|cudnn_)",
    ],
    "signal": [
        r"^(bartlett_window|blackman_window|hamming_window|hann_window|kaiser_window)(_|$|\.|\.)",
        r"^(stft|istft)(_|$|\.|\.)",
    ],
    "attention": [
        r"^(scaled_dot_product_attention|_scaled_dot_product)",
        r"^(_flash_attention|_efficient_attention|_cudnn_attention)",
        r"^(_triton_scaled_dot_attention|_triton_multi_head_attention)",
        r"^(_native_multi_head_attention|_transformer_encoder_layer_fwd)",
    ],
    "optimizer": [
        r"^(_fused_adam|_fused_adamw|_fused_sgd|_fused_adagrad)(_|$|\.|\.)",
    ],
    "autograd": [
        r"^(detach|detach_|grad|grad_|_backward|backward|requires_grad_|retain_grad|retains_grad)(_|$|\.|\.)",
        r"^(_fw_primal|_make_dual|_unpack_dual|is_leaf|output_nr)",
    ],
    "utility": [
        r"^(data|alias|copy|copy_|clone)(_|$|\.|\.)",
        r"^(set_data|rename|rename_|align_to|align_as|align_tensors)",
        r"^(lift_fresh|lift_fresh_copy)",
        r"^(_version|_assert|_print)",
    ],
    "stats": [
        r"^(histc|histogram|bincount|cov|corrcoef|aminmax|cummax|cummin)(_|$|\.|\.)",
        r"^(argsort|msort|sort|kthvalue|topk)(_|$|\.|\.)",
    ],
    "foreach": [
        r"^_foreach_",  # All foreach operations
    ],
    "backward": [
        r"_backward(_|$)",  # Backward operations
        r"backward$",
    ],
    "interop": [
        r"^(_?cudnn_|_?mkldnn_|_?miopen_|_?mps_|_?nnpack_)",  # Framework-specific ops
        r"^(cudnn_|mkldnn_|miopen_|mps_convolution)",
    ],
    "testing": [
        r"^_test_",  # Test functions
    ],
    "casting": [
        r"^_cast_",  # Type casting operations
        r"^(to|_to_copy|_to_cpu|_to_dense|_to_sparse)",
        r"^chalf$",  # Complex half
    ],
    "nested": [
        r"^_nested_",  # Nested tensor operations
        r"^nested_to_padded_tensor$",
    ],
    "upsampling": [
        r"^_upsample_",  # Upsampling operations
        r"^(upsample|interpolate)(_|$|\.|\.)",
    ],
    "validation": [
        r"^_validate_",  # Validation functions
    ],
    "packing": [
        r"pack|unpack",  # Packing/unpacking operations
        r"^_convert_weight_to_int4pack",
    ],
    "adaptive": [
        r"adaptive_[a-z]+_pool",  # Adaptive pooling
    ],
    "memory": [
        r"^(pin_memory|_pin_memory|record_stream|_resize_output_|resize_|resize_as_)",
        r"^(_coalesce|coalesce|_copy_from)",
    ],
    "distance": [
        r"^(dist|cdist|pdist|cosine_similarity|pairwise_distance)(_|$|\.|\.)",
        r"^(_cdist_|_pdist_|_euclidean_dist)",
    ],
    "histogram": [
        r"histogram",  # Histogram operations
    ],
    "solver": [
        r"^(solve|cholesky|lu_solve|triangular_solve|lstsq|inverse|pinverse)(_|$|\.|\.)",
        r"^(_cholesky_solve_helper|_linalg_solve_ex|geqrf)",
    ],
    "grid": [
        r"grid_sampler",  # Grid sampling
    ],
    "complex": [
        r"^(polar|complex|conj|_conj)",
        r"^(real|imag|angle)(_|$|\.|\.)",
    ],
    "functional": [
        r"^_functional_",  # Functional operations
    ],
    "shape": [
        r"^(split|chunk|stack|column_stack|row_stack|hstack|vstack|dstack|concat|concatenate)(_|$|\.|\.)",
        r"^(_chunk_cat|_stack|unsafe_split|unsafe_chunk)",
        r"^(dsplit|hsplit|vsplit)(_|$|\.|\.)",
    ],
    "window": [
        r"_window$",  # Window functions
    ],
    "internal": [
        r"^_(amp_|addmm_activation|rowwise_prune|lazy_clone|new_zeros_with_same_feature_meta)",
        r"^_(safe_softmax|propagate_xla_data|mixed_dtypes_linear)",
        r"^_(efficientzerotensor|local_scalar_dense|add_batch_dim|remove_batch_dim)",
        r"^_(compute_linear_combination|trilinear|spdiags)",
    ],
    "indices": [
        r"(indices|index)(_|$|\.|\.)",  # Index-related operations
        r"^(ccol_indices|col_indices|crow_indices|row_indices)",
        r"^_unsafe_(index|masked_index)",
    ],
    "fbgemm": [
        r"^fbgemm_",
    ],
    "slow": [
        r"^slow_conv",  # Slow/reference implementations
        r"^_slow_conv",
    ],
    "symbolic": [
        r"^sym_",  # Symbolic operations
        r"^_dim_arange$",
        r"^_dimI$|^_dimV$",
    ],
    "jagged": [
        r"jagged|padded",  # Jagged/padded tensor operations
    ],
    "type_info": [
        r"^(is_distributed|is_inference|is_pinned|is_set_to|is_signed|is_vulkan_available)$",
        r"^(can_cast|promote_types|result_type|type_as|qscheme|int_repr)$",
    ],
    "math_misc": [
        r"^(gcd|lcm|diff|gradient|mvlgamma|trapezoid|trapz|cumulative_trapezoid)(_|$|\.|\.)",
        r"^(logaddexp|logaddexp2|float_power)(_|$|\.|\.)",
        r"^(fix|sgn|sinc|nan_to_num)(_|$|\.|\.)",
    ],
    "rnn": [
        r"(gru_cell|lstm_cell|lstm_mps)",  # RNN cell operations
    ],
    "fused": [
        r"^_fused_",  # Fused operations
        r"^fused_moving_avg_obs_fake_quant$",
    ],
    "view": [
        r"_view(_|$)",  # View operations
    ],
    "softmax_internal": [
        r"^_(log_)?softmax$",  # Internal softmax
        r"^_masked_softmax$",
    ],
    "ctc": [
        r"ctc_loss",  # CTC loss operations
    ],
    "tensor_info": [
        r"^(size|stride|item|numel)$",  # Basic tensor info
        r"^(t_|t_copy|matrix_H|numpy_T|ravel)$",  # Transpose/reshape aliases
    ],
    "misc": [
        # Misc array operations
        r"^(argwhere|block_diag|einsum|meshgrid|vander|combinations|kron)$",
        # Fill/set operations
        r"^(fill|fill_|fill_diagonal_|zero_|range|lift|refine_names|set_)(_|$|\.|\.)",
        r"^(rsub|segment_reduce|im2col|col2im|native_channel_shuffle)$",  # Other operations
        r"^(relu6|relu6_|resolve_conj|resolve_neg)$",  # Specific activations/resolvers
        r"^(values|values_copy|item|from_file)$",  # Value operations
        r"^(choose_qparams_optimized|linear|linear_backward)$",  # Linear/quant ops
        r"^(nll_loss2d|quantile|isin)(_|$|\.|\.)",  # Specific loss/stats
        r"^_(add_relu|aminmax|autocast_|cufft_|has_|is_zerotensor|unique|prelu_kernel|shape_as_tensor)",
        r"^_(saturate_weight_to_fp16|transform_bias_rescale_qkv|mask_|pad_|use_cudnn_)",
        r"^_(linalg_check_errors|spsolve|sample_dirichlet|standard_gamma|sobol_engine_)",
        r"^_(logcumsumexp|make_dep_token|thnn_|foobar)$",
        r"^(arctan2|arctan2_)$",  # Arctan2 (not caught by arithmetic patterns)
        r"^(thnn_conv2d)$",  # THNN convolution
        r"^(logcumsumexp)(_|$|\.|\.)",  # Log cumulative sum exp
        r"^(_debug_has_internal_overlap|_masked_scale|_cummax_helper|_cummin_helper)$",
        r"^(_cslt_compress|_dirichlet_grad|_fill_mem_eff_dropout_mask_)$",
        r"^(_fake_quantize_per_tensor_affine_cachemask_tensor_qparams)$",
        r"^(_fake_quantize_learnable_per_tensor_affine|_fake_quantize_learnable_per_channel_affine)$",
        r"^(_choose_qparams_per_tensor)$",
    ],
}


def get_implemented_ops():
    """Dynamically get the set of operations implemented in torch-neuronx"""
    try:
        import torch_neuronx  # noqa: F401
        from torch_neuronx.utils import get_neuron_registered_ops

        return get_neuron_registered_ops()
    except ImportError:
        print("Warning: Could not import torch_neuronx to get implemented ops", file=sys.stderr)
        return set()
    except Exception as e:
        print(f"Warning: Error getting implemented ops: {e}", file=sys.stderr)
        return set()


def extract_function_name(func_signature):
    """Extract the function name from the function signature"""
    match = re.match(r"^([a-zA-Z0-9_]+)", func_signature)
    if match:
        return match.group(1)
    return None


def classify_function(func_name):
    """Classify a function based on name patterns"""
    categories = []

    for category, patterns in CATEGORY_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, func_name):
                categories.append(category)
                break  # Only match once per category

    return categories


def filter_by_category(functions, category):
    """Filter functions by category"""
    filtered = []

    for func in functions:
        if "func" not in func:
            continue

        func_name = extract_function_name(func["func"])
        if not func_name:
            continue

        categories = classify_function(func_name)
        if category in categories:
            filtered.append(func)

    return filtered


def get_unclassified_functions(functions):
    """Get functions that don't match any category patterns"""
    unclassified = []

    for func in functions:
        if "func" not in func:
            continue

        func_name = extract_function_name(func["func"])
        if not func_name:
            continue

        categories = classify_function(func_name)
        if not categories:
            unclassified.append(func)

    return unclassified


def analyze_categories(functions):
    """Analyze category distribution of functions"""
    category_counts = defaultdict(int)
    unclassified_count = 0

    for func in functions:
        if "func" not in func:
            continue

        func_name = extract_function_name(func["func"])
        if not func_name:
            continue

        categories = classify_function(func_name)
        if categories:
            for cat in categories:
                category_counts[cat] += 1
        else:
            unclassified_count += 1

    return category_counts, unclassified_count


def get_category_functions_map(functions):
    """Get mapping of categories to functions with core op and implementation info"""
    category_map = defaultdict(list)
    core_ops = set()
    implemented_ops = get_implemented_ops()

    # First pass: identify core ops
    for func in functions:
        if "func" not in func:
            continue

        func_name = extract_function_name(func["func"])
        if not func_name:
            continue

        # Check if this function is tagged as core
        if "tags" in func:
            tags = func["tags"]
            if (isinstance(tags, list) and "core" in tags) or tags == "core":
                core_ops.add(func_name)

    # Second pass: categorize functions
    # Use a set to track unique functions per category to avoid duplicates
    category_funcs = defaultdict(set)

    for func in functions:
        if "func" not in func:
            continue

        func_name = extract_function_name(func["func"])
        if not func_name:
            continue

        categories = classify_function(func_name)

        if categories:
            for cat in categories:
                category_funcs[cat].add(func_name)
        else:
            category_funcs["unclassified"].add(func_name)

    # Convert sets to sorted lists with core op and implementation information
    for cat, func_names in category_funcs.items():
        func_list = []
        for func_name in sorted(func_names):
            func_list.append(
                {
                    "name": func_name,
                    "is_core": func_name in core_ops,
                    "is_implemented": func_name in implemented_ops,
                }
            )
        category_map[cat] = func_list

    return category_map


def main():
    parser = argparse.ArgumentParser(
        description="Analyze ATen functions from native_functions.yaml",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  aten-functions --yaml-path /path/to/native_functions.yaml --count
  aten-functions -y ~/pytorch/aten/src/ATen/native/native_functions.yaml --count
        """,
    )

    parser.add_argument(
        "--yaml-path", "-y", type=str, required=True, help="Path to native_functions.yaml file"
    )
    parser.add_argument(
        "--count", action="store_true", help="Count the total number of ATen functions"
    )
    parser.add_argument(
        "--category",
        type=str,
        choices=list(CATEGORY_PATTERNS.keys()),
        help="Filter functions by category",
    )
    parser.add_argument(
        "--unclassified",
        action="store_true",
        help="Show functions that don't match any category patterns",
    )
    parser.add_argument(
        "--analyze-categories",
        action="store_true",
        help="Analyze category distribution of all functions",
    )
    parser.add_argument(
        "--markdown-table",
        action="store_true",
        help="Generate a markdown table with categories and their functions",
    )
    parser.add_argument(
        "--html-table",
        action="store_true",
        help="Generate an HTML table with categories and their functions (core ops in green)",
    )

    args = parser.parse_args()

    # Check if YAML path exists
    yaml_path = Path(args.yaml_path)
    if not yaml_path.exists():
        print(f"Error: YAML file not found at {yaml_path}", file=sys.stderr)
        sys.exit(1)

    # If no action specified, show help
    if not any(
        [
            args.count,
            args.category,
            args.unclassified,
            args.analyze_categories,
            args.markdown_table,
            args.html_table,
        ]
    ):
        parser.print_help()
        sys.exit(0)

    # Load the YAML file
    functions = load_native_functions(yaml_path)

    # Handle different command options
    if args.count:
        count = count_functions(functions)
        print(count)

    elif args.category:
        filtered = filter_by_category(functions, args.category)
        print(f"Found {len(filtered)} functions in category '{args.category}':")
        for func in filtered:
            func_name = extract_function_name(func["func"])
            print(f"  {func_name}")

    elif args.unclassified:
        unclassified = get_unclassified_functions(functions)
        print(f"Found {len(unclassified)} unclassified functions:")
        for func in unclassified:
            func_name = extract_function_name(func["func"])
            print(f"  {func_name}")

    elif args.analyze_categories:
        category_counts, unclassified_count = analyze_categories(functions)
        total = sum(category_counts.values()) + unclassified_count

        print(f"Category distribution of {total} functions:")
        print(f"{'Category':<20} {'Count':<10} {'Percentage'}")
        print("-" * 45)

        # Sort categories by count
        for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total) * 100
            print(f"{category:<20} {count:<10} {percentage:.1f}%")

        if unclassified_count > 0:
            percentage = (unclassified_count / total) * 100
            print(f"{'[unclassified]':<20} {unclassified_count:<10} {percentage:.1f}%")

    elif args.markdown_table:
        category_map = get_category_functions_map(functions)

        print("| Category | Functions |")
        print("|----------|-----------|")

        # Sort categories alphabetically
        for category in sorted(category_map.keys()):
            functions_list = category_map[category]
            # Extract just the function names for markdown (no core highlighting)
            func_names = [f["name"] for f in functions_list]
            functions_str = ", ".join(func_names)
            print(f"| {category} | {functions_str} |")

    elif args.html_table:
        category_map = get_category_functions_map(functions)

        # Print CSS styles for a beautiful table
        print("""<!DOCTYPE html>
<html>
<head>
<style>
table {
    border-collapse: collapse;
    width: 100%;
    font-family: Arial, sans-serif;
    margin: 20px 0;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}

th {
    background-color: #2c3e50;
    color: white;
    padding: 12px 15px;
    text-align: left;
    font-weight: bold;
    position: sticky;
    top: 0;
    z-index: 10;
}

td {
    padding: 12px 15px;
    border-bottom: 1px solid #ddd;
}

tr:nth-child(even) {
    background-color: #f8f9fa;
}

tr:hover {
    background-color: #e8f4f8;
}

td:first-child {
    font-weight: bold;
    background-color: #ecf0f1;
    width: 200px;
    vertical-align: top;
}

.core-op {
    color: #27ae60;
    font-weight: bold;
    background-color: #e8f5e9;
    padding: 2px 6px;
    border-radius: 3px;
    margin: 2px;
    display: inline-block;
}

.regular-op {
    color: #2c3e50;
    background-color: #ecf0f1;
    padding: 2px 6px;
    border-radius: 3px;
    margin: 2px;
    display: inline-block;
}

.implemented {
    text-decoration: line-through;
    opacity: 0.7;
}

.functions-cell {
    line-height: 1.8;
}

.legend {
    margin: 20px;
    padding: 15px;
    background-color: #f8f9fa;
    border: 1px solid #ddd;
    border-radius: 5px;
}

.legend-item {
    display: inline-block;
    margin-right: 20px;
}
</style>
</head>
<body>
<table>
<thead>
<tr>
<th>Category</th>
<th>Functions</th>
</tr>
</thead>
<tbody>""")

        # Sort categories alphabetically
        for category in sorted(category_map.keys()):
            functions_list = category_map[category]
            # Format functions with core ops highlighted and implemented ops struck through
            formatted_funcs = []
            for func in functions_list:
                classes = []
                if func["is_core"]:
                    classes.append("core-op")
                else:
                    classes.append("regular-op")

                if func["is_implemented"]:
                    classes.append("implemented")

                class_str = " ".join(classes)
                formatted_funcs.append(f'<span class="{class_str}">{func["name"]}</span>')

            functions_str = " ".join(
                formatted_funcs
            )  # Use space instead of comma for better visual separation
            print(f'<tr><td>{category}</td><td class="functions-cell">{functions_str}</td></tr>')

        print("""</tbody>
</table>
<div class="legend">
    <strong>Legend:</strong>
    <span class="legend-item"><span class="core-op">Core Op</span> = Core PyTorch operation</span>
    <span class="legend-item"><span class="regular-op">Regular Op</span> = Non-core operation</span>
    <span class="legend-item"><span class="regular-op implemented">Implemented</span> =
        Already implemented in torch-neuronx</span>
</div>
</body>
</html>""")


if __name__ == "__main__":
    main()
