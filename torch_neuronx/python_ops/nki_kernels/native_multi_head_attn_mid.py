"""NKI kernel for native multi-head attention middle operation (self-attention computation)"""

from .scaled_dot_product_attention import scaled_dot_product_attention_kernel


def native_multi_head_attn_mid_kernel_with_grid(
    q, k, v, attn_output, use_causal_mask=False, dropout_p=0.0, training=False
):
    """
    This version is closer to the mha.py implementation but wrapped properly
    for integration as an aten operation.
    """

    scaled_dot_product_attention_kernel(
        q,
        k,
        v,
        attn_output,
        is_causal=use_causal_mask,
        dropout_p=dropout_p,
        scale=None,
        is_gqa=False,
        training=training,
        seed=None,
        lse=None,
        should_transpose_v=True,
    )
