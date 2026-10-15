"""Shared block-extraction math for aten::im2col / aten::col2im (F.unfold/fold).

One source for both the eager and torch.compile paths, so they cannot drift.
Leaf module: imports only torch.

The index map is built from arange so it survives AOTAutograd as a constant.
col2im accumulates on axis 0, never the minor axis (an index_add into the last
dimension is far slower on Neuron). im2col takes a reshape+permute fast path when
kernel == stride with unit dilation and the blocks tile the input exactly.
"""

import torch

aten = torch.ops.aten


def _pair(value):
    return int(value[0]), int(value[1])


def _out_size(size, kernel, dilation, padding, stride):
    return (size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1


def _block_flat_indices(kh, kw, ho, wo, dh, dw, sh, sw, w_pad, device):
    """Flattened source index of every block element, shape ``(kh*kw, ho*wo)``.

    ``flat[t, l]`` is the position, in the flattened ``(H_pad*W_pad)`` image, of
    tap ``t = ki*kw + kj`` for output block ``l = oi*wo + oj``.
    """
    ki = torch.arange(kh, device=device).reshape(kh, 1)
    kj = torch.arange(kw, device=device).reshape(kw, 1)
    oi = torch.arange(ho, device=device).reshape(1, ho)
    oj = torch.arange(wo, device=device).reshape(1, wo)
    src_h = ki * dh + oi * sh  # (kh, ho)
    src_w = kj * dw + oj * sw  # (kw, wo)
    flat = src_h.reshape(kh, 1, ho, 1) * w_pad + src_w.reshape(1, kw, 1, wo)
    return flat.reshape(kh * kw, ho * wo).to(torch.int64)


def im2col_forward(x, kernel_size, dilation, padding, stride):
    """aten::im2col: extract sliding blocks into columns."""
    kh, kw = _pair(kernel_size)
    dh, dw = _pair(dilation)
    ph, pw = _pair(padding)
    sh, sw = _pair(stride)
    batched = x.ndim == 4
    inp = x if batched else x.unsqueeze(0)
    n, c, h, w = inp.shape
    ho = _out_size(h, kh, dh, ph, sh)
    wo = _out_size(w, kw, dw, pw, sw)
    padded = torch.nn.functional.pad(inp, (pw, pw, ph, ph)) if (ph or pw) else inp
    h_pad, w_pad = int(padded.shape[-2]), int(padded.shape[-1])

    # Non-overlapping tiling -> a view + permute, no gather. Every dimension of
    # the reshape divides exactly, so this is a metadata-only rearrange.
    non_overlap = (
        sh == kh and sw == kw and dh == 1 and dw == 1 and ho * kh == h_pad and wo * kw == w_pad
    )
    if non_overlap:
        blocks = padded.reshape(n, c, ho, kh, wo, kw).permute(0, 1, 3, 5, 2, 4)
        out = blocks.reshape(n, c * kh * kw, ho * wo)
    else:
        flat = _block_flat_indices(kh, kw, ho, wo, dh, dw, sh, sw, w_pad, padded.device)
        x_flat = padded.reshape(n, c, h_pad * w_pad)
        idx = flat.reshape(1, 1, kh * kw, ho * wo).expand(n, c, kh * kw, ho * wo)
        gathered = torch.gather(x_flat.unsqueeze(2).expand(n, c, kh * kw, h_pad * w_pad), 3, idx)
        out = gathered.reshape(n, c * kh * kw, ho * wo)
    return out if batched else out.squeeze(0)


def col2im_forward(cols, output_size, kernel_size, dilation, padding, stride):
    """aten::col2im: scatter columns back to an image, summing overlaps."""
    h, w = _pair(output_size)
    kh, kw = _pair(kernel_size)
    dh, dw = _pair(dilation)
    ph, pw = _pair(padding)
    sh, sw = _pair(stride)
    batched = cols.ndim == 3
    inp = cols if batched else cols.unsqueeze(0)
    n = inp.shape[0]
    c = inp.shape[1] // (kh * kw)
    ho = _out_size(h, kh, dh, ph, sh)
    wo = _out_size(w, kw, dw, pw, sw)
    h_pad, w_pad = h + 2 * ph, w + 2 * pw
    flat = _block_flat_indices(kh, kw, ho, wo, dh, dw, sh, sw, w_pad, inp.device)

    # Accumulate on axis 0, never the minor axis: index_add into the last dim is
    # far slower on Neuron. Source laid out (K, N*C) for one index_add.
    src = inp.reshape(n, c, kh * kw, ho * wo).permute(2, 3, 0, 1).reshape(kh * kw * ho * wo, n * c)
    zeros = torch.zeros(h_pad * w_pad, n * c, dtype=inp.dtype, device=inp.device)
    accum = zeros.index_add(0, flat.reshape(kh * kw * ho * wo), src)
    out = accum.reshape(h_pad, w_pad, n, c).permute(2, 3, 0, 1)
    if ph or pw:
        out = out[:, :, ph : ph + h, pw : pw + w]
    return out if batched else out.squeeze(0)
