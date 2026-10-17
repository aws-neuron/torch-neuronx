# Torch API Compatibility

`neuron_compatible` values:
- `Y`: supported on Neuron.
- `CPU_FALLBACK`: works, but the operation runs on the CPU and results are copied back to the Neuron device (slower).
- `N`: not supported on Neuron.

Terms used in `limitations`:
- Native: runs on the Neuron device.
- Host-only (no device ops): runs in Python on the host CPU and does no tensor computation on the device (for example, configuration, metadata, hooks). Works with Neuron tensors.
- Falls back (`aten::op`): the named PyTorch operator runs on the CPU.
- Round-trip via CPU: the input is copied to the CPU and back before the operation.
- Needs a process group: requires `torch.distributed.init_process_group` first.

## torch APIs

||pytorch_api|neuron_compatible|limitations|
|---|---|---|---|
|1|torch.default_generator|Y|Host-only (no device ops). This is the default CPU generator|
|2|torch.SymInt|Y|Host-only (no device ops). Represents a symbolic integer|
|3|torch.SymFloat|Y|Host-only (no device ops). Represents a symbolic floating-point value|
|4|torch.SymBool|Y|Host-only (no device ops). Represents a symbolic Boolean value|
|5|torch.Tag|Y|Host-only (no device ops). Provides operator-tag metadata|
|6|torch.is_tensor|Y|Host-only (no device ops). Checks whether an object is a tensor|
|7|torch.is_storage|Y|Host-only (no device ops). Checks whether an object is a storage object|
|8|torch.is_complex|Y|Host-only (no device ops). Reads tensor dtype metadata|
|9|torch.is_conj|Y|Host-only (no device ops). Reads the tensor conjugate metadata bit|
|10|torch.is_floating_point|Y|Host-only (no device ops). Reads tensor dtype metadata|
|11|torch.is_nonzero|Y|Native. Supports one-element tensors and returns a host Boolean|
|12|torch.set_default_dtype|Y|Host-only (no device ops). Sets the global default floating-point dtype|
|13|torch.get_default_dtype|Y|Host-only (no device ops). Returns the global default floating-point dtype|
|14|torch.set_default_device|Y|Host-only (no device ops). Sets the default device for subsequent allocations|
|15|torch.set_default_tensor_type|Y|Host-only (no device ops). Sets the process-wide default tensor type|
|16|torch.numel|Y|Host-only (no device ops). Returns the product of tensor shape metadata|
|17|torch.set_printoptions|Y|Host-only (no device ops). Sets tensor formatting options|
|18|torch.set_flush_denormal|Y|Host-only (no device ops)|
|19|torch.tensor|Y|Native. Host data is copied to the Neuron device|
|20|torch.sparse_coo_tensor|N|Not supported (`aten::_sparse_coo_tensor_with_dims_and_tensors`)|
|21|torch.sparse_csr_tensor|N|Sparse CSR tensor contents cannot be accessed on Neuron|
|22|torch.sparse_csc_tensor|N|Sparse CSC tensor contents cannot be accessed on Neuron|
|23|torch.sparse_bsr_tensor|N|Sparse BSR tensor contents cannot be accessed on Neuron|
|24|torch.sparse_bsc_tensor|N|Sparse BSC tensor contents cannot be accessed on Neuron|
|25|torch.asarray|Y|Native. Supports host data and existing Neuron tensors|
|26|torch.as_tensor|Y|Native. Supports host data and existing Neuron tensors|
|27|torch.as_strided|Y|Native|
|28|torch.from_numpy|Y|Host-only (no device ops)|
|29|torch.from_dlpack|N|DLPack import does not support Neuron tensors; import on CPU then move|
|30|torch.frombuffer|Y|Host-only (no device ops)|
|31|torch.zeros|Y|Native|
|32|torch.zeros_like|Y|Native|
|33|torch.ones|Y|Native|
|34|torch.ones_like|Y|Native|
|35|torch.arange|Y|Native|
|36|torch.range|CPU_FALLBACK|Falls back (`aten::range.out`). A `torch.compile` decomposition is registered|
|37|torch.linspace|Y|Native; `bfloat16` can differ from CPU|
|38|torch.logspace|CPU_FALLBACK|Falls back (`aten::logspace.out`). A `torch.compile` decomposition is registered|
|39|torch.eye|Y|Native|
|40|torch.empty|Y|Native|
|41|torch.empty_like|Y|Native|
|42|torch.empty_strided|Y|Native|
|43|torch.full|Y|Native|
|44|torch.full_like|Y|Native|
|45|torch.quantize_per_tensor|N|Quantized tensors not supported on Neuron (`aten::empty_quantized`)|
|46|torch.quantize_per_channel|N|Quantized tensors not supported on Neuron (`aten::empty_quantized`)|
|47|torch.dequantize|N|Quantized tensors not supported on Neuron (`aten::empty_quantized`)|
|48|torch.complex|CPU_FALLBACK|Falls back (`aten::complex.out`)|
|49|torch.polar|CPU_FALLBACK|Falls back (`aten::polar.out`)|
|50|torch.heaviside|CPU_FALLBACK|Falls back (`aten::heaviside.out`)|
|51|torch.adjoint|Y|Native|
|52|torch.argwhere|Y|Native. Not covered by `torch.compile` tests|
|53|torch.cat|Y|Native|
|54|torch.concat|Y|Native|
|55|torch.concatenate|Y|Native|
|56|torch.conj|Y|Native|
|57|torch.chunk|Y|Native. Returns tensor views|
|58|torch.dsplit|Y|Native. Returns tensor views|
|59|torch.column_stack|Y|Native|
|60|torch.dstack|Y|Native|
|61|torch.gather|Y|Native|
|62|torch.hsplit|Y|Native. Returns tensor views|
|63|torch.hstack|Y|Native|
|64|torch.index_add|Y|Native|
|65|torch.index_copy|Y|Native|
|66|torch.index_reduce|CPU_FALLBACK|Falls back (`aten::index_reduce.out`)|
|67|torch.index_select|Y|Native|
|68|torch.masked_select|Y|Native|
|69|torch.movedim|Y|Native. Returns a metadata view|
|70|torch.moveaxis|Y|Native. Returns a metadata view|
|71|torch.narrow|Y|Native. Returns a tensor view|
|72|torch.narrow_copy|Y|Native. Not covered by `torch.compile` tests|
|73|torch.nonzero|Y|Native. Not covered by `torch.compile` tests|
|74|torch.permute|Y|Native. Returns a metadata view|
|75|torch.reshape|Y|Native. Noncontiguous inputs may require a native copy|
|76|torch.row_stack|Y|Native|
|77|torch.select|Y|Native. Returns a metadata view|
|78|torch.scatter|Y|Native|
|79|torch.diagonal_scatter|Y|Native. `int64` inputs fall back (`aten::copy_`)|
|80|torch.select_scatter|Y|Native. Non-contiguous `int64` inputs can fall back (`aten::copy_`)|
|81|torch.slice_scatter|Y|Native|
|82|torch.scatter_add|Y|Native; collision-heavy cases can differ from CPU|
|83|torch.scatter_reduce|CPU_FALLBACK|Falls back (`aten::scatter_reduce.two_out`)|
|84|torch.split|Y|Native. Returns tensor views|
|85|torch.squeeze|Y|Native. Returns a metadata view|
|86|torch.stack|Y|Native|
|87|torch.swapaxes|Y|Native. Not covered by `torch.compile` tests|
|88|torch.swapdims|Y|Native. Not covered by `torch.compile` tests|
|89|torch.t|Y|Native. Returns a metadata view for tensors with at most two dimensions|
|90|torch.take|CPU_FALLBACK|Falls back (`aten::take`)|
|91|torch.take_along_dim|Y|Native. Not covered by `torch.compile` tests|
|92|torch.tensor_split|Y|Native. Returns tensor views|
|93|torch.tile|Y|Native|
|94|torch.transpose|Y|Native. Not covered by `torch.compile` tests|
|95|torch.unbind|Y|Native. Returns tensor views|
|96|torch.unsqueeze|Y|Native. Returns a metadata view|
|97|torch.vsplit|Y|Native. Returns tensor views|
|98|torch.vstack|Y|Native|
|99|torch.where|Y|Native|
|100|torch.Generator|Y|Host-only (no device ops). Creates a random-number generator for Neuron|
|101|torch.Generator.device|Y|Host-only (no device ops). Returns the device associated with the generator|
|102|torch.Generator.get_state|Y|Host-only; returns serialized Neuron generator state as a CPU `uint8` tensor|
|103|torch.Generator.initial_seed|Y|Host-only (no device ops). Returns the initial seed of a Neuron generator|
|104|torch.Generator.manual_seed|Y|Host-only (no device ops). Seeds a Neuron generator and returns that generator|
|105|torch.Generator.seed|Y|Host-only (no device ops)|
|106|torch.Generator.set_state|Y|Host-only (no device ops); restores Neuron generator from CPU `uint8` state|
|107|torch.seed|Y|Host-only (no device ops); does not seed Neuron generators|
|108|torch.manual_seed|Y|Host-only (no device ops)|
|109|torch.initial_seed|Y|Host-only (no device ops). Returns the initial seed of the default CPU generator|
|110|torch.get_rng_state|Y|Host-only (no device ops); does not read Neuron generator state|
|111|torch.set_rng_state|Y|Host-only (no device ops); sets CPU RNG state, not Neuron RNG state|
|112|torch.bernoulli|CPU_FALLBACK|Falls back (`aten::bernoulli_.Tensor`) for `float32` and `bfloat16`; `generator` must be `None`|
|113|torch.multinomial|CPU_FALLBACK|Falls back (`aten::multinomial`) for `float32` and `bfloat16`|
|114|torch.normal|CPU_FALLBACK|Falls back (`aten::normal.Tensor_Tensor`, `aten::normal.float_Tensor`, …); scalar native|
|115|torch.poisson|CPU_FALLBACK|Falls back (`aten::poisson`) for `float32`|
|116|torch.rand|Y|Native. A custom Neuron generator is unsupported|
|117|torch.rand_like|Y|Native|
|118|torch.randint|CPU_FALLBACK|`int64` falls back (`aten::random_.from`); custom Neuron generator unsupported|
|119|torch.randint_like|Y|Native. `int64` output falls back (`aten::random_.from`)|
|120|torch.randn|Y|Native. A custom Neuron generator is unsupported|
|121|torch.randn_like|Y|Native|
|122|torch.randperm|CPU_FALLBACK|Falls back (`aten::randperm.generator_out`) for `int64` and `float32`|
|123|torch.quasirandom.SobolEngine|Y|Host-only (no device ops); CPU tensors only, with no Neuron device|
|124|torch.save|Y|Host-only (no device ops)|
|125|torch.load|Y|Host-only (no device ops)|
|126|torch.get_num_threads|Y|Host-only (no device ops). Returns the CPU intra-op thread-pool setting|
|127|torch.set_num_threads|Y|Host-only (no device ops). Configures the CPU intra-op thread pool|
|128|torch.get_num_interop_threads|Y|Host-only (no device ops). Returns the CPU inter-op thread-pool setting|
|129|torch.set_num_interop_threads|Y|Host-only (no device ops). Configures the CPU inter-op thread pool|
|130|torch.no_grad|Y|Host-only (no device ops)|
|131|torch.enable_grad|Y|Host-only (no device ops)|
|132|torch.set_grad_enabled|Y|Host-only (no device ops)|
|133|torch.is_grad_enabled|Y|Host-only (no device ops)|
|134|torch.inference_mode|Y|Host-only (no device ops); unlike CPU, Neuron outputs are not marked as inference tensors|
|135|torch.is_inference_mode_enabled|Y|Host-only (no device ops)|
|136|torch.abs|Y|Native|
|137|torch.absolute|Y|Native|
|138|torch.acos|Y|Native|
|139|torch.arccos|Y|Native|
|140|torch.acosh|Y|Native|
|141|torch.arccosh|Y|Native|
|142|torch.add|Y|Native|
|143|torch.addcdiv|Y|Native|
|144|torch.addcmul|Y|Native|
|145|torch.angle|Y|Native|
|146|torch.asin|Y|Native|
|147|torch.arcsin|Y|Native|
|148|torch.asinh|Y|Native|
|149|torch.arcsinh|Y|Native|
|150|torch.atan|Y|Native|
|151|torch.arctan|Y|Native|
|152|torch.atanh|Y|Native|
|153|torch.arctanh|Y|Native|
|154|torch.atan2|CPU_FALLBACK|Falls back (`aten::atan2.out`) for `float32`, `bfloat16`, `int64`, and `bool`|
|155|torch.arctan2|CPU_FALLBACK|Falls back (`aten::atan2.out`) for `float32`, `bfloat16`, `int64`, and `bool`|
|156|torch.bitwise_not|Y|Native|
|157|torch.bitwise_and|Y|Native|
|158|torch.bitwise_or|Y|Native|
|159|torch.bitwise_xor|Y|Native|
|160|torch.bitwise_left_shift|Y|Native|
|161|torch.bitwise_right_shift|Y|Native|
|162|torch.ceil|Y|Native|
|163|torch.clamp|Y|Native|
|164|torch.clip|Y|Native. Alias for `torch.clamp`|
|165|torch.conj_physical|CPU_FALLBACK|`complex64` falls back (`aten::conj_physical.out`); real inputs native|
|166|torch.copysign|CPU_FALLBACK|Falls back (`aten::copysign.out`)|
|167|torch.cos|Y|Native|
|168|torch.cosh|Y|Native|
|169|torch.deg2rad|Y|Native. `int64` and `bool` inputs fall back for promotion (`aten::_to_copy`)|
|170|torch.div|Y|Native|
|171|torch.divide|Y|Native. Alias for `torch.div`|
|172|torch.digamma|CPU_FALLBACK|Falls back (`aten::digamma.out`)|
|173|torch.erf|Y|Native|
|174|torch.erfc|CPU_FALLBACK|Falls back (`aten::erfc.out`)|
|175|torch.erfinv|Y|Native|
|176|torch.exp|Y|Native|
|177|torch.exp2|Y|Native|
|178|torch.expm1|Y|Native|
|179|torch.fake_quantize_per_channel_affine|CPU_FALLBACK|Falls back (`aten::fake_quantize_per_channel_affine_cachemask`)|
|180|torch.fake_quantize_per_tensor_affine|CPU_FALLBACK|Falls back (`aten::fake_quantize_per_tensor_affine_cachemask`)|
|181|torch.fix|Y|Native. Alias for `torch.trunc`|
|182|torch.float_power|CPU_FALLBACK|Falls back (`aten::_to_copy`) for required `float64` conversion|
|183|torch.floor|Y|Native|
|184|torch.floor_divide|Y|Native|
|185|torch.fmod|CPU_FALLBACK|Falls back (`aten::fmod.Tensor_out`)|
|186|torch.frac|Y|Native|
|187|torch.frexp|CPU_FALLBACK|Falls back (`aten::frexp.Tensor_out`)|
|188|torch.gradient|Y|Native. `int64` inputs fall back for promotion (`aten::_to_copy`)|
|189|torch.imag|Y|Native. Returns a metadata view|
|190|torch.ldexp|Y|Native|
|191|torch.lerp|Y|Native; non-contiguous inputs fall back (`aten::lerp.Scalar`)|
|192|torch.lgamma|CPU_FALLBACK|Falls back (`aten::lgamma.out`)|
|193|torch.log|Y|Native|
|194|torch.log10|Y|Native|
|195|torch.log1p|Y|Native|
|196|torch.log2|Y|Native|
|197|torch.logaddexp|CPU_FALLBACK|Falls back (`aten::logaddexp.out`)|
|198|torch.logaddexp2|Y|Native|
|199|torch.logical_and|Y|Native|
|200|torch.logical_not|Y|Native|
|201|torch.logical_or|Y|Native|
|202|torch.logical_xor|Y|Native|
|203|torch.logit|Y|Native|
|204|torch.hypot|CPU_FALLBACK|Falls back (`aten::hypot.out`)|
|205|torch.i0|CPU_FALLBACK|Falls back (`aten::i0.out`)|
|206|torch.igamma|CPU_FALLBACK|Falls back (`aten::igamma.out`)|
|207|torch.igammac|CPU_FALLBACK|Falls back (`aten::igammac.out`)|
|208|torch.mul|Y|Native|
|209|torch.multiply|Y|Native|
|210|torch.mvlgamma|CPU_FALLBACK|Falls back (`aten::lgamma.out`)|
|211|torch.nan_to_num|CPU_FALLBACK|Falls back (`aten::nan_to_num.out`)|
|212|torch.neg|Y|Native|
|213|torch.negative|Y|Native|
|214|torch.nextafter|CPU_FALLBACK|Falls back (`aten::nextafter.out`)|
|215|torch.polygamma|CPU_FALLBACK|Falls back (`aten::polygamma.out`)|
|216|torch.positive|Y|Native|
|217|torch.pow|Y|Native|
|218|torch.quantized_batch_norm|N|Quantized tensors not supported on Neuron (`aten::empty_quantized`)|
|219|torch.quantized_max_pool1d|N|Quantized tensors not supported on Neuron (`aten::empty_quantized`)|
|220|torch.quantized_max_pool2d|N|Quantized tensors not supported on Neuron (`aten::empty_quantized`)|
|221|torch.rad2deg|Y|Native; `bool` and `int64` fall back (`aten::_to_copy`)|
|222|torch.real|Y|Native|
|223|torch.reciprocal|Y|Native|
|224|torch.remainder|Y|Native|
|225|torch.round|Y|Native|
|226|torch.rsqrt|Y|Native; `False` returns a finite value instead of infinity|
|227|torch.sigmoid|Y|Native|
|228|torch.sign|Y|Native|
|229|torch.sgn|Y|Native. Backward falls back (`aten::_efficientzerotensor`)|
|230|torch.signbit|Y|Native|
|231|torch.sin|Y|Native|
|232|torch.sinc|CPU_FALLBACK|Falls back (`aten::sinc.out`)|
|233|torch.sinh|Y|Native|
|234|torch.softmax|Y|Native; `dtype=torch.float64` and its backward fall back (`aten::_to_copy`)|
|235|torch.sqrt|Y|Native|
|236|torch.square|Y|Native|
|237|torch.sub|Y|Native|
|238|torch.subtract|Y|Native|
|239|torch.tan|Y|Native; `float32` can differ from CPU near the poles (ill-conditioned)|
|240|torch.tanh|Y|Native|
|241|torch.true_divide|Y|Native|
|242|torch.trunc|Y|Native|
|243|torch.xlogy|CPU_FALLBACK|Falls back (`aten::xlogy.OutTensor`)|
|244|torch.argmax|Y|Native|
|245|torch.argmin|Y|Native|
|246|torch.amax|Y|Native|
|247|torch.amin|Y|Native|
|248|torch.aminmax|CPU_FALLBACK|Falls back (`aten::aminmax.out`)|
|249|torch.all|Y|Native|
|250|torch.any|Y|Native|
|251|torch.max|Y|Native|
|252|torch.min|Y|Native|
|253|torch.dist|CPU_FALLBACK|Falls back (`aten::norm.out`)|
|254|torch.logsumexp|Y|Native|
|255|torch.mean|Y|Native|
|256|torch.nanmean|CPU_FALLBACK|Falls back (`aten::nansum`)|
|257|torch.median|CPU_FALLBACK|Falls back (`aten::median`)|
|258|torch.nanmedian|CPU_FALLBACK|Falls back (`aten::nanmedian`)|
|259|torch.mode|CPU_FALLBACK|Falls back (`aten::mode`)|
|260|torch.norm|Y|Native; nuclear norm falls back (`aten::_linalg_svd.U`), with no `bfloat16`|
|261|torch.nansum|CPU_FALLBACK|Falls back (`aten::nansum`)|
|262|torch.prod|Y|Native. Backward performs an internal dtype conversion on CPU (`aten::_to_copy`)|
|263|torch.quantile|CPU_FALLBACK|`float32` falls back (`aten::sort.values_stable`); no `bfloat16`|
|264|torch.nanquantile|CPU_FALLBACK|Falls back (`aten::_to_copy`, `aten::sort.values_stable`); no `bfloat16`|
|265|torch.std|Y|Native. Backward performs an internal dtype conversion on CPU (`aten::_to_copy`)|
|266|torch.std_mean|Y|Native. Backward performs an internal dtype conversion on CPU (`aten::_to_copy`)|
|267|torch.sum|Y|Native|
|268|torch.unique|CPU_FALLBACK|Falls back (`aten::_unique2`)|
|269|torch.unique_consecutive|CPU_FALLBACK|Falls back (`aten::unique_consecutive`)|
|270|torch.var|CPU_FALLBACK|Falls back (`aten::var.correction`)|
|271|torch.var_mean|CPU_FALLBACK|Falls back (`aten::var_mean.correction`)|
|272|torch.count_nonzero|CPU_FALLBACK|Falls back (`aten::count_nonzero.dim_IntList`)|
|273|torch.allclose|Y|Native|
|274|torch.argsort|CPU_FALLBACK|Falls back (`aten::sort.values_stable`); `stable=True` is native|
|275|torch.eq|Y|Native|
|276|torch.equal|Y|Native|
|277|torch.ge|Y|Native|
|278|torch.greater_equal|Y|Native|
|279|torch.gt|Y|Native|
|280|torch.greater|Y|Native|
|281|torch.isclose|Y|Native|
|282|torch.isfinite|Y|Native|
|283|torch.isin|Y|Native|
|284|torch.isinf|Y|Native|
|285|torch.isposinf|Y|Native|
|286|torch.isneginf|Y|Native|
|287|torch.isnan|Y|Native|
|288|torch.isreal|Y|Native; `complex64` conversion falls back (`aten::_to_copy`)|
|289|torch.kthvalue|CPU_FALLBACK|Falls back (`aten::kthvalue.values`)|
|290|torch.le|Y|Native|
|291|torch.less_equal|Y|Native|
|292|torch.lt|Y|Native|
|293|torch.less|Y|Native|
|294|torch.maximum|Y|Native|
|295|torch.minimum|Y|Native|
|296|torch.fmax|CPU_FALLBACK|Falls back (`aten::fmax.out`)|
|297|torch.fmin|CPU_FALLBACK|Falls back (`aten::fmin.out`)|
|298|torch.ne|Y|Native|
|299|torch.not_equal|Y|Native; scalar operands fall back (`aten::_to_copy`)|
|300|torch.sort|CPU_FALLBACK|Falls back (`aten::sort.values_stable`)|
|301|torch.topk|Y|Native|
|302|torch.msort|CPU_FALLBACK|Falls back (`aten::sort.values_stable`)|
|303|torch.stft|CPU_FALLBACK|Falls back (`aten::_fft_r2c`)|
|304|torch.istft|CPU_FALLBACK|Falls back (`aten::_fft_c2r`)|
|305|torch.bartlett_window|Y|Native|
|306|torch.blackman_window|Y|Native; `bfloat16` may differ from CPU for some lengths|
|307|torch.hamming_window|Y|Native|
|308|torch.hann_window|Y|Native|
|309|torch.kaiser_window|N|Eager unsupported; compiled `bfloat16` differs from CPU|
|310|torch.atleast_1d|Y|Native|
|311|torch.atleast_2d|Y|Native|
|312|torch.atleast_3d|Y|Native|
|313|torch.bincount|CPU_FALLBACK|Falls back (`aten::bincount`)|
|314|torch.block_diag|Y|Native; `int64` falls back (`aten::copy_`)|
|315|torch.broadcast_tensors|Y|Native|
|316|torch.broadcast_to|Y|Native|
|317|torch.broadcast_shapes|Y|Host-only (no device ops)|
|318|torch.bucketize|CPU_FALLBACK|Falls back (`aten::bucketize.Tensor`)|
|319|torch.cartesian_prod|Y|Native|
|320|torch.cdist|Y|Native; backward: `aten::_cdist_backward` fallback, other shapes use CPU `aten::_to_copy`|
|321|torch.clone|Y|Native|
|322|torch.combinations|Y|Native|
|323|torch.corrcoef|Y|Native|
|324|torch.cov|Y|Native|
|325|torch.cross|CPU_FALLBACK|Falls back (`aten::linalg_cross.out`)|
|326|torch.cummax|Y|Native|
|327|torch.cummin|Y|Native|
|328|torch.cumprod|CPU_FALLBACK|Falls back (`aten::cumprod.out`)|
|329|torch.cumsum|Y|Native|
|330|torch.diag|Y|Native; `int64` output assembly falls back (`aten::copy_`)|
|331|torch.diag_embed|Y|Native; `int64` assembly falls back (`aten::copy_`)|
|332|torch.diagflat|Y|Native; `int64` falls back (`aten::copy_`)|
|333|torch.diagonal|Y|Native|
|334|torch.diff|Y|Native|
|335|torch.einsum|Y|Native|
|336|torch.flatten|Y|Native|
|337|torch.flip|Y|Native|
|338|torch.fliplr|Y|Native|
|339|torch.flipud|Y|Native|
|340|torch.kron|Y|Native|
|341|torch.rot90|Y|Native|
|342|torch.gcd|CPU_FALLBACK|Falls back (`aten::gcd.out`)|
|343|torch.histc|Y|Native|
|344|torch.histogram|CPU_FALLBACK|Falls back (`aten::histogram.bin_ct`)|
|345|torch.histogramdd|CPU_FALLBACK|Falls back (`aten::_histogramdd_bin_edges`, `aten::_histogramdd_from_bin_cts`); no `bfloat16`|
|346|torch.meshgrid|Y|Native|
|347|torch.lcm|CPU_FALLBACK|Falls back (`aten::lcm.out`)|
|348|torch.logcumsumexp|CPU_FALLBACK|Falls back (`aten::_logcumsumexp`)|
|349|torch.ravel|Y|Native|
|350|torch.renorm|CPU_FALLBACK|Falls back (`aten::renorm.out`)|
|351|torch.repeat_interleave|Y|Native|
|352|torch.roll|Y|Native|
|353|torch.searchsorted|Y|Native|
|354|torch.tensordot|Y|Native; `int64` falls back (`aten::mm`)|
|355|torch.trace|CPU_FALLBACK|Falls back (`aten::trace`)|
|356|torch.tril|Y|Native|
|357|torch.tril_indices|Y|Native|
|358|torch.triu|Y|Native|
|359|torch.triu_indices|Y|Native|
|360|torch.unflatten|Y|Native|
|361|torch.vander|CPU_FALLBACK|Falls back (`aten::copy_` and `aten::cumprod.out`)|
|362|torch.view_as_real|Y|Native|
|363|torch.view_as_complex|Y|Native for `float32` input. `bfloat16` not supported by PyTorch|
|364|torch.resolve_conj|Y|Native|
|365|torch.resolve_neg|N|`float32` negative-bit results differ from CPU|
|366|torch.addbmm|CPU_FALLBACK|Falls back (`aten::addbmm`)|
|367|torch.addmm|Y|Native; `int64` falls back (`aten::addmm`)|
|368|torch.addmv|Y|Native|
|369|torch.addr|Y|Native|
|370|torch.baddbmm|Y|Native|
|371|torch.bmm|Y|Native; `int64` falls back (`aten::bmm`)|
|372|torch.chain_matmul|Y|Native; `int64` falls back (`aten::mm`)|
|373|torch.cholesky|CPU_FALLBACK|`float32` falls back (`aten::cholesky`); no `bfloat16`|
|374|torch.cholesky_inverse|CPU_FALLBACK|Falls back (`aten::cholesky_inverse`); no `bfloat16`|
|375|torch.cholesky_solve|CPU_FALLBACK|Falls back (`aten::_cholesky_solve_helper`) for `float32`; no `bfloat16`|
|376|torch.dot|Y|Native|
|377|torch.geqrf|CPU_FALLBACK|Falls back (`aten::geqrf`) for `float32`. `bfloat16` inputs are not supported|
|378|torch.ger|Y|Native|
|379|torch.inner|Y|Native|
|380|torch.inverse|CPU_FALLBACK|`float32` falls back (`aten::linalg_inv_ex.inverse`); no `bfloat16`|
|381|torch.det|CPU_FALLBACK|`float32` falls back (`aten::_linalg_det.result`); no `bfloat16`|
|382|torch.logdet|CPU_FALLBACK|Falls back (`aten::_linalg_slogdet.sign`); no `bfloat16`|
|383|torch.slogdet|CPU_FALLBACK|Falls back (`aten::_linalg_slogdet.sign`)|
|384|torch.lu|CPU_FALLBACK|Falls back (`aten::linalg_lu_factor_ex.out`) for `float32`; no `bfloat16`|
|385|torch.lu_solve|CPU_FALLBACK|Falls back (`aten::linalg_lu_solve.out`) for `float32`; no `bfloat16`|
|386|torch.lu_unpack|CPU_FALLBACK|Falls back (`aten::lu_unpack.out`)|
|387|torch.matmul|Y|Native|
|388|torch.matrix_power|Y|Native nonnegative `float32`; negative falls back (`aten::linalg_inv_ex.inverse`)|
|389|torch.matrix_exp|CPU_FALLBACK|Falls back (`aten::linalg_matrix_exp`)|
|390|torch.mm|Y|Native; `int64` falls back (`aten::mm`)|
|391|torch.mv|Y|Native|
|392|torch.orgqr|CPU_FALLBACK|Falls back (`aten::linalg_householder_product`)|
|393|torch.ormqr|CPU_FALLBACK|Falls back (`aten::ormqr`)|
|394|torch.outer|Y|Native|
|395|torch.pinverse|CPU_FALLBACK|Falls back (`aten::_linalg_svd.U` and `aten::_to_copy`)|
|396|torch.qr|CPU_FALLBACK|Falls back (`aten::linalg_qr.out`) for `float32`; no `bfloat16`|
|397|torch.svd|CPU_FALLBACK|Falls back (`aten::_linalg_svd.U`) for `float32`; no `bfloat16`|
|398|torch.svd_lowrank|CPU_FALLBACK|Falls back (`aten::_linalg_svd.U`, `aten::linalg_qr.out`) for `float32`; no `bfloat16`|
|399|torch.pca_lowrank|CPU_FALLBACK|Falls back (`aten::_linalg_svd.U`, `aten::linalg_qr.out`); `bfloat16` basis mismatch|
|400|torch.lobpcg|CPU_FALLBACK|Falls back (`aten::_linalg_eigh.eigenvalues`, `aten::linalg_cholesky_ex.L`, …); no `bfloat16`|
|401|torch.trapz|Y|Native for `float32`, `bfloat16`, and `int64` inputs|
|402|torch.trapezoid|Y|Native for `float32`, `bfloat16`, and `int64` inputs|
|403|torch.cumulative_trapezoid|Y|Native for `float32`, `bfloat16`, and `int64` inputs|
|404|torch.triangular_solve|CPU_FALLBACK|Falls back (`aten::triangular_solve.X`) for `float32`; no `bfloat16`|
|405|torch.vdot|CPU_FALLBACK|Falls back (`aten::vdot`)|
|406|torch.ops.aten._foreach_add.Scalar|Y|Native for `float32`, `float16`, and `bfloat16` tensors|
|407|torch.ops.aten._foreach_add_.Scalar|Y|Native for `float32`, `float16`, and `bfloat16` tensors|
|408|torch.ops.aten._foreach_add.Scalar_out|Y|Native for `float32`, `float16`, and `bfloat16` tensors with scalar `1.0`|
|409|torch.ops.aten._foreach_add.List|Y|Native|
|410|torch.ops.aten._foreach_add_.List|Y|Native|
|411|torch.ops.aten._foreach_add.List_out|Y|Native|
|412|torch.ops.aten._foreach_add.ScalarList|Y|Native|
|413|torch.ops.aten._foreach_add_.ScalarList|Y|Native|
|414|torch.ops.aten._foreach_add.ScalarList_out|Y|Native|
|415|torch.ops.aten._foreach_add.Tensor|Y|Native|
|416|torch.ops.aten._foreach_add_.Tensor|Y|Native|
|417|torch.ops.aten._foreach_add.Tensor_out|Y|Native|
|418|torch.ops.aten._foreach_mul.Scalar|Y|Native|
|419|torch.ops.aten._foreach_mul_.Scalar|Y|Native|
|420|torch.ops.aten._foreach_mul.Scalar_out|Y|Native|
|421|torch.ops.aten._foreach_mul.List|Y|Native|
|422|torch.ops.aten._foreach_mul_.List|Y|Native|
|423|torch.ops.aten._foreach_mul.List_out|Y|Native|
|424|torch.ops.aten._foreach_mul.ScalarList|Y|Native|
|425|torch.ops.aten._foreach_mul_.ScalarList|Y|Native|
|426|torch.ops.aten._foreach_mul.ScalarList_out|Y|Native|
|427|torch.ops.aten._foreach_mul.Tensor|Y|Native|
|428|torch.ops.aten._foreach_mul_.Tensor|Y|Native|
|429|torch.ops.aten._foreach_mul.Tensor_out|Y|Native|
|430|torch.ops.aten._foreach_sub.Scalar|Y|Native|
|431|torch.ops.aten._foreach_sub_.Scalar|Y|Native|
|432|torch.ops.aten._foreach_sub.Scalar_out|Y|Native|
|433|torch.ops.aten._foreach_sub.List|Y|Native|
|434|torch.ops.aten._foreach_sub_.List|Y|Native|
|435|torch.ops.aten._foreach_sub.List_out|Y|Native|
|436|torch.ops.aten._foreach_sub.ScalarList|Y|Native|
|437|torch.ops.aten._foreach_sub_.ScalarList|Y|Native|
|438|torch.ops.aten._foreach_sub.ScalarList_out|Y|Native|
|439|torch.ops.aten._foreach_div.Scalar|Y|Native|
|440|torch.ops.aten._foreach_div_.Scalar|Y|Native|
|441|torch.ops.aten._foreach_div.Scalar_out|Y|Native|
|442|torch.ops.aten._foreach_div.List|Y|Native|
|443|torch.ops.aten._foreach_div_.List|Y|Native|
|444|torch.ops.aten._foreach_div.List_out|Y|Native|
|445|torch.ops.aten._foreach_div.ScalarList|Y|Native|
|446|torch.ops.aten._foreach_div_.ScalarList|Y|Native|
|447|torch.ops.aten._foreach_div.ScalarList_out|Y|Native|
|448|torch.ops.aten._foreach_div.Tensor|Y|Native|
|449|torch.ops.aten._foreach_div_.Tensor|Y|Native|
|450|torch.ops.aten._foreach_div.Tensor_out|Y|Native|
|451|torch._foreach_norm|Y|Native|
|452|torch.compiled_with_cxx11_abi|Y|Host-only (no device ops). Reports the C++ ABI used to build PyTorch|
|453|torch.result_type|Y|Host-only (no device ops). Reports the promoted dtype for the provided operands|
|454|torch.can_cast|Y|Host-only (no device ops). Reports whether one dtype can be cast to another|
|455|torch.promote_types|Y|Host-only (no device ops). Reports the promoted dtype for two input dtypes|
|456|torch.use_deterministic_algorithms|Y|Host-only (no device ops)|
|457|torch.are_deterministic_algorithms_enabled|Y|Host-only (no device ops)|
|458|torch.is_deterministic_algorithms_warn_only_enabled|Y|Host-only (no device ops)|
|459|torch.set_deterministic_debug_mode|Y|Host-only (no device ops)|
|460|torch.get_deterministic_debug_mode|Y|Host-only (no device ops). Reads the global deterministic debug mode|
|461|torch.set_float32_matmul_precision|N|CUDA-only|
|462|torch.get_float32_matmul_precision|N|CUDA-only|
|463|torch.set_warn_always|Y|Host-only (no device ops). Sets the warning policy|
|464|torch.is_warn_always_enabled|Y|Host-only (no device ops). Reads the warning policy|
|465|torch.vmap|Y|Native|
|466|torch._assert|Y|Host-only (no device ops)|
|467|torch.sym_float|Y|Host-only (no device ops)|
|468|torch.sym_int|Y|Host-only (no device ops)|
|469|torch.sym_max|Y|Host-only (no device ops)|
|470|torch.sym_min|Y|Host-only (no device ops)|
|471|torch.sym_not|Y|Host-only (no device ops)|
|472|torch.compile|Y|Native|

## torch.nn

||pytorch_api|neuron_compatible|limitations|
|---|---|---|---|
|1|torch.nn.parameter.Parameter|Y|Host-only (no device ops)|
|2|torch.nn.parameter.UninitializedParameter|Y|Native; direct Neuron construction allocates Neuron storage|
|3|torch.nn.parameter.UninitializedParameter.cls_to_become|Y|Host-only (no device ops)|
|4|torch.nn.parameter.UninitializedBuffer|Y|Host-only (no device ops)|
|5|torch.nn.Module|Y|Host-only (no device ops)|
|6|torch.nn.Module.add_module|Y|Host-only (no device ops). Registers a child module|
|7|torch.nn.Module.apply|Y|Host-only (no device ops)|
|8|torch.nn.Module.bfloat16|Y|Native|
|9|torch.nn.Module.buffers|Y|Host-only (no device ops). Returns an iterator over registered buffers|
|10|torch.nn.Module.children|Y|Host-only (no device ops)|
|11|torch.nn.Module.compile|Y|Native|
|12|torch.nn.Module.cpu|Y|Native|
|13|torch.nn.Module.cuda|N|CUDA-only|
|14|torch.nn.Module.double|Y|Native|
|15|torch.nn.Module.eval|Y|Host-only (no device ops)|
|16|torch.nn.Module.extra_repr|Y|Host-only (no device ops). Formats module metadata without device computation|
|17|torch.nn.Module.float|Y|Native|
|18|torch.nn.Module.forward|Y|Host-only (no device ops)|
|19|torch.nn.Module.get_buffer|Y|Host-only (no device ops)|
|20|torch.nn.Module.get_extra_state|Y|Host-only (no device ops)|
|21|torch.nn.Module.get_parameter|Y|Host-only (no device ops)|
|22|torch.nn.Module.get_submodule|Y|Host-only (no device ops)|
|23|torch.nn.Module.half|Y|Native|
|24|torch.nn.Module.ipu|N|IPU-specific|
|25|torch.nn.Module.load_state_dict|Y|Native|
|26|torch.nn.Module.modules|Y|Host-only (no device ops)|
|27|torch.nn.Module.named_buffers|Y|Host-only (no device ops). Returns names and registered buffers|
|28|torch.nn.Module.named_children|Y|Host-only (no device ops)|
|29|torch.nn.Module.named_modules|Y|Host-only (no device ops)|
|30|torch.nn.Module.named_parameters|Y|Host-only (no device ops)|
|31|torch.nn.Module.parameters|Y|Host-only (no device ops)|
|32|torch.nn.Module.register_backward_hook|Y|Host-only (no device ops)|
|33|torch.nn.Module.register_buffer|Y|Host-only (no device ops). Registers a buffer without device computation|
|34|torch.nn.Module.register_forward_hook|Y|Host-only (no device ops)|
|35|torch.nn.Module.register_forward_pre_hook|Y|Host-only (no device ops)|
|36|torch.nn.Module.register_full_backward_hook|Y|Host-only (no device ops)|
|37|torch.nn.Module.register_full_backward_pre_hook|Y|Host-only (no device ops)|
|38|torch.nn.Module.register_load_state_dict_post_hook|Y|Host-only (no device ops)|
|39|torch.nn.Module.register_module|Y|Host-only (no device ops). Alias for `add_module`|
|40|torch.nn.Module.register_parameter|Y|Host-only (no device ops). Registers a parameter on the module|
|41|torch.nn.Module.register_state_dict_pre_hook|Y|Host-only (no device ops). Registers a state-dictionary pre-hook|
|42|torch.nn.Module.requires_grad_|Y|Host-only (no device ops). Updates parameter autograd metadata|
|43|torch.nn.Module.set_extra_state|Y|Host-only (no device ops). Defines restoration of extra module state|
|44|torch.nn.Module.share_memory|Y|Native|
|45|torch.nn.Module.state_dict|Y|Host-only (no device ops)|
|46|torch.nn.Module.to|Y|Native. Supports the covered `float32` and `bfloat16` parameter conversions|
|47|torch.nn.Module.to_empty|Y|Native. Supports `float32` and `bfloat16`; parameter values are uninitialized|
|48|torch.nn.Module.train|Y|Host-only (no device ops). Sets the module and its children to training mode|
|49|torch.nn.Module.type|Y|Native. Supports the covered `float32` and `bfloat16` parameter conversions|
|50|torch.nn.Module.xpu|N|Intel XPU-specific|
|51|torch.nn.Module.zero_grad|Y|Native. Clears gradients or sets them to `None`|
|52|torch.nn.Sequential|Y|Native. Support depends on the contained modules|
|53|torch.nn.Sequential.append|Y|Host-only (no device ops). Appends a module to the container|
|54|torch.nn.ModuleList|Y|Host-only (no device ops). Stores registered modules in list order|
|55|torch.nn.ModuleList.append|Y|Host-only (no device ops). Appends a module to the container|
|56|torch.nn.ModuleList.extend|Y|Host-only (no device ops). Appends modules from an iterable|
|57|torch.nn.ModuleList.insert|Y|Host-only (no device ops). Inserts a module at the requested index|
|58|torch.nn.ModuleDict|Y|Host-only (no device ops). Stores registered modules by key|
|59|torch.nn.ModuleDict.clear|Y|Host-only (no device ops). Removes all modules from the container|
|60|torch.nn.ModuleDict.items|Y|Host-only (no device ops). Returns the container's key-module pairs|
|61|torch.nn.ModuleDict.keys|Y|Host-only (no device ops). Returns the registered module names|
|62|torch.nn.ModuleDict.pop|Y|Host-only (no device ops). Removes and returns a module by key|
|63|torch.nn.ModuleDict.update|Y|Host-only (no device ops). Updates registered modules by key|
|64|torch.nn.ModuleDict.values|Y|Host-only (no device ops). Returns the registered modules|
|65|torch.nn.ParameterList|Y|Host-only (no device ops). Stores registered parameters in list order|
|66|torch.nn.ParameterList.append|Y|Host-only (no device ops). Appends a parameter to the container|
|67|torch.nn.ParameterList.extend|Y|Host-only (no device ops). Appends parameters from an iterable|
|68|torch.nn.ParameterDict|Y|Host-only (no device ops). Stores registered parameters by key|
|69|torch.nn.ParameterDict.clear|Y|Host-only (no device ops). Removes all parameters from the container|
|70|torch.nn.ParameterDict.copy|Y|Host-only (no device ops). Returns a shallow copy of the container|
|71|torch.nn.ParameterDict.fromkeys|Y|Host-only (no device ops). Creates a parameter container from the supplied keys|
|72|torch.nn.ParameterDict.get|Y|Host-only (no device ops). Returns a parameter by key or a default value|
|73|torch.nn.ParameterDict.items|Y|Host-only (no device ops). Returns the container's key-parameter pairs|
|74|torch.nn.ParameterDict.keys|Y|Host-only (no device ops). Returns the registered parameter names|
|75|torch.nn.ParameterDict.pop|Y|Host-only (no device ops). Removes and returns a parameter by key|
|76|torch.nn.ParameterDict.popitem|Y|Host-only (no device ops). Removes and returns the last key-parameter pair|
|77|torch.nn.ParameterDict.setdefault|Y|Host-only (no device ops)|
|78|torch.nn.ParameterDict.update|Y|Host-only (no device ops). Updates registered parameters by key|
|79|torch.nn.ParameterDict.values|Y|Host-only (no device ops). Returns the registered parameters|
|80|torch.nn.modules.module.register_module_forward_pre_hook|Y|Host-only (no device ops). Registers a global module forward pre-hook|
|81|torch.nn.modules.module.register_module_forward_hook|Y|Host-only (no device ops). Registers a global module forward hook|
|82|torch.nn.modules.module.register_module_backward_hook|Y|Host-only (no device ops). Registers a global module backward hook|
|83|torch.nn.modules.module.register_module_full_backward_pre_hook|Y|Host-only (no device ops)|
|84|torch.nn.modules.module.register_module_full_backward_hook|Y|Host-only (no device ops)|
|85|torch.nn.modules.module.register_module_buffer_registration_hook|Y|Host-only (no device ops). Registers a global buffer-registration hook|
|86|torch.nn.modules.module.register_module_module_registration_hook|Y|Host-only (no device ops). Registers a global module-registration hook|
|87|torch.nn.modules.module.register_module_parameter_registration_hook|Y|Host-only (no device ops). Registers a global parameter-registration hook|
|88|torch.nn.Conv1d|Y|Native|
|89|torch.nn.Conv2d|Y|Native|
|90|torch.nn.Conv3d|Y|Native|
|91|torch.nn.ConvTranspose1d|Y|Native; non-contiguous inputs round-trip via CPU|
|92|torch.nn.ConvTranspose2d|Y|Native|
|93|torch.nn.ConvTranspose3d|Y|Native|
|94|torch.nn.LazyConv1d|Y|Native; non-contiguous inputs round-trip via CPU|
|95|torch.nn.LazyConv1d.cls_to_become|Y|Host-only (no device ops)|
|96|torch.nn.LazyConv2d|Y|Native|
|97|torch.nn.LazyConv2d.cls_to_become|Y|Host-only (no device ops)|
|98|torch.nn.LazyConv3d|Y|Native|
|99|torch.nn.LazyConv3d.cls_to_become|Y|Host-only (no device ops)|
|100|torch.nn.LazyConvTranspose1d|Y|Native|
|101|torch.nn.LazyConvTranspose1d.cls_to_become|Y|Host-only (no device ops)|
|102|torch.nn.LazyConvTranspose2d|Y|Native|
|103|torch.nn.LazyConvTranspose2d.cls_to_become|Y|Host-only (no device ops)|
|104|torch.nn.LazyConvTranspose3d|Y|Native|
|105|torch.nn.LazyConvTranspose3d.cls_to_become|Y|Host-only (no device ops)|
|106|torch.nn.Unfold|Y|Native|
|107|torch.nn.Fold|Y|Native. `float32` and `bfloat16` results match CPU|
|108|torch.nn.MaxPool1d|CPU_FALLBACK|Falls back (`aten::max_pool2d_with_indices.out`) for `float32` and `bfloat16`|
|109|torch.nn.MaxPool2d|CPU_FALLBACK|Falls back (`aten::max_pool2d_with_indices.out`)|
|110|torch.nn.MaxPool3d|CPU_FALLBACK|Falls back (`aten::max_pool3d_with_indices`) for `float32`, `bfloat16` and `int64`|
|111|torch.nn.MaxUnpool1d|CPU_FALLBACK|Falls back (`aten::max_unpool2d`) for `float32` and `bfloat16`|
|112|torch.nn.MaxUnpool2d|CPU_FALLBACK|Falls back (`aten::max_unpool2d`) for `float32` and `bfloat16`|
|113|torch.nn.MaxUnpool3d|CPU_FALLBACK|Falls back (`aten::max_unpool3d`) for `float32` and `bfloat16`|
|114|torch.nn.AvgPool1d|Y|Native. `float32` and `bfloat16` results match CPU|
|115|torch.nn.AvgPool2d|Y|Native|
|116|torch.nn.AvgPool3d|CPU_FALLBACK|Falls back (`aten::avg_pool3d.out`) for `float32` and `int64`|
|117|torch.nn.FractionalMaxPool2d|CPU_FALLBACK|Falls back (`aten::fractional_max_pool2d.output`) for `float32` and `bfloat16`|
|118|torch.nn.FractionalMaxPool3d|CPU_FALLBACK|Falls back (`aten::fractional_max_pool3d.output`) for `float32` and `bfloat16`|
|119|torch.nn.LPPool1d|Y|Native. `float32` and `bfloat16` results match CPU|
|120|torch.nn.LPPool2d|Y|Native. `float32` and `bfloat16` results match CPU|
|121|torch.nn.AdaptiveMaxPool1d|CPU_FALLBACK|Falls back (`aten::adaptive_max_pool2d.out`) for `float32` and `bfloat16`|
|122|torch.nn.AdaptiveMaxPool2d|CPU_FALLBACK|Falls back (`aten::adaptive_max_pool2d.out`) for `float32` and `bfloat16`|
|123|torch.nn.AdaptiveMaxPool3d|CPU_FALLBACK|Falls back (`aten::adaptive_max_pool3d.out`) for `float32` and `bfloat16`|
|124|torch.nn.AdaptiveAvgPool1d|Y|Native; `bfloat16` can differ from CPU|
|125|torch.nn.AdaptiveAvgPool2d|Y|Native|
|126|torch.nn.AdaptiveAvgPool3d|CPU_FALLBACK|Falls back (`aten::_adaptive_avg_pool3d`) for `float32` and `bfloat16`|
|127|torch.nn.ReflectionPad1d|Y|Native|
|128|torch.nn.ReflectionPad2d|Y|Native|
|129|torch.nn.ReflectionPad3d|Y|Native|
|130|torch.nn.ReplicationPad1d|Y|Native. Backward falls back (`aten::replication_pad1d_backward.grad_input`)|
|131|torch.nn.ReplicationPad2d|Y|Native; non-contiguous uses CPU, backward falls back (`aten::replication_pad2d_backward`)|
|132|torch.nn.ReplicationPad3d|Y|Native. Backward falls back (`aten::replication_pad3d_backward`)|
|133|torch.nn.ZeroPad1d|Y|Native|
|134|torch.nn.ZeroPad2d|Y|Native|
|135|torch.nn.ZeroPad3d|Y|Native|
|136|torch.nn.ConstantPad1d|Y|Native|
|137|torch.nn.ConstantPad2d|Y|Native|
|138|torch.nn.ConstantPad3d|Y|Native|
|139|torch.nn.ELU|Y|Native|
|140|torch.nn.Hardshrink|Y|Native|
|141|torch.nn.Hardsigmoid|Y|Native|
|142|torch.nn.Hardtanh|Y|Native|
|143|torch.nn.Hardswish|Y|Native|
|144|torch.nn.LeakyReLU|Y|Native. Backward falls back (`aten::leaky_relu_backward.grad_input`)|
|145|torch.nn.LogSigmoid|Y|Native; forward and input-gradient backward match CPU|
|146|torch.nn.MultiheadAttention|Y|Forward runs natively on Neuron. Not covered by `torch.compile` tests|
|147|torch.nn.MultiheadAttention.forward|Y|Native. Not covered by `torch.compile` tests|
|148|torch.nn.MultiheadAttention.merge_masks|Y|Native|
|149|torch.nn.PReLU|CPU_FALLBACK|`float32` and `bfloat16` run on CPU through fallback (`aten::_prelu_kernel`)|
|150|torch.nn.ReLU|Y|Native|
|151|torch.nn.ReLU6|Y|Native|
|152|torch.nn.RReLU|CPU_FALLBACK|`float32` and `bfloat16` run on CPU through fallback (`aten::rrelu_with_noise`)|
|153|torch.nn.SELU|Y|Native|
|154|torch.nn.CELU|Y|Native|
|155|torch.nn.GELU|Y|Native|
|156|torch.nn.Sigmoid|Y|Native|
|157|torch.nn.SiLU|Y|Native|
|158|torch.nn.Mish|Y|Native|
|159|torch.nn.Softplus|Y|Native|
|160|torch.nn.Softshrink|CPU_FALLBACK|`float32` and `bfloat16` run on CPU through fallback (`aten::softshrink.out`)|
|161|torch.nn.Softsign|Y|Native through `torch.nn.functional.softsign`|
|162|torch.nn.Tanh|Y|Native|
|163|torch.nn.Tanhshrink|Y|Native through `torch.nn.functional.tanhshrink`|
|164|torch.nn.Threshold|Y|Native|
|165|torch.nn.GLU|Y|Native|
|166|torch.nn.Softmin|Y|Native through `torch.nn.functional.softmin`|
|167|torch.nn.Softmax|Y|Native|
|168|torch.nn.Softmax2d|Y|Native through `torch.nn.functional.softmax`|
|169|torch.nn.LogSoftmax|Y|Native|
|170|torch.nn.AdaptiveLogSoftmaxWithLoss|Y|Forward runs natively on Neuron for `float32` and `bfloat16`|
|171|torch.nn.AdaptiveLogSoftmaxWithLoss.log_prob|Y|Native for `float32` and `bfloat16`|
|172|torch.nn.AdaptiveLogSoftmaxWithLoss.predict|Y|Native for `float32` and `bfloat16`|
|173|torch.nn.BatchNorm1d|Y|Native|
|174|torch.nn.BatchNorm2d|Y|Native|
|175|torch.nn.BatchNorm3d|Y|Native|
|176|torch.nn.LazyBatchNorm1d|Y|Native|
|177|torch.nn.LazyBatchNorm1d.cls_to_become|Y|Host-only (no device ops)|
|178|torch.nn.LazyBatchNorm2d|Y|Native|
|179|torch.nn.LazyBatchNorm2d.cls_to_become|Y|Host-only (no device ops)|
|180|torch.nn.LazyBatchNorm3d|Y|Native|
|181|torch.nn.LazyBatchNorm3d.cls_to_become|Y|Host-only (no device ops)|
|182|torch.nn.GroupNorm|Y|Native|
|183|torch.nn.SyncBatchNorm|Y|Native for single-process evaluation|
|184|torch.nn.SyncBatchNorm.convert_sync_batchnorm|Y|Host-only (no device ops)|
|185|torch.nn.InstanceNorm1d|Y|Native; non-contiguous inputs round-trip via CPU|
|186|torch.nn.InstanceNorm2d|Y|Native|
|187|torch.nn.InstanceNorm3d|Y|Native|
|188|torch.nn.LazyInstanceNorm1d|Y|Native; non-contiguous inputs use CPU (`aten::contiguous`)|
|189|torch.nn.LazyInstanceNorm1d.cls_to_become|Y|Host-only (no device ops)|
|190|torch.nn.LazyInstanceNorm2d|Y|Native|
|191|torch.nn.LazyInstanceNorm2d.cls_to_become|Y|Host-only (no device ops)|
|192|torch.nn.LazyInstanceNorm3d|Y|Native|
|193|torch.nn.LazyInstanceNorm3d.cls_to_become|Y|Host-only (no device ops)|
|194|torch.nn.LayerNorm|Y|Native|
|195|torch.nn.LocalResponseNorm|Y|Native rank 3; rank 4/5 falls back (`aten::avg_pool3d.out`), no `bfloat16`|
|196|torch.nn.RNNBase|Y|Host-only (no device ops); `torch.nn.GRU` is not supported|
|197|torch.nn.RNNBase.flatten_parameters|Y|Host-only (no device ops). Checks recurrent-weight layout metadata|
|198|torch.nn.RNN|Y|Native|
|199|torch.nn.LSTM|Y|Native|
|200|torch.nn.GRU|N|Raises `NotImplementedError` (`aten::_thnn_fused_gru_cell`)|
|201|torch.nn.RNNCell|Y|Native; non-contiguous inputs use CPU (`aten::linear`)|
|202|torch.nn.LSTMCell|Y|Native|
|203|torch.nn.GRUCell|N|Raises `NotImplementedError` (`aten::_thnn_fused_gru_cell`)|
|204|torch.nn.Transformer|Y|Native; `bfloat16` can differ from CPU for some shapes|
|205|torch.nn.Transformer.forward|Y|Native; `bfloat16` results can differ slightly from CPU|
|206|torch.nn.Transformer.generate_square_subsequent_mask|Y|Native|
|207|torch.nn.TransformerEncoder|Y|Native|
|208|torch.nn.TransformerEncoder.forward|Y|Native|
|209|torch.nn.TransformerDecoder|Y|Native|
|210|torch.nn.TransformerDecoder.forward|Y|Native|
|211|torch.nn.TransformerEncoderLayer|Y|Native; `bfloat16` may differ from CPU for some shapes|
|212|torch.nn.TransformerEncoderLayer.forward|Y|Native|
|213|torch.nn.TransformerDecoderLayer|Y|Native|
|214|torch.nn.TransformerDecoderLayer.forward|Y|Native|
|215|torch.nn.Identity|Y|Native|
|216|torch.nn.Linear|Y|Native for `float32` and `bfloat16`. `int64` inputs fall back (`aten::linear`)|
|217|torch.nn.Bilinear|Y|Native; integer inputs unsupported|
|218|torch.nn.LazyLinear|Y|Native; `int64` falls back (`aten::linear`)|
|219|torch.nn.LazyLinear.cls_to_become|Y|Host-only (no device ops)|
|220|torch.nn.Dropout|Y|Native|
|221|torch.nn.Dropout1d|CPU_FALLBACK|Training falls back (`aten::bernoulli_.float`); evaluation is native|
|222|torch.nn.Dropout2d|CPU_FALLBACK|Training falls back (`aten::bernoulli_.float`); evaluation is a no-op|
|223|torch.nn.Dropout3d|CPU_FALLBACK|Training falls back (`aten::bernoulli_.float`); evaluation is a no-op|
|224|torch.nn.AlphaDropout|CPU_FALLBACK|Training falls back (`aten::bernoulli_.float`); evaluation is native|
|225|torch.nn.FeatureAlphaDropout|CPU_FALLBACK|Training falls back (`aten::bernoulli_.float`); evaluation is native|
|226|torch.nn.Embedding|Y|Native|
|227|torch.nn.Embedding.from_pretrained|Y|Host-only (no device ops)|
|228|torch.nn.EmbeddingBag|CPU_FALLBACK|Falls back (`aten::_embedding_bag`) for `float32` and `bfloat16`|
|229|torch.nn.EmbeddingBag.forward|CPU_FALLBACK|Falls back (`aten::_embedding_bag`) for `float32` and `bfloat16`|
|230|torch.nn.EmbeddingBag.from_pretrained|Y|Native|
|231|torch.nn.CosineSimilarity|Y|Native. Backward performs an internal dtype conversion on CPU (`aten::_to_copy`)|
|232|torch.nn.PairwiseDistance|CPU_FALLBACK|Falls back (`aten::norm.out`) for `float32`, `bfloat16`, and `int64`|
|233|torch.nn.L1Loss|Y|Native|
|234|torch.nn.MSELoss|Y|Native|
|235|torch.nn.CrossEntropyLoss|Y|Native|
|236|torch.nn.CTCLoss|CPU_FALLBACK|Falls back (`aten::_ctc_loss`) for `float32`|
|237|torch.nn.NLLLoss|Y|Native|
|238|torch.nn.PoissonNLLLoss|Y|Native; `log_input=False, full=True` falls back (`aten::_to_copy`)|
|239|torch.nn.GaussianNLLLoss|CPU_FALLBACK|Falls back (`aten::_to_copy`) for `float32` and `bfloat16`|
|240|torch.nn.KLDivLoss|CPU_FALLBACK|Falls back (`aten::xlogy.OutTensor`) for `float32` and `bfloat16`|
|241|torch.nn.BCELoss|CPU_FALLBACK|Falls back (`aten::binary_cross_entropy`) for `float32` and `bfloat16`|
|242|torch.nn.BCEWithLogitsLoss|Y|Native|
|243|torch.nn.MarginRankingLoss|Y|Native. Backward performs an internal dtype conversion on CPU (`aten::_to_copy`)|
|244|torch.nn.HingeEmbeddingLoss|CPU_FALLBACK|Falls back (`aten::_to_copy`) for internal dtype conversion|
|245|torch.nn.MultiLabelMarginLoss|CPU_FALLBACK|Falls back (`aten::multilabel_margin_loss_forward`) for `float32`|
|246|torch.nn.HuberLoss|CPU_FALLBACK|Eager `float32` and `bfloat16` forward execution falls back (`aten::huber_loss`)|
|247|torch.nn.SmoothL1Loss|CPU_FALLBACK|Falls back (`aten::smooth_l1_loss.out`) for `float32` and `bfloat16`|
|248|torch.nn.SoftMarginLoss|Y|Native for `float32` and `bfloat16`; forward results match CPU|
|249|torch.nn.MultiLabelSoftMarginLoss|Y|Native for `float32` and `bfloat16`; forward results match CPU|
|250|torch.nn.CosineEmbeddingLoss|Y|Native|
|251|torch.nn.MultiMarginLoss|CPU_FALLBACK|Falls back (`aten::multi_margin_loss`) for `float32`|
|252|torch.nn.TripletMarginLoss|CPU_FALLBACK|Falls back (`aten::norm.out`) for `float32`, `bfloat16`, and `int64`|
|253|torch.nn.TripletMarginWithDistanceLoss|CPU_FALLBACK|Falls back (`aten::norm.out`) for `float32`, `bfloat16`, and `int64`|
|254|torch.nn.PixelShuffle|Y|Native|
|255|torch.nn.PixelUnshuffle|Y|Native|
|256|torch.nn.Upsample|Y|Native; non-contiguous inputs can round-trip via CPU|
|257|torch.nn.UpsamplingNearest2d|Y|Native|
|258|torch.nn.UpsamplingBilinear2d|Y|Native|
|259|torch.nn.ChannelShuffle|CPU_FALLBACK|Falls back (`aten::channel_shuffle`) for `float32`, `bfloat16`, `int64`, `bool`|
|260|torch.nn.DataParallel|N|CUDA-only; use `torch.nn.parallel.DistributedDataParallel` on Neuron|
|261|torch.nn.parallel.DistributedDataParallel|Y|Native. Needs an initialized process group|
|262|torch.nn.parallel.DistributedDataParallel.join|Y|Native; flag casts fall back (`aten::_to_copy`)|
|263|torch.nn.parallel.DistributedDataParallel.join_hook|Y|Host-only (no device ops). Returns the DDP join hook used by `Join`|
|264|torch.nn.parallel.DistributedDataParallel.no_sync|Y|Native. Needs an initialized process group|
|265|torch.nn.parallel.DistributedDataParallel.register_comm_hook|Y|Host-only (no device ops)|
|266|torch.nn.parallel.data_parallel|N|CUDA-only; use `DistributedDataParallel` on Neuron|
|267|torch.nn.utils.clip_grad_norm_|Y|Native for `float32` and `bfloat16`; results match CPU|
|268|torch.nn.utils.clip_grad_value_|Y|Native for `float32` and `bfloat16`; results match CPU|
|269|torch.nn.utils.parameters_to_vector|Y|Native for `float32` and `bfloat16`; results match CPU|
|270|torch.nn.utils.vector_to_parameters|Y|Native for `float32` and `bfloat16`; results match CPU|
|271|torch.nn.utils.prune.BasePruningMethod|Y|Host-only (no device ops)|
|272|torch.nn.utils.prune.BasePruningMethod.apply|Y|Native for `float32` and `bfloat16`; pruning results match CPU|
|273|torch.nn.utils.prune.BasePruningMethod.apply_mask|Y|Native for `float32` and `bfloat16`; masked results match CPU|
|274|torch.nn.utils.prune.BasePruningMethod.compute_mask|Y|Native|
|275|torch.nn.utils.prune.BasePruningMethod.prune|Y|Native for `float32` and `bfloat16`; pruning results match CPU|
|276|torch.nn.utils.prune.BasePruningMethod.remove|Y|Native for `float32` and `bfloat16`; removal results match CPU|
|277|torch.nn.utils.prune.PruningContainer|Y|Host-only (no device ops). Constructs a pruning-method container|
|278|torch.nn.utils.prune.PruningContainer.add_pruning_method|Y|Host-only (no device ops). Records a pruning method|
|279|torch.nn.utils.prune.PruningContainer.apply|N|Not a valid standalone `PruningContainer` operation|
|280|torch.nn.utils.prune.PruningContainer.apply_mask|Y|Native for `float32` and `bfloat16`; masked results match CPU|
|281|torch.nn.utils.prune.PruningContainer.compute_mask|CPU_FALLBACK|Indexed masks fall back (`aten::_to_copy`)|
|282|torch.nn.utils.prune.PruningContainer.prune|CPU_FALLBACK|Falls back (`aten::_to_copy`) for indexed-mask handling|
|283|torch.nn.utils.prune.PruningContainer.remove|Y|Native for `float32` and `bfloat16`; removal results match CPU|
|284|torch.nn.utils.prune.Identity|Y|Host-only (no device ops). Constructs an identity pruning method|
|285|torch.nn.utils.prune.Identity.apply|Y|Native for `float32` and `bfloat16`; pruning results match CPU|
|286|torch.nn.utils.prune.Identity.apply_mask|Y|Native for `float32` and `bfloat16`; masked results match CPU|
|287|torch.nn.utils.prune.Identity.prune|Y|Native for `float32` and `bfloat16`; pruning results match CPU|
|288|torch.nn.utils.prune.Identity.remove|Y|Native for `float32` and `bfloat16`; removal results match CPU|
|289|torch.nn.utils.prune.RandomUnstructured|Y|Host-only (no device ops). Constructs a random unstructured pruning method|
|290|torch.nn.utils.prune.RandomUnstructured.apply|Y|Native for `float32` and `bfloat16`; seeded same-device pruning invariants pass|
|291|torch.nn.utils.prune.RandomUnstructured.apply_mask|Y|Native for `float32` and `bfloat16`; masked results match CPU|
|292|torch.nn.utils.prune.RandomUnstructured.prune|Y|Native for `float32` and `bfloat16`; seeded same-device pruning invariants pass|
|293|torch.nn.utils.prune.RandomUnstructured.remove|Y|Native for `float32` and `bfloat16`; removal results match CPU|
|294|torch.nn.utils.prune.L1Unstructured|Y|Host-only (no device ops)|
|295|torch.nn.utils.prune.L1Unstructured.apply|Y|Native|
|296|torch.nn.utils.prune.L1Unstructured.apply_mask|Y|Native|
|297|torch.nn.utils.prune.L1Unstructured.prune|Y|Native|
|298|torch.nn.utils.prune.L1Unstructured.remove|Y|Native|
|299|torch.nn.utils.prune.RandomStructured|Y|Host-only (no device ops)|
|300|torch.nn.utils.prune.RandomStructured.apply|Y|Native for `float32` and `bfloat16`|
|301|torch.nn.utils.prune.RandomStructured.apply_mask|Y|Native|
|302|torch.nn.utils.prune.RandomStructured.compute_mask|Y|Native for `float32` and `bfloat16`|
|303|torch.nn.utils.prune.RandomStructured.prune|Y|Native for `float32` and `bfloat16`|
|304|torch.nn.utils.prune.RandomStructured.remove|Y|Native|
|305|torch.nn.utils.prune.LnStructured|Y|Host-only (no device ops)|
|306|torch.nn.utils.prune.LnStructured.apply|Y|Native|
|307|torch.nn.utils.prune.LnStructured.apply_mask|Y|Native|
|308|torch.nn.utils.prune.LnStructured.compute_mask|Y|Native|
|309|torch.nn.utils.prune.LnStructured.prune|Y|Native|
|310|torch.nn.utils.prune.LnStructured.remove|Y|Native|
|311|torch.nn.utils.prune.CustomFromMask|Y|Host-only (no device ops). Stores the supplied pruning mask|
|312|torch.nn.utils.prune.CustomFromMask.apply|Y|Native|
|313|torch.nn.utils.prune.CustomFromMask.apply_mask|Y|Native|
|314|torch.nn.utils.prune.CustomFromMask.prune|Y|Native|
|315|torch.nn.utils.prune.CustomFromMask.remove|Y|Native|
|316|torch.nn.utils.prune.identity|Y|Native|
|317|torch.nn.utils.prune.random_unstructured|Y|Native; random masks may differ from CPU for the same seed|
|318|torch.nn.utils.prune.l1_unstructured|Y|Native|
|319|torch.nn.utils.prune.random_structured|Y|Native for `float32` and `bfloat16`|
|320|torch.nn.utils.prune.ln_structured|Y|Native|
|321|torch.nn.utils.prune.global_unstructured|CPU_FALLBACK|Falls back (`aten::_to_copy`) during indexed-mask handling|
|322|torch.nn.utils.prune.custom_from_mask|Y|Native|
|323|torch.nn.utils.prune.remove|Y|Native|
|324|torch.nn.utils.prune.is_pruned|Y|Host-only (no device ops). Inspects module hooks for pruning reparameterizations|
|325|torch.nn.utils.weight_norm|Y|Native; registration falls back (`aten::norm.out`)|
|326|torch.nn.utils.remove_weight_norm|Y|Native. Registering weight normalization first falls back (`aten::norm.out`)|
|327|torch.nn.utils.spectral_norm|Y|Native|
|328|torch.nn.utils.remove_spectral_norm|Y|Native|
|329|torch.nn.utils.skip_init|Y|Native|
|330|torch.nn.utils.parametrizations.orthogonal|CPU_FALLBACK|Falls back (`aten::geqrf`, `aten::linalg_householder_product`, …); no `bfloat16`|
|331|torch.nn.utils.parametrizations.spectral_norm|CPU_FALLBACK|Falls back (`aten::vdot`)|
|332|torch.nn.utils.parametrize.register_parametrization|Y|Native|
|333|torch.nn.utils.parametrize.remove_parametrizations|Y|Native|
|334|torch.nn.utils.parametrize.cached|Y|Host-only (no device ops). Controls parametrization caching|
|335|torch.nn.utils.parametrize.is_parametrized|Y|Host-only (no device ops). Inspects module parametrization metadata|
|336|torch.nn.utils.parametrize.ParametrizationList|Y|Native|
|337|torch.nn.utils.parametrize.ParametrizationList.right_inverse|Y|Native|
|338|torch.nn.utils.stateless.functional_call|Y|Native. Support depends on the called module using supported tensor operations|
|339|torch.nn.utils.rnn.PackedSequence|Y|Host-only (no device ops)|
|340|torch.nn.utils.rnn.PackedSequence.batch_sizes|Y|Host-only (no device ops). Returns the stored batch-size metadata|
|341|torch.nn.utils.rnn.PackedSequence.count|Y|Host-only (no device ops). Uses the inherited tuple count operation|
|342|torch.nn.utils.rnn.PackedSequence.data|Y|Host-only (no device ops). Returns the stored packed tensor|
|343|torch.nn.utils.rnn.PackedSequence.index|Y|Host-only (no device ops). Uses the inherited tuple index operation|
|344|torch.nn.utils.rnn.PackedSequence.is_cuda|Y|Host-only (no device ops). Reports whether the packed tensor is on CUDA|
|345|torch.nn.utils.rnn.PackedSequence.is_pinned|Y|Host-only (no device ops). Reports whether the packed data is in pinned memory|
|346|torch.nn.utils.rnn.PackedSequence.sorted_indices|Y|Host-only (no device ops). Returns stored sorting metadata|
|347|torch.nn.utils.rnn.PackedSequence.to|Y|Native|
|348|torch.nn.utils.rnn.PackedSequence.unsorted_indices|Y|Host-only (no device ops). Returns stored inverse-sorting metadata|
|349|torch.nn.utils.rnn.pack_padded_sequence|Y|Native. The `lengths` argument must remain on CPU, as required by PyTorch|
|350|torch.nn.utils.rnn.pad_packed_sequence|Y|Native|
|351|torch.nn.utils.rnn.pad_sequence|Y|Native|
|352|torch.nn.utils.rnn.pack_sequence|Y|Native|
|353|torch.nn.utils.rnn.unpack_sequence|Y|Native|
|354|torch.nn.utils.rnn.unpad_sequence|Y|Native. The `lengths` argument remains on CPU|
|355|torch.nn.Flatten|Y|Native|
|356|torch.nn.Unflatten|Y|Native|
|357|torch.nn.modules.lazy.LazyModuleMixin|Y|Host-only (no device ops). Provides lazy-module initialization behavior|
|358|torch.nn.modules.lazy.LazyModuleMixin.has_uninitialized_params|Y|Host-only (no device ops). Inspects lazy-parameter metadata|
|359|torch.nn.modules.lazy.LazyModuleMixin.initialize_parameters|Y|Native|

## torch.nn.functional

||pytorch_api|neuron_compatible|limitations|
|---|---|---|---|
|1|F.conv1d|Y|Native|
|2|F.conv2d|Y|Native; non-contiguous inputs round-trip via CPU|
|3|F.conv3d|Y|Native|
|4|F.conv_transpose1d|Y|Native|
|5|F.conv_transpose2d|Y|Native|
|6|F.conv_transpose3d|Y|Native|
|7|F.unfold|Y|Native|
|8|F.fold|Y|Native; non-contiguous inputs round-trip via CPU|
|9|F.avg_pool1d|Y|Native|
|10|F.avg_pool2d|Y|Native|
|11|F.avg_pool3d|CPU_FALLBACK|Falls back (`aten::avg_pool3d.out`) for `float32`; no `bfloat16`|
|12|F.max_pool1d|CPU_FALLBACK|Falls back (`aten::max_pool2d_with_indices.out`)|
|13|F.max_pool2d|CPU_FALLBACK|Falls back (`aten::max_pool2d_with_indices.out`)|
|14|F.max_pool3d|CPU_FALLBACK|Falls back (`aten::max_pool3d_with_indices`)|
|15|F.max_unpool1d|CPU_FALLBACK|Falls back (`aten::max_unpool2d`)|
|16|F.max_unpool2d|CPU_FALLBACK|Falls back (`aten::max_unpool2d`)|
|17|F.max_unpool3d|CPU_FALLBACK|Falls back (`aten::max_unpool3d`)|
|18|F.lp_pool1d|Y|Native|
|19|F.lp_pool2d|Y|Native|
|20|F.adaptive_max_pool1d|CPU_FALLBACK|Falls back (`aten::adaptive_max_pool2d.out`)|
|21|F.adaptive_max_pool2d|CPU_FALLBACK|Falls back (`aten::adaptive_max_pool2d.out`)|
|22|F.adaptive_max_pool3d|CPU_FALLBACK|Falls back (`aten::adaptive_max_pool3d.out`)|
|23|F.adaptive_avg_pool1d|Y|Native; `bfloat16` can differ from CPU|
|24|F.adaptive_avg_pool2d|Y|Native|
|25|F.adaptive_avg_pool3d|CPU_FALLBACK|Falls back (`aten::_adaptive_avg_pool3d`)|
|26|F.fractional_max_pool2d|CPU_FALLBACK|Falls back (`aten::fractional_max_pool2d.output`)|
|27|F.fractional_max_pool3d|CPU_FALLBACK|Falls back (`aten::fractional_max_pool3d.output`)|
|28|F.scaled_dot_product_attention|Y|Native; non-contiguous Q/K/V round-trip via CPU|
|29|F.threshold|Y|Native|
|30|F.threshold_|Y|Native|
|31|F.relu|Y|Native|
|32|F.relu_|Y|Native|
|33|F.hardtanh|Y|Native|
|34|F.hardtanh_|Y|Native|
|35|F.hardswish|Y|Native|
|36|F.relu6|Y|Native|
|37|F.elu|Y|Native|
|38|F.elu_|Y|Native|
|39|F.selu|Y|Native|
|40|F.celu|Y|Native|
|41|F.leaky_relu|Y|Native. Backward falls back (`aten::leaky_relu_backward.grad_input`)|
|42|F.leaky_relu_|Y|Native. Backward falls back (`aten::leaky_relu_backward.grad_input`)|
|43|F.prelu|CPU_FALLBACK|Falls back (`aten::_prelu_kernel`)|
|44|F.rrelu|CPU_FALLBACK|Falls back (`aten::rrelu_with_noise`)|
|45|F.rrelu_|CPU_FALLBACK|Falls back (`aten::rrelu_with_noise_`)|
|46|F.glu|Y|Native|
|47|F.gelu|Y|Native|
|48|F.logsigmoid|Y|Native|
|49|F.hardshrink|Y|Native|
|50|F.tanhshrink|Y|Native|
|51|F.softsign|Y|Native|
|52|F.softplus|Y|Native|
|53|F.softmin|Y|Native without `dtype`; `float64` forward and backward fall back (`aten::_to_copy`)|
|54|F.softmax|Y|Native; `dtype=torch.float64` and its backward fall back (`aten::_to_copy`)|
|55|F.softshrink|CPU_FALLBACK|Falls back (`aten::softshrink.out`)|
|56|F.gumbel_softmax|CPU_FALLBACK|Falls back (`aten::exponential_`)|
|57|F.log_softmax|Y|Native|
|58|F.tanh|Y|Native|
|59|F.sigmoid|Y|Native|
|60|F.hardsigmoid|Y|Native|
|61|F.silu|Y|Native|
|62|F.mish|Y|Native|
|63|F.batch_norm|Y|Native, including rank-5 inputs and their backward pass|
|64|F.group_norm|Y|Native|
|65|F.instance_norm|Y|Native|
|66|F.layer_norm|Y|Native|
|67|F.local_response_norm|Y|Native rank 3; rank 4/5 falls back (`aten::avg_pool3d.out`), no `bfloat16`|
|68|F.normalize|Y|Native. Backward performs an internal dtype conversion on CPU (`aten::_to_copy`)|
|69|F.linear|Y|Native. `int64` inputs fall back for `aten::linear`|
|70|F.bilinear|Y|Native. `int64` inputs fall back for `aten::bmm`|
|71|F.dropout|Y|Native|
|72|F.alpha_dropout|CPU_FALLBACK|`training=True` falls back (`aten::bernoulli_.float`); `training=False` is a no-op|
|73|F.feature_alpha_dropout|CPU_FALLBACK|Training falls back (`aten::bernoulli_.float`); evaluation is a no-op|
|74|F.dropout1d|CPU_FALLBACK|Falls back (`aten::bernoulli_.float`) in training mode|
|75|F.dropout2d|CPU_FALLBACK|Falls back (`aten::bernoulli_.float`) in training mode|
|76|F.dropout3d|CPU_FALLBACK|Training falls back (`aten::bernoulli_.float`); non-contiguous uses CPU|
|77|F.embedding|Y|Native|
|78|F.embedding_bag|CPU_FALLBACK|Falls back (`aten::_embedding_bag_forward_only`)|
|79|F.one_hot|Y|Native|
|80|F.pairwise_distance|CPU_FALLBACK|Falls back (`aten::norm.out`)|
|81|F.cosine_similarity|Y|Native. Backward performs an internal dtype conversion on CPU (`aten::_to_copy`)|
|82|F.pdist|Y|Native. Backward falls back (`aten::_pdist_backward`)|
|83|F.binary_cross_entropy|CPU_FALLBACK|Falls back (`aten::binary_cross_entropy`)|
|84|F.binary_cross_entropy_with_logits|Y|Native|
|85|F.poisson_nll_loss|Y|Native; `log_input=False, full=True` falls back (`aten::_to_copy`)|
|86|F.cosine_embedding_loss|CPU_FALLBACK|Falls back (`aten::_to_copy`)|
|87|F.cross_entropy|Y|Native|
|88|F.ctc_loss|CPU_FALLBACK|Falls back (`aten::_ctc_loss`)|
|89|F.gaussian_nll_loss|CPU_FALLBACK|Falls back (`aten::_to_copy`)|
|90|F.hinge_embedding_loss|CPU_FALLBACK|Falls back (`aten::_to_copy`)|
|91|F.kl_div|CPU_FALLBACK|`log_target=False` falls back (`aten::xlogy.OutTensor`); `log_target=True` is native|
|92|F.l1_loss|Y|Native|
|93|F.mse_loss|Y|Native|
|94|F.margin_ranking_loss|Y|Native. Backward performs an internal dtype conversion on CPU (`aten::_to_copy`)|
|95|F.multilabel_margin_loss|CPU_FALLBACK|Falls back (`aten::multilabel_margin_loss_forward`) for `float32`; no `bfloat16`|
|96|F.multilabel_soft_margin_loss|Y|Native|
|97|F.multi_margin_loss|CPU_FALLBACK|Falls back (`aten::multi_margin_loss`) for `float32`; no `bfloat16`|
|98|F.nll_loss|Y|Native for 2D; 3D falls back (`aten::nll_loss2d_forward`)|
|99|F.huber_loss|CPU_FALLBACK|Falls back (`aten::huber_loss`)|
|100|F.smooth_l1_loss|CPU_FALLBACK|Falls back (`aten::smooth_l1_loss.out`)|
|101|F.soft_margin_loss|Y|Native|
|102|F.triplet_margin_loss|CPU_FALLBACK|Falls back (`aten::norm.out`)|
|103|F.triplet_margin_with_distance_loss|CPU_FALLBACK|Falls back (`aten::norm.out`)|
|104|F.pixel_shuffle|Y|Native|
|105|F.pixel_unshuffle|Y|Native|
|106|F.pad|Y|Native; `replicate` backward falls back (`aten::replication_pad1d_backward`, …)|
|107|F.interpolate|Y|Native; 3D nearest-neighbor falls back (`aten::upsample_nearest3d.out`)|
|108|F.upsample|Y|Native; 3D nearest falls back (`aten::upsample_nearest3d.out`)|
|109|F.upsample_nearest|Y|Native; 5D inputs fall back (`aten::upsample_nearest3d.out`)|
|110|F.upsample_bilinear|Y|Native; non-contiguous inputs use CPU (`aten::upsample_bilinear2d`)|
|111|F.grid_sample|Y|Native|
|112|F.affine_grid|CPU_FALLBACK|Falls back for a noncontiguous `aten::copy_` scatter while constructing the grid|

## torch.Tensor

||pytorch_api|neuron_compatible|limitations|
|---|---|---|---|
|1|Tensor.T|Y|Native|
|2|Tensor.H|Y|Native|
|3|Tensor.mT|Y|Native|
|4|Tensor.mH|Y|Native|
|5|Tensor.new_tensor|Y|Native|
|6|Tensor.new_full|Y|Native|
|7|Tensor.new_empty|Y|Native|
|8|Tensor.new_ones|Y|Native|
|9|Tensor.new_zeros|Y|Native|
|10|Tensor.is_cuda|Y|Host-only (no device ops). Returns backend metadata|
|11|Tensor.is_quantized|Y|Host-only (no device ops). Returns quantization metadata|
|12|Tensor.is_meta|Y|Host-only (no device ops). Returns backend metadata|
|13|Tensor.device|Y|Host-only (no device ops). Returns device metadata|
|14|Tensor.grad|Y|Host-only (no device ops)|
|15|Tensor.ndim|Y|Host-only (no device ops). Returns shape metadata|
|16|Tensor.real|Y|Native|
|17|Tensor.imag|Y|Native|
|18|Tensor.nbytes|Y|Host-only (no device ops). Computes storage size from tensor metadata|
|19|Tensor.itemsize|Y|Host-only (no device ops). Returns dtype metadata|
|20|Tensor.abs|Y|Native|
|21|Tensor.abs_|Y|Native|
|22|Tensor.absolute|Y|Native|
|23|Tensor.absolute_|Y|Native|
|24|Tensor.acos|Y|Native|
|25|Tensor.acos_|Y|Native; `bool` and `int64` are rejected by in-place casting rules|
|26|Tensor.arccos|Y|Native|
|27|Tensor.arccos_|Y|Native; `bool` and `int64` are rejected by in-place casting rules|
|28|Tensor.add|Y|Native|
|29|Tensor.add_|Y|Native|
|30|Tensor.addbmm|CPU_FALLBACK|Falls back (`aten::addbmm`)|
|31|Tensor.addbmm_|CPU_FALLBACK|Falls back (`aten::addbmm_`)|
|32|Tensor.addcdiv|Y|Native|
|33|Tensor.addcdiv_|Y|Native|
|34|Tensor.addcmul|Y|Native|
|35|Tensor.addcmul_|Y|Native|
|36|Tensor.addmm|Y|Native. `int64` inputs fall back for `aten::addmm`|
|37|Tensor.addmm_|Y|Native. `int64` inputs fall back for `aten::addmm.out`|
|38|Tensor.sspaddmm|N|Requires sparse COO tensors, which Neuron does not support|
|39|Tensor.addmv|Y|Native|
|40|Tensor.addmv_|Y|Native|
|41|Tensor.addr|Y|Native|
|42|Tensor.addr_|Y|Native|
|43|Tensor.adjoint|Y|Native|
|44|Tensor.allclose|Y|Native|
|45|Tensor.amax|Y|Native|
|46|Tensor.amin|Y|Native|
|47|Tensor.aminmax|CPU_FALLBACK|Falls back (`aten::aminmax.out`)|
|48|Tensor.angle|Y|Native|
|49|Tensor.apply_|N|Works only on CPU tensors|
|50|Tensor.argmax|Y|Native|
|51|Tensor.argmin|Y|Native|
|52|Tensor.argsort|CPU_FALLBACK|Falls back (`aten::sort.values_stable`); `stable=True` is native|
|53|Tensor.argwhere|Y|Native|
|54|Tensor.asin|Y|Native|
|55|Tensor.asin_|Y|Native; `bool` and `int64` inputs rejected by result-casting rules|
|56|Tensor.arcsin|Y|Native|
|57|Tensor.arcsin_|Y|Native; `bool` and `int64` are rejected by in-place casting rules|
|58|Tensor.as_strided|Y|Native|
|59|Tensor.atan|Y|Native|
|60|Tensor.atan_|Y|Native; `bool` and `int64` rejected by in-place casting rules|
|61|Tensor.arctan|Y|Native|
|62|Tensor.arctan_|Y|Native; `bool` and `int64` inputs rejected by result-casting rules|
|63|Tensor.atan2|CPU_FALLBACK|Falls back (`aten::atan2.out`)|
|64|Tensor.atan2_|CPU_FALLBACK|Falls back (`aten::atan2.out`)|
|65|Tensor.arctan2|CPU_FALLBACK|Falls back (`aten::atan2.out`)|
|66|Tensor.arctan2_|CPU_FALLBACK|Falls back (`aten::atan2.out`)|
|67|Tensor.all|Y|Native|
|68|Tensor.any|Y|Native|
|69|Tensor.backward|Y|Native|
|70|Tensor.baddbmm|Y|Native|
|71|Tensor.baddbmm_|Y|Native|
|72|Tensor.bernoulli|CPU_FALLBACK|Falls back (`aten::bernoulli_.Tensor`)|
|73|Tensor.bernoulli_|CPU_FALLBACK|Falls back (`aten::bernoulli_.float`)|
|74|Tensor.bfloat16|Y|Native|
|75|Tensor.bincount|CPU_FALLBACK|Falls back (`aten::bincount`)|
|76|Tensor.bitwise_not|Y|Native|
|77|Tensor.bitwise_not_|Y|Native|
|78|Tensor.bitwise_and|Y|Native|
|79|Tensor.bitwise_and_|Y|Native|
|80|Tensor.bitwise_or|Y|Native|
|81|Tensor.bitwise_or_|Y|Native|
|82|Tensor.bitwise_xor|Y|Native|
|83|Tensor.bitwise_xor_|Y|Native|
|84|Tensor.bitwise_left_shift|Y|Native|
|85|Tensor.bitwise_left_shift_|Y|Native|
|86|Tensor.bitwise_right_shift|Y|Native|
|87|Tensor.bitwise_right_shift_|Y|Native|
|88|Tensor.bmm|Y|Native. `int64` inputs fall back (`aten::bmm`)|
|89|Tensor.bool|Y|Native|
|90|Tensor.byte|Y|Native|
|91|Tensor.broadcast_to|Y|Native|
|92|Tensor.cauchy_|CPU_FALLBACK|Falls back (`aten::cauchy_`)|
|93|Tensor.ceil|Y|Native|
|94|Tensor.ceil_|Y|Native|
|95|Tensor.char|Y|Native|
|96|Tensor.cholesky|CPU_FALLBACK|Falls back (`aten::cholesky`)|
|97|Tensor.cholesky_inverse|CPU_FALLBACK|Falls back (`aten::cholesky_inverse`) for `float32`; no `bfloat16`|
|98|Tensor.cholesky_solve|CPU_FALLBACK|`float32` falls back (`aten::_cholesky_solve_helper`); no `bfloat16`|
|99|Tensor.chunk|Y|Native|
|100|Tensor.clamp|Y|Native|
|101|Tensor.clamp_|Y|Native|
|102|Tensor.clip|Y|Native|
|103|Tensor.clip_|Y|Native|
|104|Tensor.clone|Y|Native|
|105|Tensor.contiguous|Y|Native|
|106|Tensor.copy_|Y|Native|
|107|Tensor.conj|Y|Native, including the tested `complex64` inputs|
|108|Tensor.conj_physical|CPU_FALLBACK|Falls back (`aten::conj_physical.out`) for `complex64`; real inputs native|
|109|Tensor.conj_physical_|N|`complex64` input raises because the Neuron dispatch kernel is unavailable|
|110|Tensor.resolve_conj|Y|Native, including the tested `complex64` conjugate views|
|111|Tensor.resolve_neg|N|`float32` negative-bit results differ from CPU|
|112|Tensor.copysign|CPU_FALLBACK|Falls back (`aten::copysign.out`)|
|113|Tensor.copysign_|CPU_FALLBACK|Falls back (`aten::copysign.out`); `bool`/`int64` raise cast errors|
|114|Tensor.cos|Y|Native|
|115|Tensor.cos_|Y|Native. `bool` and `int64` inputs raise result-cast errors|
|116|Tensor.cosh|Y|Native|
|117|Tensor.cosh_|Y|Native. `bool` and `int64` inputs raise result-cast errors|
|118|Tensor.corrcoef|Y|Native|
|119|Tensor.count_nonzero|CPU_FALLBACK|Falls back (`aten::count_nonzero.dim_IntList`)|
|120|Tensor.cov|Y|Native|
|121|Tensor.acosh|Y|Native|
|122|Tensor.acosh_|Y|Native. `bool` and `int64` inputs raise result-cast errors|
|123|Tensor.arccosh|Y|Native|
|124|Tensor.arccosh_|Y|Native. `bool` and `int64` inputs raise result-cast errors|
|125|Tensor.cpu|Y|Native. Copies the tensor to CPU|
|126|Tensor.cross|CPU_FALLBACK|Falls back (`aten::linalg_cross.out`)|
|127|Tensor.cuda|N|CUDA-only|
|128|Tensor.logcumsumexp|CPU_FALLBACK|Falls back (`aten::_logcumsumexp`)|
|129|Tensor.cummax|Y|Native|
|130|Tensor.cummin|Y|Native|
|131|Tensor.cumprod|CPU_FALLBACK|Falls back (`aten::cumprod.out`)|
|132|Tensor.cumprod_|CPU_FALLBACK|Falls back (`aten::cumprod.out`)|
|133|Tensor.cumsum|Y|Native|
|134|Tensor.cumsum_|Y|Native|
|135|Tensor.chalf|Y|Native|
|136|Tensor.cfloat|Y|Native|
|137|Tensor.cdouble|Y|Native|
|138|Tensor.data_ptr|Y|Host-only (no device ops)|
|139|Tensor.deg2rad|Y|Native. `bool` and `int64` inputs fall back (`aten::_to_copy`)|
|140|Tensor.dequantize|CPU_FALLBACK|Falls back (`aten::dequantize.self`); quantized Neuron tensors unavailable|
|141|Tensor.det|CPU_FALLBACK|Falls back (`aten::_linalg_det.result`); no `bfloat16`|
|142|Tensor.dense_dim|Y|Host-only (no device ops)|
|143|Tensor.detach|Y|Host-only (no device ops)|
|144|Tensor.detach_|Y|Host-only (no device ops)|
|145|Tensor.diag|Y|Native; noncontiguous `int64` can fall back (`aten::copy_`)|
|146|Tensor.diag_embed|Y|Native; non-contiguous `int64` may fall back (`aten::copy_`)|
|147|Tensor.diagflat|Y|Native; non-contiguous `int64` may fall back (`aten::copy_`)|
|148|Tensor.diagonal|Y|Native|
|149|Tensor.diagonal_scatter|Y|Native; non-contiguous `int64` copies may fall back (`aten::copy_`)|
|150|Tensor.fill_diagonal_|Y|Native|
|151|Tensor.fmax|CPU_FALLBACK|Falls back (`aten::fmax.out`)|
|152|Tensor.fmin|CPU_FALLBACK|Falls back (`aten::fmin.out`)|
|153|Tensor.diff|Y|Native|
|154|Tensor.digamma|CPU_FALLBACK|Falls back (`aten::digamma.out`)|
|155|Tensor.digamma_|CPU_FALLBACK|Floating inputs fall back (`aten::digamma.out`); `bool`/`int64` cast errors|
|156|Tensor.dim|Y|Host-only (no device ops)|
|157|Tensor.dim_order|Y|Host-only (no device ops)|
|158|Tensor.dist|CPU_FALLBACK|Falls back (`aten::norm.out`)|
|159|Tensor.div|Y|Native|
|160|Tensor.div_|Y|Native. `bool` and `int64` inputs raise result-cast errors|
|161|Tensor.divide|Y|Native|
|162|Tensor.divide_|Y|Native. `bool` and `int64` inputs raise result-cast errors|
|163|Tensor.dot|Y|Native|
|164|Tensor.double|Y|Native|
|165|Tensor.dsplit|Y|Native|
|166|Tensor.element_size|Y|Host-only (no device ops)|
|167|Tensor.eq|Y|Native|
|168|Tensor.eq_|Y|Native. `int64` inputs fall back (`aten::_to_copy`)|
|169|Tensor.equal|Y|Native|
|170|Tensor.erf|Y|Native|
|171|Tensor.erf_|Y|Native. `bool` and `int64` inputs raise result-cast errors|
|172|Tensor.erfc|CPU_FALLBACK|Falls back (`aten::erfc.out`)|
|173|Tensor.erfc_|CPU_FALLBACK|Falls back (`aten::erfc.out`) for floats; `bool`/`int64` raise result-cast errors|
|174|Tensor.erfinv|Y|Native|
|175|Tensor.erfinv_|Y|Native. `bool` and `int64` inputs raise result-cast errors|
|176|Tensor.exp|Y|Native|
|177|Tensor.exp_|Y|Native. `bool` and `int64` inputs raise result-cast errors|
|178|Tensor.expm1|Y|Native|
|179|Tensor.expm1_|Y|Native. `bool` and `int64` inputs raise result-cast errors|
|180|Tensor.expand|Y|Native|
|181|Tensor.expand_as|Y|Native|
|182|Tensor.exponential_|CPU_FALLBACK|Falls back (`aten::exponential_`)|
|183|Tensor.fix|Y|Native|
|184|Tensor.fix_|Y|Native|
|185|Tensor.fill_|Y|Native|
|186|Tensor.flatten|Y|Native|
|187|Tensor.flip|Y|Native|
|188|Tensor.fliplr|Y|Native|
|189|Tensor.flipud|Y|Native|
|190|Tensor.float|Y|Native|
|191|Tensor.float_power|CPU_FALLBACK|Falls back (`aten::_to_copy`) for `float64` promotion|
|192|Tensor.float_power_|CPU_FALLBACK|`float64` falls back (`aten::_to_copy`); `float32`/`bfloat16`/`int64`/`bool` raise|
|193|Tensor.floor|Y|Native|
|194|Tensor.floor_|Y|Native|
|195|Tensor.floor_divide|Y|Native|
|196|Tensor.floor_divide_|CPU_FALLBACK|Falls back (`aten::floor_divide_.Tensor`)|
|197|Tensor.fmod|CPU_FALLBACK|Falls back (`aten::fmod.Tensor_out`)|
|198|Tensor.fmod_|CPU_FALLBACK|Falls back (`aten::fmod.Tensor_out`)|
|199|Tensor.frac|Y|Native|
|200|Tensor.frac_|Y|Native|
|201|Tensor.frexp|CPU_FALLBACK|Falls back (`aten::frexp.Tensor_out`)|
|202|Tensor.gather|Y|Native|
|203|Tensor.gcd|CPU_FALLBACK|Falls back (`aten::gcd.out`)|
|204|Tensor.gcd_|CPU_FALLBACK|Falls back (`aten::gcd.out`)|
|205|Tensor.ge|Y|Native|
|206|Tensor.ge_|Y|Native. `int64` inputs fall back (`aten::_to_copy`)|
|207|Tensor.greater_equal|Y|Native|
|208|Tensor.greater_equal_|Y|Native. `int64` inputs fall back (`aten::_to_copy`)|
|209|Tensor.geometric_|CPU_FALLBACK|Falls back (`aten::geometric_`)|
|210|Tensor.geqrf|CPU_FALLBACK|Falls back (`aten::geqrf`) for `float32`; no `bfloat16`|
|211|Tensor.ger|Y|Native|
|212|Tensor.get_device|Y|Host-only (no device ops)|
|213|Tensor.gt|Y|Native|
|214|Tensor.gt_|Y|Native. `int64` inputs fall back (`aten::_to_copy`)|
|215|Tensor.greater|Y|Native|
|216|Tensor.greater_|Y|Native for tensor floats; scalar or `int64` operands fall back (`aten::_to_copy`)|
|217|Tensor.half|Y|Native|
|218|Tensor.hardshrink|Y|Native|
|219|Tensor.heaviside|CPU_FALLBACK|Falls back (`aten::heaviside.out`)|
|220|Tensor.histc|Y|Native|
|221|Tensor.histogram|CPU_FALLBACK|Falls back (`aten::histogram.bin_ct` and `aten::histogram.bins_tensor`)|
|222|Tensor.hsplit|Y|Native|
|223|Tensor.hypot|CPU_FALLBACK|Falls back (`aten::hypot.out`)|
|224|Tensor.hypot_|CPU_FALLBACK|Falls back (`aten::hypot.out`)|
|225|Tensor.i0|CPU_FALLBACK|Falls back (`aten::i0.out`)|
|226|Tensor.i0_|CPU_FALLBACK|Falls back (`aten::i0.out`) for floats; `bool`/`int64` raise result-cast errors|
|227|Tensor.igamma|CPU_FALLBACK|Falls back (`aten::igamma.out`)|
|228|Tensor.igamma_|CPU_FALLBACK|Falls back (`aten::igamma.out`)|
|229|Tensor.igammac|CPU_FALLBACK|Falls back (`aten::igammac.out`)|
|230|Tensor.igammac_|CPU_FALLBACK|Falls back (`aten::igammac.out`)|
|231|Tensor.index_add_|Y|Native|
|232|Tensor.index_add|Y|Native|
|233|Tensor.index_copy_|Y|Native|
|234|Tensor.index_copy|Y|Native|
|235|Tensor.index_fill_|Y|Native|
|236|Tensor.index_fill|Y|Native|
|237|Tensor.index_put_|Y|Native; duplicate indices with `accumulate=False` may differ from CPU|
|238|Tensor.index_put|Y|Native; duplicate indices with `accumulate=False` may differ from CPU|
|239|Tensor.index_reduce_|CPU_FALLBACK|Falls back (`aten::index_reduce.out`)|
|240|Tensor.index_reduce|CPU_FALLBACK|Falls back (`aten::index_reduce.out`)|
|241|Tensor.index_select|Y|Native|
|242|Tensor.indices|N|Requires sparse COO tensors, which Neuron does not support|
|243|Tensor.inner|Y|Native|
|244|Tensor.int|Y|Native|
|245|Tensor.int_repr|N|Quantized tensors not supported on Neuron|
|246|Tensor.inverse|CPU_FALLBACK|Falls back (`aten::linalg_inv_ex.inverse`)|
|247|Tensor.isclose|Y|Native|
|248|Tensor.isfinite|Y|Native|
|249|Tensor.isinf|Y|Native|
|250|Tensor.isposinf|Y|Native|
|251|Tensor.isneginf|Y|Native|
|252|Tensor.isnan|Y|Native|
|253|Tensor.is_contiguous|Y|Host-only (no device ops)|
|254|Tensor.is_complex|Y|Host-only (no device ops)|
|255|Tensor.is_conj|Y|Host-only (no device ops)|
|256|Tensor.is_floating_point|Y|Host-only (no device ops)|
|257|Tensor.is_inference|Y|Host-only (no device ops)|
|258|Tensor.is_leaf|Y|Host-only (no device ops)|
|259|Tensor.is_pinned|Y|Host-only (no device ops)|
|260|Tensor.is_set_to|CPU_FALLBACK|Falls back (`aten::is_set_to`); the boolean result is returned on the host|
|261|Tensor.is_shared|Y|Host-only (no device ops)|
|262|Tensor.is_signed|Y|Host-only (no device ops)|
|263|Tensor.is_sparse|Y|Host-only (no device ops)|
|264|Tensor.istft|CPU_FALLBACK|Falls back (`aten::_fft_c2r` and `aten::unfold_backward`)|
|265|Tensor.isreal|Y|Native for real inputs; `complex64` falls back (`aten::_to_copy`)|
|266|Tensor.item|Y|Native|
|267|Tensor.kthvalue|CPU_FALLBACK|Falls back (`aten::kthvalue.values`)|
|268|Tensor.lcm|CPU_FALLBACK|Falls back (`aten::lcm.out`)|
|269|Tensor.lcm_|CPU_FALLBACK|Falls back (`aten::lcm.out`)|
|270|Tensor.ldexp|Y|Native|
|271|Tensor.ldexp_|Y|Native; `int64` falls back (`aten::_to_copy`) and succeeds unlike CPU|
|272|Tensor.le|Y|Native|
|273|Tensor.le_|Y|Native. `int64` inputs fall back (`aten::_to_copy`)|
|274|Tensor.less_equal|Y|Native|
|275|Tensor.less_equal_|Y|Native. `int64` inputs fall back (`aten::_to_copy`)|
|276|Tensor.lerp|Y|Native|
|277|Tensor.lerp_|Y|Native; non-contiguous inputs fall back (`aten::lerp_.Scalar`)|
|278|Tensor.lgamma|CPU_FALLBACK|Falls back (`aten::lgamma.out`)|
|279|Tensor.lgamma_|CPU_FALLBACK|Floating inputs fall back (`aten::lgamma.out`); `bool`/`int64` cast errors|
|280|Tensor.log|Y|Native|
|281|Tensor.log_|Y|Native. `bool` and `int64` inputs raise result-cast errors|
|282|Tensor.logdet|CPU_FALLBACK|Falls back (`aten::_linalg_slogdet.sign`) for `float32` inputs|
|283|Tensor.log10|Y|Native|
|284|Tensor.log10_|Y|Native. `bool` and `int64` inputs raise PyTorch result-cast errors|
|285|Tensor.log1p|Y|Native|
|286|Tensor.log1p_|Y|`int64` falls back (`aten::_to_copy`) and truncates; `bool` accepted unlike CPU|
|287|Tensor.log2|Y|Native|
|288|Tensor.log2_|Y|Native; `int64` falls back (`aten::_to_copy`) and truncates, `bool` succeeds unlike CPU|
|289|Tensor.log_normal_|CPU_FALLBACK|Falls back (`aten::log_normal_`) for `bfloat16` and `float32` inputs|
|290|Tensor.logaddexp|CPU_FALLBACK|Falls back (`aten::logaddexp.out`) for `bfloat16` and `float32` inputs|
|291|Tensor.logaddexp2|Y|Native|
|292|Tensor.logsumexp|Y|Native|
|293|Tensor.logical_and|Y|Native|
|294|Tensor.logical_and_|Y|Native. `int64` inputs fall back (`aten::_to_copy`)|
|295|Tensor.logical_not|Y|Native|
|296|Tensor.logical_not_|Y|Native. `int64` inputs fall back (`aten::_to_copy`)|
|297|Tensor.logical_or|Y|Native|
|298|Tensor.logical_or_|Y|Native. `int64` inputs fall back (`aten::_to_copy`)|
|299|Tensor.logical_xor|Y|Native|
|300|Tensor.logical_xor_|Y|Native. `int64` inputs fall back (`aten::_to_copy`)|
|301|Tensor.logit|Y|Native|
|302|Tensor.logit_|Y|Native. `bool` and `int64` inputs raise PyTorch result-cast errors|
|303|Tensor.long|Y|Native for `float32`, `bfloat16`, and `bool` inputs|
|304|Tensor.lt|Y|Native|
|305|Tensor.lt_|Y|Native. `int64` inputs fall back (`aten::_to_copy`)|
|306|Tensor.less|Y|Native|
|307|Tensor.less_|Y|Native; scalars and `int64` tensors fall back (`aten::_to_copy`)|
|308|Tensor.lu|CPU_FALLBACK|Falls back (`aten::linalg_lu_factor_ex.out`) for `float32` inputs|
|309|Tensor.lu_solve|CPU_FALLBACK|Falls back (`aten::linalg_lu_solve.out`) for `float32` inputs|
|310|Tensor.as_subclass|Y|Native|
|311|Tensor.map_|N|Works only on CPU tensors|
|312|Tensor.masked_scatter_|Y|Native|
|313|Tensor.masked_scatter|Y|Native|
|314|Tensor.masked_fill_|Y|Native|
|315|Tensor.masked_fill|Y|Native|
|316|Tensor.masked_select|Y|Native|
|317|Tensor.matmul|Y|Native|
|318|Tensor.matrix_power|Y|Native n≥0; negative `float32` falls back (`aten::linalg_inv_ex.inverse`), no `bfloat16`|
|319|Tensor.matrix_exp|CPU_FALLBACK|Falls back (`aten::linalg_matrix_exp`) for `bfloat16` and `float32` inputs|
|320|Tensor.max|Y|Native|
|321|Tensor.maximum|Y|Native|
|322|Tensor.mean|Y|Native|
|323|Tensor.module_load|Y|Native|
|324|Tensor.nanmean|CPU_FALLBACK|Falls back (`aten::nansum`) for `bfloat16` and `float32` inputs|
|325|Tensor.median|CPU_FALLBACK|Falls back (`aten::median`) for `bfloat16`, `float32`, and `int64` inputs|
|326|Tensor.nanmedian|CPU_FALLBACK|Falls back (`aten::nanmedian`, `aten::nanmedian.dim_values`)|
|327|Tensor.min|Y|Native|
|328|Tensor.minimum|Y|Native|
|329|Tensor.mm|Y|Native. `int64` inputs fall back (`aten::mm`)|
|330|Tensor.smm|N|Requires sparse tensors, which Neuron does not support|
|331|Tensor.mode|CPU_FALLBACK|Falls back (`aten::mode`)|
|332|Tensor.movedim|Y|Native|
|333|Tensor.moveaxis|Y|Native|
|334|Tensor.msort|CPU_FALLBACK|Falls back (`aten::sort.values_stable`)|
|335|Tensor.mul|Y|Native|
|336|Tensor.mul_|Y|Native|
|337|Tensor.multiply|Y|Native|
|338|Tensor.multiply_|Y|Native|
|339|Tensor.multinomial|CPU_FALLBACK|Falls back (`aten::multinomial`) for `bfloat16` and `float32` inputs|
|340|Tensor.mv|Y|Native|
|341|Tensor.mvlgamma|CPU_FALLBACK|Falls back (`aten::lgamma.out`) for `bfloat16`, `float32`, and `int64` inputs|
|342|Tensor.mvlgamma_|CPU_FALLBACK|`float32`/`bfloat16` fall back (`aten::mvlgamma.out`); `int64` falls back then raises|
|343|Tensor.nansum|CPU_FALLBACK|Falls back (`aten::nansum`)|
|344|Tensor.narrow|Y|Native|
|345|Tensor.narrow_copy|Y|Native|
|346|Tensor.ndimension|Y|Host-only (no device ops). Returns the tensor rank|
|347|Tensor.nan_to_num|CPU_FALLBACK|Falls back (`aten::nan_to_num.out`)|
|348|Tensor.nan_to_num_|CPU_FALLBACK|Falls back (`aten::nan_to_num.out`)|
|349|Tensor.ne|Y|Native|
|350|Tensor.ne_|Y|Native. `int64` inputs fall back (`aten::_to_copy`)|
|351|Tensor.not_equal|Y|Native|
|352|Tensor.not_equal_|Y|Native. `int64` inputs fall back (`aten::_to_copy`)|
|353|Tensor.neg|Y|Native|
|354|Tensor.neg_|Y|Native|
|355|Tensor.negative|Y|Native|
|356|Tensor.negative_|Y|Native|
|357|Tensor.nelement|Y|Host-only (no device ops). Returns the number of tensor elements|
|358|Tensor.nextafter|CPU_FALLBACK|Falls back (`aten::nextafter.out`) for `bfloat16` and `float32` inputs|
|359|Tensor.nextafter_|CPU_FALLBACK|Falls back (`aten::nextafter.out`) for `bfloat16` and `float32` inputs|
|360|Tensor.nonzero|Y|Native|
|361|Tensor.norm|Y|Native; nuclear norm falls back (`aten::_linalg_svd.U`), no `bfloat16`; backward falls back (`aten::_to_copy`)|
|362|Tensor.normal_|Y|Native|
|363|Tensor.numel|Y|Host-only (no device ops). Returns the number of tensor elements|
|364|Tensor.numpy|N|Works only on CPU tensors; call `Tensor.cpu()` first|
|365|Tensor.orgqr|CPU_FALLBACK|Falls back (`aten::linalg_householder_product`) for `float32` inputs|
|366|Tensor.ormqr|CPU_FALLBACK|Falls back (`aten::ormqr`) for `float32` inputs|
|367|Tensor.outer|Y|Native|
|368|Tensor.permute|Y|Native|
|369|Tensor.pin_memory|N|Pins host memory for CPU tensors|
|370|Tensor.pinverse|CPU_FALLBACK|Falls back (`aten::_linalg_svd.U` and `aten::_to_copy`) for `float32` inputs|
|371|Tensor.polygamma|CPU_FALLBACK|Falls back (`aten::polygamma.out`)|
|372|Tensor.polygamma_|CPU_FALLBACK|Floats fall back (`aten::polygamma.out`); `bool`/`int64` cast errors|
|373|Tensor.positive|Y|Native|
|374|Tensor.pow|Y|Native, including the tested backward passes|
|375|Tensor.pow_|Y|Native|
|376|Tensor.prod|Y|Native. Backward performs an internal dtype conversion on CPU (`aten::_to_copy`)|
|377|Tensor.put_|CPU_FALLBACK|Falls back (`aten::put_`)|
|378|Tensor.qr|CPU_FALLBACK|Falls back (`aten::linalg_qr.out`) for `float32` inputs|
|379|Tensor.qscheme|N|Quantized tensors not supported on Neuron|
|380|Tensor.quantile|CPU_FALLBACK|Falls back (`aten::sort.values_stable`) for `float32` inputs|
|381|Tensor.nanquantile|CPU_FALLBACK|Falls back (`aten::_to_copy`, `aten::sort.values_stable`); no `bfloat16`|
|382|Tensor.q_scale|N|Quantized tensors not supported on Neuron|
|383|Tensor.q_zero_point|N|Quantized tensors not supported on Neuron|
|384|Tensor.q_per_channel_scales|N|Quantized tensors not supported on Neuron|
|385|Tensor.q_per_channel_zero_points|N|Quantized tensors not supported on Neuron|
|386|Tensor.q_per_channel_axis|N|Quantized tensors not supported on Neuron|
|387|Tensor.rad2deg|Y|Native. `bool` and `int64` inputs fall back (`aten::_to_copy`)|
|388|Tensor.random_|Y|Native|
|389|Tensor.ravel|Y|Native|
|390|Tensor.reciprocal|Y|Native|
|391|Tensor.reciprocal_|Y|Native. `bool` and `int64` inputs raise PyTorch result-cast errors|
|392|Tensor.record_stream|Y|Native|
|393|Tensor.register_hook|Y|Host-only (no device ops)|
|394|Tensor.register_post_accumulate_grad_hook|Y|Host-only (no device ops)|
|395|Tensor.remainder|Y|Native|
|396|Tensor.remainder_|Y|Native|
|397|Tensor.renorm|CPU_FALLBACK|Falls back (`aten::renorm.out`) for `bfloat16` and `float32` inputs|
|398|Tensor.renorm_|CPU_FALLBACK|Falls back (`aten::renorm.out`) for `bfloat16` and `float32` inputs|
|399|Tensor.repeat|Y|Native|
|400|Tensor.repeat_interleave|Y|Native|
|401|Tensor.requires_grad|Y|Host-only (no device ops)|
|402|Tensor.requires_grad_|Y|Host-only (no device ops). Updates autograd metadata|
|403|Tensor.reshape|Y|Native|
|404|Tensor.reshape_as|Y|Native|
|405|Tensor.resize_|Y|Native|
|406|Tensor.resize_as_|Y|Native|
|407|Tensor.retain_grad|Y|Host-only (no device ops)|
|408|Tensor.retains_grad|Y|Host-only (no device ops). Returns autograd metadata|
|409|Tensor.roll|Y|Native|
|410|Tensor.rot90|Y|Native|
|411|Tensor.round|Y|Native|
|412|Tensor.round_|Y|Native|
|413|Tensor.rsqrt|Y|Native; `bool` false returns finite max instead of infinity|
|414|Tensor.rsqrt_|Y|Native. `bool` and `int64` inputs raise PyTorch result-cast errors|
|415|Tensor.scatter|Y|Native|
|416|Tensor.scatter_|Y|Native|
|417|Tensor.scatter_add_|Y|Native; large `float32` and `bfloat16` cases can differ from CPU|
|418|Tensor.scatter_add|Y|Native; colliding `bfloat16` indices can differ from CPU|
|419|Tensor.scatter_reduce_|CPU_FALLBACK|Falls back (`aten::scatter_reduce.two_out`)|
|420|Tensor.scatter_reduce|CPU_FALLBACK|Falls back (`aten::scatter_reduce.two_out`)|
|421|Tensor.select|Y|Native|
|422|Tensor.select_scatter|Y|Native. `int64` inputs fall back (`aten::copy_`)|
|423|Tensor.set_|Y|Native|
|424|Tensor.share_memory_|Y|Host-only (no device ops). Shared-memory management applies to CPU storage|
|425|Tensor.short|Y|Native|
|426|Tensor.sigmoid|Y|Native|
|427|Tensor.sigmoid_|Y|Native. `bool` and `int64` inputs raise PyTorch result-cast errors|
|428|Tensor.sign|Y|Native|
|429|Tensor.sign_|Y|Native|
|430|Tensor.signbit|Y|Native|
|431|Tensor.sgn|Y|Native. Backward falls back (`aten::_efficientzerotensor`)|
|432|Tensor.sgn_|Y|Native. Backward falls back (`aten::_efficientzerotensor`)|
|433|Tensor.sin|Y|Native|
|434|Tensor.sin_|Y|Native. `bool` and `int64` inputs raise PyTorch result-cast errors|
|435|Tensor.sinc|CPU_FALLBACK|Falls back (`aten::sinc.out`)|
|436|Tensor.sinc_|CPU_FALLBACK|`float32`/`bfloat16` fall back (`aten::sinc.out`); `bool`/`int64` raise first|
|437|Tensor.sinh|Y|Native|
|438|Tensor.sinh_|Y|Native. `bool` and `int64` inputs raise PyTorch result-cast errors|
|439|Tensor.asinh|Y|Native|
|440|Tensor.asinh_|Y|Native. `bool` and `int64` inputs raise PyTorch result-cast errors|
|441|Tensor.arcsinh|Y|Native|
|442|Tensor.arcsinh_|Y|Native. `bool` and `int64` inputs raise PyTorch result-cast errors|
|443|Tensor.shape|Y|Host-only (no device ops). Returns tensor shape metadata|
|444|Tensor.size|Y|Host-only (no device ops). Returns tensor size metadata|
|445|Tensor.slogdet|CPU_FALLBACK|Falls back (`aten::_linalg_slogdet.sign`) for `float32`; no `bfloat16`|
|446|Tensor.slice_scatter|Y|Native|
|447|Tensor.softmax|Y|Base `float32` native; `float64` output and backward fall back (`aten::_to_copy`)|
|448|Tensor.sort|CPU_FALLBACK|Falls back (`aten::sort.values_stable`)|
|449|Tensor.split|Y|Native|
|450|Tensor.sparse_mask|N|Requires sparse tensors, which Neuron does not support|
|451|Tensor.sparse_dim|Y|Host-only (no device ops). Returns tensor layout metadata|
|452|Tensor.sqrt|Y|Native|
|453|Tensor.sqrt_|Y|Native. `bool` and `int64` inputs raise PyTorch result-cast errors|
|454|Tensor.square|Y|Native|
|455|Tensor.square_|Y|Native. `bool` input raises a PyTorch result-cast error|
|456|Tensor.squeeze|Y|Native|
|457|Tensor.squeeze_|Y|Native|
|458|Tensor.std|Y|Native. Backward performs an internal dtype conversion on CPU (`aten::_to_copy`)|
|459|Tensor.stft|CPU_FALLBACK|Falls back (`aten::_fft_r2c`); no `bfloat16`|
|460|Tensor.storage|Y|Host-only (no device ops)|
|461|Tensor.untyped_storage|Y|Host-only (no device ops)|
|462|Tensor.storage_offset|Y|Host-only (no device ops). Returns storage-offset metadata|
|463|Tensor.storage_type|Y|Host-only (no device ops). This storage query is deprecated|
|464|Tensor.stride|Y|Host-only (no device ops). Returns stride metadata|
|465|Tensor.sub|Y|Native|
|466|Tensor.sub_|Y|Native|
|467|Tensor.subtract|Y|Native|
|468|Tensor.subtract_|Y|Native|
|469|Tensor.sum|Y|Native; large `bfloat16` reductions can differ from CPU|
|470|Tensor.sum_to_size|Y|Native|
|471|Tensor.svd|CPU_FALLBACK|Falls back (`aten::_linalg_svd.U`) for `float32` inputs|
|472|Tensor.swapaxes|Y|Native|
|473|Tensor.swapdims|Y|Native|
|474|Tensor.t|Y|Native|
|475|Tensor.t_|Y|Native|
|476|Tensor.tensor_split|Y|Native. Tensor-valued split indices must remain on CPU|
|477|Tensor.tile|Y|Native|
|478|Tensor.to|Y|Native|
|479|Tensor.to_mkldnn|N|Converts to the CPU MKL-DNN layout|
|480|Tensor.take|CPU_FALLBACK|Falls back (`aten::take`)|
|481|Tensor.take_along_dim|Y|Native|
|482|Tensor.tan|Y|Native; `float32` may differ from CPU near tangent poles|
|483|Tensor.tan_|Y|Native; near poles `float32` can differ from CPU|
|484|Tensor.tanh|Y|Native|
|485|Tensor.tanh_|Y|Native. `bool` and `int64` inputs raise PyTorch result-cast errors|
|486|Tensor.atanh|Y|Native|
|487|Tensor.atanh_|Y|Native; `bool` and integer inputs raise dtype-cast errors|
|488|Tensor.arctanh|Y|Native as an alias of `Tensor.atanh`|
|489|Tensor.arctanh_|Y|Native; `bool` and integer inputs raise dtype-cast errors|
|490|Tensor.tolist|Y|Native. Copies tensor values to a Python list|
|491|Tensor.topk|Y|Native; some `bfloat16` results differ from CPU|
|492|Tensor.to_dense|N|Sparse tensors unsupported; dense tensors are a no-op|
|493|Tensor.to_sparse|N|Sparse COO tensors not supported on Neuron; raises `NotImplementedError`|
|494|Tensor.to_sparse_csr|N|Sparse CSR unsupported; `aten::_to_sparse_csr` falls back then raises|
|495|Tensor.to_sparse_csc|N|Sparse CSC unsupported (`aten::_to_sparse_csc`); then raises|
|496|Tensor.to_sparse_bsr|N|Sparse BSR unsupported; falls back (`aten::_to_sparse_bsr`) then raises|
|497|Tensor.to_sparse_bsc|N|Sparse BSC unsupported; falls back (`aten::_to_sparse_bsc`) then raises|
|498|Tensor.trace|CPU_FALLBACK|Falls back (`aten::trace`)|
|499|Tensor.transpose|Y|Native as a metadata-only view|
|500|Tensor.transpose_|Y|Native as an in-place metadata update|
|501|Tensor.triangular_solve|CPU_FALLBACK|Falls back (`aten::triangular_solve.X`). Applies to the tested `float32` input|
|502|Tensor.tril|Y|Native|
|503|Tensor.tril_|Y|Native|
|504|Tensor.triu|Y|Native|
|505|Tensor.triu_|Y|Native|
|506|Tensor.true_divide|Y|Native|
|507|Tensor.true_divide_|Y|Native; `bool` and integer inputs raise dtype-cast errors|
|508|Tensor.trunc|Y|Native|
|509|Tensor.trunc_|Y|Native|
|510|Tensor.type|Y|Native; no-argument form is host-only|
|511|Tensor.type_as|Y|Native for dtype conversion|
|512|Tensor.unbind|Y|Native and returns Neuron views|
|513|Tensor.unflatten|Y|Native as a metadata-only view|
|514|Tensor.unfold|Y|Output falls back (`aten::contiguous`); backward falls back (`aten::unfold_backward`)|
|515|Tensor.uniform_|Y|Native for `float32` and `bfloat16`. Uses seeded eager reproducibility behavior|
|516|Tensor.unique|CPU_FALLBACK|Falls back (`aten::_unique2`)|
|517|Tensor.unique_consecutive|CPU_FALLBACK|Falls back (`aten::unique_consecutive`)|
|518|Tensor.unsqueeze|Y|Native as a metadata-only view|
|519|Tensor.unsqueeze_|Y|Native as an in-place metadata update|
|520|Tensor.values|N|Requires sparse tensors, which Neuron does not support|
|521|Tensor.var|CPU_FALLBACK|Falls back (`aten::var.correction`) for `float32` and `bfloat16`|
|522|Tensor.vdot|CPU_FALLBACK|Falls back (`aten::vdot`) for `bfloat16`/`float32`/`int64`|
|523|Tensor.view|Y|Native|
|524|Tensor.view_as|Y|Native as a metadata-only view|
|525|Tensor.vsplit|Y|Native and returns Neuron slice views|
|526|Tensor.where|Y|Native|
|527|Tensor.xlogy|CPU_FALLBACK|Falls back (`aten::xlogy.OutTensor`)|
|528|Tensor.xlogy_|CPU_FALLBACK|Falls back (`aten::xlogy.OutTensor`); `bool`/integers raise cast errors|
|529|Tensor.xpu|N|Intel XPU-specific|
|530|Tensor.zero_|Y|Native|

## Tensor Attributes

||pytorch_api|neuron_compatible|limitations|
|---|---|---|---|
|1|torch.dtype|Y|Host-only (no device ops)|
|2|torch.device|Y|Host-only (no device ops)|
|3|torch.layout|Y|Host-only (no device ops)|
|4|torch.memory_format|Y|Host-only (no device ops)|

## torch.autograd

||pytorch_api|neuron_compatible|limitations|
|---|---|---|---|
|1|torch.autograd.backward|Y|Native|
|2|torch.autograd.grad|Y|Native|
|3|torch.no_grad|Y|Host-only (no device ops)|
|4|torch.enable_grad|Y|Host-only (no device ops)|
|5|torch.set_grad_enabled|Y|Host-only (no device ops)|
|6|torch.is_grad_enabled|Y|Host-only (no device ops). Reads thread-local autograd state|
|7|torch.inference_mode|Y|Host-only (no device ops); unlike CPU, Neuron outputs are not marked as inference tensors|
|8|torch.is_inference_mode_enabled|Y|Host-only (no device ops)|
|9|torch.autograd.grad_mode.set_multithreading_enabled|Y|Host-only (no device ops)|
|10|torch.autograd.Function|Y|Native|
|11|Function.forward|Y|Runs supported tensor operations natively on Neuron in custom autograd functions|
|12|Function.backward|Y|Native|
|13|FunctionCtx.save_for_backward|Y|Host-only (no device ops)|
|14|FunctionCtx.mark_dirty|Y|Host-only (no device ops)|
|15|FunctionCtx.mark_non_differentiable|Y|Host-only (no device ops)|
|16|FunctionCtx.set_materialize_grads|Y|Host-only (no device ops)|
|17|function.once_differentiable|Y|Host-only (no device ops)|
|18|Tensor.grad|Y|Host-only (no device ops)|
|19|Tensor.requires_grad|Y|Host-only (no device ops)|
|20|Tensor.is_leaf|Y|Host-only (no device ops)|
|21|Tensor.backward|Y|Native|
|22|Tensor.detach|Y|Host-only (no device ops)|
|23|Tensor.detach_|Y|Host-only (no device ops)|
|24|Tensor.register_hook|Y|Host-only (no device ops)|
|25|Tensor.register_post_accumulate_grad_hook|Y|Host-only (no device ops)|
|26|Tensor.retain_grad|Y|Host-only (no device ops)|
|27|torch.autograd.detect_anomaly|Y|Host-only (no device ops)|
|28|torch.autograd.set_detect_anomaly|Y|Host-only (no device ops)|
|29|graph.Node.name|Y|Host-only (no device ops)|
|30|graph.Node.metadata|Y|Host-only (no device ops)|
|31|graph.Node.next_functions|Y|Host-only (no device ops)|
|32|graph.Node.register_hook|Y|Host-only (no device ops)|
|33|graph.Node.register_prehook|Y|Host-only (no device ops)|
|34|graph.increment_version|Y|Host-only (no device ops)|
|35|graph.saved_tensors_hooks|Y|Host-only (no device ops)|
|36|graph.save_on_cpu|Y|Host-only (no device ops); intentionally offloads saved tensors to CPU|
|37|graph.disable_saved_tensors_hooks|Y|Host-only (no device ops)|
|38|graph.register_multi_grad_hook|Y|Host-only (no device ops)|
|39|graph.allow_mutation_on_saved_tensors|Y|Host-only (no device ops)|
|40|graph.GradientEdge|Y|Host-only (no device ops)|
|41|graph.get_gradient_edge|Y|Host-only (no device ops)|
|42|forward_ad.dual_level|Y|Host-only (no device ops)|
|43|forward_ad.make_dual|Y|Native|
|44|forward_ad.unpack_dual|Y|Native|
|45|functional.jacobian|Y|Native. Differentiated operations and their gradients must also be supported|
|46|functional.hessian|Y|Native|
|47|functional.vjp|Y|Native. Differentiated operations must also be supported|
|48|functional.jvp|Y|Native. Differentiated operations and required gradients must also be supported|
|49|functional.vhp|Y|Native|
|50|functional.hvp|Y|Native|
|51|profiler.profile|Y|Host-only (no device ops); does not record Neuron device timing|
|52|profiler.profile.export_chrome_trace|Y|Host-only (no device ops); Neuron timing appears only if profiled|
|53|profiler.profile.key_averages|Y|Host-only (no device ops); no Neuron device timing|
|54|profiler.profile.total_average|Y|Host-only (no device ops); no Neuron device timing|
|55|profiler.record_function|Y|Host-only (no device ops)|
|56|profiler.emit_nvtx|N|NVIDIA NVTX-specific instrumentation|
|57|profiler.emit_itt|Y|Host-only (no device ops)|
|58|profiler.load_nvprof|N|Reads NVIDIA nvprof output|
|59|profiler.parse_nvprof_trace|N|Parses NVIDIA nvprof output|
|60|graph.set_warn_on_accumulate_grad_stream_mismatch|Y|Host-only (no device ops)|
|61|torch.autograd.gradcheck|CPU_FALLBACK|Falls back (`aten::_to_copy`, `aten::copy_`); default `float64` check fails for `sin`|
|62|torch.autograd.gradgradcheck|CPU_FALLBACK|Falls back (`aten::_to_copy`, `aten::copy_`); `float64` `sin` check fails|

## torch.library

||pytorch_api|neuron_compatible|limitations|
|---|---|---|---|
|1|torch.library.Library|Y|Host-only (no device ops)|
|2|torch.library.Library.define|Y|Host-only (no device ops)|
|3|torch.library.Library.impl|Y|Host-only (no device ops)|
|4|torch.library.fallthrough_kernel|Y|Host-only (no device ops)|
|5|torch.library.Library.fallback|Y|Host-only (no device ops)|
|6|torch.library.define|Y|Host-only (no device ops)|
|7|torch.library.impl|Y|Host-only (no device ops)|
|8|torch.library.impl_abstract|Y|Host-only; deprecated alias—use `torch.library.register_fake`|
|9|torch.library.register_kernel|Y|Host-only (no device ops)|
|10|torch.library.register_autocast|Y|Host-only (no device ops)|
|11|torch.library.register_fake|Y|Host-only (no device ops)|
|12|torch.library.register_autograd|Y|Host-only (no device ops)|
|13|torch.library.register_torch_dispatch|Y|Host-only (no device ops)|
|14|torch.library.register_vmap|Y|Host-only (no device ops)|
|15|torch.library.get_ctx|Y|Host-only (no device ops)|
|16|torch.library.get_kernel|Y|Host-only (no device ops)|
|17|torch.library.custom_op|Y|Host-only (no device ops)|
|18|torch.library.infer_schema|Y|Host-only (no device ops)|
|19|torch.library.opcheck|N|Full check supports only CPU, CUDA and XPU; partial checks run on Neuron|
|20|torch.library.triton_op|N|Triton-specific|
|21|torch.library.wrap_triton|N|Triton-specific|

## torch.backends

||pytorch_api|neuron_compatible|limitations|
|---|---|---|---|
|1|torch.backends.cpu.get_cpu_capability|Y|Host-only (no device ops)|
|2|torch.backends.cuda.is_built|Y|Host-only (no device ops)|
|3|torch.backends.cuda.matmul.allow_tf32|N|Controls CUDA cuBLAS TF32 behavior|
|4|torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction|N|Controls CUDA cuBLAS FP16 reduction behavior|
|5|torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction_split_k|N|Reads the CUDA cuBLAS FP16 split-K reduction setting|
|6|torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction|N|Controls CUDA cuBLAS BF16 reduction behavior|
|7|torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction_split_k|N|Reads the CUDA cuBLAS BF16 split-K reduction setting|
|8|torch.backends.cuda.matmul.allow_fp16_accumulation|N|Controls CUDA cuBLAS FP16 accumulation|
|9|torch.backends.cuda.matmul.fp32_precision|N|Controls CUDA matrix-multiplication FP32 precision|
|10|torch.backends.cuda.cufft_plan_cache.size|N|Reads a CUDA cuFFT plan cache and is unavailable when CUDA is not built|
|11|torch.backends.cuda.cufft_plan_cache.max_size|N|Controls a CUDA cuFFT plan cache and is unavailable when CUDA is not built|
|12|torch.backends.cuda.cufft_plan_cache.clear|N|Clears a CUDA cuFFT plan cache and is unavailable when CUDA is not built|
|13|torch.backends.cuda.preferred_linalg_library|N|Controls CUDA cuSOLVER/MAGMA selection|
|14|torch.backends.cuda.SDPBackend|N|CUDA scaled-dot-product-attention backend enum; does not select Neuron kernels|
|15|torch.backends.cuda.cudnn_sdp_enabled|N|Reads a CUDA cuDNN attention setting|
|16|torch.backends.cuda.enable_cudnn_sdp|N|Changes a CUDA cuDNN attention setting|
|17|torch.backends.cuda.flash_sdp_enabled|N|Reads a CUDA Flash Attention setting|
|18|torch.backends.cuda.enable_flash_sdp|N|Changes a CUDA Flash Attention setting|
|19|torch.backends.cuda.mem_efficient_sdp_enabled|N|Reads a CUDA efficient-attention setting|
|20|torch.backends.cuda.enable_mem_efficient_sdp|N|Changes a CUDA efficient-attention setting|
|21|torch.backends.cuda.math_sdp_enabled|N|Reads a CUDA math-attention setting|
|22|torch.backends.cuda.enable_math_sdp|N|Changes a CUDA math-attention setting|
|23|torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp|N|Changes CUDA math-attention reduction behavior|
|24|torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed|N|Reads CUDA math-attention reduction behavior|
|25|torch.backends.cuda.sdp_kernel|N|CUDA-only; deprecated in favor of `torch.nn.attention.sdpa_kernel`|
|26|torch.backends.cudnn.version|Y|Host-only (no device ops)|
|27|torch.backends.cudnn.is_available|Y|Host-only (no device ops); returns `False`|
|28|torch.backends.cudnn.enabled|N|CuDNN-specific|
|29|torch.backends.cudnn.allow_tf32|N|CuDNN-specific|
|30|torch.backends.cudnn.deterministic|N|CuDNN-specific|
|31|torch.backends.cudnn.benchmark|N|CuDNN-specific|
|32|torch.backends.cudnn.benchmark_limit|N|CuDNN-specific|
|33|torch.backends.mps.is_available|Y|Host-only (no device ops)|
|34|torch.backends.mps.is_built|Y|Host-only (no device ops)|
|35|torch.backends.mkl.is_available|Y|Host-only (no device ops)|
|36|torch.backends.mkl.verbose|N|Controls Intel MKL logging for CPU operations|
|37|torch.backends.mkldnn.is_available|Y|Host-only (no device ops); returns `True`|
|38|torch.backends.mkldnn.verbose|N|Controls oneDNN logging for CPU operations|
|39|torch.backends.openmp.is_available|Y|Host-only (no device ops)|
|40|torch.backends.opt_einsum.is_available|Y|Host-only (no device ops)|
|41|torch.backends.opt_einsum.get_opt_einsum|Y|Host-only (no device ops); `opt_einsum` unavailable|
|42|torch.backends.opt_einsum.enabled|Y|Host-only (no device ops); resolves to `False`|
|43|torch.backends.opt_einsum.strategy|Y|Host-only (no device ops)|

## torch.distributed

||PyTorch API|Compatibility|Limitations|
|---|---|---|---|
|1|torch.distributed.is_available|Y|Host-only (no device ops)|
|2|torch.distributed.is_initialized|Y|Host-only (no device ops)|
|3|torch.distributed.is_mpi_available|Y|Host-only (no device ops); returns `False`|
|4|torch.distributed.is_nccl_available|Y|Host-only (no device ops)|
|5|torch.distributed.is_gloo_available|Y|Host-only (no device ops)|
|6|torch.distributed.is_ucc_available|Y|Host-only (no device ops)|
|7|torch.distributed.is_xccl_available|Y|Host-only (no device ops); returns `False`|
|8|torch.distributed.is_backend_available|Y|Host-only (no device ops)|
|9|torch.distributed.is_torchelastic_launched|Y|Host-only (no device ops)|
|10|torch.distributed.init_process_group|Y|Host-only (no device ops)|
|11|torch.distributed.destroy_process_group|Y|Host-only (no device ops). Needs an initialized process group|
|12|torch.distributed.Backend|Y|Host-only (no device ops); `Backend("neuron")` resolves to `neuron`|
|13|torch.distributed.Backend.register_backend|Y|Host-only (no device ops)|
|14|torch.distributed.BackendConfig|Y|Host-only (no device ops)|
|15|torch.distributed.get_backend|Y|Host-only (no device ops)|
|16|torch.distributed.get_backend_config|Y|Host-only (no device ops); returns `neuron:neuron`|
|17|torch.distributed.get_default_backend_for_device|Y|Host-only (no device ops)|
|18|torch.distributed.get_rank|Y|Host-only (no device ops)|
|19|torch.distributed.get_world_size|Y|Host-only (no device ops)|
|20|torch.distributed.get_group_rank|Y|Host-only (no device ops)|
|21|torch.distributed.get_global_rank|Y|Host-only (no device ops)|
|22|torch.distributed.get_process_group_ranks|Y|Host-only (no device ops)|
|23|torch.distributed.get_node_local_rank|Y|Host-only (no device ops)|
|24|torch.distributed.get_pg_count|Y|Host-only (no device ops)|
|25|torch.distributed.Store|Y|Host-only (no device ops)|
|26|torch.distributed.TCPStore|Y|Host-only (no device ops)|
|27|torch.distributed.HashStore|Y|Host-only (no device ops)|
|28|torch.distributed.FileStore|Y|Host-only (no device ops)|
|29|torch.distributed.PrefixStore|Y|Host-only (no device ops)|
|30|torch.distributed.Store.set|Y|Host-only (no device ops). Sets a key-value pair in the selected c10d store|
|31|torch.distributed.Store.get|Y|Host-only (no device ops). Reads a value from the selected c10d store|
|32|torch.distributed.Store.add|Y|Host-only (no device ops)|
|33|torch.distributed.Store.compare_set|Y|Host-only (no device ops)|
|34|torch.distributed.Store.wait|Y|Host-only (no device ops). Blocks until the requested store keys are available|
|35|torch.distributed.Store.num_keys|Y|Host-only (no device ops). Returns the number of keys in the selected c10d store|
|36|torch.distributed.Store.delete_key|Y|Host-only (no device ops)|
|37|torch.distributed.Store.set_timeout|Y|Host-only (no device ops). Updates the selected c10d store timeout|
|38|torch.distributed.ReduceOp|Y|Host-only (no device ops)|
|39|torch.distributed.reduce_op|Y|Host-only (no device ops)|
|40|torch.distributed.DistBackendError|Y|Host-only (no device ops)|
|41|torch.distributed.GroupMember|Y|Host-only (no device ops)|
|42|torch.distributed.GroupName|Y|Host-only (no device ops)|
|43|torch.distributed.ProcessGroup|Y|Host-only (no device ops). This is the public c10d process-group interface|
|44|torch.distributed.Work|Y|Host-only (no device ops)|
|45|torch.distributed.DebugLevel|Y|Host-only (no device ops). This is distributed-debug configuration metadata|
|46|torch.distributed.get_debug_level|Y|Host-only (no device ops). Returns the distributed-debug level|
|47|torch.distributed.default_pg_timeout|Y|Host-only (no device ops). This is the default process-group timeout value|
|48|torch.distributed.group|Y|Host-only (no device ops). Exposes the current `WORLD` process group|
|49|torch.distributed.supports_complex|Y|Host-only (no device ops)|
|50|torch.distributed.AllreduceOptions|Y|Host-only (no device ops). Configures an all-reduce operation|
|51|torch.distributed.AllreduceCoalescedOptions|Y|Host-only (no device ops). Configures a coalesced all-reduce operation|
|52|torch.distributed.AllToAllOptions|Y|Host-only (no device ops). Configures an all-to-all operation|
|53|torch.distributed.BarrierOptions|Y|Host-only (no device ops). Configures a barrier operation|
|54|torch.distributed.BroadcastOptions|Y|Host-only (no device ops). Configures a broadcast operation|
|55|torch.distributed.GatherOptions|Y|Host-only (no device ops). Configures a gather operation|
|56|torch.distributed.ReduceOptions|Y|Host-only (no device ops). Configures a reduce operation|
|57|torch.distributed.ReduceScatterOptions|Y|Host-only (no device ops). Configures a reduce-scatter operation|
|58|torch.distributed.ScatterOptions|Y|Host-only (no device ops). Configures a scatter operation|
|59|torch.distributed.all_reduce|Y|Native. Needs an initialized process group with all ranks participating|
|60|torch.distributed.all_reduce_coalesced|Y|Native; two-tensor MAX is a known failure|
|61|torch.distributed.broadcast|Y|Native|
|62|torch.distributed.all_gather|Y|Native. Needs an initialized process group with all ranks participating|
|63|torch.distributed.all_gather_into_tensor|Y|Native. Needs an initialized process group with all ranks participating|
|64|torch.distributed.all_gather_coalesced|Y|Native; `int64` falls back (`aten::_to_copy`)|
|65|torch.distributed.reduce|Y|Native. Needs an initialized process group with a coordinated destination rank|
|66|torch.distributed.reduce_scatter|Y|Native|
|67|torch.distributed.reduce_scatter_tensor|Y|Native; input size must be divisible by group size|
|68|torch.distributed.gather|Y|Native|
|69|torch.distributed.scatter|Y|Native|
|70|torch.distributed.all_to_all|Y|Native|
|71|torch.distributed.all_to_all_single|Y|Native|
|72|torch.distributed.barrier|Y|Native. Needs an initialized process group with all ranks participating|
|73|torch.distributed.broadcast_object_list|Y|Native; objects serialize on host, and `aten::_to_copy` is staging, not fallback|
|74|torch.distributed.all_gather_object|CPU_FALLBACK|Falls back (`aten::_to_copy`); object serialization is host-side|
|75|torch.distributed.gather_object|CPU_FALLBACK|Falls back (`aten::_to_copy`)|
|76|torch.distributed.scatter_object_list|CPU_FALLBACK|Falls back (`aten::_to_copy`)|
|77|torch.distributed.new_group|Y|Host-only (no device ops). Every process must create subgroups in the same order|
|78|torch.distributed.new_subgroups|Y|Host-only (no device ops)|
|79|torch.distributed.new_subgroups_by_enumeration|Y|Host-only (no device ops)|
|80|torch.distributed.split_group|N|The Neuron process-group backend does not support group splitting|
|81|torch.distributed.shrink_group|N|The Neuron process-group backend does not support group shrinking|
|82|torch.distributed.send|Y|Native; tensor must be contiguous; send to self only via `batch_isend_irecv`|
|83|torch.distributed.recv|Y|Native; any-source receive is not supported|
|84|torch.distributed.isend|Y|Native; posting `isend` and `irecv` to one peer at once hangs, use `batch_isend_irecv`|
|85|torch.distributed.irecv|Y|Native; no any-source receive; posting `isend` and `irecv` to one peer at once hangs, use `batch_isend_irecv`|
|86|torch.distributed.batch_isend_irecv|Y|Native|
|87|torch.distributed.P2POp|Y|Host-only (no device ops)|
|88|torch.distributed.send_object_list|Y|Native; objects serialize on host|
|89|torch.distributed.recv_object_list|Y|Native; any-source unsupported and deserialization is host-side|
|90|torch.distributed.monitored_barrier|N|Requires a CPU-capable process-group backend such as Gloo|

## torch.distributed.algorithms.join

||PyTorch API|Compatibility|Limitations|
|---|---|---|---|
|1|Join|Y|Native; dtype conversion falls back (`aten::_to_copy`)|
|2|Join.notify_join_context|Y|Native|
|3|Joinable|Y|Host-only (no device ops). Defines the abstract interface for joinable objects|
|4|Joinable.join_device|Y|Host-only (no device ops). Returns the device used by join collectives|
|5|Joinable.join_hook|Y|Host-only (no device ops)|
|6|Joinable.join_process_group|Y|Host-only (no device ops). Returns process-group metadata|
|7|JoinHook|Y|Host-only (no device ops)|
|8|JoinHook.main_hook|Y|Host-only (no device ops)|
|9|JoinHook.post_hook|Y|Host-only (no device ops)|

## torch.distributed.fsdp

||PyTorch API|Compatibility|Limitations|
|---|---|---|---|
|1|torch.distributed.fsdp.FullyShardedDataParallel|Y|Native. Performs FSDP1 parameter sharding and multi-rank collectives|
|2|torch.distributed.fsdp.FullyShardedDataParallel.forward|Y|Native|
|3|torch.distributed.fsdp.FullyShardedDataParallel.clip_grad_norm_|Y|Native|
|4|torch.distributed.fsdp.FullyShardedDataParallel.full_optim_state_dict|Y|Native; internal state copies fall back (`aten::_to_copy`)|
|5|torch.distributed.fsdp.FullyShardedDataParallel.optim_state_dict|Y|Native; internal state conversion falls back (`aten::_to_copy`)|
|6|torch.distributed.fsdp.FullyShardedDataParallel.register_comm_hook|Y|Host-only (no device ops)|
|7|torch.distributed.fsdp.FullyShardedDataParallel.sharded_optim_state_dict|Y|Native; optimizer-state copy falls back (`aten::_to_copy`)|
|8|torch.distributed.fsdp.FullyShardedDataParallel.summon_full_params|Y|Native. All-gathers full parameters on context entry and reshards them on exit|
|9|torch.distributed.fsdp.FullyShardedDataParallel.apply|Y|Native. Unshards full parameters before recursively applying the host callback|
|10|torch.distributed.fsdp.FullyShardedDataParallel.flatten_sharded_optim_state_dict|Y|Native. All-gathers sharded optimizer tensors before flattening them|
|11|torch.distributed.fsdp.FullyShardedDataParallel.optim_state_dict_to_load|Y|Native|
|12|torch.distributed.fsdp.FullyShardedDataParallel.scatter_full_optim_state_dict|Y|Native; state copies to Neuron fall back (`aten::_to_copy`)|
|13|torch.distributed.fsdp.FullyShardedDataParallel.shard_full_optim_state_dict|Y|Native|
|14|torch.distributed.fsdp.FullyShardedDataParallel.fsdp_modules|Y|Host-only (no device ops). Traverses the module tree and returns FSDP modules|
|15|torch.distributed.fsdp.FullyShardedDataParallel.get_state_dict_type|Y|Host-only (no device ops). Returns the FSDP state-dict configuration|
|16|torch.distributed.fsdp.FullyShardedDataParallel.module|Y|Host-only (no device ops). Returns the wrapped module|
|17|torch.distributed.fsdp.FullyShardedDataParallel.named_buffers|Y|Host-only (no device ops)|
|18|torch.distributed.fsdp.FullyShardedDataParallel.named_parameters|Y|Host-only (no device ops)|
|19|torch.distributed.fsdp.FullyShardedDataParallel.no_sync|Y|Host-only (no device ops)|
|20|torch.distributed.fsdp.FullyShardedDataParallel.rekey_optim_state_dict|Y|Host-only (no device ops)|
|21|torch.distributed.fsdp.FullyShardedDataParallel.set_state_dict_type|Y|Host-only (no device ops)|
|22|torch.distributed.fsdp.FullyShardedDataParallel.state_dict_type|Y|Host-only (no device ops)|
|23|BackwardPrefetch|Y|Host-only (no device ops). Defines backward-prefetch configuration values|
|24|ShardingStrategy|Y|Host-only (no device ops). Defines FSDP sharding-strategy configuration values|
|25|MixedPrecision|Y|Host-only (no device ops). Stores mixed-precision configuration|
|26|CPUOffload|Y|Host-only (no device ops)|
|27|StateDictType|Y|Host-only (no device ops). Defines state-dict type configuration values|
|28|OptimStateKeyType|Y|Host-only (no device ops). Defines optimizer-state key-type configuration values|
|29|StateDictConfig|Y|Host-only (no device ops). Stores state-dict configuration|
|30|FullStateDictConfig|Y|Host-only (no device ops). Stores full-state-dict configuration|
|31|ShardedStateDictConfig|Y|Host-only (no device ops). Stores sharded-state-dict configuration|
|32|LocalStateDictConfig|Y|Host-only (no device ops). Stores local-state-dict configuration|
|33|OptimStateDictConfig|Y|Host-only (no device ops). Defines optimizer state-dictionary configuration|
|34|FullOptimStateDictConfig|Y|Host-only (no device ops). Defines full optimizer state-dictionary configuration|
|35|ShardedOptimStateDictConfig|Y|Host-only (no device ops)|
|36|LocalOptimStateDictConfig|Y|Host-only (no device ops)|
|37|StateDictSettings|Y|Host-only (no device ops). Stores state-dictionary settings|
|38|fully_shard|Y|Native|
|39|FSDPModule|Y|Native|
|40|UnshardHandle|Y|Host-only (no device ops). Synchronizes an asynchronous FSDP unshard|
|41|MixedPrecisionPolicy|Y|Host-only (no device ops). Defines FSDP2 mixed-precision configuration|
|42|OffloadPolicy|Y|Host-only (no device ops). Defines FSDP2 offload configuration|
|43|CPUOffloadPolicy|Y|Host-only (no device ops)|
|44|register_fsdp_forward_method|Y|Host-only (no device ops)|
|45|share_comm_ctx|Y|Host-only (no device ops). Assigns a shared FSDP communication context|

## torch.distributed.optim

||PyTorch API|Compatibility|Limitations|
|---|---|---|---|
|1|DistributedOptimizer|Y|Native|
|2|DistributedOptimizer.step|Y|Native|
|3|PostLocalSGDOptimizer|Y|Host-only (no device ops)|
|4|PostLocalSGDOptimizer.load_state_dict|Y|Host-only (no device ops)|
|5|PostLocalSGDOptimizer.state_dict|Y|Host-only (no device ops). Returns the wrapped optimizer state and averager step|
|6|PostLocalSGDOptimizer.step|Y|Native|
|7|torch.distributed.optim.ZeroRedundancyOptimizer|Y|Native|
|8|torch.distributed.optim.ZeroRedundancyOptimizer.add_param_group|Y|Host-only (no device ops)|
|9|torch.distributed.optim.ZeroRedundancyOptimizer.consolidate_state_dict|Y|Native; optimizer state is intentionally materialized on CPU|
|10|torch.distributed.optim.ZeroRedundancyOptimizer.join_hook|Y|Host-only (no device ops)|
|11|torch.distributed.optim.ZeroRedundancyOptimizer.load_state_dict|Y|Native|
|12|torch.distributed.optim.ZeroRedundancyOptimizer.state_dict|Y|Host-only (no device ops)|
|13|torch.distributed.optim.ZeroRedundancyOptimizer.step|Y|Native|
|14|as_functional_optim|Y|Host-only (no device ops)|

## torch.distributed.tensor.parallel

||PyTorch API|Compatibility|Limitations|
|---|---|---|---|
|1|parallelize_module|Y|Native|
|2|ColwiseParallel|Y|Native|
|3|RowwiseParallel|Y|Native|
|4|SequenceParallel|Y|Native|
|5|ParallelStyle|Y|Host-only (no device ops). Defines the abstract tensor-parallel plan interface|
|6|PrepareModuleInput|Y|Native|
|7|PrepareModuleOutput|Y|Native|
|8|PrepareModuleInputOutput|Y|Native|
|9|loss_parallel|Y|Native; one backward schedule and multidimensional plain targets unsupported|

## torch.distributed.checkpoint

||PyTorch API|Compatibility|Limitations|
|---|---|---|---|
|1|save|Y|Native|
|2|load|Y|Native|
|3|save_state_dict|Y|Native; deprecated in favor of `save`|
|4|load_state_dict|Y|Native; deprecated, use `load`|
|5|async_save|Y|Native; distributed mode requires a CPU backend|
|6|CheckpointException|Y|Host-only (no device ops). Provides a Python exception type|
|7|load_sharded_optimizer_state_dict|Y|Native; optimizer-state copy falls back (`aten::_to_copy`)|
|8|torch.distributed.checkpoint.StorageWriter|Y|Host-only (no device ops). Defines the abstract checkpoint writer contract|
|9|torch.distributed.checkpoint.StorageReader|Y|Host-only (no device ops). Defines the abstract checkpoint reader contract|
|10|torch.distributed.checkpoint.SavePlanner|Y|Host-only (no device ops)|
|11|torch.distributed.checkpoint.LoadPlanner|Y|Host-only (no device ops)|
|12|torch.distributed.checkpoint.metadata.ChunkStorageMetadata|Y|Host-only (no device ops). Stores checkpoint chunk offsets and sizes|
|13|torch.distributed.checkpoint.metadata.TensorProperties|Y|Host-only (no device ops)|
|14|torch.distributed.checkpoint.metadata.TensorStorageMetadata|Y|Host-only (no device ops)|
|15|torch.distributed.checkpoint.metadata.BytesStorageMetadata|Y|Host-only (no device ops). Marks byte-oriented checkpoint entries|
|16|torch.distributed.checkpoint.metadata.StorageMeta|Y|Host-only (no device ops). Stores checkpoint, save, load, and module identifiers|
|17|torch.distributed.checkpoint.metadata.Metadata|Y|Host-only (no device ops)|
|18|torch.distributed.checkpoint.metadata.MetadataIndex|Y|Host-only (no device ops)|
|19|torch.distributed.checkpoint.planner.BytesIOWriteData|Y|Host-only (no device ops). Stores the byte count for a planned byte-stream write|
|20|torch.distributed.checkpoint.planner.TensorWriteData|Y|Host-only (no device ops)|
|21|torch.distributed.checkpoint.planner.WriteItem|Y|Host-only (no device ops)|
|22|torch.distributed.checkpoint.planner.ReadItem|Y|Host-only (no device ops)|
|23|torch.distributed.checkpoint.planner.SavePlan|Y|Host-only (no device ops)|
|24|torch.distributed.checkpoint.planner.LoadPlan|Y|Host-only (no device ops)|
|25|torch.distributed.checkpoint.storage.WriteResult|Y|Host-only (no device ops)|
|26|torch.distributed.checkpoint.DefaultSavePlanner|Y|Host-only (no device ops). Builds and deduplicates checkpoint save plans|
|27|torch.distributed.checkpoint.DefaultLoadPlanner|Y|Host-only (no device ops)|
|28|torch.distributed.checkpoint.FileSystemReader|Y|Host-only (no device ops)|
|29|torch.distributed.checkpoint.FileSystemWriter|Y|Host-only (no device ops)|
|30|torch.distributed.checkpoint.HuggingFaceStorageReader|Y|Host-only (no device ops); requires `safetensors`|
|31|torch.distributed.checkpoint.HuggingFaceStorageWriter|Y|Host-only (no device ops); requires `safetensors`|

## torch.distributions

||PyTorch API|Compatibility|Limitations|
|---|---|---|---|
|1|Distribution|Y|Host-only (no device ops)|
|2|ExponentialFamily|Y|Host-only (no device ops)|
|3|torch.distributions.Bernoulli|CPU_FALLBACK|Sampling falls back (`aten::bernoulli_.Tensor`); `log_prob` is native|
|4|torch.distributions.Beta|CPU_FALLBACK|Falls back: `aten::_sample_dirichlet`, `aten::lgamma.out`, …; no `bfloat16` sampling|
|5|torch.distributions.Binomial|CPU_FALLBACK|Sampling and `log_prob` fall back (`aten::binomial`, `aten::lgamma.out`); no `bfloat16` sampling|
|6|torch.distributions.Categorical|CPU_FALLBACK|Sampling falls back (`aten::multinomial`); `float32` `log_prob` native|
|7|torch.distributions.Cauchy|CPU_FALLBACK|`rsample` falls back (`aten::cauchy_`); `log_prob` is native|
|8|torch.distributions.Chi2|CPU_FALLBACK|Falls back: `aten::_standard_gamma`, `aten::lgamma.out`, …; no `bfloat16` sampling|
|9|torch.distributions.ContinuousBernoulli|Y|Native|
|10|torch.distributions.Dirichlet|CPU_FALLBACK|Falls back (`aten::_sample_dirichlet`, `aten::lgamma.out`, …); no `bfloat16` `rsample`|
|11|torch.distributions.Exponential|CPU_FALLBACK|`rsample` falls back (`aten::exponential_`); `log_prob` is native|
|12|torch.distributions.FisherSnedecor|CPU_FALLBACK|Falls back (`aten::_standard_gamma`, `aten::lgamma.out`); no `bfloat16` sampling|
|13|torch.distributions.Gamma|CPU_FALLBACK|Falls back (`aten::_standard_gamma`, `aten::lgamma.out`, …); no `bfloat16` `rsample`|
|14|torch.distributions.Geometric|CPU_FALLBACK|Sampling and `log_prob` falls back (`aten::_to_copy`) for scalar conversion|
|15|torch.distributions.Gumbel|Y|Native|
|16|torch.distributions.HalfCauchy|CPU_FALLBACK|`rsample` falls back (`aten::cauchy_`); `log_prob` falls back (`aten::_to_copy`)|
|17|torch.distributions.HalfNormal|CPU_FALLBACK|`rsample` is native; `float32` `log_prob` falls back (`aten::_to_copy`)|
|18|torch.distributions.Independent|Y|Native|
|19|torch.distributions.InverseGamma|CPU_FALLBACK|Falls back (`aten::_standard_gamma`, `aten::lgamma.out`, …); no `bfloat16` `rsample`|
|20|torch.distributions.Kumaraswamy|Y|Native|
|21|torch.distributions.Laplace|Y|Native|
|22|torch.distributions.LKJCholesky|CPU_FALLBACK|Falls back (`aten::_sample_dirichlet`, `aten::lgamma.out`, …); no `bfloat16` sampling|
|23|torch.distributions.LogNormal|Y|Native|
|24|torch.distributions.LogisticNormal|CPU_FALLBACK|`rsample` falls back (`aten::cumprod.out`); `log_prob` falls back (`aten::_to_copy`)|
|25|torch.distributions.LowRankMultivariateNormal|CPU_FALLBACK|Falls back (`aten::linalg_cholesky_ex.L`) for `float32`; no `bfloat16`|
|26|torch.distributions.MixtureSameFamily|CPU_FALLBACK|`sample` falls back (`aten::multinomial`, `aten::normal.Tensor_Tensor`); `log_prob` native|
|27|torch.distributions.Multinomial|CPU_FALLBACK|Sampling/log_prob fall back (`aten::multinomial`, `aten::_to_copy`, …)|
|28|torch.distributions.MultivariateNormal|Y|Native|
|29|torch.distributions.NegativeBinomial|CPU_FALLBACK|Sampling and `log_prob` fall back (`aten::_standard_gamma`, `aten::poisson`, …)|
|30|torch.distributions.Normal|Y|Native|
|31|torch.distributions.OneHotCategorical|CPU_FALLBACK|Sampling/log_prob fall back (`aten::multinomial`, `aten::_to_copy`)|
|32|torch.distributions.OneHotCategoricalStraightThrough|CPU_FALLBACK|Falls back: rsample (`aten::multinomial`, `aten::_to_copy`); log_prob (`aten::_to_copy`)|
|33|torch.distributions.Pareto|CPU_FALLBACK|`rsample` falls back (`aten::exponential_`); `float32` `log_prob` is native|
|34|torch.distributions.Poisson|CPU_FALLBACK|Sampling and `log_prob` fall back (`aten::poisson`, `aten::lgamma.out`, …)|
|35|torch.distributions.RelaxedBernoulli|Y|Native|
|36|torch.distributions.RelaxedOneHotCategorical|CPU_FALLBACK|`log_prob` falls back (`aten::_to_copy`, `aten::lgamma.out`); `rsample` is native|
|37|torch.distributions.StudentT|CPU_FALLBACK|Falls back (`aten::_standard_gamma`, `aten::lgamma.out`); no `bfloat16` sampling|
|38|torch.distributions.TransformedDistribution|Y|Native|
|39|torch.distributions.Uniform|Y|Native|
|40|torch.distributions.VonMises|Y|Native `log_prob`; `sample` may fall back (`aten::_to_copy`)|
|41|torch.distributions.Weibull|CPU_FALLBACK|Falls back (`aten::exponential_`)|
|42|torch.distributions.Wishart|CPU_FALLBACK|Falls back (`aten::_cholesky_solve_helper`, `aten::_linalg_slogdet.sign`, …); no `bfloat16`|
|43|torch.distributions.GeneralizedPareto|CPU_FALLBACK|`float32` `log_prob` falls back (`aten::_to_copy`); `bfloat16` has a dtype mismatch|
|44|torch.distributions.transforms.Transform|Y|Host-only (no device ops)|
|45|torch.distributions.transforms.identity_transform|Y|Host-only (no device ops)|
|46|AbsTransform|Y|Native|
|47|AffineTransform|Y|Native|
|48|ExpTransform|Y|Native|
|49|PowerTransform|Y|Native|
|50|ReshapeTransform|Y|`ReshapeTransform` changes tensor metadata without device computation|
|51|SigmoidTransform|Y|Native|
|52|SoftmaxTransform|Y|Native|
|53|SoftplusTransform|Y|Native|
|54|StickBreakingTransform|CPU_FALLBACK|`StickBreakingTransform` falls back (`aten::cumprod.out`)|
|55|TanhTransform|Y|Native|
|56|LowerCholeskyTransform|Y|Native|
|57|CorrCholeskyTransform|CPU_FALLBACK|`CorrCholeskyTransform` falls back (`aten::cumprod.out`)|
|58|PositiveDefiniteTransform|Y|Native|
|59|CatTransform|Y|Native|
|60|StackTransform|Y|Native|
|61|ComposeTransform|Y|Native|
|62|IndependentTransform|Y|Native|
|63|CumulativeDistributionTransform|Y|Native|
|64|constraints|Y|Host-only (no device ops)|
|65|constraint_registry.biject_to|Y|Host-only (no device ops)|
|66|constraint_registry.transform_to|Y|Host-only (no device ops)|
|67|kl_divergence|Y|Native; Bernoulli/Categorical pairs fall back (`aten::_to_copy`)|
|68|register_kl|Y|Host-only (no device ops)|
## torch.futures

||PyTorch API|Compatibility|Limitations|
|---|---|---|---|
|1|Future|Y|Host-only (no device ops)|
|2|Future.done|Y|Host-only (no device ops)|
|3|Future.wait|Y|Host-only (no device ops)|
|4|Future.value|Y|Host-only (no device ops)|
|5|Future.then|Y|Host-only (no device ops)|
|6|Future.add_done_callback|Y|Host-only (no device ops)|
|7|Future.set_result|Y|Host-only (no device ops)|
|8|Future.set_exception|Y|Host-only (no device ops)|
|9|collect_all|Y|Host-only (no device ops)|
|10|wait_all|Y|Host-only (no device ops)|

## torch.linalg

||pytorch_api|neuron_compatible|limitations|
|---|---|---|---|
|1|torch.linalg.norm|Y|Native; spectral and nuclear norms fall back (`aten::_linalg_svd.U`), no `bfloat16`; backward falls back (`aten::_to_copy`)|
|2|torch.linalg.vector_norm|Y|Native. Backward performs an internal dtype conversion on CPU (`aten::_to_copy`)|
|3|torch.linalg.matrix_norm|Y|Native; spectral/nuclear fallback: `aten::_linalg_svd.U`, backward CPU: `aten::_to_copy`|
|4|torch.linalg.diagonal|Y|Native|
|5|torch.linalg.det|CPU_FALLBACK|Falls back (`aten::_linalg_det.result`) for `float32`; no `bfloat16`|
|6|torch.linalg.slogdet|CPU_FALLBACK|Falls back (`aten::_linalg_slogdet.sign`)|
|7|torch.linalg.cond|CPU_FALLBACK|`float32` falls back (`aten::_linalg_svd.U`); no `bfloat16`|
|8|torch.linalg.matrix_rank|CPU_FALLBACK|Falls back (`aten::_linalg_eigh.eigenvalues`, `aten::_linalg_svd.U`, …); no `bfloat16`|
|9|torch.linalg.cholesky|CPU_FALLBACK|Falls back (`aten::linalg_cholesky_ex.L`)|
|10|torch.linalg.qr|CPU_FALLBACK|Falls back (`aten::linalg_qr.out`) for `float32`; no `bfloat16`|
|11|torch.linalg.lu|CPU_FALLBACK|Falls back (`aten::linalg_lu.out`) for `float32`; no `bfloat16`|
|12|torch.linalg.lu_factor|CPU_FALLBACK|`float32` falls back (`aten::linalg_lu_factor_ex.out`); no `bfloat16`|
|13|torch.linalg.eig|CPU_FALLBACK|Falls back (`aten::linalg_eig`); no `bfloat16`|
|14|torch.linalg.eigvals|CPU_FALLBACK|Falls back (`aten::_linalg_eigvals`) for `float32`; no `bfloat16`|
|15|torch.linalg.eigh|CPU_FALLBACK|`float32` falls back (`aten::_linalg_eigh.eigenvalues`); no `bfloat16`|
|16|torch.linalg.eigvalsh|CPU_FALLBACK|`float32` falls back (`aten::_linalg_eigh.eigenvalues`); no `bfloat16`|
|17|torch.linalg.svd|CPU_FALLBACK|Falls back (`aten::_linalg_svd.U`)|
|18|torch.linalg.svdvals|CPU_FALLBACK|Falls back (`aten::_linalg_svd.U`) for `float32`. `bfloat16` is not supported|
|19|torch.linalg.solve|CPU_FALLBACK|Falls back (`aten::_linalg_solve_ex.result`)|
|20|torch.linalg.solve_triangular|Y|Native|
|21|torch.linalg.lu_solve|CPU_FALLBACK|Falls back (`aten::linalg_lu_solve.out`); no `bfloat16`|
|22|torch.linalg.lstsq|CPU_FALLBACK|Falls back (`aten::linalg_lstsq.out`)|
|23|torch.linalg.inv|CPU_FALLBACK|Falls back (`aten::linalg_inv_ex.inverse`) for `float32`; no `bfloat16`|
|24|torch.linalg.pinv|CPU_FALLBACK|Falls back (`aten::_linalg_eigh.eigenvalues`, `aten::_linalg_svd.U`, …)|
|25|torch.linalg.matrix_exp|CPU_FALLBACK|Falls back (`aten::linalg_matrix_exp`)|
|26|torch.linalg.matrix_power|Y|Native n≥0; negative `float32` falls back (`aten::linalg_inv_ex.inverse`), no `bfloat16`|
|27|torch.linalg.cross|CPU_FALLBACK|Falls back (`aten::linalg_cross.out`)|
|28|torch.linalg.matmul|Y|Native; `int64` paths fall back (`aten::mm`, `aten::bmm`)|
|29|torch.linalg.vecdot|CPU_FALLBACK|Falls back (`aten::vdot`)|
|30|torch.linalg.multi_dot|Y|Native; `int64` falls back (`aten::mm`)|
|31|torch.linalg.householder_product|CPU_FALLBACK|`float32` falls back (`aten::linalg_householder_product`); no `bfloat16`|
|32|torch.linalg.tensorinv|CPU_FALLBACK|`float32` falls back (`aten::linalg_inv_ex.inverse`); no `bfloat16`|
|33|torch.linalg.tensorsolve|CPU_FALLBACK|Falls back (`aten::_linalg_solve_ex.result`); no `bfloat16`|
|34|torch.linalg.vander|CPU_FALLBACK|Falls back (`aten::cumprod.out`)|
|35|torch.linalg.cholesky_ex|CPU_FALLBACK|Falls back (`aten::linalg_cholesky_ex.L`)|
|36|torch.linalg.inv_ex|CPU_FALLBACK|Falls back (`aten::linalg_inv_ex.inverse`) for `float32`; no `bfloat16`|
|37|torch.linalg.solve_ex|CPU_FALLBACK|`float32` falls back (`aten::_linalg_solve_ex.result`); no `bfloat16`|
|38|torch.linalg.lu_factor_ex|CPU_FALLBACK|Falls back (`aten::linalg_lu_factor_ex.out`)|
|39|torch.linalg.ldl_factor|CPU_FALLBACK|`float32` falls back (`aten::linalg_ldl_factor_ex.out`); no `bfloat16`|
|40|torch.linalg.ldl_factor_ex|CPU_FALLBACK|Falls back (`aten::linalg_ldl_factor_ex.out`); no `bfloat16`|
|41|torch.linalg.ldl_solve|CPU_FALLBACK|Falls back (`aten::linalg_ldl_solve.out`) for `float32`; no `bfloat16`|

## torch.signal

||pytorch_api|neuron_compatible|limitations|
|---|---|---|---|
|1|torch.signal.windows.bartlett|Y|Native|
|2|torch.signal.windows.blackman|Y|Native|
|3|torch.signal.windows.cosine|Y|Native|
|4|torch.signal.windows.exponential|Y|Native|
|5|torch.signal.windows.gaussian|Y|Native|
|6|torch.signal.windows.general_cosine|Y|Native|
|7|torch.signal.windows.general_hamming|Y|Native|
|8|torch.signal.windows.hamming|Y|Native|
|9|torch.signal.windows.hann|Y|Native|
|10|torch.signal.windows.kaiser|CPU_FALLBACK|Falls back (`aten::i0.out`) for nonempty `float32` windows|
|11|torch.signal.windows.nuttall|Y|Native|

## torch.special

||pytorch_api|neuron_compatible|limitations|
|---|---|---|---|
|1|torch.special.bessel_j0|CPU_FALLBACK|`float32` falls back (`aten::special_bessel_j0.out`); no `bfloat16`|
|2|torch.special.bessel_j1|CPU_FALLBACK|`float32` falls back (`aten::special_bessel_j1.out`); no `bfloat16`|
|3|torch.special.digamma|CPU_FALLBACK|Falls back (`aten::digamma.out`)|
|4|torch.special.psi|CPU_FALLBACK|Falls back (`aten::digamma.out`)|
|5|torch.special.entr|CPU_FALLBACK|Falls back (`aten::special_entr.out`)|
|6|torch.special.erf|Y|Native|
|7|torch.special.erfc|CPU_FALLBACK|Falls back (`aten::erfc.out`)|
|8|torch.special.erfcx|CPU_FALLBACK|Falls back (`aten::special_erfcx.out`); no `bfloat16`|
|9|torch.special.erfinv|Y|Native|
|10|torch.special.exp2|Y|Native|
|11|torch.special.expit|Y|Native|
|12|torch.special.expm1|Y|Native|
|13|torch.special.gammainc|CPU_FALLBACK|Falls back (`aten::igamma.out`)|
|14|torch.special.gammaincc|CPU_FALLBACK|Falls back (`aten::igammac.out`)|
|15|torch.special.gammaln|CPU_FALLBACK|Falls back (`aten::lgamma.out`)|
|16|torch.special.i0|CPU_FALLBACK|Falls back (`aten::i0.out`)|
|17|torch.special.i0e|CPU_FALLBACK|Falls back (`aten::special_i0e.out`)|
|18|torch.special.i1|CPU_FALLBACK|Falls back (`aten::special_i1.out`)|
|19|torch.special.i1e|Y|Native. Backward falls back (`aten::special_i0e.out`)|
|20|torch.special.laguerre_polynomial_l|CPU_FALLBACK|Falls back (`aten::special_laguerre_polynomial_l.out`) for `float32`; no `bfloat16`|
|21|torch.special.log1p|Y|Native|
|22|torch.special.log_ndtr|CPU_FALLBACK|Falls back (`aten::special_log_ndtr.out`)|
|23|torch.special.log_softmax|Y|Native; non-contiguous inputs round-trip via CPU (`aten::_log_softmax`, `aten::_to_copy`)|
|24|torch.special.softmax|Y|Base `float32` native; `float64` output and backward fall back (`aten::_to_copy`)|
|25|torch.special.logit|Y|Native|
|26|torch.special.logsumexp|Y|Native|
|27|torch.special.multigammaln|CPU_FALLBACK|Falls back (`aten::lgamma.out`); `bfloat16` can differ from CPU|
|28|torch.special.ndtr|Y|Native|
|29|torch.special.ndtri|CPU_FALLBACK|Falls back (`aten::special_ndtri.out`) for `float32`; no `bfloat16`|
|30|torch.special.round|Y|Native|
|31|torch.special.sinc|CPU_FALLBACK|Falls back (`aten::sinc.out`)|
|32|torch.special.spherical_bessel_j0|CPU_FALLBACK|Falls back (`aten::special_spherical_bessel_j0.out`); no `bfloat16`|
|33|torch.special.xlog1py|CPU_FALLBACK|Falls back (`aten::special_xlog1py.out`)|
|34|torch.special.xlogy|CPU_FALLBACK|Falls back (`aten::xlogy.OutTensor`)|
|35|torch.special.zeta|CPU_FALLBACK|`float32` falls back (`aten::special_zeta.out`); no `bfloat16`|
|36|torch.special.airy_ai|CPU_FALLBACK|Falls back (`aten::special_airy_ai.out`); no `bfloat16`|
|37|torch.special.bessel_y0|CPU_FALLBACK|Falls back (`aten::special_bessel_y0.out`) for `float32`; no `bfloat16`|
|38|torch.special.bessel_y1|CPU_FALLBACK|`float32` falls back (`aten::special_bessel_y1.out`); no `bfloat16`|
|39|torch.special.chebyshev_polynomial_t|CPU_FALLBACK|`float32` falls back (`aten::special_chebyshev_polynomial_t.out`); no `bfloat16`|
|40|torch.special.chebyshev_polynomial_u|CPU_FALLBACK|Falls back (`aten::special_chebyshev_polynomial_u.out`); no `bfloat16`|
|41|torch.special.chebyshev_polynomial_v|CPU_FALLBACK|Falls back (`aten::special_chebyshev_polynomial_v.out`)|
|42|torch.special.chebyshev_polynomial_w|CPU_FALLBACK|Falls back (`aten::special_chebyshev_polynomial_w.out`) for `float32`; no `bfloat16`|
|43|torch.special.hermite_polynomial_h|CPU_FALLBACK|Falls back (`aten::special_hermite_polynomial_h.out`)|
|44|torch.special.hermite_polynomial_he|CPU_FALLBACK|Falls back (`aten::special_hermite_polynomial_he.out`); no `bfloat16`|
|45|torch.special.legendre_polynomial_p|CPU_FALLBACK|`float32` falls back (`aten::special_legendre_polynomial_p.out`); no `bfloat16`|
|46|torch.special.modified_bessel_i0|CPU_FALLBACK|Falls back (`aten::special_modified_bessel_i0.out`); no `bfloat16`|
|47|torch.special.modified_bessel_i1|CPU_FALLBACK|Falls back (`aten::special_modified_bessel_i1.out`) for `float32`; no `bfloat16`|
|48|torch.special.modified_bessel_k0|CPU_FALLBACK|Falls back (`aten::special_modified_bessel_k0.out`); no `bfloat16`|
|49|torch.special.modified_bessel_k1|CPU_FALLBACK|`float32` falls back (`aten::special_modified_bessel_k1.out`); no `bfloat16`|
|50|torch.special.scaled_modified_bessel_k0|CPU_FALLBACK|Falls back (`aten::special_scaled_modified_bessel_k0.out`); no `bfloat16`|
|51|torch.special.scaled_modified_bessel_k1|CPU_FALLBACK|Falls back (`aten::special_scaled_modified_bessel_k1.out`) for `float32`; no `bfloat16`|
|52|torch.special.shifted_chebyshev_polynomial_t|CPU_FALLBACK|Falls back (`aten::special_shifted_chebyshev_polynomial_t.out`); no `bfloat16`|
|53|torch.special.shifted_chebyshev_polynomial_u|CPU_FALLBACK|`float32` falls back (`aten::special_shifted_chebyshev_polynomial_u.out`); no `bfloat16`|
|54|torch.special.shifted_chebyshev_polynomial_v|CPU_FALLBACK|Falls back (`aten::special_shifted_chebyshev_polynomial_v.out`); no `bfloat16`|
|55|torch.special.shifted_chebyshev_polynomial_w|CPU_FALLBACK|Falls back (`aten::special_shifted_chebyshev_polynomial_w.out`)|
|56|torch.special.polygamma|CPU_FALLBACK|Falls back (`aten::polygamma.out`)|

## torch.nn.init

||pytorch_api|neuron_compatible|limitations|
|---|---|---|---|
|1|calculate_gain|Y|Host-only (no device ops)|
|2|uniform_|Y|Native|
|3|normal_|Y|Native|
|4|constant_|Y|Native|
|5|ones_|Y|Native|
|6|zeros_|Y|Native|
|7|eye_|Y|Native|
|8|dirac_|Y|Native|
|9|xavier_uniform_|Y|Native|
|10|xavier_normal_|Y|Native|
|11|kaiming_uniform_|Y|Native|
|12|kaiming_normal_|Y|Native|
|13|trunc_normal_|Y|Native|
|14|orthogonal_|CPU_FALLBACK|`orthogonal_` falls back (`aten::linalg_qr.out`)|
|15|sparse_|Y|Native|

## torch.optim

||pytorch_api|neuron_compatible|limitations|
|---|---|---|---|
|1|torch.optim.Optimizer|Y|Host-only (no device ops)|
|2|Optimizer.add_param_group|Y|Host-only (no device ops)|
|3|Optimizer.load_state_dict|Y|Host-only (no device ops)|
|4|Optimizer.state_dict|Y|Host-only (no device ops)|
|5|Optimizer.step|Y|Host-only (no device ops)|
|6|Optimizer.zero_grad|Y|Native|
|7|torch.optim.Adadelta|Y|Host-only (no device ops)|
|8|torch.optim.Adadelta.step|Y|Native|
|9|torch.optim.Adagrad|Y|Native|
|10|torch.optim.Adagrad.step|Y|Native|
|11|torch.optim.Adam|Y|Host-only (no device ops)|
|12|torch.optim.Adam.step|Y|Native|
|13|torch.optim.AdamW|Y|Host-only (no device ops)|
|14|torch.optim.AdamW.step|Y|Native|
|15|torch.optim.SparseAdam|Y|Host-only (no device ops)|
|16|torch.optim.SparseAdam.step|N|Sparse gradients not supported on Neuron|
|17|torch.optim.Adamax|Y|Host-only (no device ops)|
|18|torch.optim.Adamax.step|Y|Native|
|19|torch.optim.ASGD|Y|Host-only (no device ops)|
|20|torch.optim.ASGD.step|Y|Native|
|21|torch.optim.LBFGS|Y|Host-only (no device ops)|
|22|torch.optim.LBFGS.step|Y|Native for a one-iteration `float32` and `bfloat16` update|
|23|torch.optim.NAdam|Y|Host-only (no device ops)|
|24|torch.optim.NAdam.step|Y|Native|
|25|torch.optim.RAdam|Y|Host-only (no device ops)|
|26|torch.optim.RAdam.step|Y|Native|
|27|torch.optim.RMSprop|Y|Host-only (no device ops)|
|28|torch.optim.RMSprop.step|Y|Native|
|29|torch.optim.Rprop|Y|Host-only (no device ops)|
|30|torch.optim.Rprop.step|CPU_FALLBACK|Falls back (`aten::_to_copy`) for `float32` and `bfloat16` updates|
|31|torch.optim.SGD|Y|Host-only (no device ops)|
|32|torch.optim.SGD.step|Y|Native|
|33|torch.optim.Adafactor|Y|Host-only (no device ops)|
|34|torch.optim.Adafactor.step|Y|Native|
|35|torch.optim.Muon|Y|Host-only (no device ops)|
|36|torch.optim.Muon.step|N|`float32` parameter updates differ from CPU|
|37|Optimizer.register_state_dict_pre_hook|Y|Host-only (no device ops)|
|38|Optimizer.register_state_dict_post_hook|Y|Host-only (no device ops)|
|39|Optimizer.register_load_state_dict_pre_hook|Y|Host-only (no device ops). Registers an optimizer load-state pre-hook|
|40|Optimizer.register_load_state_dict_post_hook|Y|Host-only (no device ops). Registers an optimizer load-state post-hook|
|41|Optimizer.register_step_pre_hook|Y|Host-only (no device ops). Registers an optimizer step pre-hook|
|42|Optimizer.register_step_post_hook|Y|Host-only (no device ops). Registers an optimizer step post-hook|
|43|torch.optim.optimizer.register_optimizer_step_pre_hook|Y|Host-only (no device ops). Registers a global optimizer step pre-hook|
|44|torch.optim.optimizer.register_optimizer_step_post_hook|Y|Host-only (no device ops). Registers a global optimizer step post-hook|
|45|torch.optim.lr_scheduler.LRScheduler|Y|Host-only (no device ops). Base class for learning-rate schedulers|
|46|torch.optim.lr_scheduler.LambdaLR|Y|Host-only (no device ops). Adjusts the optimizer's Python `float` learning rate|
|47|torch.optim.lr_scheduler.MultiplicativeLR|Y|Host-only (no device ops). Adjusts the optimizer's Python `float` learning rate|
|48|torch.optim.lr_scheduler.StepLR|Y|Host-only (no device ops). Adjusts the optimizer's Python `float` learning rate|
|49|torch.optim.lr_scheduler.MultiStepLR|Y|Host-only (no device ops). Adjusts the optimizer's Python `float` learning rate|
|50|torch.optim.lr_scheduler.ConstantLR|Y|Host-only (no device ops). Adjusts the optimizer's Python `float` learning rate|
|51|torch.optim.lr_scheduler.LinearLR|Y|Host-only (no device ops). Adjusts the optimizer's Python `float` learning rate|
|52|torch.optim.lr_scheduler.ExponentialLR|Y|Host-only (no device ops). Adjusts the optimizer's Python `float` learning rate|
|53|torch.optim.lr_scheduler.SequentialLR|Y|Host-only (no device ops). Adjusts the optimizer's Python `float` learning rate|
|54|torch.optim.lr_scheduler.CosineAnnealingLR|Y|Host-only (no device ops). Adjusts the optimizer's Python `float` learning rate|
|55|torch.optim.lr_scheduler.ChainedScheduler|Y|Host-only (no device ops). Adjusts the optimizer's Python `float` learning rate|
|56|torch.optim.lr_scheduler.ReduceLROnPlateau|Y|Host-only (no device ops). Adjusts the optimizer's Python `float` learning rate|
|57|torch.optim.lr_scheduler.CyclicLR|Y|Host-only (no device ops). Adjusts the optimizer's Python `float` learning rate|
|58|torch.optim.lr_scheduler.CosineAnnealingWarmRestarts|Y|Host-only (no device ops). Adjusts the optimizer's Python `float` learning rate|
|59|torch.optim.lr_scheduler.OneCycleLR|Y|Host-only (no device ops). Adjusts the optimizer's Python `float` learning rate|
|60|torch.optim.lr_scheduler.PolynomialLR|Y|Host-only (no device ops). Adjusts the optimizer's Python `float` learning rate|

## DDP Communication Hooks

||PyTorch API|Compatibility|Limitations|
|---|---|---|---|
|1|DDPCommHookType|Y|Host-only (no device ops). Selects a DDP communication-hook implementation|
|2|register_ddp_comm_hook|Y|Host-only (no device ops). Execution support depends on the selected hook|
|3|default.allreduce_hook|Y|Native. Supports `float32`; requires a configured DDP process group|
|4|default.fp16_compress_hook|Y|Native; `float32` to `float16` compression falls back (`aten::_to_copy`)|
|5|default.bf16_compress_hook|N|Raises `TypeError` because the hook requires CUDA with NCCL or XPU with XCCL|
|6|default.fp16_compress_wrapper|Y|Native|
|7|default.bf16_compress_wrapper|N|Requires CUDA/NCCL or XPU/XCCL|
|8|powerSGD.PowerSGDState|Y|Host-only (no device ops). Stores PowerSGD hook configuration and state|
|9|powerSGD.powerSGD_hook|Y|Native. Needs an initialized process group and a `DistributedDataParallel` model|
|10|powerSGD.batched_powerSGD_hook|Y|Native; some small models raise `AssertionError`|
|11|quantization.quantization_pertensor_hook|CPU_FALLBACK|Falls back (`aten::aminmax.out`)|
|12|quantization.quantization_perchannel_hook|CPU_FALLBACK|Falls back (`aten::aminmax.out`)|
|13|debugging.noop_hook|Y|Native. Returns the local gradient bucket without communication|
|14|post_localSGD.PostLocalSGDState|Y|Host-only (no device ops); pass `subgroup`, default requires CUDA|
|15|post_localSGD.post_localSGD_hook|Y|Native. Needs an initialized process group and a `DistributedDataParallel` model|
|16|ddp_zero_hook.hook_with_zero_step|Y|Native|
|17|ddp_zero_hook.hook_with_zero_step_interleaved|Y|Native|

## Pipeline Parallelism

||PyTorch API|Compatibility|Limitations|
|---|---|---|---|
|1|torch.distributed.pipelining.SplitPoint|Y|Host-only (no device ops)|
|2|torch.distributed.pipelining.pipeline|Y|Native|
|3|torch.distributed.pipelining.Pipe|Y|Native. Local `Pipe` execution supports `float32` and `bfloat16`|
|4|torch.distributed.pipelining.pipe_split|Y|Host-only (no device ops). Marks a pipeline split in the captured graph|
|5|torch.distributed.pipelining.microbatch.TensorChunkSpec|Y|Host-only (no device ops)|
|6|torch.distributed.pipelining.microbatch.split_args_kwargs_into_chunks|Y|Native. Splits `float32` and `bfloat16` tensors into microbatches|
|7|torch.distributed.pipelining.microbatch.merge_chunks|Y|Native. Merges `float32` and `bfloat16` microbatches through `aten::cat`|
|8|torch.distributed.pipelining.PipelineStage|Y|Native|
|9|torch.distributed.pipelining.build_stage|Y|Host-only (no device ops)|
|10|torch.distributed.pipelining.ScheduleGPipe|Y|Native. Needs an initialized process group for multi-rank scheduling|
|11|torch.distributed.pipelining.Schedule1F1B|Y|Native. Needs an initialized process group for multi-rank scheduling|
|12|torch.distributed.pipelining.ScheduleInterleaved1F1B|Y|Native|
|13|torch.distributed.pipelining.ScheduleLoopedBFS|Y|Native|
|14|torch.distributed.pipelining.ScheduleInterleavedZeroBubble|Y|Native|
|15|torch.distributed.pipelining.ScheduleZBVZeroBubble|Y|Native|
|16|torch.distributed.pipelining.ScheduleDualPipeV|Y|Native|
|17|torch.distributed.pipelining.schedules.PipelineScheduleSingle|Y|Native|
|18|torch.distributed.pipelining.schedules.PipelineScheduleSingle.step|Y|Native for concrete schedules|
|19|torch.distributed.pipelining.schedules.PipelineScheduleMulti|Y|Native for concrete multi-stage schedules|
|20|torch.distributed.pipelining.schedules.PipelineScheduleMulti.step|Y|Native|
|21|torch.distributed.pipelining.schedules.get_schedule_class|Y|Host-only (no device ops). Returns a pipeline schedule class by name|

## Quantization

||pytorch_api|neuron_compatible|limitations|
|---|---|---|---|
|1|torch.ao.quantization.quantize|CPU_FALLBACK|Falls back (`aten::aminmax.out`)|
|2|torch.ao.quantization.quantize_dynamic|Y|Native; converted quantized execution has separate limits|
|3|torch.ao.quantization.quantize_qat|N|Quantized tensors not supported on Neuron (`aten::empty_quantized`)|
|4|torch.ao.quantization.prepare|Y|Native; `inplace=False` clones Neuron parameters without fallback|
|5|torch.ao.quantization.prepare_qat|Y|Native; observer initialization copies host tensors to Neuron|
|6|torch.ao.quantization.convert|Y|Host-only (no device ops)|
|7|torch.ao.quantization.fuse_modules|Y|Host-only (no device ops)|
|8|torch.ao.quantization.QuantStub|Y|Native; converted quantized execution has separate limits|
|9|torch.ao.quantization.DeQuantStub|Y|Native|
|10|torch.ao.quantization.QuantWrapper|Y|Native before conversion; converted quantized support is separate|
|11|torch.ao.quantization.add_quant_dequant|Y|Host-only (no device ops)|
|12|torch.ao.quantization.swap_module|Y|Host-only (no device ops)|
|13|torch.ao.quantization.propagate_qconfig_|Y|Host-only (no device ops)|
|14|torch.ao.quantization.default_eval_fn|Y|Host-only (no device ops)|
|15|torch.ao.quantization.quantize_fx.prepare_fx|Y|Native; observer initialization allocates and transfers on Neuron|
|16|torch.ao.quantization.quantize_fx.prepare_qat_fx|Y|Host-only (no device ops)|
|17|torch.ao.quantization.quantize_fx.convert_fx|Y|Host-only (no device ops)|
|18|torch.ao.quantization.quantize_fx.fuse_fx|Y|Host-only (no device ops)|
|19|QConfigMapping|Y|Host-only (no device ops). Stores quantization configuration mappings|
|20|BackendConfig|Y|Host-only (no device ops). Describes quantization backend capabilities|
|21|BackendPatternConfig|Y|Host-only (no device ops)|
|22|DTypeConfig|Y|Host-only (no device ops). Describes supported quantization data types|
|23|ObservationType|Y|Host-only (no device ops). Selects observer placement behavior|
|24|FuseCustomConfig|Y|Host-only (no device ops). Stores custom FX fusion configuration|
|25|PrepareCustomConfig|Y|Host-only (no device ops). Stores custom FX preparation configuration|
|26|ConvertCustomConfig|Y|Host-only (no device ops). Stores custom FX conversion configuration|
|27|torch.ao.quantization.fx.custom_config.StandaloneModuleConfigEntry|Y|Host-only (no device ops)|
|28|torch.ao.quantization.observer.ObserverBase|Y|Host-only (no device ops); concrete observers fall back (`aten::aminmax.out`)|
|29|torch.ao.quantization.observer.ObserverBase.with_args|Y|Host-only (no device ops). Returns an observer constructor with bound arguments|
|30|torch.ao.quantization.observer.ObserverBase.with_callable_args|Y|Host-only (no device ops)|
|31|torch.ao.quantization.observer.MinMaxObserver|CPU_FALLBACK|Falls back (`aten::aminmax.out`)|
|32|torch.ao.quantization.observer.MinMaxObserver.calculate_qparams|Y|Native. Requires initialized observer statistics|
|33|torch.ao.quantization.observer.MinMaxObserver.forward|CPU_FALLBACK|Falls back (`aten::aminmax.out`)|
|34|torch.ao.quantization.observer.MinMaxObserver.reset_min_max_vals|Y|Native|
|35|torch.ao.quantization.observer.MovingAverageMinMaxObserver|CPU_FALLBACK|Falls back (`aten::aminmax.out`)|
|36|torch.ao.quantization.observer.PerChannelMinMaxObserver|CPU_FALLBACK|Falls back (`aten::aminmax.out`)|
|37|torch.ao.quantization.observer.PerChannelMinMaxObserver.reset_min_max_vals|Y|Host-only (no device ops). Resets `min_val` and `max_val` to empty CPU tensors|
|38|torch.ao.quantization.observer.MovingAveragePerChannelMinMaxObserver|CPU_FALLBACK|Falls back (`aten::aminmax.out`)|
|39|torch.ao.quantization.observer.HistogramObserver|CPU_FALLBACK|Falls back (`aten::aminmax.out`); `aten::histc` is native|
|40|torch.ao.quantization.observer.PlaceholderObserver|Y|Native. Acts as an identity operation; `calculate_qparams` is unsupported|
|41|torch.ao.quantization.observer.RecordingObserver|Y|Native; `calculate_qparams` is unsupported|
|42|torch.ao.quantization.observer.NoopObserver|Y|Native. Acts as an identity operation; `calculate_qparams` is unsupported|
|43|torch.ao.quantization.observer.get_observer_state_dict|Y|Host-only (no device ops)|
|44|torch.ao.quantization.observer.load_observer_state_dict|Y|Native. Loads observer state into Neuron buffers without CPU fallback|
|45|torch.ao.quantization.observer.default_observer|Y|Host-only (no device ops)|
|46|torch.ao.quantization.observer.default_placeholder_observer|Y|Host-only (no device ops)|
|47|torch.ao.quantization.observer.default_debug_observer|Y|Host-only (no device ops)|
|48|torch.ao.quantization.observer.default_weight_observer|Y|Host-only (no device ops)|
|49|torch.ao.quantization.observer.default_histogram_observer|Y|Host-only (no device ops)|
|50|torch.ao.quantization.observer.default_per_channel_weight_observer|Y|Host-only (no device ops)|
|51|torch.ao.quantization.observer.default_dynamic_quant_observer|Y|Host-only (no device ops)|
|52|torch.ao.quantization.observer.default_float_qparams_observer|Y|Host-only (no device ops)|
|53|torch.ao.quantization.fake_quantize.FakeQuantizeBase|Y|Host-only (no device ops); concrete `FakeQuantize` falls back|
|54|torch.ao.quantization.fake_quantize.FakeQuantize|CPU_FALLBACK|Falls back (`aten::_fake_quantize_per_tensor_affine_cachemask_tensor_qparams`, `aten::aminmax.out`)|
|55|torch.ao.quantization.fake_quantize.FixedQParamsFakeQuantize|CPU_FALLBACK|Falls back (`aten::_fake_quantize_per_tensor_affine_cachemask_tensor_qparams`)|
|56|torch.ao.quantization.fake_quantize.FusedMovingAvgObsFakeQuantize|CPU_FALLBACK|Falls back (`aten::_fused_moving_avg_obs_fq_helper`); no `bfloat16`|
|57|torch.ao.quantization.fake_quantize.default_fake_quant|Y|Host-only (no device ops)|
|58|torch.ao.quantization.fake_quantize.default_weight_fake_quant|Y|Host-only (no device ops)|
|59|torch.ao.quantization.fake_quantize.default_per_channel_weight_fake_quant|Y|Host-only (no device ops)|
|60|torch.ao.quantization.fake_quantize.default_histogram_fake_quant|Y|Host-only (no device ops)|
|61|torch.ao.quantization.fake_quantize.default_fused_act_fake_quant|Y|Host-only (no device ops)|
|62|torch.ao.quantization.fake_quantize.default_fused_wt_fake_quant|Y|Host-only (no device ops)|
|63|torch.ao.quantization.fake_quantize.default_fused_per_channel_wt_fake_quant|Y|Host-only (no device ops)|
|64|torch.ao.quantization.fake_quantize.disable_fake_quant|Y|Native|
|65|torch.ao.quantization.fake_quantize.enable_fake_quant|Y|Host-only (no device ops). Updates fake-quantization module state|
|66|torch.ao.quantization.fake_quantize.disable_observer|Y|Native. Disables observation by updating the module state without CPU fallback|
|67|torch.ao.quantization.fake_quantize.enable_observer|Y|Native. Enables observation by updating the module state without CPU fallback|
|68|torch.ao.quantization.qconfig.QConfig|Y|Host-only (no device ops)|
|69|torch.ao.quantization.qconfig.default_qconfig|Y|Host-only (no device ops)|
|70|torch.ao.quantization.qconfig.default_debug_qconfig|Y|Host-only (no device ops)|
|71|torch.ao.quantization.qconfig.default_per_channel_qconfig|Y|Host-only (no device ops)|
|72|torch.ao.quantization.qconfig.default_dynamic_qconfig|Y|Host-only (no device ops)|
|73|torch.ao.quantization.qconfig.float16_dynamic_qconfig|Y|Host-only (no device ops)|
|74|torch.ao.quantization.qconfig.float16_static_qconfig|Y|Host-only (no device ops)|
|75|torch.ao.quantization.qconfig.per_channel_dynamic_qconfig|Y|Host-only (no device ops)|
|76|torch.ao.quantization.qconfig.float_qparams_weight_only_qconfig|Y|Host-only (no device ops)|
|77|torch.ao.quantization.qconfig.default_qat_qconfig|Y|Host-only (no device ops)|
|78|torch.ao.quantization.qconfig.default_weight_only_qconfig|Y|Host-only (no device ops)|
|79|torch.ao.quantization.qconfig.default_activation_only_qconfig|Y|Host-only (no device ops)|
|80|torch.ao.quantization.qconfig.default_qat_qconfig_v2|Y|Host-only (no device ops)|
|81|torch.ao.nn.intrinsic.ConvReLU1d|Y|Native|
|82|torch.ao.nn.intrinsic.ConvReLU2d|Y|Native|
|83|torch.ao.nn.intrinsic.ConvReLU3d|Y|Native; non-contiguous inputs round-trip via CPU|
|84|torch.ao.nn.intrinsic.LinearReLU|Y|Native|
|85|torch.ao.nn.intrinsic.ConvBn1d|Y|Native|
|86|torch.ao.nn.intrinsic.ConvBn2d|Y|Native|
|87|torch.ao.nn.intrinsic.ConvBn3d|Y|Native|
|88|torch.ao.nn.intrinsic.ConvBnReLU1d|Y|Native|
|89|torch.ao.nn.intrinsic.ConvBnReLU2d|Y|Native|
|90|torch.ao.nn.intrinsic.ConvBnReLU3d|Y|Native|
|91|torch.ao.nn.intrinsic.BNReLU2d|Y|Native|
|92|torch.ao.nn.intrinsic.BNReLU3d|Y|Native|
|93|torch.ao.nn.intrinsic.qat.LinearReLU|CPU_FALLBACK|Fake quantization falls back (`aten::_fake_quantize_per_tensor_affine_cachemask_tensor_qparams`, …)|
|94|torch.ao.nn.intrinsic.qat.ConvBn1d|CPU_FALLBACK|`float32` fake quantization falls back (`aten::_fused_moving_avg_obs_fq_helper`); non-contiguous inputs round-trip via CPU|
|95|torch.ao.nn.intrinsic.qat.ConvBnReLU1d|CPU_FALLBACK|Fake quantization falls back (`aten::_fake_quantize_per_tensor_affine_cachemask_tensor_qparams`, …)|
|96|torch.ao.nn.intrinsic.qat.ConvBn2d|CPU_FALLBACK|`float32` QAT falls back (`aten::_fused_moving_avg_obs_fq_helper`)|
|97|torch.ao.nn.intrinsic.qat.ConvBnReLU2d|CPU_FALLBACK|Falls back (`aten::_fused_moving_avg_obs_fq_helper`) for `float32` QAT|
|98|torch.ao.nn.intrinsic.qat.ConvReLU2d|CPU_FALLBACK|`float32` falls back (`aten::_fused_moving_avg_obs_fq_helper`)|
|99|torch.ao.nn.intrinsic.qat.ConvBn3d|CPU_FALLBACK|Falls back (`aten::_fused_moving_avg_obs_fq_helper`)|
|100|torch.ao.nn.intrinsic.qat.ConvBnReLU3d|CPU_FALLBACK|`float32` QAT falls back (`aten::_fused_moving_avg_obs_fq_helper`); no `bfloat16`|
|101|torch.ao.nn.intrinsic.qat.ConvReLU3d|CPU_FALLBACK|Fake quantization falls back (`aten::_fake_quantize_per_tensor_affine_cachemask_tensor_qparams`, …)|
|102|torch.ao.nn.intrinsic.qat.update_bn_stats|Y|Host-only (no device ops). Enables batch-normalization statistics updates|
|103|torch.ao.nn.intrinsic.qat.freeze_bn_stats|Y|Host-only (no device ops). Freezes batch-normalization statistics updates|
|104|torch.ao.nn.qat.Conv2d|CPU_FALLBACK|Fake quantization falls back (`aten::_fake_quantize_per_tensor_affine_cachemask_tensor_qparams`, …)|
|105|torch.ao.nn.qat.Conv3d|CPU_FALLBACK|Fake quantization falls back (`aten::_fake_quantize_per_tensor_affine_cachemask_tensor_qparams`, …)|
|106|torch.ao.nn.qat.Linear|CPU_FALLBACK|`float32` QAT falls back (`aten::_fused_moving_avg_obs_fq_helper`); `bfloat16` raises|
|107|torch.ao.nn.qat.dynamic.Linear|CPU_FALLBACK|Fake quantization falls back (`aten::_fake_quantize_per_tensor_affine_cachemask_tensor_qparams`, …)|
|108|torch.ao.nn.quantized.ReLU6|N|Quantized tensors not supported on Neuron (`aten::empty_quantized`)|
|109|torch.ao.nn.quantized.Hardswish|N|Quantized tensors not supported on Neuron (`aten::empty_quantized`)|
|110|torch.ao.nn.quantized.ELU|N|Quantized tensors not supported on Neuron (`aten::empty_quantized`)|
|111|torch.ao.nn.quantized.LeakyReLU|N|Quantized tensors not supported on Neuron (`aten::empty_quantized`)|
|112|torch.ao.nn.quantized.Sigmoid|N|Quantized tensors not supported on Neuron (`aten::empty_quantized`)|
|113|torch.ao.nn.quantized.BatchNorm2d|N|Quantized tensors not supported on Neuron (`aten::empty_quantized`)|
|114|torch.ao.nn.quantized.BatchNorm3d|N|Quantized tensors not supported on Neuron (`aten::empty_quantized`)|
|115|torch.ao.nn.quantized.Conv1d|N|Quantized tensors not supported on Neuron (`aten::empty_quantized`)|
|116|torch.ao.nn.quantized.Conv1d.from_float|Y|Host-only (no device ops); quantized tensors unavailable on Neuron|
|117|torch.ao.nn.quantized.Conv2d|N|Quantized tensors not supported on Neuron (`aten::empty_quantized`)|
|118|torch.ao.nn.quantized.Conv2d.from_float|Y|Host-only (no device ops); quantized tensors unavailable on Neuron|
|119|torch.ao.nn.quantized.Conv3d|N|Quantized tensors not supported on Neuron (`aten::empty_quantized`)|
|120|torch.ao.nn.quantized.Conv3d.from_float|Y|Host-only (no device ops); quantized tensors unavailable on Neuron|
|121|torch.ao.nn.quantized.ConvTranspose1d|N|Quantized tensors not supported on Neuron (`aten::empty_quantized`)|
|122|torch.ao.nn.quantized.ConvTranspose2d|N|Quantized tensors not supported on Neuron (`aten::empty_quantized`)|
|123|torch.ao.nn.quantized.ConvTranspose3d|N|Quantized tensors not supported on Neuron (`aten::empty_quantized`)|
|124|torch.ao.nn.quantized.Embedding|N|Quantized tensors not supported on Neuron (`aten::empty_quantized`)|
|125|torch.ao.nn.quantized.Embedding.from_float|Y|Host-only (no device ops); quantized tensors unavailable on Neuron|
|126|torch.ao.nn.quantized.EmbeddingBag|CPU_FALLBACK|Falls back (`quantized::embedding_bag_byte`, `aten::empty_quantized`)|
|127|torch.ao.nn.quantized.EmbeddingBag.from_float|Y|Host-only (no device ops); quantized tensors unavailable on Neuron|
|128|torch.ao.nn.quantized.FloatFunctional|Y|Native|
|129|torch.ao.nn.quantized.FXFloatFunctional|Y|Native|
|130|torch.ao.nn.quantized.QFunctional|N|Quantized tensors not supported on Neuron (`aten::empty_quantized`)|
|131|torch.ao.nn.quantized.Linear|N|Quantized tensors not supported on Neuron (`aten::empty_quantized`)|
|132|torch.ao.nn.quantized.Linear.from_float|Y|Host-only (no device ops); quantized tensors unavailable on Neuron|
|133|torch.ao.nn.quantized.Linear.from_reference|Y|Host-only (no device ops); quantized tensors unavailable on Neuron|
|134|torch.ao.nn.quantized.LayerNorm|N|Quantized tensors not supported on Neuron (`aten::empty_quantized`)|
|135|torch.ao.nn.quantized.GroupNorm|N|Quantized tensors not supported on Neuron (`aten::empty_quantized`)|
|136|torch.ao.nn.quantized.InstanceNorm1d|N|Quantized tensors not supported on Neuron (`aten::empty_quantized`)|
|137|torch.ao.nn.quantized.InstanceNorm2d|N|Quantized tensors not supported on Neuron (`aten::empty_quantized`)|
|138|torch.ao.nn.quantized.InstanceNorm3d|N|Quantized tensors not supported on Neuron (`aten::empty_quantized`)|
|139|torch.ao.nn.quantized.functional.avg_pool2d|N|Quantized tensors not supported on Neuron (`aten::empty_quantized`)|
|140|torch.ao.nn.quantized.functional.avg_pool3d|N|Quantized tensors not supported on Neuron (`aten::empty_quantized`)|
|141|torch.ao.nn.quantized.functional.adaptive_avg_pool2d|N|Quantized tensors not supported on Neuron (`aten::empty_quantized`)|
|142|torch.ao.nn.quantized.functional.adaptive_avg_pool3d|N|Quantized tensors not supported on Neuron (`aten::empty_quantized`)|
|143|torch.ao.nn.quantized.functional.conv1d|N|Quantized tensors not supported on Neuron (`aten::empty_quantized`)|
|144|torch.ao.nn.quantized.functional.conv2d|N|Quantized tensors not supported on Neuron (`aten::empty_quantized`)|
|145|torch.ao.nn.quantized.functional.conv3d|N|Quantized tensors not supported on Neuron (`aten::empty_quantized`)|
|146|torch.ao.nn.quantized.functional.interpolate|N|Quantized tensors not supported on Neuron (`aten::empty_quantized`)|
|147|torch.ao.nn.quantized.functional.linear|N|Quantized tensors not supported on Neuron (`aten::empty_quantized`)|
|148|torch.ao.nn.quantized.functional.max_pool1d|N|Quantized tensors not supported on Neuron (`aten::empty_quantized`)|
|149|torch.ao.nn.quantized.functional.max_pool2d|N|Quantized tensors not supported on Neuron (`aten::empty_quantized`)|
|150|torch.ao.nn.quantized.functional.celu|N|Quantized tensors not supported on Neuron (`aten::empty_quantized`)|
|151|torch.ao.nn.quantized.functional.leaky_relu|N|Quantized tensors not supported on Neuron (`aten::empty_quantized`)|
|152|torch.ao.nn.quantized.functional.hardtanh|N|Quantized tensors not supported on Neuron (`aten::empty_quantized`)|
|153|torch.ao.nn.quantized.functional.hardswish|N|Quantized tensors not supported on Neuron (`aten::empty_quantized`)|
|154|torch.ao.nn.quantized.functional.threshold|N|Quantized tensors not supported on Neuron (`aten::empty_quantized`)|
|155|torch.ao.nn.quantized.functional.elu|N|Quantized tensors not supported on Neuron (`aten::empty_quantized`)|
|156|torch.ao.nn.quantized.functional.hardsigmoid|N|Quantized tensors not supported on Neuron (`aten::empty_quantized`)|
|157|torch.ao.nn.quantized.functional.clamp|N|Quantized tensors not supported on Neuron (`aten::empty_quantized`)|
|158|torch.ao.nn.quantized.functional.upsample|N|Quantized tensors not supported on Neuron (`aten::empty_quantized`)|
|159|torch.ao.nn.quantized.functional.upsample_bilinear|N|Quantized tensors not supported on Neuron (`aten::empty_quantized`)|
|160|torch.ao.nn.quantized.functional.upsample_nearest|N|Quantized tensors not supported on Neuron (`aten::empty_quantized`)|
|161|torch.ao.nn.quantizable.LSTM|Y|Native; `bfloat16` unsupported because the implicit hidden state is `float32`|
|162|torch.ao.nn.quantizable.MultiheadAttention|Y|Native. Pre-conversion `float32` and `bfloat16` forward match CPU|
|163|torch.ao.nn.quantizable.MultiheadAttention.dequantize|Y|Host-only (no device ops); reinitializes Q/K/V projection weights|
|164|torch.ao.nn.quantizable.MultiheadAttention.forward|Y|Native for `float32` and `bfloat16` inputs|
|165|torch.ao.nn.quantized.dynamic.Linear|CPU_FALLBACK|`float32` falls back (`quantized::linear_dynamic`); no `bfloat16`|
|166|torch.ao.nn.quantized.dynamic.Linear.from_float|Y|Host-only (no device ops). Converts a module to dynamic quantized form|
|167|torch.ao.nn.quantized.dynamic.Linear.from_reference|Y|Host-only (no device ops). Converts a module to dynamic quantized form|
|168|torch.ao.nn.quantized.dynamic.LSTM|CPU_FALLBACK|Falls back (`aten::quantized_lstm.input`)|
|169|torch.ao.nn.quantized.dynamic.GRU|CPU_FALLBACK|Falls back (`aten::quantized_gru.input`); no `bfloat16`|
|170|torch.ao.nn.quantized.dynamic.RNNCell|CPU_FALLBACK|Falls back (`quantized::quantized_rnn_tanh_cell_dynamic`); `bfloat16` raises|
|171|torch.ao.nn.quantized.dynamic.LSTMCell|CPU_FALLBACK|Falls back (`quantized::quantized_lstm_cell_dynamic`)|
|172|torch.ao.nn.quantized.dynamic.GRUCell|CPU_FALLBACK|Falls back (`quantized::quantized_gru_cell_dynamic`)|
|173|torch.ao.ns._numeric_suite.compare_weights|Y|Host-only (no device ops)|
|174|torch.ao.ns._numeric_suite.get_logger_dict|Y|Host-only (no device ops)|
|175|torch.ao.ns._numeric_suite.Logger|Y|Host-only (no device ops)|
|176|torch.ao.ns._numeric_suite.Logger.forward|Y|Host-only (no device ops)|
|177|torch.ao.ns._numeric_suite.ShadowLogger|Y|Host-only (no device ops)|
|178|torch.ao.ns._numeric_suite.ShadowLogger.forward|Y|Host-only (no device ops)|
|179|torch.ao.ns._numeric_suite.OutputLogger|Y|Host-only (no device ops)|
|180|torch.ao.ns._numeric_suite.OutputLogger.forward|Y|Host-only (no device ops)|
|181|torch.ao.ns._numeric_suite.Shadow|Y|Host-only (no device ops)|
|182|torch.ao.ns._numeric_suite.Shadow.forward|Y|Native|
|183|torch.ao.ns._numeric_suite.Shadow.add|CPU_FALLBACK|Falls back (`aten::dequantize.self`); the remaining arithmetic runs on Neuron|
|184|torch.ao.ns._numeric_suite.Shadow.add_scalar|CPU_FALLBACK|Falls back (`aten::dequantize.self`); the remaining arithmetic runs on Neuron|
|185|torch.ao.ns._numeric_suite.Shadow.mul|CPU_FALLBACK|Falls back (`aten::dequantize.self`); the remaining arithmetic runs on Neuron|
|186|torch.ao.ns._numeric_suite.Shadow.mul_scalar|CPU_FALLBACK|Falls back (`aten::dequantize.self`); the remaining arithmetic runs on Neuron|
|187|torch.ao.ns._numeric_suite.Shadow.cat|CPU_FALLBACK|Falls back (`aten::dequantize.self`); the remaining arithmetic runs on Neuron|
|188|torch.ao.ns._numeric_suite.Shadow.add_relu|CPU_FALLBACK|Falls back (`aten::dequantize.self`); non-contiguous uses CPU (`aten::add.Tensor`)|
|189|torch.ao.ns._numeric_suite.prepare_model_with_stubs|Y|Host-only (no device ops)|
|190|torch.ao.ns._numeric_suite.compare_model_stub|Y|Native|
|191|torch.ao.ns._numeric_suite.get_matching_activations|Y|Host-only (no device ops)|
|192|torch.ao.ns._numeric_suite.prepare_model_outputs|Y|Host-only (no device ops)|
|193|torch.ao.ns._numeric_suite.compare_model_outputs|Y|Native|
|194|torch.ao.ns._numeric_suite_fx.OutputLogger|Y|Host-only (no device ops)|
|195|torch.ao.ns._numeric_suite_fx.OutputLogger.forward|Y|Host-only (no device ops)|
|196|torch.ao.ns._numeric_suite_fx.OutputComparisonLogger|Y|Host-only (no device ops)|
|197|torch.ao.ns._numeric_suite_fx.OutputComparisonLogger.forward|Y|Native|
|198|torch.ao.ns._numeric_suite_fx.NSTracer|Y|Host-only (no device ops)|
|199|torch.ao.ns._numeric_suite_fx.NSTracer.is_leaf_module|Y|Host-only (no device ops)|
|200|torch.ao.ns._numeric_suite_fx.extract_weights|Y|Host-only (no device ops)|
|201|torch.ao.ns._numeric_suite_fx.add_loggers|Y|Host-only (no device ops)|
|202|torch.ao.ns._numeric_suite_fx.extract_logger_info|Y|Host-only (no device ops)|
|203|torch.ao.ns._numeric_suite_fx.add_shadow_loggers|Y|Host-only (no device ops)|
|204|torch.ao.ns._numeric_suite_fx.extract_shadow_logger_info|Y|Host-only (no device ops)|
|205|torch.ao.ns._numeric_suite_fx.extend_logger_results_with_comparison|Y|Native when the supplied comparison function uses supported tensor operations|
|206|torch.ao.ns._numeric_suite_fx.prepare_n_shadows_model|Y|Native|
|207|torch.ao.ns._numeric_suite_fx.loggers_set_enabled|Y|Host-only (no device ops)|
|208|torch.ao.ns._numeric_suite_fx.loggers_set_save_activations|Y|Host-only (no device ops)|
|209|torch.ao.ns._numeric_suite_fx.convert_n_shadows_model|Y|Host-only (no device ops)|
|210|torch.ao.ns._numeric_suite_fx.extract_results_n_shadows_model|Y|Host-only (no device ops)|
|211|torch.ao.ns._numeric_suite_fx.print_comparisons_n_shadows_model|Y|Host-only (no device ops)|
|212|torch.ao.ns.fx.utils.compute_sqnr|Y|Native for `float32`; `bfloat16` unsupported and returns `None`|
|213|torch.ao.ns.fx.utils.compute_normalized_l2_error|Y|Native for `float32`; `bfloat16` unsupported and returns `None`|
|214|torch.ao.ns.fx.utils.compute_cosine_similarity|Y|Native for `float32`; `bfloat16` unsupported and returns `None`|

## torch.random

||pytorch_api|neuron_compatible|limitations|
|---|---|---|---|
|1|torch.random.fork_rng|Y|Native; selecting a Neuron generator transfers via `aten::_to_copy`|
|2|torch.random.get_rng_state|Y|Host-only (no device ops). Returns the CPU default-generator state|
|3|torch.random.initial_seed|Y|Host-only (no device ops). Returns the CPU default-generator seed|
|4|torch.random.manual_seed|Y|Native|
|5|torch.random.seed|Y|Host-only (no device ops)|
|6|torch.random.set_rng_state|Y|Host-only (no device ops). Updates the CPU default-generator state|

## torch.Storage

TypedStorage is deprecated in PyTorch 2.13.

||pytorch_api|neuron_compatible|limitations|
|---|---|---|---|
|1|torch.TypedStorage|Y|Host-only (no device ops); supports Neuron-backed storage|
|2|torch.TypedStorage.bfloat16|Y|Native. Converts Neuron-backed storage to `bfloat16`|
|3|torch.TypedStorage.bool|Y|Native. Converts Neuron-backed storage to `bool`|
|4|torch.TypedStorage.byte|Y|Native. Converts Neuron-backed storage to `uint8`|
|5|torch.TypedStorage.char|Y|Native. Converts Neuron-backed storage to `int8`|
|6|torch.TypedStorage.clone|Y|Native for `float32` and `bfloat16` storage|
|7|torch.TypedStorage.complex_double|Y|Native. Converts Neuron-backed storage to `complex128`|
|8|torch.TypedStorage.complex_float|Y|Native. Converts Neuron-backed storage to `complex64`|
|9|torch.TypedStorage.copy_|Y|Native for `float32` and `bfloat16` storage|
|10|torch.TypedStorage.cpu|Y|Native|
|11|torch.TypedStorage.cuda|N|CUDA-only|
|12|torch.TypedStorage.data_ptr|Y|Host-only (no device ops)|
|13|torch.TypedStorage.device|Y|Host-only (no device ops)|
|14|torch.TypedStorage.double|Y|Native. Converts Neuron-backed storage to `float64`|
|15|torch.TypedStorage.dtype|Y|Host-only (no device ops)|
|16|torch.TypedStorage.element_size|Y|Host-only (no device ops)|
|17|torch.TypedStorage.fill_|Y|Native for `float32` and `bfloat16` storage|
|18|torch.TypedStorage.float|Y|Native. Converts Neuron-backed storage to `float32`|
|19|torch.TypedStorage.from_buffer|Y|Host-only (no device ops). Constructs CPU storage from host memory|
|20|torch.TypedStorage.get_device|Y|Host-only (no device ops)|
|21|torch.TypedStorage.half|Y|Native. Converts Neuron-backed storage to `float16`|
|22|torch.TypedStorage.hpu|N|HPU-only|
|23|torch.TypedStorage.int|Y|Native. Converts Neuron-backed storage to `int32`|
|24|torch.TypedStorage.is_cuda|Y|Host-only (no device ops)|
|25|torch.TypedStorage.is_hpu|Y|Host-only (no device ops)|
|26|torch.TypedStorage.is_pinned|Y|Host-only (no device ops). Returns whether the Neuron-backed storage is pinned|
|27|torch.TypedStorage.is_shared|Y|Host-only (no device ops)|
|28|torch.TypedStorage.is_sparse|Y|Host-only (no device ops)|
|29|torch.TypedStorage.long|Y|Native. Converts Neuron-backed storage to `int64`|
|30|torch.TypedStorage.nbytes|Y|Host-only (no device ops)|
|31|torch.TypedStorage.pickle_storage_type|Y|Host-only (no device ops)|
|32|torch.TypedStorage.pin_memory|N|Is a CPU host-memory facility and does not apply to Neuron-backed storage|
|33|torch.TypedStorage.resize_|Y|Native-backed storage; resizing performs no tensor computation|
|34|torch.TypedStorage.share_memory_|Y|Host-only (no device ops)|
|35|torch.TypedStorage.short|Y|Native. Converts Neuron-backed storage to `int16`|
|36|torch.TypedStorage.size|Y|Host-only (no device ops)|
|37|torch.TypedStorage.tolist|Y|Host-only (no device ops)|
|38|torch.TypedStorage.type|Y|Host-only (no device ops)|
|39|torch.TypedStorage.untyped|Y|Host-only (no device ops)|
|40|torch.UntypedStorage|Y|Host-only (no device ops)|
|41|torch.UntypedStorage.bfloat16|Y|Native. Converts Neuron-backed storage to `bfloat16`|
|42|torch.UntypedStorage.bool|Y|Native. Converts Neuron-backed storage to `bool`|
|43|torch.UntypedStorage.byte|Y|Native. Converts Neuron-backed storage to `uint8`|
|44|torch.UntypedStorage.byteswap|N|`torch.UntypedStorage.byteswap` terminates the process for Neuron-backed storage|
|45|torch.UntypedStorage.char|Y|Native. Converts Neuron-backed storage to `int8`|
|46|torch.UntypedStorage.clone|Y|Native for `float32` and `bfloat16` storage|
|47|torch.UntypedStorage.complex_double|Y|Native. Converts Neuron-backed storage to `complex128`|
|48|torch.UntypedStorage.complex_float|Y|Native. Converts Neuron-backed storage to `complex64`|
|49|torch.UntypedStorage.copy_|Y|Native for `float32` and `bfloat16` storage|
|50|torch.UntypedStorage.cpu|Y|Host-only (no device ops)|
|51|torch.UntypedStorage.cuda|N|CUDA-only|
|52|torch.UntypedStorage.data_ptr|Y|Host-only (no device ops)|
|53|torch.UntypedStorage.device|Y|Host-only (no device ops)|
|54|torch.UntypedStorage.double|Y|Native. Converts Neuron-backed storage to `float64`|
|55|torch.UntypedStorage.element_size|Y|Host-only (no device ops)|
|56|torch.UntypedStorage.fill_|Y|Native for `float32` and `bfloat16` storage|
|57|torch.UntypedStorage.float|Y|Host-only (no device ops); conversion transfers data through CPU|
|58|torch.UntypedStorage.from_buffer|Y|Host-only (no device ops). Constructs CPU storage from host memory|
|59|torch.UntypedStorage.from_file|Y|Host-only (no device ops). Constructs CPU storage from a host file|
|60|torch.UntypedStorage.get_device|Y|Host-only (no device ops)|
|61|torch.UntypedStorage.half|Y|Native; `float16` conversion copies through the host|
|62|torch.UntypedStorage.hpu|N|HPU-only|
|63|torch.UntypedStorage.int|Y|Native. Converts Neuron-backed storage to `int32`|
|64|torch.UntypedStorage.is_cuda|Y|Host-only (no device ops)|
|65|torch.UntypedStorage.is_hpu|Y|Host-only (no device ops)|
|66|torch.UntypedStorage.is_pinned|Y|Native; returns `False` for Neuron-backed storage|
|67|torch.UntypedStorage.is_shared|Y|Host-only (no device ops)|
|68|torch.UntypedStorage.is_sparse|Y|Host-only (no device ops)|
|69|torch.UntypedStorage.is_sparse_csr|Y|Host-only (no device ops)|
|70|torch.UntypedStorage.long|Y|Native. Converts Neuron-backed storage to `int64`|
|71|torch.UntypedStorage.mps|N|MPS-only|
|72|torch.UntypedStorage.nbytes|Y|Host-only (no device ops)|
|73|torch.UntypedStorage.new|Y|Creates empty storage on the same device; it performs no tensor computation|
|74|torch.UntypedStorage.pin_memory|N|Is a CPU host-memory facility and does not apply to Neuron-backed storage|
|75|torch.UntypedStorage.resize_|Y|Native-backed storage; resizing performs no tensor computation|
|76|torch.UntypedStorage.share_memory_|Y|Host-only (no device ops)|
|77|torch.UntypedStorage.short|Y|Native. Converts Neuron-backed storage to `int16`|
|78|torch.UntypedStorage.size|Y|Host-only (no device ops)|
|79|torch.UntypedStorage.tolist|Y|Host-only (no device ops)|
|80|torch.UntypedStorage.type|Y|Host-only (no device ops)|
|81|torch.UntypedStorage.untyped|Y|Host-only (no device ops)|
|82|torch.DoubleStorage|Y|Host-only (no device ops). This deprecated legacy storage alias is CPU-only|
|83|torch.FloatStorage|Y|Host-only (no device ops). This deprecated legacy storage alias is CPU-only|
|84|torch.HalfStorage|Y|Host-only (no device ops). This deprecated legacy storage alias is CPU-only|
|85|torch.LongStorage|Y|Host-only (no device ops). This deprecated legacy storage alias is CPU-only|
|86|torch.IntStorage|Y|Host-only (no device ops). This deprecated legacy storage alias is CPU-only|
|87|torch.ShortStorage|Y|Host-only (no device ops). This deprecated legacy storage alias is CPU-only|
|88|torch.CharStorage|Y|Host-only (no device ops). This deprecated legacy storage alias is CPU-only|
|89|torch.ByteStorage|Y|Host-only (no device ops). This deprecated legacy storage alias is CPU-only|
|90|torch.BoolStorage|Y|Host-only (no device ops). This deprecated legacy storage alias is CPU-only|
|91|torch.BFloat16Storage|Y|Host-only (no device ops). This deprecated legacy storage alias is CPU-only|
|92|torch.ComplexDoubleStorage|Y|Host-only (no device ops). This deprecated legacy storage alias is CPU-only|
|93|torch.ComplexFloatStorage|Y|Host-only (no device ops). This deprecated legacy storage alias is CPU-only|
|94|torch.QUInt8Storage|Y|Host-only (no device ops). This deprecated legacy storage alias is CPU-only|
|95|torch.QInt8Storage|Y|Host-only (no device ops). This deprecated legacy storage alias is CPU-only|
|96|torch.QInt32Storage|Y|Host-only (no device ops). This deprecated legacy storage alias is CPU-only|
|97|torch.QUInt4x2Storage|Y|Host-only (no device ops). This deprecated legacy storage alias is CPU-only|
|98|torch.QUInt2x4Storage|Y|Host-only (no device ops). This deprecated legacy storage alias is CPU-only|

## torch.utils.data

||pytorch_api|neuron_compatible|limitations|
|---|---|---|---|
|1|DataLoader|Y|Host-only (no device ops); single-process only, move batches to Neuron|
|2|Dataset|Y|Host-only (no device ops)|
|3|IterableDataset|Y|Host-only (no device ops)|
|4|TensorDataset|Y|Native when indexing `float32` or `bfloat16` Neuron-backed tensors|
|5|StackDataset|Y|Host-only (no device ops)|
|6|ConcatDataset|Y|Host-only (no device ops)|
|7|ChainDataset|Y|Host-only (no device ops)|
|8|Subset|Y|Host-only (no device ops)|
|9|default_collate|Y|Native for `float32` and `bfloat16` tensor batches (`aten::stack`)|
|10|default_convert|Y|Host-only (no device ops). `default_convert` provides data-loading orchestration|
|11|get_worker_info|Y|Host-only (no device ops). `get_worker_info` provides data-loading orchestration|
|12|random_split|Y|Host-only (no device ops). Produces dataset subsets from host-generated indices|
|13|Sampler|Y|Host-only (no device ops)|
|14|SequentialSampler|Y|Host-only (no device ops)|
|15|RandomSampler|Y|Host-only (no device ops). `RandomSampler` generates or groups dataset indices|
|16|SubsetRandomSampler|Y|Host-only (no device ops)|
|17|WeightedRandomSampler|Y|Host-only (no device ops)|
|18|BatchSampler|Y|Host-only (no device ops). `BatchSampler` generates or groups dataset indices|
|19|DistributedSampler|Y|Host-only (no device ops)|

## Type Info

||pytorch_api|neuron_compatible|limitations|
|---|---|---|---|
|1|torch.finfo|Y|Host-only (no device ops). Returns floating-point dtype metadata|
|2|torch.iinfo|Y|Host-only (no device ops). Returns integer dtype metadata|

## Removed APIs

||pytorch_api|neuron_compatible|limitations|
|---|---|---|---|
|1|torch.distributed.broadcast_multigpu|N|Removed in PyTorch v2.11|
|2|torch.distributed.all_reduce_multigpu|N|Removed in PyTorch v2.11|
|3|torch.distributed.reduce_multigpu|N|Removed in PyTorch v2.11|
|4|torch.distributed.all_gather_multigpu|N|Removed in PyTorch v2.11|
|5|torch.distributed.reduce_scatter_multigpu|N|Removed in PyTorch v2.11|
|6|PairwiseParallel|N|Removed in PyTorch v2.11|
|7|enable_2d_with_fsdp|N|Removed in PyTorch v2.11|
|8|pre_dp_module_transform|N|Removed in PyTorch v2.11|
