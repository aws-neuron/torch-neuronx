# Torch API Compatibility


## torch APIs

||pytorch_api|neuron_compatible|limitations|
|---|---|---|---|
|1|torch.default_generator|Y|Pure Python; Neuron registers NeuronGeneratorImpl|
|2|torch.SymInt|Y|Pure Python symbolic type|
|3|torch.SymFloat|Y|Pure Python symbolic type|
|4|torch.SymBool|Y|Pure Python symbolic type|
|5|torch.Tag|Y|Pure Python enum|
|6|torch.is_tensor|Y|Pure Python type check|
|7|torch.is_storage|Y|Pure Python type check|
|8|torch.is_complex|Y|Pure Python dtype check|
|9|torch.is_conj|Y|Pure Python metadata check|
|10|torch.is_floating_point|Y|Pure Python dtype check|
|11|torch.is_nonzero|Y|Uses item() internally; copies to CPU|
|12|torch.set_default_dtype|Y|Pure Python global setting|
|13|torch.get_default_dtype|Y|Pure Python global setting|
|14|torch.set_default_device|Y|Pure Python global setting|
|15|torch.set_default_tensor_type|Y|Pure Python global setting|
|16|torch.numel|Y|Metadata only|
|17|torch.set_printoptions|Y|Pure Python|
|18|torch.set_flush_denormal|Y|Pure Python CPU setting|
|19|torch.tensor|Y|Creates tensor; uses copy_ to move to device|
|20|torch.sparse_coo_tensor|CPU_FALLBACK|Sparse not supported on Neuron|
|21|torch.sparse_csr_tensor|CPU_FALLBACK|Sparse not supported on Neuron|
|22|torch.sparse_csc_tensor|CPU_FALLBACK|Sparse not supported on Neuron|
|23|torch.sparse_bsr_tensor|CPU_FALLBACK|Sparse not supported on Neuron|
|24|torch.sparse_bsc_tensor|CPU_FALLBACK|Sparse not supported on Neuron|
|25|torch.asarray|Y|Pure Python; delegates to tensor/as_tensor|
|26|torch.as_tensor|Y|Pure Python; shares storage if possible|
|27|torch.as_strided|Y|Native C++ impl (as_strided_neuron)|
|28|torch.from_numpy|Y|Creates CPU tensor from numpy; user moves to device|
|29|torch.from_dlpack|CPU_FALLBACK|DLPack not supported on Neuron|
|30|torch.frombuffer|Y|Creates CPU tensor from buffer|
|31|torch.zeros|Y|Native MLIR impl|
|32|torch.zeros_like|Y|Uses empty + zero_|
|33|torch.ones|Y|Native MLIR impl|
|34|torch.ones_like|Y|Native MLIR impl|
|35|torch.arange|Y|Native MLIR impl|
|36|torch.range|Y|Deprecated; delegates to arange|
|37|torch.linspace|CPU_FALLBACK|aten::linspace not registered|
|38|torch.logspace|CPU_FALLBACK|aten::logspace not registered|
|39|torch.eye|Y|Native MLIR impl|
|40|torch.empty|Y|Native C++ impl|
|41|torch.empty_like|Y|Uses empty internally|
|42|torch.empty_strided|Y|Native C++ impl|
|43|torch.full|Y|Native MLIR impl|
|44|torch.full_like|Y|Uses full internally|
|45|torch.quantize_per_tensor|CPU_FALLBACK|Quantization not natively supported|
|46|torch.quantize_per_channel|CPU_FALLBACK|Quantization not natively supported|
|47|torch.dequantize|CPU_FALLBACK|Quantization not natively supported|
|48|torch.complex|CPU_FALLBACK|Complex not supported on Neuron|
|49|torch.polar|CPU_FALLBACK|Complex not supported on Neuron|
|50|torch.heaviside|CPU_FALLBACK|aten::heaviside not registered|
|51|torch.adjoint|Y|View operation|
|52|torch.argwhere|Y|Decomposes to nonzero (supported)|
|53|torch.cat|Y|Native MLIR impl|
|54|torch.concat|Y|Alias for cat|
|55|torch.concatenate|Y|Alias for cat|
|56|torch.conj|Y|View operation|
|57|torch.chunk|Y|Decomposes to split|
|58|torch.dsplit|Y|Decomposes to split|
|59|torch.column_stack|Y|CIA; decomposes to cat + reshape|
|60|torch.dstack|Y|CIA; decomposes to cat + reshape|
|61|torch.gather|Y|Native MLIR impl|
|62|torch.hsplit|Y|Decomposes to split|
|63|torch.hstack|Y|CIA; decomposes to cat|
|64|torch.index_add|Y|Native MLIR impl|
|65|torch.index_copy|Y|Native MLIR impl|
|66|torch.index_reduce|CPU_FALLBACK|aten::index_reduce not registered|
|67|torch.index_select|Y|Native MLIR impl|
|68|torch.masked_select|Y|Native MLIR impl|
|69|torch.movedim|Y|View operation|
|70|torch.moveaxis|Y|Alias for movedim|
|71|torch.narrow|Y|View operation (slice)|
|72|torch.narrow_copy|CPU_FALLBACK|aten::narrow_copy not registered|
|73|torch.nonzero|Y|Native Python impl|
|74|torch.permute|Y|View operation|
|75|torch.reshape|Y|Decomposes to view or clone+view|
|76|torch.row_stack|Y|Alias for vstack; decomposes to cat|
|77|torch.select|Y|View operation|
|78|torch.scatter|Y|Native MLIR impl|
|79|torch.diagonal_scatter|CPU_FALLBACK|aten::diagonal_scatter not registered|
|80|torch.select_scatter|CPU_FALLBACK|aten::select_scatter not registered|
|81|torch.slice_scatter|CPU_FALLBACK|aten::slice_scatter not registered|
|82|torch.scatter_add|Y|Native MLIR impl|
|83|torch.scatter_reduce|CPU_FALLBACK|aten::scatter_reduce not registered|
|84|torch.split|Y|View operation|
|85|torch.squeeze|Y|View operation|
|86|torch.stack|Y|Native MLIR impl|
|87|torch.swapaxes|Y|Alias for transpose; view operation|
|88|torch.swapdims|Y|Alias for transpose; view operation|
|89|torch.t|Y|View operation|
|90|torch.take|CPU_FALLBACK|aten::take not registered|
|91|torch.take_along_dim|Y|Decomposes to gather (supported)|
|92|torch.tensor_split|Y|View operation|
|93|torch.tile|Y|Decomposes to repeat|
|94|torch.transpose|Y|View operation|
|95|torch.unbind|Y|View operation|
|96|torch.unsqueeze|Y|View operation|
|97|torch.vsplit|Y|Decomposes to split|
|98|torch.vstack|Y|CIA; decomposes to cat|
|99|torch.where|Y|Native MLIR impl|
|100|torch.Generator|Y|Neuron registers NeuronGeneratorImpl (Philox-based)|
|101|torch.Generator.device|Y|Returns neuron device|
|102|torch.Generator.get_state|Y|Returns generator state|
|103|torch.Generator.initial_seed|Y|Returns initial seed|
|104|torch.Generator.manual_seed|Y|Sets seed|
|105|torch.Generator.seed|Y|Sets non-deterministic seed|
|106|torch.Generator.set_state|Y|Sets generator state|
|107|torch.seed|Y|Pure Python; sets seed on all devices|
|108|torch.manual_seed|Y|Pure Python; sets seed on all devices|
|109|torch.initial_seed|Y|Pure Python|
|110|torch.get_rng_state|Y|Returns CPU RNG state|
|111|torch.set_rng_state|Y|Sets CPU RNG state|
|112|torch.bernoulli|CPU_FALLBACK|aten::bernoulli not registered on Neuron|
|113|torch.multinomial|CPU_FALLBACK|aten::multinomial not registered|
|114|torch.normal|CPU_FALLBACK|aten::normal_ not registered|
|115|torch.poisson|CPU_FALLBACK|aten::poisson not registered|
|116|torch.rand|CPU_FALLBACK|aten::rand not registered|
|117|torch.rand_like|CPU_FALLBACK|aten::rand_like not registered|
|118|torch.randint|CPU_FALLBACK|aten::randint not registered|
|119|torch.randint_like|CPU_FALLBACK|aten::randint_like not registered|
|120|torch.randn|CPU_FALLBACK|aten::randn not registered|
|121|torch.randn_like|CPU_FALLBACK|aten::randn_like not registered|
|122|torch.randperm|CPU_FALLBACK|aten::randperm not registered|
|123|torch.quasirandom.SobolEngine|Y|Pure Python|
|124|torch.save|Y|Pure Python serialization|
|125|torch.load|Y|Pure Python serialization|
|126|torch.get_num_threads|Y|Pure Python CPU setting|
|127|torch.set_num_threads|Y|Pure Python CPU setting|
|128|torch.get_num_interop_threads|Y|Pure Python CPU setting|
|129|torch.set_num_interop_threads|Y|Pure Python CPU setting|
|130|torch.no_grad|Y|Pure Python context manager|
|131|torch.enable_grad|Y|Pure Python context manager|
|132|torch.set_grad_enabled|Y|Pure Python|
|133|torch.is_grad_enabled|Y|Pure Python|
|134|torch.inference_mode|Y|Pure Python context manager|
|135|torch.is_inference_mode_enabled|Y|Pure Python|
|136|torch.abs|Y|Native MLIR impl|
|137|torch.absolute|Y|Alias for abs|
|138|torch.acos|CPU_FALLBACK|aten::acos not registered|
|139|torch.arccos|CPU_FALLBACK|Alias for acos|
|140|torch.acosh|CPU_FALLBACK|aten::acosh not registered|
|141|torch.arccosh|CPU_FALLBACK|Alias for acosh|
|142|torch.add|Y|Native MLIR impl; supports alpha|
|143|torch.addcdiv|Y|Native MLIR impl|
|144|torch.addcmul|Y|Native MLIR impl|
|145|torch.angle|CPU_FALLBACK|aten::angle not registered|
|146|torch.asin|CPU_FALLBACK|aten::asin not registered|
|147|torch.arcsin|CPU_FALLBACK|Alias for asin|
|148|torch.asinh|CPU_FALLBACK|aten::asinh not registered|
|149|torch.arcsinh|CPU_FALLBACK|Alias for asinh|
|150|torch.atan|CPU_FALLBACK|aten::atan not registered|
|151|torch.arctan|CPU_FALLBACK|Alias for atan|
|152|torch.atanh|CPU_FALLBACK|aten::atanh not registered|
|153|torch.arctanh|CPU_FALLBACK|Alias for atanh|
|154|torch.atan2|CPU_FALLBACK|aten::atan2 not registered|
|155|torch.arctan2|CPU_FALLBACK|Alias for atan2|
|156|torch.bitwise_not|Y|Native MLIR impl|
|157|torch.bitwise_and|Y|Native MLIR impl|
|158|torch.bitwise_or|Y|Native MLIR impl|
|159|torch.bitwise_xor|Y|Native MLIR impl|
|160|torch.bitwise_left_shift|Y|Native MLIR impl|
|161|torch.bitwise_right_shift|Y|Native MLIR impl|
|162|torch.ceil|Y|Native MLIR impl|
|163|torch.clamp|Y|Native MLIR impl; supports Tensor min/max|
|164|torch.clip|Y|Alias for clamp|
|165|torch.conj_physical|CPU_FALLBACK|Complex not supported|
|166|torch.copysign|CPU_FALLBACK|aten::copysign not registered|
|167|torch.cos|Y|Native MLIR impl|
|168|torch.cosh|CPU_FALLBACK|aten::cosh not registered|
|169|torch.deg2rad|CPU_FALLBACK|aten::deg2rad not registered|
|170|torch.div|Y|Native MLIR impl; supports rounding_mode|
|171|torch.divide|Y|Alias for div|
|172|torch.digamma|CPU_FALLBACK|aten::digamma not registered|
|173|torch.erf|Y|Native MLIR impl|
|174|torch.erfc|CPU_FALLBACK|aten::erfc not registered|
|175|torch.erfinv|Y|Native MLIR impl (polynomial approximation)|
|176|torch.exp|Y|Native MLIR impl|
|177|torch.exp2|CPU_FALLBACK|aten::exp2 not registered|
|178|torch.expm1|CPU_FALLBACK|aten::expm1 not registered|
|179|torch.fake_quantize_per_channel_affine|CPU_FALLBACK|Quantization not supported|
|180|torch.fake_quantize_per_tensor_affine|CPU_FALLBACK|Quantization not supported|
|181|torch.fix|Y|Alias for trunc; native MLIR impl|
|182|torch.float_power|Y|Decomposes to pow + dtype cast|
|183|torch.floor|CPU_FALLBACK|aten::floor not registered|
|184|torch.floor_divide|Y|Native MLIR impl|
|185|torch.fmod|CPU_FALLBACK|aten::fmod not registered|
|186|torch.frac|CPU_FALLBACK|aten::frac not registered|
|187|torch.frexp|CPU_FALLBACK|aten::frexp not registered|
|188|torch.gradient|CPU_FALLBACK|Pure Python but uses unsupported ops|
|189|torch.imag|Y|Metadata operation|
|190|torch.ldexp|CPU_FALLBACK|aten::ldexp not registered|
|191|torch.lerp|Y|Native MLIR impl|
|192|torch.lgamma|CPU_FALLBACK|aten::lgamma not registered|
|193|torch.log|Y|Native MLIR impl|
|194|torch.log10|CPU_FALLBACK|aten::log10 not registered|
|195|torch.log1p|CPU_FALLBACK|aten::log1p not registered|
|196|torch.log2|CPU_FALLBACK|aten::log2 not registered|
|197|torch.logaddexp|CPU_FALLBACK|aten::logaddexp not registered|
|198|torch.logaddexp2|CPU_FALLBACK|aten::logaddexp2 not registered|
|199|torch.logical_and|Y|Native MLIR impl|
|200|torch.logical_not|Y|Native MLIR impl|
|201|torch.logical_or|Y|Native MLIR impl|
|202|torch.logical_xor|Y|Native MLIR impl|
|203|torch.logit|CPU_FALLBACK|aten::logit not registered|
|204|torch.hypot|CPU_FALLBACK|aten::hypot not registered|
|205|torch.i0|CPU_FALLBACK|aten::i0 not registered|
|206|torch.igamma|CPU_FALLBACK|aten::igamma not registered|
|207|torch.igammac|CPU_FALLBACK|aten::igammac not registered|
|208|torch.mul|Y|Native MLIR impl|
|209|torch.multiply|Y|Alias for mul|
|210|torch.mvlgamma|CPU_FALLBACK|aten::mvlgamma not registered|
|211|torch.nan_to_num|CPU_FALLBACK|aten::nan_to_num not registered|
|212|torch.neg|Y|Native MLIR impl|
|213|torch.negative|Y|Alias for neg|
|214|torch.nextafter|CPU_FALLBACK|aten::nextafter not registered|
|215|torch.polygamma|CPU_FALLBACK|aten::polygamma not registered|
|216|torch.positive|Y|Returns self (no-op)|
|217|torch.pow|Y|Native MLIR impl|
|218|torch.quantized_batch_norm|CPU_FALLBACK|Quantization not supported|
|219|torch.quantized_max_pool1d|CPU_FALLBACK|Quantization not supported|
|220|torch.quantized_max_pool2d|CPU_FALLBACK|Quantization not supported|
|221|torch.rad2deg|CPU_FALLBACK|aten::rad2deg not registered|
|222|torch.real|Y|Metadata operation|
|223|torch.reciprocal|Y|Native MLIR impl|
|224|torch.remainder|Y|Native MLIR impl|
|225|torch.round|CPU_FALLBACK|aten::round not registered|
|226|torch.rsqrt|Y|Native MLIR impl|
|227|torch.sigmoid|Y|Native MLIR impl|
|228|torch.sign|Y|Native MLIR impl|
|229|torch.sgn|Y|Native MLIR impl|
|230|torch.signbit|Y|Native MLIR impl|
|231|torch.sin|Y|Native MLIR impl|
|232|torch.sinc|CPU_FALLBACK|aten::sinc not registered|
|233|torch.sinh|CPU_FALLBACK|aten::sinh not registered|
|234|torch.softmax|Y|Native MLIR impl|
|235|torch.sqrt|Y|Native MLIR impl|
|236|torch.square|Y|Decomposes to pow(self, 2)|
|237|torch.sub|Y|Native MLIR impl|
|238|torch.subtract|Y|Alias for sub|
|239|torch.tan|CPU_FALLBACK|aten::tan not registered|
|240|torch.tanh|Y|Native MLIR impl|
|241|torch.true_divide|Y|Alias for div|
|242|torch.trunc|Y|Native MLIR impl|
|243|torch.xlogy|CPU_FALLBACK|aten::xlogy not registered|
|244|torch.argmax|CPU_FALLBACK|aten::argmax not registered|
|245|torch.argmin|CPU_FALLBACK|aten::argmin not registered|
|246|torch.amax|Y|Native MLIR impl|
|247|torch.amin|Y|Native MLIR impl|
|248|torch.aminmax|CPU_FALLBACK|aten::aminmax not registered|
|249|torch.all|Y|Native MLIR impl|
|250|torch.any|Y|Native MLIR impl|
|251|torch.max|Y|Native MLIR impl|
|252|torch.min|CPU_FALLBACK|aten::min not registered (only max is)|
|253|torch.dist|CPU_FALLBACK|aten::dist not registered|
|254|torch.logsumexp|CPU_FALLBACK|aten::logsumexp not registered|
|255|torch.mean|Y|Native MLIR impl|
|256|torch.nanmean|CPU_FALLBACK|aten::nanmean not registered|
|257|torch.median|CPU_FALLBACK|aten::median not registered|
|258|torch.nanmedian|CPU_FALLBACK|aten::nanmedian not registered|
|259|torch.mode|CPU_FALLBACK|aten::mode not registered|
|260|torch.norm|Y|Decomposes to linalg_vector_norm (registered)|
|261|torch.nansum|CPU_FALLBACK|aten::nansum not registered|
|262|torch.prod|CPU_FALLBACK|aten::prod not registered|
|263|torch.quantile|CPU_FALLBACK|aten::quantile not registered|
|264|torch.nanquantile|CPU_FALLBACK|aten::nanquantile not registered|
|265|torch.std|CPU_FALLBACK|aten::std not registered|
|266|torch.std_mean|CPU_FALLBACK|aten::std_mean not registered|
|267|torch.sum|Y|Native MLIR impl|
|268|torch.unique|CPU_FALLBACK|aten::unique not registered|
|269|torch.unique_consecutive|CPU_FALLBACK|aten::unique_consecutive not registered|
|270|torch.var|CPU_FALLBACK|aten::var not registered|
|271|torch.var_mean|CPU_FALLBACK|aten::var_mean not registered|
|272|torch.count_nonzero|CPU_FALLBACK|aten::count_nonzero not registered|
|273|torch.allclose|CPU_FALLBACK|aten::allclose not registered|
|274|torch.argsort|CPU_FALLBACK|aten::sort not registered|
|275|torch.eq|Y|Native MLIR impl|
|276|torch.equal|CPU_FALLBACK|aten::equal not registered (returns single bool)|
|277|torch.ge|Y|Native MLIR impl|
|278|torch.greater_equal|Y|Alias for ge|
|279|torch.gt|Y|Native MLIR impl|
|280|torch.greater|Y|Alias for gt|
|281|torch.isclose|Y|CIA; decomposes to eq, abs, mul, add, le (all supported)|
|282|torch.isfinite|Y|Native MLIR impl|
|283|torch.isin|CPU_FALLBACK|aten::isin not registered|
|284|torch.isinf|Y|Native MLIR impl|
|285|torch.isposinf|CPU_FALLBACK|aten::isposinf not registered|
|286|torch.isneginf|Y|Native MLIR impl|
|287|torch.isnan|Y|Native MLIR impl|
|288|torch.isreal|Y|Metadata check only|
|289|torch.kthvalue|CPU_FALLBACK|aten::kthvalue not registered|
|290|torch.le|Y|Native MLIR impl|
|291|torch.less_equal|Y|Alias for le|
|292|torch.lt|Y|Native MLIR impl|
|293|torch.less|Y|Alias for lt|
|294|torch.maximum|CPU_FALLBACK|aten::maximum not registered|
|295|torch.minimum|CPU_FALLBACK|aten::minimum not registered|
|296|torch.fmax|CPU_FALLBACK|aten::fmax not registered|
|297|torch.fmin|CPU_FALLBACK|aten::fmin not registered|
|298|torch.ne|Y|Native MLIR impl|
|299|torch.not_equal|Y|Alias for ne|
|300|torch.sort|CPU_FALLBACK|aten::sort not registered|
|301|torch.topk|Y|Native MLIR impl|
|302|torch.msort|CPU_FALLBACK|Decomposes to sort; sort not registered|
|303|torch.stft|CPU_FALLBACK|aten::stft not registered|
|304|torch.istft|CPU_FALLBACK|aten::istft not registered|
|305|torch.bartlett_window|CPU_FALLBACK|aten::bartlett_window has backend dispatch|
|306|torch.blackman_window|CPU_FALLBACK|aten::blackman_window has backend dispatch|
|307|torch.hamming_window|CPU_FALLBACK|aten::hamming_window has backend dispatch|
|308|torch.hann_window|CPU_FALLBACK|aten::hann_window has backend dispatch|
|309|torch.kaiser_window|CPU_FALLBACK|aten::kaiser_window has backend dispatch|
|310|torch.atleast_1d|Y|CIA; view operation|
|311|torch.atleast_2d|Y|CIA; view operation|
|312|torch.atleast_3d|Y|CIA; view operation|
|313|torch.bincount|CPU_FALLBACK|aten::bincount not registered|
|314|torch.block_diag|CPU_FALLBACK|aten::block_diag not registered|
|315|torch.broadcast_tensors|Y|CIA; uses expand (view op)|
|316|torch.broadcast_to|Y|CIA; uses expand (view op)|
|317|torch.broadcast_shapes|Y|Pure Python|
|318|torch.bucketize|CPU_FALLBACK|aten::bucketize not registered|
|319|torch.cartesian_prod|CPU_FALLBACK|aten::cartesian_prod not registered|
|320|torch.cdist|CPU_FALLBACK|aten::cdist not registered|
|321|torch.clone|Y|Native C++ impl|
|322|torch.combinations|CPU_FALLBACK|aten::combinations not registered|
|323|torch.corrcoef|CPU_FALLBACK|aten::corrcoef not registered|
|324|torch.cov|CPU_FALLBACK|aten::cov not registered|
|325|torch.cross|CPU_FALLBACK|aten::linalg_cross has backend dispatch|
|326|torch.cummax|CPU_FALLBACK|aten::cummax not registered|
|327|torch.cummin|CPU_FALLBACK|aten::cummin not registered|
|328|torch.cumprod|CPU_FALLBACK|aten::cumprod not registered|
|329|torch.cumsum|Y|Native MLIR impl|
|330|torch.diag|Y|CIA; decomposes to diag_embed or diagonal_copy|
|331|torch.diag_embed|CPU_FALLBACK|aten::diag_embed not registered|
|332|torch.diagflat|CPU_FALLBACK|Decomposes to flatten + diag_embed; diag_embed not registered|
|333|torch.diagonal|Y|View operation|
|334|torch.diff|Y|CIA; decomposes to narrow + sub (both supported)|
|335|torch.einsum|Y|CIA; decomposes to mm/bmm/sum (supported)|
|336|torch.flatten|Y|Decomposes to reshape|
|337|torch.flip|Y|Native MLIR impl|
|338|torch.fliplr|Y|Decomposes to flip|
|339|torch.flipud|Y|Decomposes to flip|
|340|torch.kron|CPU_FALLBACK|aten::kron not registered|
|341|torch.rot90|Y|CIA; decomposes to flip + transpose (supported)|
|342|torch.gcd|CPU_FALLBACK|aten::gcd not registered|
|343|torch.histc|CPU_FALLBACK|aten::histc has backend dispatch|
|344|torch.histogram|CPU_FALLBACK|aten::histogram not registered|
|345|torch.histogramdd|CPU_FALLBACK|aten::histogramdd not registered|
|346|torch.meshgrid|Y|CIA; uses expand + reshape (supported)|
|347|torch.lcm|CPU_FALLBACK|aten::lcm not registered|
|348|torch.logcumsumexp|CPU_FALLBACK|aten::logcumsumexp not registered|
|349|torch.ravel|Y|Decomposes to flatten|
|350|torch.renorm|CPU_FALLBACK|aten::renorm not registered|
|351|torch.repeat_interleave|Y|Native MLIR impl with preprocessing|
|352|torch.roll|CPU_FALLBACK|aten::roll not registered|
|353|torch.searchsorted|Y|Decomposed into comparison + sum|
|354|torch.tensordot|CPU_FALLBACK|aten::tensordot not registered|
|355|torch.trace|CPU_FALLBACK|aten::trace not registered|
|356|torch.tril|Y|Native MLIR impl|
|357|torch.tril_indices|CPU_FALLBACK|aten::tril_indices not registered|
|358|torch.triu|Y|Native MLIR impl|
|359|torch.triu_indices|CPU_FALLBACK|aten::triu_indices not registered|
|360|torch.unflatten|Y|Decomposes to reshape|
|361|torch.vander|Y|CIA (explicit); Vandermonde using supported ops|
|362|torch.view_as_real|CPU_FALLBACK|Complex not supported|
|363|torch.view_as_complex|CPU_FALLBACK|Complex not supported|
|364|torch.resolve_conj|Y|Metadata operation|
|365|torch.resolve_neg|Y|Metadata operation|
|366|torch.addbmm|CPU_FALLBACK|aten::addbmm not registered|
|367|torch.addmm|Y|Native MLIR impl|
|368|torch.addmv|Y|Native MLIR impl|
|369|torch.addr|CPU_FALLBACK|aten::addr not registered|
|370|torch.baddbmm|CPU_FALLBACK|aten::baddbmm not registered|
|371|torch.bmm|Y|Native MLIR impl|
|372|torch.chain_matmul|Y|Deprecated; decomposes to chain of mm|
|373|torch.cholesky|CPU_FALLBACK|aten::cholesky not registered|
|374|torch.cholesky_inverse|CPU_FALLBACK|aten::cholesky_inverse not registered|
|375|torch.cholesky_solve|CPU_FALLBACK|aten::cholesky_solve not registered|
|376|torch.dot|Y|Native MLIR impl|
|377|torch.geqrf|CPU_FALLBACK|aten::geqrf not registered|
|378|torch.ger|Y|CIA; decomposes to reshape + mul (supported)|
|379|torch.inner|Y|CIA; decomposes to mul + sum (supported)|
|380|torch.inverse|CPU_FALLBACK|aten::linalg_inv_ex not registered|
|381|torch.det|CPU_FALLBACK|aten::_linalg_det not registered|
|382|torch.logdet|CPU_FALLBACK|Decomposes to linalg_slogdet; not registered|
|383|torch.slogdet|CPU_FALLBACK|aten::_linalg_slogdet not registered|
|384|torch.lu|CPU_FALLBACK|aten::linalg_lu not registered|
|385|torch.lu_solve|CPU_FALLBACK|aten::linalg_lu_solve not registered|
|386|torch.lu_unpack|CPU_FALLBACK|aten::lu_unpack not registered|
|387|torch.matmul|Y|CIA; decomposes to mm/bmm (registered)|
|388|torch.matrix_power|CPU_FALLBACK|Uses linalg_inv for negative powers|
|389|torch.matrix_exp|CPU_FALLBACK|aten::linalg_matrix_exp not registered|
|390|torch.mm|Y|Native MLIR impl|
|391|torch.mv|Y|Native MLIR impl|
|392|torch.orgqr|CPU_FALLBACK|aten::linalg_householder_product not registered|
|393|torch.ormqr|CPU_FALLBACK|aten::ormqr not registered|
|394|torch.outer|Y|CIA; decomposes to reshape + mul (supported)|
|395|torch.pinverse|CPU_FALLBACK|aten::linalg_pinv not registered|
|396|torch.qr|CPU_FALLBACK|aten::linalg_qr not registered|
|397|torch.svd|CPU_FALLBACK|aten::_linalg_svd not registered|
|398|torch.svd_lowrank|CPU_FALLBACK|Uses SVD internally|
|399|torch.pca_lowrank|CPU_FALLBACK|Uses SVD internally|
|400|torch.lobpcg|CPU_FALLBACK|Complex iterative algorithm|
|401|torch.trapz|CPU_FALLBACK|Deprecated; decomposes to trapezoid|
|402|torch.trapezoid|Y|CIA; decomposes to mul + sum (supported)|
|403|torch.cumulative_trapezoid|Y|CIA; decomposes to diff + mul + cumsum (supported)|
|404|torch.triangular_solve|CPU_FALLBACK|aten::triangular_solve not registered|
|405|torch.vdot|CPU_FALLBACK|aten::vdot not registered|
|406|torch._foreach_add (all variants)|Y|Native MLIR impl|
|407|torch._foreach_mul (all variants)|Y|Native MLIR impl|
|408|torch._foreach_sub (all variants)|Y|Native MLIR impl|
|409|torch._foreach_div (all variants)|Y|Native MLIR impl|
|410|torch._foreach_norm|Y|Native MLIR impl|
|411|torch._foreach_* (other variants)|CPU_FALLBACK|Only add/mul/sub/div/norm registered|
|412|torch.compiled_with_cxx11_abi|Y|Pure Python|
|413|torch.result_type|Y|Pure Python|
|414|torch.can_cast|Y|Pure Python|
|415|torch.promote_types|Y|Pure Python|
|416|torch.use_deterministic_algorithms|Y|Pure Python|
|417|torch.are_deterministic_algorithms_enabled|Y|Pure Python|
|418|torch.is_deterministic_algorithms_warn_only_enabled|Y|Pure Python|
|419|torch.set_deterministic_debug_mode|Y|Pure Python|
|420|torch.get_deterministic_debug_mode|Y|Pure Python|
|421|torch.set_float32_matmul_precision|Y|Pure Python global setting|
|422|torch.get_float32_matmul_precision|Y|Pure Python global setting|
|423|torch.set_warn_always|Y|Pure Python|
|424|torch.is_warn_always_enabled|Y|Pure Python|
|425|torch.vmap|Y|Pure Python functional transform|
|426|torch._assert|Y|Pure Python|
|427|torch.sym_float|Y|Pure Python symbolic type|
|428|torch.sym_int|Y|Pure Python symbolic type|
|429|torch.sym_max|Y|Pure Python symbolic type|
|430|torch.sym_min|Y|Pure Python symbolic type|
|431|torch.sym_not|Y|Pure Python symbolic type|
|432|torch.compile|Y|Pure Python; Neuron has custom dynamo backend|

## torch.nn

||pytorch_api|neuron_compatible|limitations|
|---|---|---|---|
|1|torch.nn.parameter.Parameter|Y|Pure Python container/infrastructure|
|2|torch.nn.parameter.UninitializedParameter|Y|Pure Python container/infrastructure|
|3|torch.nn.parameter.UninitializedParameter.cls_to_become|Y|Pure Python module method|
|4|torch.nn.parameter.UninitializedBuffer|Y|Pure Python container/infrastructure|
|5|torch.nn.Module|Y|Pure Python container/infrastructure|
|6|torch.nn.Module.add_module|Y|Pure Python module method|
|7|torch.nn.Module.apply|Y|Pure Python module method|
|8|torch.nn.Module.bfloat16|Y|Pure Python module method|
|9|torch.nn.Module.buffers|Y|Pure Python module method|
|10|torch.nn.Module.children|Y|Pure Python module method|
|11|torch.nn.Module.compile|Y|Pure Python module method|
|12|torch.nn.Module.cpu|Y|Pure Python module method|
|13|torch.nn.Module.cuda|N/A|Other backend specific|
|14|torch.nn.Module.double|Y|Pure Python module method|
|15|torch.nn.Module.eval|Y|Pure Python; sets module to eval mode|
|16|torch.nn.Module.extra_repr|Y|Pure Python module method|
|17|torch.nn.Module.float|Y|Pure Python module method|
|18|torch.nn.Module.forward|Y|Pure Python module method|
|19|torch.nn.Module.get_buffer|Y|Pure Python module method|
|20|torch.nn.Module.get_extra_state|Y|Pure Python module method|
|21|torch.nn.Module.get_parameter|Y|Pure Python module method|
|22|torch.nn.Module.get_submodule|Y|Pure Python module method|
|23|torch.nn.Module.half|Y|Pure Python module method|
|24|torch.nn.Module.ipu|N/A|Other backend specific|
|25|torch.nn.Module.load_state_dict|Y|Pure Python; loads parameters from dict|
|26|torch.nn.Module.modules|Y|Pure Python module method|
|27|torch.nn.Module.named_buffers|Y|Pure Python module method|
|28|torch.nn.Module.named_children|Y|Pure Python module method|
|29|torch.nn.Module.named_modules|Y|Pure Python module method|
|30|torch.nn.Module.named_parameters|Y|Pure Python module method|
|31|torch.nn.Module.parameters|Y|Pure Python; returns parameter iterator|
|32|torch.nn.Module.register_backward_hook|Y|Pure Python module method|
|33|torch.nn.Module.register_buffer|Y|Pure Python module method|
|34|torch.nn.Module.register_forward_hook|Y|Pure Python module method|
|35|torch.nn.Module.register_forward_pre_hook|Y|Pure Python module method|
|36|torch.nn.Module.register_full_backward_hook|Y|Pure Python module method|
|37|torch.nn.Module.register_full_backward_pre_hook|Y|Pure Python module method|
|38|torch.nn.Module.register_load_state_dict_post_hook|Y|Pure Python module method|
|39|torch.nn.Module.register_module|Y|Pure Python module method|
|40|torch.nn.Module.register_parameter|Y|Pure Python module method|
|41|torch.nn.Module.register_state_dict_pre_hook|Y|Pure Python module method|
|42|torch.nn.Module.requires_grad_|Y|Pure Python module method|
|43|torch.nn.Module.set_extra_state|Y|Pure Python module method|
|44|torch.nn.Module.share_memory|Y|Pure Python module method|
|45|torch.nn.Module.state_dict|Y|Pure Python; returns parameter dict|
|46|torch.nn.Module.to|Y|Pure Python; moves module to device/dtype|
|47|torch.nn.Module.to_empty|Y|Pure Python module method|
|48|torch.nn.Module.train|Y|Pure Python; sets module to training mode|
|49|torch.nn.Module.type|Y|Pure Python module method|
|50|torch.nn.Module.xpu|N/A|Other backend specific|
|51|torch.nn.Module.zero_grad|Y|Forward runs on Neuron|
|52|torch.nn.Sequential|Y|Forward runs on Neuron|
|53|torch.nn.Sequential.append|Y|Pure Python module method|
|54|torch.nn.ModuleList|Y|Pure Python container/infrastructure|
|55|torch.nn.ModuleList.append|Y|Pure Python module method|
|56|torch.nn.ModuleList.extend|Y|Pure Python module method|
|57|torch.nn.ModuleList.insert|Y|Pure Python module method|
|58|torch.nn.ModuleDict|Y|Pure Python container/infrastructure|
|59|torch.nn.ModuleDict.clear|Y|Pure Python module methods|
|60|torch.nn.ModuleDict.items|Y|Pure Python module methods|
|61|torch.nn.ModuleDict.keys|Y|Pure Python module methods|
|62|torch.nn.ModuleDict.pop|Y|Pure Python module methods|
|63|torch.nn.ModuleDict.update|Y|Pure Python module methods|
|64|torch.nn.ModuleDict.values|Y|Pure Python module methods|
|65|torch.nn.ParameterList|Y|Pure Python container/infrastructure|
|66|torch.nn.ParameterList.append|Y|Pure Python module methods|
|67|torch.nn.ParameterList.extend|Y|Pure Python module methods|
|68|torch.nn.ParameterDict|Y|Pure Python container/infrastructure|
|69|torch.nn.ParameterDict.clear|Y|Pure Python module methods|
|70|torch.nn.ParameterDict.copy|Y|Pure Python module methods|
|71|torch.nn.ParameterDict.fromkeys|Y|Pure Python module methods|
|72|torch.nn.ParameterDict.get|Y|Pure Python module methods|
|73|torch.nn.ParameterDict.items|Y|Pure Python module methods|
|74|torch.nn.ParameterDict.keys|Y|Pure Python module methods|
|75|torch.nn.ParameterDict.pop|Y|Pure Python module methods|
|76|torch.nn.ParameterDict.popitem|Y|Pure Python module methods|
|77|torch.nn.ParameterDict.setdefault|Y|Pure Python module methods|
|78|torch.nn.ParameterDict.update|Y|Pure Python module methods|
|79|torch.nn.ParameterDict.values|Y|Pure Python module methods|
|80|torch.nn.modules.module.register_module_forward_pre_hook|Y|Pure Python hook registration|
|81|torch.nn.modules.module.register_module_forward_hook|Y|Pure Python hook registration|
|82|torch.nn.modules.module.register_module_backward_hook|Y|Pure Python hook registration|
|83|torch.nn.modules.module.register_module_full_backward_pre_hook|Y|Pure Python hook registration|
|84|torch.nn.modules.module.register_module_full_backward_hook|Y|Pure Python hook registration|
|85|torch.nn.modules.module.register_module_buffer_registration_hook|Y|Pure Python hook registration|
|86|torch.nn.modules.module.register_module_module_registration_hook|Y|Pure Python hook registration|
|87|torch.nn.modules.module.register_module_parameter_registration_hook|Y|Pure Python hook registration|
|88|torch.nn.Conv1d|Y|Forward runs on Neuron|
|89|torch.nn.Conv2d|Y|Forward runs on Neuron|
|90|torch.nn.Conv3d|Y|Forward runs on Neuron|
|91|torch.nn.ConvTranspose1d|Y|Forward runs on Neuron|
|92|torch.nn.ConvTranspose2d|Y|Forward runs on Neuron|
|93|torch.nn.ConvTranspose3d|Y|Forward runs on Neuron (via F.conv_transpose3d)|
|94|torch.nn.LazyConv1d|Y|Forward runs on Neuron (via F.conv1d)|
|95|torch.nn.LazyConv1d.cls_to_become|Y|Pure Python module method|
|96|torch.nn.LazyConv2d|Y|Forward runs on Neuron (via F.conv2d)|
|97|torch.nn.LazyConv2d.cls_to_become|Y|Pure Python module method|
|98|torch.nn.LazyConv3d|Y|Forward runs on Neuron (via F.conv3d)|
|99|torch.nn.LazyConv3d.cls_to_become|Y|Pure Python module method|
|100|torch.nn.LazyConvTranspose1d|Y|Forward runs on Neuron (via F.conv_transpose1d)|
|101|torch.nn.LazyConvTranspose1d.cls_to_become|Y|Pure Python module method|
|102|torch.nn.LazyConvTranspose2d|Y|Forward runs on Neuron (via F.conv_transpose2d)|
|103|torch.nn.LazyConvTranspose2d.cls_to_become|Y|Pure Python module method|
|104|torch.nn.LazyConvTranspose3d|Y|Forward runs on Neuron (via F.conv_transpose3d)|
|105|torch.nn.LazyConvTranspose3d.cls_to_become|Y|Pure Python module method|
|106|torch.nn.Unfold|CPU_FALLBACK|Falls back (aten::im2col)|
|107|torch.nn.Fold|CPU_FALLBACK|Falls back (aten::col2im)|
|108|torch.nn.MaxPool1d|CPU_FALLBACK|Falls back (aten::max_pool2d_with_indices.out)|
|109|torch.nn.MaxPool2d|CPU_FALLBACK|Falls back (aten::max_pool2d_with_indices.out)|
|110|torch.nn.MaxPool3d|CPU_FALLBACK|Falls back (aten::max_pool3d_with_indices)|
|111|torch.nn.MaxUnpool1d|CPU_FALLBACK|Falls back (aten::max_unpool)|
|112|torch.nn.MaxUnpool2d|CPU_FALLBACK|Falls back (aten::max_unpool)|
|113|torch.nn.MaxUnpool3d|CPU_FALLBACK|Falls back (aten::max_unpool)|
|114|torch.nn.AvgPool1d|CPU_FALLBACK|Falls back (aten::avg_pool2d.out)|
|115|torch.nn.AvgPool2d|CPU_FALLBACK|Falls back (aten::avg_pool2d.out)|
|116|torch.nn.AvgPool3d|CPU_FALLBACK|Falls back (aten::avg_pool3d.out)|
|117|torch.nn.FractionalMaxPool2d|CPU_FALLBACK|Falls back (aten::fractional_max_pool)|
|118|torch.nn.FractionalMaxPool3d|CPU_FALLBACK|Falls back (aten::fractional_max_pool)|
|119|torch.nn.LPPool1d|CPU_FALLBACK|Falls back (aten::avg_pool2d.out)|
|120|torch.nn.LPPool2d|CPU_FALLBACK|Falls back (aten::avg_pool2d.out)|
|121|torch.nn.AdaptiveMaxPool1d|CPU_FALLBACK|Falls back (aten::adaptive_max_pool)|
|122|torch.nn.AdaptiveMaxPool2d|CPU_FALLBACK|Falls back (aten::adaptive_max_pool)|
|123|torch.nn.AdaptiveMaxPool3d|CPU_FALLBACK|Falls back (aten::adaptive_max_pool)|
|124|torch.nn.AdaptiveAvgPool1d|CPU_FALLBACK|Falls back (aten::_adaptive_avg_pool)|
|125|torch.nn.AdaptiveAvgPool2d|CPU_FALLBACK|Falls back (aten::_adaptive_avg_pool)|
|126|torch.nn.AdaptiveAvgPool3d|CPU_FALLBACK|Falls back (aten::_adaptive_avg_pool)|
|127|torch.nn.ReflectionPad1d|Y|Forward runs on Neuron|
|128|torch.nn.ReflectionPad2d|Y|Forward runs on Neuron|
|129|torch.nn.ReflectionPad3d|Y|Forward runs on Neuron|
|130|torch.nn.ReplicationPad1d|Y|Forward runs on Neuron|
|131|torch.nn.ReplicationPad2d|Y|Forward runs on Neuron|
|132|torch.nn.ReplicationPad3d|Y|Forward runs on Neuron|
|133|torch.nn.ZeroPad1d|Y|Forward runs on Neuron|
|134|torch.nn.ZeroPad2d|Y|Forward runs on Neuron|
|135|torch.nn.ZeroPad3d|Y|Forward runs on Neuron|
|136|torch.nn.ConstantPad1d|Y|Forward runs on Neuron|
|137|torch.nn.ConstantPad2d|Y|Forward runs on Neuron|
|138|torch.nn.ConstantPad3d|Y|Forward runs on Neuron|
|139|torch.nn.ELU|CPU_FALLBACK|Falls back (aten::elu.out)|
|140|torch.nn.Hardshrink|CPU_FALLBACK|Falls back (aten::hardshrink.out)|
|141|torch.nn.Hardsigmoid|CPU_FALLBACK|Falls back (aten::hardsigmoid.out)|
|142|torch.nn.Hardtanh|CPU_FALLBACK|Falls back (aten::hardtanh)|
|143|torch.nn.Hardswish|CPU_FALLBACK|Falls back (aten::hardswish)|
|144|torch.nn.LeakyReLU|CPU_FALLBACK|Falls back (aten::leaky_relu.out)|
|145|torch.nn.LogSigmoid|CPU_FALLBACK|Falls back (aten::log_sigmoid_forward)|
|146|torch.nn.MultiheadAttention|Y|Forward runs on Neuron|
|147|torch.nn.MultiheadAttention.forward|Y|Pure Python module method|
|148|torch.nn.MultiheadAttention.merge_masks|Y|Forward runs on Neuron|
|149|torch.nn.PReLU|CPU_FALLBACK|Falls back (aten::_prelu_kernel)|
|150|torch.nn.ReLU|Y|Forward runs on Neuron|
|151|torch.nn.ReLU6|CPU_FALLBACK|Falls back (aten::hardtanh)|
|152|torch.nn.RReLU|CPU_FALLBACK|Falls back (aten::rrelu_with_noise)|
|153|torch.nn.SELU|CPU_FALLBACK|Falls back (aten::elu.out)|
|154|torch.nn.CELU|CPU_FALLBACK|Falls back (aten::elu.out)|
|155|torch.nn.GELU|Y|Forward runs on Neuron|
|156|torch.nn.Sigmoid|Y|Forward runs on Neuron|
|157|torch.nn.SiLU|Y|Forward runs on Neuron|
|158|torch.nn.Mish|CPU_FALLBACK|Falls back (aten::mish.out)|
|159|torch.nn.Softplus|Y|Forward runs on Neuron|
|160|torch.nn.Softshrink|CPU_FALLBACK|Falls back (aten::softshrink.out)|
|161|torch.nn.Softsign|Y|Forward runs on Neuron (via F.softsign)|
|162|torch.nn.Tanh|Y|Forward runs on Neuron|
|163|torch.nn.Tanhshrink|Y|Forward runs on Neuron (via F.tanhshrink)|
|164|torch.nn.Threshold|Y|Forward runs on Neuron|
|165|torch.nn.GLU|CPU_FALLBACK|Falls back (aten::glu.out)|
|166|torch.nn.Softmin|Y|Forward runs on Neuron (via F.softmin)|
|167|torch.nn.Softmax|Y|Forward runs on Neuron|
|168|torch.nn.Softmax2d|Y|Forward runs on Neuron (via F.softmax)|
|169|torch.nn.LogSoftmax|Y|Forward runs on Neuron|
|170|torch.nn.AdaptiveLogSoftmaxWithLoss|CPU_FALLBACK|Falls back (aten::index_fill_.int_Scalar)|
|171|torch.nn.AdaptiveLogSoftmaxWithLoss.log_prob|CPU_FALLBACK|Falls back (aten::index_fill_.int_Scalar)|
|172|torch.nn.AdaptiveLogSoftmaxWithLoss.predict|CPU_FALLBACK|Falls back (aten::index_fill_.int_Scalar)|
|173|torch.nn.BatchNorm1d|CPU_FALLBACK|Falls back (aten::native_batch_norm)|
|174|torch.nn.BatchNorm2d|CPU_FALLBACK|Falls back (aten::native_batch_norm)|
|175|torch.nn.BatchNorm3d|CPU_FALLBACK|Falls back (aten::native_batch_norm)|
|176|torch.nn.LazyBatchNorm1d|CPU_FALLBACK|Falls back (aten::native_batch_norm)|
|177|torch.nn.LazyBatchNorm1d.cls_to_become|Y|Pure Python module method|
|178|torch.nn.LazyBatchNorm2d|CPU_FALLBACK|Falls back (aten::native_batch_norm)|
|179|torch.nn.LazyBatchNorm2d.cls_to_become|Y|Pure Python module method|
|180|torch.nn.LazyBatchNorm3d|CPU_FALLBACK|Falls back (aten::native_batch_norm)|
|181|torch.nn.LazyBatchNorm3d.cls_to_become|Y|Pure Python module method|
|182|torch.nn.GroupNorm|CPU_FALLBACK|Falls back (aten::native_batch_norm)|
|183|torch.nn.SyncBatchNorm|CPU_FALLBACK|Falls back (aten::native_batch_norm)|
|184|torch.nn.SyncBatchNorm.convert_sync_batchnorm|CPU_FALLBACK|Falls back (aten::native_batch_norm)|
|185|torch.nn.InstanceNorm1d|CPU_FALLBACK|Falls back (aten::native_batch_norm)|
|186|torch.nn.InstanceNorm2d|CPU_FALLBACK|Falls back (aten::native_batch_norm)|
|187|torch.nn.InstanceNorm3d|CPU_FALLBACK|Falls back (aten::native_batch_norm)|
|188|torch.nn.LazyInstanceNorm1d|CPU_FALLBACK|Falls back (aten::native_batch_norm)|
|189|torch.nn.LazyInstanceNorm1d.cls_to_become|Y|Pure Python module method|
|190|torch.nn.LazyInstanceNorm2d|CPU_FALLBACK|Falls back (aten::native_batch_norm)|
|191|torch.nn.LazyInstanceNorm2d.cls_to_become|Y|Pure Python module method|
|192|torch.nn.LazyInstanceNorm3d|CPU_FALLBACK|Falls back (aten::native_batch_norm)|
|193|torch.nn.LazyInstanceNorm3d.cls_to_become|Y|Pure Python module method|
|194|torch.nn.LayerNorm|Y|Forward runs on Neuron|
|195|torch.nn.LocalResponseNorm|CPU_FALLBACK|Falls back (aten::avg_pool3d.out)|
|196|torch.nn.RNNBase|Y||
|197|torch.nn.RNNBase.flatten_parameters|Y||
|198|torch.nn.RNN|Y|Forward runs on Neuron|
|199|torch.nn.LSTM|CPU_FALLBACK|Could not run 'aten::_thnn_fused_lstm_cell'|
|200|torch.nn.GRU|CPU_FALLBACK|Could not run 'aten::_thnn_fused_gru_cell'|
|201|torch.nn.RNNCell|Y|Forward runs on Neuron|
|202|torch.nn.LSTMCell|CPU_FALLBACK|Could not run 'aten::_thnn_fused_lstm_cell'|
|203|torch.nn.GRUCell|CPU_FALLBACK|Could not run 'aten::_thnn_fused_gru_cell'|
|204|torch.nn.Transformer|Y|Forward runs on Neuron|
|205|torch.nn.Transformer.forward|Y|Pure Python module method|
|206|torch.nn.Transformer.generate_square_subsequent_mask|Y|Forward runs on Neuron|
|207|torch.nn.TransformerEncoder|Y|Forward runs on Neuron|
|208|torch.nn.TransformerEncoder.forward|Y|Pure Python module method|
|209|torch.nn.TransformerDecoder|Y|Forward runs on Neuron|
|210|torch.nn.TransformerDecoder.forward|Y|Pure Python module method|
|211|torch.nn.TransformerEncoderLayer|Y|Forward runs on Neuron|
|212|torch.nn.TransformerEncoderLayer.forward|Y|Pure Python module method|
|213|torch.nn.TransformerDecoderLayer|Y|Forward runs on Neuron|
|214|torch.nn.TransformerDecoderLayer.forward|Y|Pure Python module method|
|215|torch.nn.Identity|Y|Forward runs on Neuron|
|216|torch.nn.Linear|Y|Forward runs on Neuron|
|217|torch.nn.Bilinear|CPU_FALLBACK|In-place add_ on non-contiguous tensor falls back to CPU|
|218|torch.nn.LazyLinear|CPU_FALLBACK|Can't access shape of uninitialized parameter|
|219|torch.nn.LazyLinear.cls_to_become|Y|Pure Python module method|
|220|torch.nn.Dropout|Y|Forward runs on Neuron; dropout mask generated on CPU then copied to device|
|221|torch.nn.Dropout1d|CPU_FALLBACK|Falls back (aten::bernoulli_.float)|
|222|torch.nn.Dropout2d|CPU_FALLBACK|Falls back (aten::bernoulli_.float)|
|223|torch.nn.Dropout3d|CPU_FALLBACK|Falls back (aten::bernoulli_.float)|
|224|AlphaDropout|CPU_FALLBACK|Falls back (aten::bernoulli_.float)|
|225|FeatureAlphaDropout|CPU_FALLBACK|Falls back (aten::bernoulli_.float)|
|226|torch.nn.Embedding|Y|Uses aten::embedding (registered on Neuron)|
|227|torch.nn.Embedding.from_pretrained|Y|Pure Python; creates Embedding from weight tensor|
|228|torch.nn.EmbeddingBag|CPU_FALLBACK|Falls back (aten::_embedding_bag)|
|229|torch.nn.EmbeddingBag.forward|CPU_FALLBACK|Falls back (aten::_embedding_bag)|
|230|torch.nn.EmbeddingBag.from_pretrained|CPU_FALLBACK|Falls back (aten::_embedding_bag)|
|231|torch.nn.CosineSimilarity|Y|Forward runs on Neuron (via F.cosine_similarity)|
|232|torch.nn.PairwiseDistance|CPU_FALLBACK|Falls back (aten::norm.out)|
|233|torch.nn.L1Loss|Y|Forward runs on Neuron|
|234|torch.nn.MSELoss|CPU_FALLBACK|Falls back (aten::mse_loss.out)|
|235|torch.nn.CrossEntropyLoss|Y|Forward runs on Neuron|
|236|torch.nn.CTCLoss|CPU_FALLBACK|Falls back (aten::_ctc_loss)|
|237|torch.nn.NLLLoss|Y|Forward runs on Neuron|
|238|torch.nn.PoissonNLLLoss|Y|Forward runs on Neuron|
|239|torch.nn.GaussianNLLLoss|Y|Forward runs on Neuron|
|240|torch.nn.KLDivLoss|CPU_FALLBACK|Falls back (aten::xlogy.OutTensor)|
|241|torch.nn.BCELoss|CPU_FALLBACK|Falls back (aten::binary_cross_entropy)|
|242|torch.nn.BCEWithLogitsLoss|CPU_FALLBACK|Falls back (aten::log_sigmoid_forward)|
|243|torch.nn.MarginRankingLoss|Y|Forward runs on Neuron|
|244|torch.nn.HingeEmbeddingLoss|Y|Forward runs on Neuron|
|245|torch.nn.MultiLabelMarginLoss|CPU_FALLBACK|Falls back (aten::multilabel_margin_loss_forward)|
|246|torch.nn.HuberLoss|CPU_FALLBACK|Falls back (aten::huber_loss)|
|247|torch.nn.SmoothL1Loss|CPU_FALLBACK|Falls back (aten::smooth_l1_loss.out)|
|248|torch.nn.SoftMarginLoss|CPU_FALLBACK|Falls back (aten::log1p.out)|
|249|torch.nn.MultiLabelSoftMarginLoss|CPU_FALLBACK|Falls back (aten::log_sigmoid_forward)|
|250|torch.nn.CosineEmbeddingLoss|Y|Forward runs on Neuron|
|251|torch.nn.MultiMarginLoss|CPU_FALLBACK|Falls back (aten::multi_margin_loss)|
|252|torch.nn.TripletMarginLoss|CPU_FALLBACK|Falls back (aten::norm.out)|
|253|TripletMarginWithDistanceLoss|CPU_FALLBACK|Falls back (aten::norm.out)|
|254|torch.nn.PixelShuffle|Y|Forward runs on Neuron|
|255|torch.nn.PixelUnshuffle|Y|Forward runs on Neuron (via F.pixel_unshuffle)|
|256|torch.nn.Upsample|CPU_FALLBACK|Falls back (aten::upsample)|
|257|UpsamplingNearest2d|CPU_FALLBACK|Falls back (aten::upsample)|
|258|UpsamplingBilinear2d|CPU_FALLBACK|Falls back (aten::upsample)|
|259|torch.nn.ChannelShuffle|CPU_FALLBACK|Falls back (aten::channel_shuffle)|
|260|torch.nn.DataParallel|Y|Pure Python container/infrastructure|
|261|torch.nn.parallel.DistributedDataParallel|Y|Pure Python distributed methods|
|262|torch.nn.parallel.DistributedDataParallel.join|Y|Pure Python distributed methods|
|263|torch.nn.parallel.DistributedDataParallel.join_hook|Y|Pure Python distributed methods|
|264|torch.nn.parallel.DistributedDataParallel.no_sync|Y|Pure Python distributed methods|
|265|torch.nn.parallel.DistributedDataParallel.register_comm_hook|Y|Pure Python distributed methods|
|266|torch.nn.parallel.data_parallel|N|Not supported; requires CUDA scatter/gather primitives|
|267|torch.nn.utils.clip_grad_norm_|Y|Runs on Neuron|
|268|torch.nn.utils.clip_grad_value_|Y|Runs on Neuron|
|269|torch.nn.utils.parameters_to_vector|Y||
|270|torch.nn.utils.vector_to_parameters|Y||
|271|torch.nn.utils.prune.BasePruningMethod|Y|Pure Python pruning utility|
|272|torch.nn.utils.prune.BasePruningMethod.apply|Y|Pure Python module method|
|273|torch.nn.utils.prune.BasePruningMethod.apply_mask|Y|Pure Python pruning utility|
|274|torch.nn.utils.prune.BasePruningMethod.compute_mask|Y|Pure Python pruning utility|
|275|torch.nn.utils.prune.BasePruningMethod.prune|Y|Pure Python pruning utility|
|276|torch.nn.utils.prune.BasePruningMethod.remove|Y|Pure Python pruning utility|
|277|torch.nn.utils.prune.PruningContainer|Y|Pure Python pruning utility|
|278|torch.nn.utils.prune.PruningContainer.add_pruning_method|Y|Pure Python pruning utility|
|279|torch.nn.utils.prune.PruningContainer.apply|Y|Pure Python module method|
|280|torch.nn.utils.prune.PruningContainer.apply_mask|Y|Pure Python pruning utility|
|281|torch.nn.utils.prune.PruningContainer.compute_mask|Y|Pure Python pruning utility|
|282|torch.nn.utils.prune.PruningContainer.prune|Y|Pure Python pruning utility|
|283|torch.nn.utils.prune.PruningContainer.remove|Y|Pure Python pruning utility|
|284|torch.nn.utils.prune.Identity|Y|Pure Python pruning utility|
|285|torch.nn.utils.prune.Identity.apply|Y|Pure Python module method|
|286|torch.nn.utils.prune.Identity.apply_mask|Y|Pure Python pruning utility|
|287|torch.nn.utils.prune.Identity.prune|Y|Pure Python pruning utility|
|288|torch.nn.utils.prune.Identity.remove|Y|Pure Python pruning utility|
|289|torch.nn.utils.prune.RandomUnstructured|Y|Pure Python pruning utility|
|290|torch.nn.utils.prune.RandomUnstructured.apply|Y|Pure Python module method|
|291|torch.nn.utils.prune.RandomUnstructured.apply_mask|Y|Pure Python pruning utility|
|292|torch.nn.utils.prune.RandomUnstructured.prune|Y|Pure Python pruning utility|
|293|torch.nn.utils.prune.RandomUnstructured.remove|Y|Pure Python pruning utility|
|294|torch.nn.utils.prune.L1Unstructured|Y|Pure Python pruning utility|
|295|torch.nn.utils.prune.L1Unstructured.apply|Y|Pure Python module method|
|296|torch.nn.utils.prune.L1Unstructured.apply_mask|Y|Pure Python pruning utility|
|297|torch.nn.utils.prune.L1Unstructured.prune|Y|Pure Python pruning utility|
|298|torch.nn.utils.prune.L1Unstructured.remove|Y|Pure Python pruning utility|
|299|torch.nn.utils.prune.RandomStructured|Y|Pure Python pruning utility|
|300|torch.nn.utils.prune.RandomStructured.apply|Y|Pure Python module method|
|301|torch.nn.utils.prune.RandomStructured.apply_mask|Y|Pure Python pruning utility|
|302|torch.nn.utils.prune.RandomStructured.compute_mask|Y|Pure Python pruning utility|
|303|torch.nn.utils.prune.RandomStructured.prune|Y|Pure Python pruning utility|
|304|torch.nn.utils.prune.RandomStructured.remove|Y|Pure Python pruning utility|
|305|torch.nn.utils.prune.LnStructured|Y|Pure Python pruning utility|
|306|torch.nn.utils.prune.LnStructured.apply|Y|Pure Python module method|
|307|torch.nn.utils.prune.LnStructured.apply_mask|Y|Pure Python pruning utility|
|308|torch.nn.utils.prune.LnStructured.compute_mask|Y|Pure Python pruning utility|
|309|torch.nn.utils.prune.LnStructured.prune|Y|Pure Python pruning utility|
|310|torch.nn.utils.prune.LnStructured.remove|Y|Pure Python pruning utility|
|311|torch.nn.utils.prune.CustomFromMask|Y|Pure Python pruning utility|
|312|torch.nn.utils.prune.CustomFromMask.apply|Y|Pure Python module method|
|313|torch.nn.utils.prune.CustomFromMask.apply_mask|Y|Pure Python pruning utility|
|314|torch.nn.utils.prune.CustomFromMask.prune|Y|Pure Python pruning utility|
|315|torch.nn.utils.prune.CustomFromMask.remove|Y|Pure Python pruning utility|
|316|torch.nn.utils.prune.identity|Y|Pure Python pruning utility|
|317|torch.nn.utils.prune.random_unstructured|Y|Pure Python pruning utility|
|318|torch.nn.utils.prune.l1_unstructured|Y|Pure Python pruning utility|
|319|torch.nn.utils.prune.random_structured|Y|Pure Python pruning utility|
|320|torch.nn.utils.prune.ln_structured|Y|Pure Python pruning utility|
|321|torch.nn.utils.prune.global_unstructured|Y|Pure Python pruning utility|
|322|torch.nn.utils.prune.custom_from_mask|Y|Pure Python pruning utility|
|323|torch.nn.utils.prune.remove|Y|Pure Python pruning utility|
|324|torch.nn.utils.prune.is_pruned|Y|Pure Python pruning utility|
|325|torch.nn.utils.weight_norm|CPU_FALLBACK|Falls back (aten::_weight_norm_interface)|
|326|torch.nn.utils.remove_weight_norm|CPU_FALLBACK|Falls back (aten::_weight_norm_interface)|
|327|torch.nn.utils.spectral_norm|Y||
|328|torch.nn.utils.remove_spectral_norm|Y||
|329|torch.nn.utils.skip_init|Y||
|330|torch.nn.utils.parametrizations.orthogonal|Y|Pure Python parametrization utility|
|331|torch.nn.utils.parametrizations.spectral_norm|Y|Pure Python parametrization utility|
|332|torch.nn.utils.parametrize.register_parametrization|Y|Pure Python|
|333|torch.nn.utils.parametrize.remove_parametrizations|Y|Pure Python|
|334|torch.nn.utils.parametrize.cached|Y|Pure Python|
|335|torch.nn.utils.parametrize.is_parametrized|Y|Pure Python|
|336|torch.nn.utils.parametrize.ParametrizationList|Y|Pure Python parametrization utility|
|337|torch.nn.utils.parametrize.ParametrizationList.right_inverse|Y|Pure Python parametrization utility|
|338|torch.nn.utils.stateless.functional_call|Y||
|339|torch.nn.utils.rnn.PackedSequence|Y|Pure Python data class attribute|
|340|torch.nn.utils.rnn.PackedSequence.batch_sizes|Y|Pure Python data class attribute|
|341|torch.nn.utils.rnn.PackedSequence.count|Y|Pure Python data class attribute|
|342|torch.nn.utils.rnn.PackedSequence.data|Y|Pure Python data class attribute|
|343|torch.nn.utils.rnn.PackedSequence.index|Y|Pure Python data class attribute|
|344|torch.nn.utils.rnn.PackedSequence.is_cuda|Y|Pure Python data class attribute|
|345|torch.nn.utils.rnn.PackedSequence.is_pinned|Y|Pure Python data class attribute|
|346|torch.nn.utils.rnn.PackedSequence.sorted_indices|Y|Pure Python data class attribute|
|347|torch.nn.utils.rnn.PackedSequence.to|Y|Pure Python module method|
|348|torch.nn.utils.rnn.PackedSequence.unsorted_indices|Y|Pure Python data class attribute|
|349|torch.nn.utils.rnn.pack_padded_sequence|Y|Runs on Neuron; decomposes to index_select, scatter, cat, arange|
|350|torch.nn.utils.rnn.pad_packed_sequence|Y|Runs on Neuron; decomposes to index_select, full, copy_|
|351|torch.nn.utils.rnn.pad_sequence|Y|Runs on Neuron; decomposes to full + copy_|
|352|torch.nn.utils.rnn.pack_sequence|Y|Runs on Neuron; decomposes to index_select, scatter, cat, arange|
|353|torch.nn.utils.rnn.unpack_sequence|Y|Pure Python; unpacks PackedSequence into list of tensors|
|354|torch.nn.utils.rnn.unpad_sequence|Y|Pure Python; removes padding from padded tensor using lengths|
|355|torch.nn.Flatten|Y|Forward runs on Neuron|
|356|torch.nn.Unflatten|Y|Forward runs on Neuron|
|357|torch.nn.modules.lazy.LazyModuleMixin|Y|Pure Python|
|358|torch.nn.modules.lazy.LazyModuleMixin.has_uninitialized_params|Y|Pure Python|
|359|torch.nn.modules.lazy.LazyModuleMixin.initialize_parameters|Y|Pure Python|

## torch.nn.functional

||pytorch_api|neuron_compatible|limitations|
|---|---|---|---|
|1|F.conv1d|Y|Uses aten::convolution (registered on Neuron)|
|2|F.conv2d|Y|Uses aten::convolution (registered on Neuron)|
|3|F.conv3d|Y|Uses aten::convolution (registered on Neuron)|
|4|F.conv_transpose{1d|Y|Uses aten::convolution (registered on Neuron)|
|5|2d|Y|Uses aten::convolution (registered on Neuron)|
|6|3d}|Y|Uses aten::convolution (registered on Neuron)|
|7|F.unfold|CPU_FALLBACK|Falls back to CPU (aten::im2col)|
|8|F.fold|CPU_FALLBACK|Falls back to CPU (aten::col2im)|
|9|F.avg_pool1d|CPU_FALLBACK|Falls back to CPU (aten::avg_pool)|
|10|F.avg_pool2d|CPU_FALLBACK|Falls back to CPU (aten::avg_pool)|
|11|F.avg_pool3d|CPU_FALLBACK|Falls back to CPU (aten::avg_pool)|
|12|F.max_pool1d|CPU_FALLBACK|Falls back to CPU (aten::max_pool)|
|13|F.max_pool2d|CPU_FALLBACK|Falls back to CPU (aten::max_pool)|
|14|F.max_pool3d|CPU_FALLBACK|Falls back to CPU (aten::max_pool)|
|15|F.max_unpool1d|CPU_FALLBACK|Falls back to CPU|
|16|F.max_unpool2d|CPU_FALLBACK|Falls back to CPU|
|17|F.max_unpool3d|CPU_FALLBACK|Falls back to CPU|
|18|F.lp_pool1d|CPU_FALLBACK|Falls back to CPU (aten::avg_pool2d.out)|
|19|F.lp_pool2d|CPU_FALLBACK|Falls back to CPU (aten::avg_pool2d.out)|
|20|F.adaptive_max_pool1d|CPU_FALLBACK|Falls back to CPU|
|21|F.adaptive_max_pool2d|CPU_FALLBACK|Falls back to CPU|
|22|F.adaptive_max_pool3d|CPU_FALLBACK|Falls back to CPU|
|23|F.adaptive_avg_pool1d|CPU_FALLBACK|Falls back to CPU|
|24|F.adaptive_avg_pool2d|CPU_FALLBACK|Falls back to CPU|
|25|F.adaptive_avg_pool3d|CPU_FALLBACK|Falls back to CPU|
|26|F.fractional_max_pool2d|CPU_FALLBACK|Falls back to CPU|
|27|F.fractional_max_pool3d|CPU_FALLBACK|Falls back to CPU|
|28|F.scaled_dot_product_attention|Y|Native Neuron SDPA implementation|
|29|F.threshold|Y|Uses aten::threshold (registered on Neuron)|
|30|F.threshold_|CPU_FALLBACK|torch_threshold() got unexpected keyword argument 'out'|
|31|F.relu|Y|Uses aten::relu (registered on Neuron)|
|32|F.relu_|Y|In-place variant|
|33|F.hardtanh|CPU_FALLBACK|Falls back to CPU (aten::hardtanh)|
|34|F.hardtanh_|CPU_FALLBACK|Falls back to CPU (aten::hardtanh)|
|35|F.hardswish|CPU_FALLBACK|Falls back to CPU (aten::hardswish)|
|36|F.relu6|CPU_FALLBACK|Falls back to CPU (aten::hardtanh)|
|37|F.elu|CPU_FALLBACK|Falls back to CPU (aten::elu.out)|
|38|F.elu_|CPU_FALLBACK|Falls back to CPU (aten::elu.out)|
|39|F.selu|CPU_FALLBACK|Falls back to CPU (aten::elu.out)|
|40|F.celu|CPU_FALLBACK|Falls back to CPU (aten::elu.out)|
|41|F.leaky_relu|CPU_FALLBACK|Falls back to CPU (aten::leaky_relu.out)|
|42|F.leaky_relu_|CPU_FALLBACK|Falls back to CPU (aten::leaky_relu.out)|
|43|F.prelu|CPU_FALLBACK|Falls back to CPU (aten::_prelu_kernel)|
|44|F.rrelu|CPU_FALLBACK|Falls back to CPU (aten::rrelu_with_noise)|
|45|F.rrelu_|CPU_FALLBACK|Falls back to CPU (aten::rrelu_with_noise)|
|46|F.glu|CPU_FALLBACK|Falls back to CPU (aten::glu.out)|
|47|F.gelu|Y|Uses aten::gelu (registered on Neuron)|
|48|F.logsigmoid|CPU_FALLBACK|Falls back to CPU (aten::log_sigmoid_forward)|
|49|F.hardshrink|CPU_FALLBACK|Falls back to CPU (aten::hardshrink.out)|
|50|F.tanhshrink|Y|Decomposes to tanh + sub (supported)|
|51|F.softsign|Y|Decomposes to div + abs (supported)|
|52|F.softplus|Y|Uses aten::softplus (registered on Neuron)|
|53|F.softmin|Y|Decomposes to softmax (supported)|
|54|F.softmax|Y|Uses aten::_softmax (registered on Neuron)|
|55|F.softshrink|CPU_FALLBACK|Falls back to CPU (aten::softshrink.out)|
|56|F.gumbel_softmax|CPU_FALLBACK|Falls back to CPU (aten::exponential_)|
|57|F.log_softmax|Y|Uses aten::_log_softmax (registered on Neuron)|
|58|F.tanh|Y|Uses aten::tanh (registered on Neuron)|
|59|F.sigmoid|Y|Uses aten::sigmoid (registered on Neuron)|
|60|F.hardsigmoid|CPU_FALLBACK|Falls back to CPU (aten::hardsigmoid.out)|
|61|F.silu|Y|Uses aten::silu (registered on Neuron)|
|62|F.mish|CPU_FALLBACK|Falls back to CPU (aten::mish.out)|
|63|F.batch_norm|CPU_FALLBACK|Falls back to CPU (aten::native_batch_norm)|
|64|F.group_norm|CPU_FALLBACK|Falls back to CPU (aten::native_batch_norm)|
|65|F.instance_norm|CPU_FALLBACK|Falls back to CPU (aten::native_batch_norm)|
|66|F.layer_norm|Y|Uses aten::native_layer_norm (registered on Neuron)|
|67|F.local_response_norm|CPU_FALLBACK|Falls back to CPU (aten::avg_pool3d.out)|
|68|F.normalize|Y|Decomposes to div + norm (supported)|
|69|F.linear|Y|Uses aten::linear (registered on Neuron)|
|70|F.bilinear|CPU_FALLBACK|In-place add_ on non-contiguous tensor falls back to CPU|
|71|F.dropout|Y|Uses aten::native_dropout (registered on Neuron)|
|72|F.alpha_dropout, F.feature_alpha_dropout, F.dropout1d|CPU_FALLBACK|Falls back to CPU (aten::bernoulli_.float)|
|73|F.alpha_dropout, F.feature_alpha_dropout, F.dropout2d|CPU_FALLBACK|Falls back to CPU (aten::bernoulli_.float)|
|74|F.alpha_dropout, F.feature_alpha_dropout, F.dropout3d|CPU_FALLBACK|Falls back to CPU (aten::bernoulli_.float)|
|77|F.embedding|Y|Uses aten::embedding (registered on Neuron)|
|78|F.embedding_bag|CPU_FALLBACK|Falls back to CPU (aten::_embedding_bag_forward_only)|
|79|F.one_hot|CPU_FALLBACK|Falls back to CPU (aten::min)|
|80|F.pairwise_distance|CPU_FALLBACK|Falls back to CPU (aten::norm.out)|
|81|F.cosine_similarity|Y|Decomposes to div + sum + mul (supported)|
|82|F.pdist|CPU_FALLBACK|Falls back to CPU (aten::_pdist_forward)|
|83|F.binary_cross_entropy|CPU_FALLBACK|Falls back to CPU (aten::binary_cross_entropy)|
|84|F.binary_cross_entropy_with_logits|CPU_FALLBACK|Falls back to CPU (aten::log_sigmoid_forward)|
|85|F.poisson_nll_loss|Y|Decomposes to exp + sub + mul (supported)|
|86|F.cosine_embedding_loss|Y|Decomposes to cosine_similarity + clamp (supported)|
|87|F.cross_entropy|Y|Decomposes to log_softmax + nll_loss (supported)|
|88|F.ctc_loss|CPU_FALLBACK|Falls back to CPU (aten::_ctc_loss)|
|89|F.gaussian_nll_loss|Y|Decomposes to log + mul + add (supported)|
|90|F.hinge_embedding_loss|Y|Decomposes to clamp + mean (supported)|
|91|F.kl_div|CPU_FALLBACK|Falls back to CPU (aten::xlogy.OutTensor)|
|92|F.l1_loss|Y|Decomposes to abs + mean (supported)|
|93|F.mse_loss|CPU_FALLBACK|Falls back to CPU (aten::mse_loss.out)|
|94|F.margin_ranking_loss|Y|Decomposes to clamp + sub (supported)|
|95|F.multilabel_margin_loss|CPU_FALLBACK|Falls back to CPU|
|96|F.multilabel_soft_margin_loss|CPU_FALLBACK|Falls back to CPU|
|97|F.multi_margin_loss|CPU_FALLBACK|Falls back to CPU|
|98|F.nll_loss|Y|Uses aten::nll_loss_forward (registered on Neuron)|
|99|F.huber_loss|CPU_FALLBACK|Falls back to CPU|
|100|F.smooth_l1_loss|CPU_FALLBACK|Falls back to CPU|
|101|F.soft_margin_loss|CPU_FALLBACK|Falls back to CPU|
|102|F.triplet_margin_loss|CPU_FALLBACK|Falls back to CPU (aten::norm.out)|
|103|F.triplet_margin_with_distance_loss|CPU_FALLBACK|Falls back to CPU (aten::norm.out)|
|104|F.pixel_shuffle|Y|Decomposes to reshape + permute (supported)|
|105|F.pixel_unshuffle|Y|Decomposes to reshape + permute (supported)|
|106|F.pad|Y|Uses aten::constant_pad_nd (registered on Neuron)|
|107|F.interpolate|CPU_FALLBACK|Falls back to CPU|
|108|F.upsample|CPU_FALLBACK|Falls back to CPU|
|109|F.upsample_nearest|CPU_FALLBACK|Falls back to CPU|
|110|F.upsample_bilinear|CPU_FALLBACK|Falls back to CPU|
|111|F.grid_sample|CPU_FALLBACK|Falls back to CPU (aten::grid_sampler_2d)|
|112|F.affine_grid|CPU_FALLBACK|In-place fill_ on non-contiguous tensor falls back to CPU|

## torch.Tensor

||pytorch_api|neuron_compatible|limitations|
|---|---|---|---|
|1|Tensor.T|Y|View operation, no device compute|
|2|Tensor.H|Y|View operation, no device compute|
|3|Tensor.mT|Y|View operation, no device compute|
|4|Tensor.mH|Y|View operation, no device compute|
|5|Tensor.new_tensor|Y|Uses copy_/fill_/empty internally|
|6|Tensor.new_full|Y|Uses copy_/fill_/empty internally|
|7|Tensor.new_empty|Y|Uses copy_/fill_/empty internally|
|8|Tensor.new_ones|Y|Uses copy_/fill_/empty internally|
|9|Tensor.new_zeros|Y|Uses copy_/fill_/empty internally|
|10|Tensor.is_cuda|Y|Returns False on Neuron|
|11|Tensor.is_quantized|Y|Returns False on Neuron|
|12|Tensor.is_meta|Y|Returns False on Neuron|
|13|Tensor.device|Y|Returns neuron device|
|14|Tensor.grad|Y|Metadata only|
|15|Tensor.ndim|Y|Metadata only|
|16|Tensor.real|Y|Metadata only|
|17|Tensor.imag|Y|Metadata only|
|18|Tensor.nbytes|Y|Metadata only|
|19|Tensor.itemsize|Y|Metadata only|
|20|Tensor.abs|Y|Native MLIR impl; float64/int64 auto-cast|
|21|Tensor.abs_|Y|Native MLIR impl; float64/int64 auto-cast|
|22|Tensor.absolute|Y|Native MLIR impl; float64/int64 auto-cast|
|23|Tensor.absolute_|Y|Native MLIR impl; float64/int64 auto-cast|
|24|Tensor.acos|CPU_FALLBACK|Falls back to CPU (aten::acos.out)|
|25|Tensor.acos_|CPU_FALLBACK|Falls back to CPU (aten::acos.out)|
|26|Tensor.arccos|CPU_FALLBACK|Falls back to CPU (aten::acos.out)|
|27|Tensor.arccos_|CPU_FALLBACK|Falls back to CPU (aten::acos.out)|
|28|Tensor.add|Y|Native MLIR impl; supports alpha parameter|
|29|Tensor.add_|Y|Native MLIR impl; supports alpha parameter|
|30|Tensor.addbmm|CPU_FALLBACK|No native Neuron impl|
|31|Tensor.addbmm_|CPU_FALLBACK|No native Neuron impl|
|32|Tensor.addcdiv|Y|Native MLIR impl / In-place variant|
|33|Tensor.addcdiv_|Y|Native MLIR impl / In-place variant|
|34|Tensor.addcmul|Y|Native MLIR impl / In-place variant|
|35|Tensor.addcmul_|Y|Native MLIR impl / In-place variant|
|36|Tensor.addmm|Y|Native MLIR impl; supports beta, alpha|
|37|Tensor.addmm_|Y|Native MLIR impl; supports beta, alpha|
|38|Tensor.sspaddmm|CPU_FALLBACK|Sparse; no native Neuron impl|
|39|Tensor.addmv|CPU_FALLBACK|No native Neuron impl|
|40|Tensor.addmv_|CPU_FALLBACK|No native Neuron impl|
|41|Tensor.addr|CPU_FALLBACK|No native Neuron impl|
|42|Tensor.addr_|CPU_FALLBACK|No native Neuron impl|
|43|Tensor.adjoint|Y|View operation|
|44|Tensor.allclose|CPU_FALLBACK|No native Neuron impl|
|45|Tensor.amax|Y|Native MLIR impl|
|46|Tensor.amin|Y|Native MLIR impl|
|47|Tensor.aminmax|CPU_FALLBACK|No native Neuron impl|
|48|Tensor.angle|CPU_FALLBACK|No native Neuron impl|
|49|Tensor.apply_|CPU_FALLBACK|Python function application|
|50|Tensor.argmax|CPU_FALLBACK|Falls back to CPU|
|51|Tensor.argmin|CPU_FALLBACK|Falls back to CPU|
|52|Tensor.argsort|CPU_FALLBACK|Falls back to CPU|
|53|Tensor.argwhere|Y|Decomposes to nonzero (supported)|
|54|Tensor.asin|CPU_FALLBACK|Falls back to CPU (aten::asin.out)|
|55|Tensor.asin_|CPU_FALLBACK|Falls back to CPU (aten::asin.out)|
|56|Tensor.arcsin|CPU_FALLBACK|Falls back to CPU (aten::asin.out)|
|57|Tensor.arcsin_|CPU_FALLBACK|Falls back to CPU (aten::asin.out)|
|58|Tensor.as_strided|Y|Native C++ impl (as_strided_neuron)|
|59|Tensor.atan|CPU_FALLBACK|Falls back to CPU|
|60|Tensor.atan_|CPU_FALLBACK|Falls back to CPU|
|61|Tensor.arctan|CPU_FALLBACK|Falls back to CPU|
|62|Tensor.arctan_|CPU_FALLBACK|Falls back to CPU|
|63|Tensor.atan2|CPU_FALLBACK|Falls back to CPU|
|64|Tensor.atan2_|CPU_FALLBACK|Falls back to CPU|
|65|Tensor.arctan2|CPU_FALLBACK|Falls back to CPU|
|66|Tensor.arctan2_|CPU_FALLBACK|Falls back to CPU|
|67|Tensor.all|Y|Native MLIR impl|
|68|Tensor.any|Y|Native MLIR impl; uint8 input returns uint8|
|69|Tensor.backward|CPU_FALLBACK|Falls back to CPU (aten::normal_)|
|70|Tensor.baddbmm|CPU_FALLBACK|No native Neuron impl|
|71|Tensor.baddbmm_|CPU_FALLBACK|No native Neuron impl|
|72|Tensor.bernoulli|CPU_FALLBACK|No native Neuron impl|
|73|Tensor.bernoulli_|CPU_FALLBACK|No native Neuron impl|
|74|Tensor.bfloat16|Y|Uses to(bfloat16)|
|75|Tensor.bincount|CPU_FALLBACK|No native Neuron impl|
|76|Tensor.bitwise_not|Y|Native MLIR impl / In-place variants|
|77|Tensor.bitwise_not_|Y|Native MLIR impl / In-place variants|
|78|Tensor.bitwise_and|Y|Native MLIR impl / In-place variants|
|79|Tensor.bitwise_and_|Y|Native MLIR impl / In-place variants|
|80|Tensor.bitwise_or|Y|Native MLIR impl / In-place variants|
|81|Tensor.bitwise_or_|Y|Native MLIR impl / In-place variants|
|82|Tensor.bitwise_xor|Y|Native MLIR impl / In-place variants|
|83|Tensor.bitwise_xor_|Y|Native MLIR impl / In-place variants|
|84|Tensor.bitwise_left_shift|Y|Native MLIR impl / In-place variants|
|85|Tensor.bitwise_left_shift_|Y|Native MLIR impl / In-place variants|
|86|Tensor.bitwise_right_shift|Y|Native MLIR impl / In-place variants|
|87|Tensor.bitwise_right_shift_|Y|Native MLIR impl / In-place variants|
|88|Tensor.bmm|Y|Native MLIR impl|
|89|Tensor.bool|Y|Uses to(bool/uint8)|
|90|Tensor.byte|Y|Uses to(bool/uint8)|
|91|Tensor.broadcast_to|Y|View operation (expand)|
|92|Tensor.cauchy_|CPU_FALLBACK|No native Neuron impl|
|93|Tensor.ceil|Y|Native MLIR impl / In-place|
|94|Tensor.ceil_|Y|Native MLIR impl / In-place|
|95|Tensor.char|Y|Uses to(int8)|
|96|Tensor.cholesky|CPU_FALLBACK|No native Neuron impl|
|97|Tensor.cholesky_inverse|CPU_FALLBACK|No native Neuron impl|
|98|Tensor.cholesky_solve|CPU_FALLBACK|No native Neuron impl|
|99|Tensor.chunk|Y|Decomposed to split|
|100|Tensor.clamp|Y|Native MLIR impl; supports Tensor min/max|
|101|Tensor.clamp_|Y|Native MLIR impl; supports Tensor min/max|
|102|Tensor.clip|Y|Native MLIR impl; supports Tensor min/max|
|103|Tensor.clip_|Y|Native MLIR impl; supports Tensor min/max|
|104|Tensor.clone|Y|Native C++ impl (clone_neuron)|
|105|Tensor.contiguous|Y|Native impl with MLIR|
|106|Tensor.copy_|Y|Native MLIR impl via @neuron_op; handles cross-device copies|
|107|Tensor.conj|Y|View operation|
|108|Tensor.conj_physical|CPU_FALLBACK|No native Neuron impl|
|109|Tensor.conj_physical_|CPU_FALLBACK|No native Neuron impl|
|110|Tensor.resolve_conj|Y|Metadata operation|
|111|Tensor.resolve_neg|Y|Metadata operation|
|112|Tensor.copysign|CPU_FALLBACK|No native Neuron impl|
|113|Tensor.copysign_|CPU_FALLBACK|No native Neuron impl|
|114|Tensor.cos|Y|Native MLIR impl / In-place|
|115|Tensor.cos_|Y|Native MLIR impl / In-place|
|116|Tensor.cosh|CPU_FALLBACK|Falls back to CPU (aten::cosh.out)|
|117|Tensor.cosh_|CPU_FALLBACK|Falls back to CPU (aten::cosh.out)|
|118|Tensor.corrcoef|CPU_FALLBACK|No native Neuron impl|
|119|Tensor.count_nonzero|CPU_FALLBACK|No native Neuron impl|
|120|Tensor.cov|CPU_FALLBACK|No native Neuron impl|
|121|Tensor.acosh|CPU_FALLBACK|Falls back to CPU (aten::acosh.out)|
|122|Tensor.acosh_|CPU_FALLBACK|Falls back to CPU (aten::acosh.out)|
|123|Tensor.arccosh|CPU_FALLBACK|Falls back to CPU (aten::acosh.out)|
|124|Tensor.arccosh_|CPU_FALLBACK|Falls back to CPU (aten::acosh.out)|
|125|Tensor.cpu|Y|Copies tensor to CPU|
|126|Tensor.cross|CPU_FALLBACK|Falls back to CPU (aten::linalg_cross.out)|
|127|Tensor.cuda|N|Neuron is not CUDA; use .to('neuron') instead|
|128|Tensor.logcumsumexp|CPU_FALLBACK|No native Neuron impl|
|129|Tensor.cummax|CPU_FALLBACK|No native Neuron impl|
|130|Tensor.cummin|CPU_FALLBACK|No native Neuron impl|
|131|Tensor.cumprod|CPU_FALLBACK|No native Neuron impl|
|132|Tensor.cumprod_|CPU_FALLBACK|No native Neuron impl|
|133|Tensor.cumsum|Y|Native MLIR impl|
|134|Tensor.cumsum_|Y|Native MLIR impl|
|135|Tensor.chalf|CPU_FALLBACK|Complex types not supported|
|136|Tensor.cfloat|CPU_FALLBACK|Complex types not supported|
|137|Tensor.cdouble|CPU_FALLBACK|Complex types not supported|
|138|Tensor.data_ptr|Y|Returns pointer|
|139|Tensor.deg2rad|CPU_FALLBACK|No native Neuron impl|
|140|Tensor.dequantize|CPU_FALLBACK|Quantization not natively supported|
|141|Tensor.det|CPU_FALLBACK|Decomposes to linalg_det; CPU fallback|
|142|Tensor.dense_dim|Y|Returns ndim for dense tensors|
|143|Tensor.detach|Y|Metadata only|
|144|Tensor.detach_|Y|Metadata only|
|145|Tensor.diag|Y|Decomposes to diag_embed (1D) or diagonal_copy (2D)|
|146|Tensor.diag_embed|CPU_FALLBACK|No native Neuron impl|
|147|Tensor.diagflat|Y|Decomposes to flatten + diag_embed|
|148|Tensor.diagonal|Y|View operation|
|149|Tensor.diagonal_scatter|CPU_FALLBACK|No native Neuron impl|
|150|Tensor.fill_diagonal_|CPU_FALLBACK|No native Neuron impl|
|151|Tensor.fmax|CPU_FALLBACK|No native Neuron impl|
|152|Tensor.fmin|CPU_FALLBACK|No native Neuron impl|
|153|Tensor.diff|Y|Decomposes to narrow + sub (both supported)|
|154|Tensor.digamma|CPU_FALLBACK|No native Neuron impl|
|155|Tensor.digamma_|CPU_FALLBACK|No native Neuron impl|
|156|Tensor.dim|Y|Metadata only|
|157|Tensor.dim_order|Y|Metadata only|
|158|Tensor.dist|CPU_FALLBACK|No native Neuron impl|
|159|Tensor.div|Y|Native MLIR impl; supports rounding_mode|
|160|Tensor.div_|Y|Native MLIR impl; supports rounding_mode|
|161|Tensor.divide|Y|Native MLIR impl; supports rounding_mode|
|162|Tensor.divide_|Y|Native MLIR impl; supports rounding_mode|
|163|Tensor.dot|Y|Native MLIR impl|
|164|Tensor.double|Y|Auto-cast: float64 → float32 with warning|
|165|Tensor.dsplit|Y|Decomposed to split|
|166|Tensor.element_size|Y|Metadata only|
|167|Tensor.eq|Y|Native MLIR impl / In-place|
|168|Tensor.eq_|Y|Native MLIR impl / In-place|
|169|Tensor.equal|CPU_FALLBACK|No native Neuron impl (returns single bool)|
|170|Tensor.erf|Y/CPU_FALLBACK|erf/erfinv native; erfc falls back|
|171|Tensor.erf_|Y/CPU_FALLBACK|erf/erfinv native; erfc falls back|
|172|Tensor.erfc|Y/CPU_FALLBACK|erf/erfinv native; erfc falls back|
|173|Tensor.erfc_|Y/CPU_FALLBACK|erf/erfinv native; erfc falls back|
|174|Tensor.erfinv|Y/CPU_FALLBACK|erf/erfinv native; erfc falls back|
|175|Tensor.erfinv_|Y/CPU_FALLBACK|erf/erfinv native; erfc falls back|
|176|Tensor.exp|Y|Native MLIR impl / In-place|
|177|Tensor.exp_|Y|Native MLIR impl / In-place|
|178|Tensor.expm1|CPU_FALLBACK|Falls back to CPU (aten::expm1.out)|
|179|Tensor.expm1_|CPU_FALLBACK|Falls back to CPU (aten::expm1.out)|
|180|Tensor.expand|Y|View operation, no copy|
|181|Tensor.expand_as|Y|View operation, no copy|
|182|Tensor.exponential_|CPU_FALLBACK|No native Neuron impl|
|183|Tensor.fix|Y|Alias for trunc; native MLIR impl|
|184|Tensor.fix_|Y|Alias for trunc; native MLIR impl|
|185|Tensor.fill_|Y|Native MLIR impl; Scalar and Tensor variants|
|186|Tensor.flatten|Y|Decomposed to reshape|
|187|Tensor.flip|Y|Native MLIR impl / Decomposed to flip|
|188|Tensor.fliplr|Y|Native MLIR impl / Decomposed to flip|
|189|Tensor.flipud|Y|Native MLIR impl / Decomposed to flip|
|190|Tensor.float|Y|Uses to(float32)|
|191|Tensor.float_power|Y|Decomposes to pow|
|192|Tensor.float_power_|Y|Decomposes to pow|
|193|Tensor.floor|CPU_FALLBACK|Falls back to CPU (aten::floor.out)|
|194|Tensor.floor_|CPU_FALLBACK|Falls back to CPU (aten::floor.out)|
|195|Tensor.floor_divide|Y|Native MLIR impl / In-place|
|196|Tensor.floor_divide_|Y|Native MLIR impl / In-place|
|197|Tensor.fmod|CPU_FALLBACK|Falls back to CPU|
|198|Tensor.fmod_|CPU_FALLBACK|Falls back to CPU|
|199|Tensor.frac|CPU_FALLBACK|Falls back to CPU|
|200|Tensor.frac_|CPU_FALLBACK|Falls back to CPU|
|201|Tensor.frexp|CPU_FALLBACK|No native Neuron impl|
|202|Tensor.gather|Y|Native MLIR impl + Python impl registered|
|203|Tensor.gcd|CPU_FALLBACK|No native Neuron impl|
|204|Tensor.gcd_|CPU_FALLBACK|No native Neuron impl|
|205|Tensor.ge|Y|Native MLIR impl|
|206|Tensor.ge_|Y|Native MLIR impl|
|207|Tensor.greater_equal|Y|Native MLIR impl|
|208|Tensor.greater_equal_|Y|Native MLIR impl|
|209|Tensor.geometric_|CPU_FALLBACK|No native Neuron impl|
|210|Tensor.geqrf|CPU_FALLBACK|No native Neuron impl|
|211|Tensor.ger|Y|Decomposes to outer → reshape + mul (supported)|
|212|Tensor.get_device|Y|Returns device index|
|213|Tensor.gt|Y|Native MLIR impl|
|214|Tensor.gt_|Y|Native MLIR impl|
|215|Tensor.greater|Y|Native MLIR impl|
|216|Tensor.greater_|Y|Native MLIR impl|
|217|Tensor.half|Y|Uses to(float16)|
|218|Tensor.hardshrink|CPU_FALLBACK|No native Neuron impl|
|219|Tensor.heaviside|CPU_FALLBACK|No native Neuron impl|
|220|Tensor.histc|CPU_FALLBACK|Falls back to CPU|
|221|Tensor.histogram|CPU_FALLBACK|Falls back to CPU|
|222|Tensor.hsplit|Y|Decomposed to split|
|223|Tensor.hypot|CPU_FALLBACK|No native Neuron impl|
|224|Tensor.hypot_|CPU_FALLBACK|No native Neuron impl|
|225|Tensor.i0|CPU_FALLBACK|No native Neuron impl|
|226|Tensor.i0_|CPU_FALLBACK|No native Neuron impl|
|227|Tensor.igamma|CPU_FALLBACK|No native Neuron impl|
|228|Tensor.igamma_|CPU_FALLBACK|No native Neuron impl|
|229|Tensor.igammac|CPU_FALLBACK|No native Neuron impl|
|230|Tensor.igammac_|CPU_FALLBACK|No native Neuron impl|
|231|Tensor.index_add_|Y|Native MLIR impl|
|232|Tensor.index_add|Y|Native MLIR impl|
|233|Tensor.index_copy_|Y|Native MLIR impl|
|234|Tensor.index_copy|Y|Native MLIR impl|
|235|Tensor.index_fill_|CPU_FALLBACK|No native Neuron impl|
|236|Tensor.index_fill|CPU_FALLBACK|No native Neuron impl|
|237|Tensor.index_put_|Y|Supported via decomposition/torch-mlir lowering|
|238|Tensor.index_put|Y|Supported via decomposition/torch-mlir lowering|
|239|Tensor.index_reduce_|CPU_FALLBACK|No native Neuron impl|
|240|Tensor.index_reduce|CPU_FALLBACK|No native Neuron impl|
|241|Tensor.index_select|Y|Native MLIR impl; validates bounds|
|242|Tensor.indices|Y|Sparse tensor metadata|
|243|Tensor.inner|Y|Decomposes to mul/tensordot (supported)|
|244|Tensor.int|Y|Uses to(int32)|
|245|Tensor.int_repr|CPU_FALLBACK|Quantization not natively supported|
|246|Tensor.inverse|CPU_FALLBACK|CPU fallback registered for linalg_inv_ex|
|247|Tensor.isclose|Y|Decomposes to eq, abs, mul, add, le (all supported)|
|248|Tensor.isfinite|Y|Native MLIR impl|
|249|Tensor.isinf|Y|Native MLIR impl|
|250|Tensor.isposinf|CPU_FALLBACK|No native Neuron impl|
|251|Tensor.isneginf|Y|Native MLIR impl|
|252|Tensor.isnan|Y|Native MLIR impl|
|253|Tensor.is_contiguous|Y|Metadata/dtype checks|
|254|Tensor.is_complex|Y|Metadata/dtype checks|
|255|Tensor.is_conj|Y|Metadata/dtype checks|
|256|Tensor.is_floating_point|Y|Metadata/dtype checks|
|257|Tensor.is_inference|Y|Metadata/dtype checks|
|258|Tensor.is_leaf|Y|Metadata/dtype checks|
|259|Tensor.is_pinned|Y|Metadata/dtype checks|
|260|Tensor.is_set_to|Y|Metadata/dtype checks|
|261|Tensor.is_shared|Y|Metadata/dtype checks|
|262|Tensor.is_signed|Y|Metadata/dtype checks|
|263|Tensor.is_sparse|Y|Metadata/dtype checks|
|264|Tensor.istft|CPU_FALLBACK|No native Neuron impl|
|265|Tensor.isreal|Y|Metadata check only|
|266|Tensor.item|Y|Native C++ impl; copies to CPU|
|267|Tensor.kthvalue|CPU_FALLBACK|No native Neuron impl|
|268|Tensor.lcm|CPU_FALLBACK|No native Neuron impl|
|269|Tensor.lcm_|CPU_FALLBACK|No native Neuron impl|
|270|Tensor.ldexp|CPU_FALLBACK|No native Neuron impl|
|271|Tensor.ldexp_|CPU_FALLBACK|No native Neuron impl|
|272|Tensor.le|Y|Native MLIR impl|
|273|Tensor.le_|Y|Native MLIR impl|
|274|Tensor.less_equal|Y|Native MLIR impl|
|275|Tensor.less_equal_|Y|Native MLIR impl|
|276|Tensor.lerp|Y|Native MLIR impl|
|277|Tensor.lerp_|Y|Native MLIR impl|
|278|Tensor.lgamma|CPU_FALLBACK|No native Neuron impl|
|279|Tensor.lgamma_|CPU_FALLBACK|No native Neuron impl|
|280|Tensor.log|Y|Native MLIR impl / In-place|
|281|Tensor.log_|Y|Native MLIR impl / In-place|
|282|Tensor.logdet|CPU_FALLBACK|Decomposes to linalg_slogdet; CPU fallback|
|283|Tensor.log10|CPU_FALLBACK|Falls back to CPU|
|284|Tensor.log10_|CPU_FALLBACK|Falls back to CPU|
|285|Tensor.log1p|CPU_FALLBACK|Falls back to CPU|
|286|Tensor.log1p_|CPU_FALLBACK|Falls back to CPU|
|287|Tensor.log2|CPU_FALLBACK|Falls back to CPU|
|288|Tensor.log2_|CPU_FALLBACK|Falls back to CPU|
|289|Tensor.log_normal_|CPU_FALLBACK|No native Neuron impl|
|290|Tensor.logaddexp|CPU_FALLBACK|No native Neuron impl|
|291|Tensor.logaddexp2|CPU_FALLBACK|No native Neuron impl|
|292|Tensor.logsumexp|CPU_FALLBACK|No native Neuron impl|
|293|Tensor.logical_and|Y|Native MLIR impl / In-place|
|294|Tensor.logical_and_|Y|Native MLIR impl / In-place|
|295|Tensor.logical_not|Y|Native MLIR impl / In-place|
|296|Tensor.logical_not_|Y|Native MLIR impl / In-place|
|297|Tensor.logical_or|Y|Native MLIR impl / In-place|
|298|Tensor.logical_or_|Y|Native MLIR impl / In-place|
|299|Tensor.logical_xor|Y|Native MLIR impl / In-place|
|300|Tensor.logical_xor_|Y|Native MLIR impl / In-place|
|301|Tensor.logit|CPU_FALLBACK|No native Neuron impl|
|302|Tensor.logit_|CPU_FALLBACK|No native Neuron impl|
|303|Tensor.long|Y|Auto-cast: int64 → int32 with warning|
|304|Tensor.lt|Y|Native MLIR impl|
|305|Tensor.lt_|Y|Native MLIR impl|
|306|Tensor.less|Y|Native MLIR impl|
|307|Tensor.less_|Y|Native MLIR impl|
|308|Tensor.lu|CPU_FALLBACK|CPU fallback registered|
|309|Tensor.lu_solve|CPU_FALLBACK|CPU fallback registered|
|310|Tensor.as_subclass|Y|Metadata only|
|311|Tensor.map_|CPU_FALLBACK|No native Neuron impl|
|312|Tensor.masked_scatter_|CPU_FALLBACK|No native Neuron impl|
|313|Tensor.masked_scatter|CPU_FALLBACK|No native Neuron impl|
|314|Tensor.masked_fill_|Y|Native MLIR impl|
|315|Tensor.masked_fill|Y|Native MLIR impl|
|316|Tensor.masked_select|Y|Supported via decomposition/torch-mlir lowering|
|317|Tensor.matmul|Y|Decomposed to mm/bmm by PyTorch|
|318|Tensor.matrix_power|CPU_FALLBACK|CPU fallback|
|319|Tensor.matrix_exp|CPU_FALLBACK|CPU fallback|
|320|Tensor.max|Y|Native MLIR impl; supports unary and dim variants|
|321|Tensor.maximum|CPU_FALLBACK|Falls back to CPU (aten::maximum.out)|
|322|Tensor.mean|Y|Native MLIR impl; supports dim, keepdim, dtype|
|323|Tensor.module_load|Y|Module loading utility|
|324|Tensor.nanmean|CPU_FALLBACK|Falls back to CPU|
|325|Tensor.median|CPU_FALLBACK|Falls back to CPU|
|326|Tensor.nanmedian|CPU_FALLBACK|Falls back to CPU|
|327|Tensor.min|CPU_FALLBACK|Falls back to CPU|
|328|Tensor.minimum|CPU_FALLBACK|Falls back to CPU|
|329|Tensor.mm|Y|Native MLIR impl|
|330|Tensor.smm|CPU_FALLBACK|No native Neuron impl|
|331|Tensor.mode|CPU_FALLBACK|No native Neuron impl|
|332|Tensor.movedim|Y|View operation / Alias|
|333|Tensor.moveaxis|Y|View operation / Alias|
|334|Tensor.msort|CPU_FALLBACK|Decomposes to sort; sort falls back to CPU|
|335|Tensor.mul|Y|Native MLIR impl|
|336|Tensor.mul_|Y|Native MLIR impl|
|337|Tensor.multiply|Y|Native MLIR impl|
|338|Tensor.multiply_|Y|Native MLIR impl|
|339|Tensor.multinomial|CPU_FALLBACK|No native Neuron impl|
|340|Tensor.mv|Y|Native MLIR impl|
|341|Tensor.mvlgamma|CPU_FALLBACK|No native Neuron impl|
|342|Tensor.mvlgamma_|CPU_FALLBACK|No native Neuron impl|
|343|Tensor.nansum|CPU_FALLBACK|No native Neuron impl|
|344|Tensor.narrow|Y|View operation (slice)|
|345|Tensor.narrow_copy|CPU_FALLBACK|No native Neuron impl|
|346|Tensor.ndimension|Y|Alias for dim|
|347|Tensor.nan_to_num|CPU_FALLBACK|No native Neuron impl|
|348|Tensor.nan_to_num_|CPU_FALLBACK|No native Neuron impl|
|349|Tensor.ne|Y|Native MLIR impl|
|350|Tensor.ne_|Y|Native MLIR impl|
|351|Tensor.not_equal|Y|Native MLIR impl|
|352|Tensor.not_equal_|Y|Native MLIR impl|
|353|Tensor.neg|Y|Native MLIR impl|
|354|Tensor.neg_|Y|Native MLIR impl|
|355|Tensor.negative|Y|Native MLIR impl|
|356|Tensor.negative_|Y|Native MLIR impl|
|357|Tensor.nelement|Y|Metadata only|
|358|Tensor.nextafter|CPU_FALLBACK|No native Neuron impl|
|359|Tensor.nextafter_|CPU_FALLBACK|No native Neuron impl|
|360|Tensor.nonzero|Y|Native Python impl registered; dynamic output shape|
|361|Tensor.norm|Y|Supported via decomposition/torch-mlir lowering|
|362|Tensor.normal_|CPU_FALLBACK|No native Neuron impl|
|363|Tensor.numel|Y|Metadata only|
|364|Tensor.numpy|Y|Copies to CPU then converts; requires .cpu() first|
|365|Tensor.orgqr|Y|Decomposes to linalg_householder_product (composite)|
|366|Tensor.ormqr|CPU_FALLBACK|No native Neuron impl|
|367|Tensor.outer|Y|Decomposes to reshape + mul (supported)|
|368|Tensor.permute|Y|View operation; contiguous called if needed|
|369|Tensor.pin_memory|CPU_FALLBACK|No native Neuron impl|
|370|Tensor.pinverse|CPU_FALLBACK|No native Neuron impl|
|371|Tensor.polygamma|CPU_FALLBACK|No native Neuron impl|
|372|Tensor.polygamma_|CPU_FALLBACK|No native Neuron impl|
|373|Tensor.positive|Y|Returns self (no-op)|
|374|Tensor.pow|Y|Native MLIR impl|
|375|Tensor.pow_|Y|Native MLIR impl|
|376|Tensor.prod|CPU_FALLBACK|Falls back to CPU (aten::prod)|
|377|Tensor.put_|CPU_FALLBACK|No native Neuron impl|
|378|Tensor.qr|CPU_FALLBACK|Quantization/linalg not natively supported|
|379|Tensor.qscheme|CPU_FALLBACK|Quantization/linalg not natively supported|
|380|Tensor.quantile|CPU_FALLBACK|Quantization/linalg not natively supported|
|381|Tensor.nanquantile|CPU_FALLBACK|Quantization/linalg not natively supported|
|382|Tensor.q_scale|CPU_FALLBACK|Quantization/linalg not natively supported|
|383|Tensor.q_zero_point|CPU_FALLBACK|Quantization/linalg not natively supported|
|384|Tensor.q_per_channel_scales|CPU_FALLBACK|Quantization/linalg not natively supported|
|385|Tensor.q_per_channel_zero_points|CPU_FALLBACK|Quantization/linalg not natively supported|
|386|Tensor.q_per_channel_axis|CPU_FALLBACK|Quantization/linalg not natively supported|
|387|Tensor.rad2deg|CPU_FALLBACK|No native Neuron impl|
|388|Tensor.random_|CPU_FALLBACK|No native Neuron impl|
|389|Tensor.ravel|Y|Decomposed to flatten|
|390|Tensor.reciprocal|Y|Native MLIR impl|
|391|Tensor.reciprocal_|Y|Native MLIR impl|
|392|Tensor.record_stream|Y|Native C++ impl|
|393|Tensor.register_hook|Y|Autograd hooks|
|394|Tensor.register_post_accumulate_grad_hook|Y|Autograd hooks|
|395|Tensor.remainder|Y|Native MLIR impl|
|396|Tensor.remainder_|Y|Native MLIR impl|
|397|Tensor.renorm|CPU_FALLBACK|No native Neuron impl|
|398|Tensor.renorm_|CPU_FALLBACK|No native Neuron impl|
|399|Tensor.repeat|Y|Native MLIR impl / CompositeImplicitAutograd|
|400|Tensor.repeat_interleave|Y|Native MLIR impl / CompositeImplicitAutograd|
|401|Tensor.requires_grad|Y|Metadata only|
|402|Tensor.requires_grad_|Y|Metadata only|
|403|Tensor.reshape|Y|Decomposed to view or clone+view|
|404|Tensor.reshape_as|Y|Decomposed to view or clone+view|
|405|Tensor.resize_|Y|Native C++ impl (resize_neuron)|
|406|Tensor.resize_as_|Y|Native C++ impl (resize_neuron)|
|407|Tensor.retain_grad|Y|Autograd metadata|
|408|Tensor.retains_grad|Y|Autograd metadata|
|409|Tensor.roll|CPU_FALLBACK|Falls back to CPU (aten::roll)|
|410|Tensor.rot90|Y|Supported via decomposition/torch-mlir lowering|
|411|Tensor.round|CPU_FALLBACK|Falls back to CPU (aten::round.out)|
|412|Tensor.round_|CPU_FALLBACK|Falls back to CPU (aten::round.out)|
|413|Tensor.rsqrt|Y|Native MLIR impl|
|414|Tensor.rsqrt_|Y|Native MLIR impl|
|415|Tensor.scatter|Y|Native MLIR impl|
|416|Tensor.scatter_|Y|Native MLIR impl|
|417|Tensor.scatter_add_|Y|Native MLIR impl|
|418|Tensor.scatter_add|Y|Native MLIR impl|
|419|Tensor.scatter_reduce_|CPU_FALLBACK|No native Neuron impl|
|420|Tensor.scatter_reduce|CPU_FALLBACK|No native Neuron impl|
|421|Tensor.select|Y|View operation|
|422|Tensor.select_scatter|CPU_FALLBACK|No native Neuron impl|
|423|Tensor.set_|Y|Native C++ impl; all variants|
|424|Tensor.share_memory_|CPU_FALLBACK|Not applicable to Neuron|
|425|Tensor.short|Y|Uses to(int16)|
|426|Tensor.sigmoid|Y|Native MLIR impl|
|427|Tensor.sigmoid_|Y|Native MLIR impl|
|428|Tensor.sign|Y|Native MLIR impl|
|429|Tensor.sign_|Y|Native MLIR impl|
|430|Tensor.signbit|Y|Native MLIR impl|
|431|Tensor.sgn|Y|Native MLIR impl|
|432|Tensor.sgn_|Y|Native MLIR impl|
|433|Tensor.sin|Y|Native MLIR impl|
|434|Tensor.sin_|Y|Native MLIR impl|
|435|Tensor.sinc|CPU_FALLBACK|Falls back to CPU|
|436|Tensor.sinc_|CPU_FALLBACK|Falls back to CPU|
|437|Tensor.sinh|CPU_FALLBACK|Falls back to CPU|
|438|Tensor.sinh_|CPU_FALLBACK|Falls back to CPU|
|439|Tensor.asinh|CPU_FALLBACK|Falls back to CPU|
|440|Tensor.asinh_|CPU_FALLBACK|Falls back to CPU|
|441|Tensor.arcsinh|CPU_FALLBACK|Falls back to CPU|
|442|Tensor.arcsinh_|CPU_FALLBACK|Falls back to CPU|
|443|Tensor.shape|Y|Metadata only|
|444|Tensor.size|Y|Metadata only|
|445|Tensor.slogdet|CPU_FALLBACK|CPU fallback registered|
|446|Tensor.slice_scatter|CPU_FALLBACK|No native Neuron impl|
|447|Tensor.softmax|Y|Native MLIR impl|
|448|Tensor.sort|CPU_FALLBACK|Falls back to CPU|
|449|Tensor.split|Y|View operation|
|450|Tensor.sparse_mask|CPU_FALLBACK|Sparse not supported|
|451|Tensor.sparse_dim|Y|Returns 0 for dense tensors|
|452|Tensor.sqrt|Y|Native MLIR impl|
|453|Tensor.sqrt_|Y|Native MLIR impl|
|454|Tensor.square|Y|Native MLIR impl|
|455|Tensor.square_|Y|Native MLIR impl|
|456|Tensor.squeeze|Y|View operation|
|457|Tensor.squeeze_|Y|View operation|
|458|Tensor.std|CPU_FALLBACK|Falls back to CPU (aten::std.correction)|
|459|Tensor.stft|CPU_FALLBACK|No native Neuron impl|
|460|Tensor.storage|Y|Metadata/storage queries|
|461|Tensor.untyped_storage|Y|Metadata/storage queries|
|462|Tensor.storage_offset|Y|Metadata/storage queries|
|463|Tensor.storage_type|Y|Metadata/storage queries|
|464|Tensor.stride|Y|Metadata/storage queries|
|465|Tensor.sub|Y|Native MLIR impl|
|466|Tensor.sub_|Y|Native MLIR impl|
|467|Tensor.subtract|Y|Native MLIR impl|
|468|Tensor.subtract_|Y|Native MLIR impl|
|469|Tensor.sum|Y|Native MLIR impl|
|470|Tensor.sum_to_size|Y|Native MLIR impl|
|471|Tensor.svd|CPU_FALLBACK|CPU fallback registered|
|472|Tensor.swapaxes|Y|Alias for transpose|
|473|Tensor.swapdims|Y|Alias for transpose|
|474|Tensor.t|Y|View operation for 2D tensors|
|475|Tensor.t_|Y|View operation for 2D tensors|
|476|Tensor.tensor_split|Y|View operation / Decomposed to repeat|
|477|Tensor.tile|Y|View operation / Decomposed to repeat|
|478|Tensor.to|Y|Decomposed to _to_copy; native MLIR impl|
|479|Tensor.to_mkldnn|N|MKL-DNN specific|
|480|Tensor.take|CPU_FALLBACK|No native Neuron impl|
|481|Tensor.take_along_dim|Y|Decomposes to gather (supported)|
|482|Tensor.tan|CPU_FALLBACK|Falls back to CPU (aten::tan.out)|
|483|Tensor.tan_|CPU_FALLBACK|Falls back to CPU (aten::tan.out)|
|484|Tensor.tanh|Y|Native MLIR impl|
|485|Tensor.tanh_|Y|Native MLIR impl|
|486|Tensor.atanh|CPU_FALLBACK|Falls back to CPU (aten::atanh.out)|
|487|Tensor.atanh_|CPU_FALLBACK|Falls back to CPU (aten::atanh.out)|
|488|Tensor.arctanh|CPU_FALLBACK|Falls back to CPU (aten::atanh.out)|
|489|Tensor.arctanh_|CPU_FALLBACK|Falls back to CPU (aten::atanh.out)|
|490|Tensor.tolist|Y|Copies to CPU then converts|
|491|Tensor.topk|Y|Supported via decomposition/torch-mlir lowering|
|492|Tensor.to_dense|CPU_FALLBACK|Sparse conversion|
|493|Tensor.to_sparse|CPU_FALLBACK|Sparse conversion|
|494|Tensor.to_sparse_csr|CPU_FALLBACK|Sparse conversion|
|495|Tensor.to_sparse_csc|CPU_FALLBACK|Sparse conversion|
|496|Tensor.to_sparse_bsr|CPU_FALLBACK|Sparse conversion|
|497|Tensor.to_sparse_bsc|CPU_FALLBACK|Sparse conversion|
|498|Tensor.trace|CPU_FALLBACK|No native Neuron impl|
|499|Tensor.transpose|Y|View operation (metadata change)|
|500|Tensor.transpose_|Y|View operation (metadata change)|
|501|Tensor.triangular_solve|CPU_FALLBACK|CPU fallback registered|
|502|Tensor.tril|Y|Native MLIR impl|
|503|Tensor.tril_|Y|Native MLIR impl|
|504|Tensor.triu|Y|Native MLIR impl|
|505|Tensor.triu_|Y|Native MLIR impl|
|506|Tensor.true_divide|Y|Alias for div|
|507|Tensor.true_divide_|Y|Alias for div|
|508|Tensor.trunc|Y|Native MLIR impl|
|509|Tensor.trunc_|Y|Native MLIR impl|
|510|Tensor.type|Y|Uses to() internally|
|511|Tensor.type_as|Y|Uses to() internally|
|512|Tensor.unbind|Y|View operation; returns tuple of slices|
|513|Tensor.unflatten|Y|Decomposed to reshape|
|514|Tensor.unfold|Y|Native C++ impl (unfold_neuron)|
|515|Tensor.uniform_|CPU_FALLBACK|No native Neuron impl|
|516|Tensor.unique|CPU_FALLBACK|No native Neuron impl|
|517|Tensor.unique_consecutive|CPU_FALLBACK|No native Neuron impl|
|518|Tensor.unsqueeze|Y|View operation|
|519|Tensor.unsqueeze_|Y|View operation|
|520|Tensor.values|Y|Sparse tensor metadata|
|521|Tensor.var|CPU_FALLBACK|Falls back to CPU (aten::var.correction)|
|522|Tensor.vdot|CPU_FALLBACK|No native Neuron impl|
|523|Tensor.view|Y|Native C++ impl (view_neuron)|
|524|Tensor.view_as|Y|Native C++ impl (view_neuron)|
|525|Tensor.vsplit|Y|Decomposed to split|
|526|Tensor.where|Y|Decomposed to torch.where|
|527|Tensor.xlogy|CPU_FALLBACK|No native Neuron impl|
|528|Tensor.xlogy_|CPU_FALLBACK|No native Neuron impl|
|529|Tensor.xpu|N|Neuron is not XPU|
|530|Tensor.zero_|Y|Native MLIR impl|

## Tensor Attributes

||pytorch_api|neuron_compatible|limitations|
|---|---|---|---|
|1|torch.dtype|Y|Pure Python enum; Neuron supports: float16, bfloat16, float32, int8, int16, int32, uint8, bool. float64/int64 auto-cast to float32/int32|
|2|torch.device|Y|Pure Python class; Neuron registers as PrivateUse1 device type|
|3|torch.layout|Y|Pure Python enum; only torch.strided supported (no sparse layouts on Neuron)|
|4|torch.memory_format|Y|Pure Python enum; only contiguous and preserve formats supported on Neuron|

## torch.autograd

||pytorch_api|neuron_compatible|limitations|
|---|---|---|---|
|1|torch.autograd.backward|Y|Pure Python/C++ autograd engine; AutogradPrivateUse1 fallback redispatches ops|
|2|torch.autograd.grad|Y|Same mechanism as backward() but returns gradients directly|
|3|torch.no_grad|Y|Pure Python context managers/queries|
|4|torch.enable_grad|Y|Pure Python context managers/queries|
|5|torch.set_grad_enabled|Y|Pure Python context managers/queries|
|6|torch.is_grad_enabled|Y|Pure Python context managers/queries|
|7|torch.inference_mode|Y|Pure Python context managers/queries|
|8|torch.is_inference_mode_enabled|Y|Pure Python context managers/queries|
|9|torch.autograd.grad_mode.set_multithreading_enabled|Y|Pure Python; controls multithreaded backward|
|10|torch.autograd.Function|Y|Base class for custom autograd ops; torch_neuronx uses it for contiguous operations|
|11|Function.forward|Y|User-defined; can use Neuron tensors freely|
|12|Function.backward|Y|User-defined; can use Neuron tensors freely|
|13|FunctionCtx.save_for_backward|Y|No device dependency|
|14|FunctionCtx.mark_dirty|Y|No device dependency|
|15|FunctionCtx.mark_non_differentiable|Y|No device dependency|
|16|FunctionCtx.set_materialize_grads|Y|No device dependency|
|17|function.once_differentiable|Y|Decorator for backward; no device dependency|
|18|Tensor.grad|Y|Autograd properties and hooks|
|19|Tensor.requires_grad|Y|Autograd properties and hooks|
|20|Tensor.is_leaf|Y|Autograd properties and hooks|
|21|Tensor.backward|Y|Autograd properties and hooks|
|22|Tensor.detach|Y|Autograd properties and hooks|
|23|Tensor.detach_|Y|Autograd properties and hooks|
|24|Tensor.register_hook|Y|Autograd properties and hooks|
|25|Tensor.register_post_accumulate_grad_hook|Y|Autograd properties and hooks|
|26|Tensor.retain_grad|Y|Autograd properties and hooks|
|27|torch.autograd.detect_anomaly|Y|Pure Python; no device dependency|
|28|torch.autograd.set_detect_anomaly|Y|Pure Python; no device dependency|
|29|graph.Node.name|Y|No device dependency|
|30|graph.Node.metadata|Y|No device dependency|
|31|graph.Node.next_functions|Y|No device dependency|
|32|graph.Node.register_hook|Y|No device dependency|
|33|graph.Node.register_prehook|Y|No device dependency|
|34|graph.increment_version|Y|No device dependency|
|35|graph.saved_tensors_hooks|Y|No device dependency|
|36|graph.save_on_cpu|Y|No device dependency|
|37|graph.disable_saved_tensors_hooks|Y|No device dependency|
|38|graph.register_multi_grad_hook|Y|No device dependency|
|39|graph.allow_mutation_on_saved_tensors|Y|No device dependency|
|40|graph.GradientEdge|Y|No device dependency|
|41|graph.get_gradient_edge|Y|No device dependency|
|42|forward_ad.dual_level|Y|Pure Python forward-mode AD|
|43|forward_ad.make_dual|Y|Pure Python forward-mode AD|
|44|forward_ad.unpack_dual|Y|Pure Python forward-mode AD|
|45|functional.jacobian|Y|Depends on aten op support|
|46|functional.hessian|Y|Depends on aten op support|
|47|functional.vjp|Y|Depends on aten op support|
|48|functional.jvp|Y|Depends on aten op support|
|49|functional.vhp|Y|Depends on aten op support|
|50|functional.hvp|Y|Depends on aten op support|
|51|profiler.profile|Y|Custom NeuronActivityProfiler registered|
|52|profiler.profile.export_chrome_trace|Y|Custom NeuronActivityProfiler registered|
|53|profiler.profile.key_averages|Y|Custom NeuronActivityProfiler registered|
|54|profiler.profile.total_average|Y|Custom NeuronActivityProfiler registered|
|55|profiler.record_function|Y|Custom NeuronActivityProfiler registered|
|56|profiler.emit_nvtx|N|NVIDIA/Intel specific|
|57|profiler.emit_itt|N|NVIDIA/Intel specific|
|58|profiler.load_nvprof|N|NVIDIA/Intel specific|
|59|profiler.parse_nvprof_trace|N|NVIDIA/Intel specific|
|60|graph.set_warn_on_accumulate_grad_stream_mismatch|N|CUDA-specific|
|61|torch.autograd.gradcheck|CPU_FALLBACK|Some intermediate ops may fall back to CPU|
|62|torch.autograd.gradgradcheck|CPU_FALLBACK|Some intermediate ops may fall back to CPU|
|63|profiler utilities|CPU_FALLBACK|Various CPU fallback considerations|
|64|forward_ad low-level|CPU_FALLBACK|Various CPU fallback considerations|
|65|deprecated functions|CPU_FALLBACK|Various CPU fallback considerations|

## torch.library

||pytorch_api|neuron_compatible|limitations|
|---|---|---|---|
|1|torch.library.Library|Y|Pure Python; op registration infrastructure. Neuron uses this to register all its ops|
|2|torch.library.Library.define|Y|Pure Python; defines op schema|
|3|torch.library.Library.impl|Y|Pure Python; registers op implementation for a dispatch key|
|4|torch.library.fallthrough_kernel|Y|Pure Python; creates a fallthrough kernel for dispatch|

## torch.backends

||pytorch_api|neuron_compatible|limitations|
|---|---|---|---|
|1|torch.backends.cpu.get_cpu_capability|Y|CPU query; works on any platform|
|2|torch.backends.cuda.is_built|Y|Returns build-time CUDA availability|
|3|torch.backends.cuda.matmul.*|N/A|CUDA-specific settings; no effect on Neuron|
|4|torch.backends.cuda.cufft_plan_cache.*|N/A|CUDA-specific settings; no effect on Neuron|
|5|torch.backends.cuda.preferred_linalg_library|N/A|CUDA-specific settings; no effect on Neuron|
|6|torch.backends.cuda.SDPBackend|N/A|CUDA-specific settings; no effect on Neuron|
|7|torch.backends.cuda.*_sdp_*|N/A|CUDA-specific settings; no effect on Neuron|
|8|torch.backends.cuda.sdp_kernel|N/A|CUDA-specific settings; no effect on Neuron|
|19|torch.backends.cudnn.version|N/A|cuDNN-specific; no effect on Neuron|
|20|torch.backends.cudnn.is_available|N/A|cuDNN-specific; no effect on Neuron|
|21|torch.backends.cudnn.enabled|N/A|cuDNN-specific; no effect on Neuron|
|22|torch.backends.cudnn.allow_tf32|N/A|cuDNN-specific; no effect on Neuron|
|23|torch.backends.cudnn.deterministic|N/A|cuDNN-specific; no effect on Neuron|
|24|torch.backends.cudnn.benchmark|N/A|cuDNN-specific; no effect on Neuron|
|25|torch.backends.cudnn.benchmark_limit|N/A|cuDNN-specific; no effect on Neuron|
|26|torch.backends.mps.is_available|N/A|MPS-specific; returns False on Neuron|
|27|torch.backends.mps.is_built|N/A|MPS-specific; returns False on Neuron|
|28|torch.backends.mkl.is_available|Y|CPU queries; work on any platform|
|29|torch.backends.mkl.verbose|Y|CPU queries; work on any platform|
|30|torch.backends.mkldnn.is_available|Y|CPU queries; work on any platform|
|31|torch.backends.mkldnn.verbose|Y|CPU queries; work on any platform|
|32|torch.backends.openmp.is_available|Y|CPU query; works on any platform|
|33|torch.backends.opt_einsum.is_available|Y|Pure Python; works on any platform|
|34|torch.backends.opt_einsum.get_opt_einsum|Y|Pure Python; works on any platform|
|35|torch.backends.opt_einsum.enabled|Y|Pure Python; works on any platform|
|36|torch.backends.opt_einsum.strategy|Y|Pure Python; works on any platform|

## torch.distributed

||PyTorch API|Compatibility|Limitations|
|---|---|---|---|
|1|torch.distributed.is_available|Y|Pure Python checks; no backend call|
|2|torch.distributed.is_initialized|Y|Pure Python checks; no backend call|
|3|torch.distributed.is_mpi_available|Y|Pure Python checks; no backend call|
|4|torch.distributed.is_nccl_available|Y|Pure Python checks; no backend call|
|5|torch.distributed.is_gloo_available|Y|Pure Python checks; no backend call|
|6|torch.distributed.is_ucc_available|Y|Pure Python checks; no backend call|
|7|torch.distributed.is_xccl_available|Y|Pure Python checks; no backend call|
|8|torch.distributed.is_backend_available|Y|Pure Python checks; no backend call|
|9|torch.distributed.is_torchelastic_launched|Y|Pure Python checks; no backend call|
|10|torch.distributed.init_process_group|Y|backend="neuron". Creates ProcessGroupNeuron|
|11|torch.distributed.destroy_process_group|Y|Pure Python teardown|
|12|torch.distributed.Backend|Y|Pure Python enum. Neuron registered via Backend.register_backend("neuron")|
|13|torch.distributed.Backend.register_backend|Y|Pure Python enum. Neuron registered via Backend.register_backend("neuron")|
|14|torch.distributed.BackendConfig|Y|Pure Python queries|
|15|torch.distributed.get_backend|Y|Pure Python queries|
|16|torch.distributed.get_backend_config|Y|Pure Python queries|
|17|torch.distributed.get_default_backend_for_device|Y|Pure Python queries|
|18|torch.distributed.get_rank|Y|Pure Python queries|
|19|torch.distributed.get_world_size|Y|Pure Python queries|
|20|torch.distributed.get_group_rank|Y|Pure Python queries|
|21|torch.distributed.get_global_rank|Y|Pure Python queries|
|22|torch.distributed.get_process_group_ranks|Y|Pure Python queries|
|23|torch.distributed.get_node_local_rank|Y|Pure Python queries|
|24|torch.distributed.get_pg_count|Y|Pure Python queries|
|25|torch.distributed.Store|Y|C++ store classes; no backend dependency|
|26|torch.distributed.TCPStore|Y|C++ store classes; no backend dependency|
|27|torch.distributed.HashStore|Y|C++ store classes; no backend dependency|
|28|torch.distributed.FileStore|Y|C++ store classes; no backend dependency|
|29|torch.distributed.PrefixStore|Y|C++ store classes; no backend dependency|
|30|torch.distributed.Store.set|Y|C++ store classes; no backend dependency|
|31|torch.distributed.Store.get|Y|C++ store classes; no backend dependency|
|32|torch.distributed.Store.add|Y|C++ store classes; no backend dependency|
|33|torch.distributed.Store.compare_set|Y|C++ store classes; no backend dependency|
|34|torch.distributed.Store.wait|Y|C++ store classes; no backend dependency|
|35|torch.distributed.Store.num_keys|Y|C++ store classes; no backend dependency|
|36|torch.distributed.Store.delete_key|Y|C++ store classes; no backend dependency|
|37|torch.distributed.Store.set_timeout|Y|C++ store classes; no backend dependency|
|38|torch.distributed.ReduceOp|Y|Pure Python enum|
|39|torch.distributed.reduce_op|Y|Pure Python enum|
|40|torch.distributed.DistBackendError|Y|Infrastructure classes|
|41|torch.distributed.GroupMember|Y|Infrastructure classes|
|42|torch.distributed.GroupName|Y|Infrastructure classes|
|43|torch.distributed.ProcessGroup|Y|Infrastructure classes|
|44|torch.distributed.Work|Y|Infrastructure classes|
|45|torch.distributed.DebugLevel|Y|Infrastructure classes|
|46|torch.distributed.get_debug_level|Y|Infrastructure classes|
|47|torch.distributed.default_pg_timeout|Y|Infrastructure classes|
|48|torch.distributed.group|Y|Infrastructure classes|
|49|torch.distributed.supports_complex|Y|Options classes|
|50|torch.distributed.AllreduceOptions|Y|Options classes|
|51|torch.distributed.AllreduceCoalescedOptions|Y|Options classes|
|52|torch.distributed.AllToAllOptions|Y|Options classes|
|53|torch.distributed.BarrierOptions|Y|Options classes|
|54|torch.distributed.BroadcastOptions|Y|Options classes|
|55|torch.distributed.GatherOptions|Y|Options classes|
|56|torch.distributed.ReduceOptions|Y|Options classes|
|57|torch.distributed.ReduceScatterOptions|Y|Options classes|
|58|torch.distributed.ScatterOptions|Y|Options classes|
|59|torch.distributed.all_reduce|Y|Native impl. Supports SUM/PRODUCT/MIN/MAX|
|60|torch.distributed.all_reduce_coalesced|Y|Native impl|
|61|torch.distributed.broadcast|Y|Impl via all_reduce SUM|
|62|torch.distributed.all_gather|Y|Native impl with split optimization|
|63|torch.distributed.all_gather_into_tensor|Y|Native impl with split optimization|
|64|torch.distributed.all_gather_coalesced|Y|Native impl with split optimization|
|65|torch.distributed.reduce|Y|Impl via all_reduce|
|66|torch.distributed.reduce_scatter|Y|Native impl with split optimization|
|67|torch.distributed.reduce_scatter_tensor|Y|Native impl with split optimization|
|68|torch.distributed.gather|Y|Impl via all_gather|
|69|torch.distributed.scatter|Y|Impl via reduce_scatter SUM|
|70|torch.distributed.all_to_all|Y|Native impl; even split sizes only for all_to_all_single|
|71|torch.distributed.all_to_all_single|Y|Native impl; even split sizes only for all_to_all_single|
|72|torch.distributed.barrier|Y|Native impl via nrt_barrier(); default PG only|
|73|torch.distributed.broadcast_object_list|Y|Pure Python wrappers using supported collectives|
|74|torch.distributed.all_gather_object|Y|Pure Python wrappers using supported collectives|
|75|torch.distributed.gather_object|Y|Pure Python wrappers using supported collectives|
|76|torch.distributed.scatter_object_list|Y|Pure Python wrappers using supported collectives|
|77|torch.distributed.new_group|Y|Creates new ProcessGroupNeuron|
|78|torch.distributed.new_subgroups|Y|Creates new ProcessGroupNeuron|
|79|torch.distributed.new_subgroups_by_enumeration|Y|Creates new ProcessGroupNeuron|
|80|torch.distributed.split_group|N|Not implemented on Neuron|
|81|torch.distributed.shrink_group|N|Not implemented on Neuron|
|82|torch.distributed.send|N|No P2P support. Neuron NRT only supports collective operations|
|83|torch.distributed.recv|N|No P2P support. Neuron NRT only supports collective operations|
|84|torch.distributed.isend|N|No P2P support. Neuron NRT only supports collective operations|
|85|torch.distributed.irecv|N|No P2P support. Neuron NRT only supports collective operations|
|86|torch.distributed.batch_isend_irecv|N|No P2P support. Neuron NRT only supports collective operations|
|87|torch.distributed.P2POp|N|No P2P support. Neuron NRT only supports collective operations|
|88|torch.distributed.send_object_list|N|No P2P support. Neuron NRT only supports collective operations|
|89|torch.distributed.recv_object_list|N|No P2P support. Neuron NRT only supports collective operations|
|90|torch.distributed.monitored_barrier|N||
|91|torch.distributed.broadcast_multigpu|Removed|Removed in PyTorch v2.11|
|92|torch.distributed.all_reduce_multigpu|Removed|Removed in PyTorch v2.11|
|93|torch.distributed.reduce_multigpu|Removed|Removed in PyTorch v2.11|
|94|torch.distributed.all_gather_multigpu|Removed|Removed in PyTorch v2.11|
|95|torch.distributed.reduce_scatter_multigpu|Removed|Removed in PyTorch v2.11|

## torch.distributed.algorithms.join

||PyTorch API|Compatibility|Limitations|
|---|---|---|---|
|1|Join|Y|Pure Python; uses dist.all_reduce() internally|
|2|Join.notify_join_context|Y|Pure Python; uses dist.all_reduce() internally|
|3|Joinable|Y|Pure Python ABC|
|4|Joinable.join_device|Y|Pure Python ABC|
|5|Joinable.join_hook|Y|Pure Python ABC|
|6|Joinable.join_process_group|Y|Pure Python ABC|
|7|JoinHook|Y|Pure Python base class; no-op defaults|
|8|JoinHook.main_hook|Y|Pure Python base class; no-op defaults|
|9|JoinHook.post_hook|Y|Pure Python base class; no-op defaults|

## torch.distributed.fsdp

||PyTorch API|Compatibility|Limitations|
|---|---|---|---|
|1|torch.distributed.fsdp.FullyShardedDataParallel|Y|FSDP1 main class. Uses all_gather_into_tensor for unshard, reduce_scatter_tensor for grad sync|
|2|torch.distributed.fsdp.FullyShardedDataParallel.forward|Y|Calls _pre_forward (unshard via all_gather) and _post_forward (reshard)|
|3|torch.distributed.fsdp.FullyShardedDataParallel.clip_grad_norm_|Y|Pure Python. Computes local grad norms, then calls dist.all_reduce|
|4|torch.distributed.fsdp.FullyShardedDataParallel.full_optim_state_dict|Y|Gathers full optimizer state via all_gather_into_tensor and broadcast_object_list|
|5|torch.distributed.fsdp.FullyShardedDataParallel.optim_state_dict|Y|Unified API for getting optimizer state dict|
|6|torch.distributed.fsdp.FullyShardedDataParallel.register_comm_hook|Y|Registers callable hook for gradient communication|
|7|torch.distributed.fsdp.FullyShardedDataParallel.sharded_optim_state_dict|Y|Returns sharded optimizer state dict|
|8|torch.distributed.fsdp.FullyShardedDataParallel.summon_full_params|Y|Context manager. Unshards params via all_gather, reshards on exit|
|9|torch.distributed.fsdp.FullyShardedDataParallel.apply|Y|Pure Python override of nn.Module.apply|
|10|torch.distributed.fsdp.FullyShardedDataParallel.flatten_sharded_optim_state_dict|Y|Flattens sharded optim state dict for loading|
|11|torch.distributed.fsdp.FullyShardedDataParallel.optim_state_dict_to_load|Y|Converts optim state dict to loadable format|
|12|torch.distributed.fsdp.FullyShardedDataParallel.scatter_full_optim_state_dict|Y|Scatters full optim state to sharded form|
|13|torch.distributed.fsdp.FullyShardedDataParallel.shard_full_optim_state_dict|Y|Shards full optim state dict|
|14|torch.distributed.fsdp.FullyShardedDataParallel.fsdp_modules|Y|Returns list of FSDP modules|
|15|torch.distributed.fsdp.FullyShardedDataParallel.get_state_dict_type|Y|Returns current StateDictSettings|
|16|torch.distributed.fsdp.FullyShardedDataParallel.module|Y|Returns the wrapped nn.Module|
|17|torch.distributed.fsdp.FullyShardedDataParallel.named_buffers|Y|Yields named buffers with FSDP-aware deduplication|
|18|torch.distributed.fsdp.FullyShardedDataParallel.named_parameters|Y|Yields named parameters with FSDP-aware deduplication|
|19|torch.distributed.fsdp.FullyShardedDataParallel.no_sync|Y|Context manager. Disables gradient sync|
|20|torch.distributed.fsdp.FullyShardedDataParallel.rekey_optim_state_dict|Y|Rekeys optim state dict between param ID and param name formats|
|21|torch.distributed.fsdp.FullyShardedDataParallel.set_state_dict_type|Y|Sets StateDictType for the FSDP module tree|
|22|torch.distributed.fsdp.FullyShardedDataParallel.state_dict_type|Y|Context manager. Temporarily sets state dict type|
|23|BackwardPrefetch|Y|Pure Python enums/dataclasses|
|24|ShardingStrategy|Y|Pure Python enums/dataclasses|
|25|MixedPrecision|Y|Pure Python enums/dataclasses|
|26|CPUOffload|Y|Pure Python enums/dataclasses|
|27|StateDictType|Y|Pure Python enums/dataclasses|
|28|OptimStateKeyType|Y|Pure Python enums/dataclasses|
|29|StateDictConfig|Y|Pure Python enums/dataclasses|
|30|FullStateDictConfig|Y|Pure Python enums/dataclasses|
|31|ShardedStateDictConfig|Y|Pure Python enums/dataclasses|
|32|LocalStateDictConfig|Y|Pure Python enums/dataclasses|
|33|OptimStateDictConfig|Y|Pure Python enums/dataclasses|
|34|FullOptimStateDictConfig|Y|Pure Python enums/dataclasses|
|35|ShardedOptimStateDictConfig|Y|Pure Python enums/dataclasses|
|36|LocalOptimStateDictConfig|Y|Pure Python enums/dataclasses|
|37|StateDictSettings|Y|Pure Python enums/dataclasses|
|38|fully_shard|Y|FSDP2. Uses all_gather and reduce_scatter implemented on Neuron|
|39|FSDPModule|Y|FSDP2. Uses all_gather and reduce_scatter implemented on Neuron|
|40|UnshardHandle|Y|Pure Python|
|41|MixedPrecisionPolicy|Y|Pure Python|
|42|OffloadPolicy|Y|Pure Python|
|43|CPUOffloadPolicy|Y|Pure Python|
|44|register_fsdp_forward_method|Y|Pure Python|
|45|share_comm_ctx|Y|Pure Python|

## torch.distributed.optim

||PyTorch API|Compatibility|Limitations|
|---|---|---|---|
|1|DistributedOptimizer|N|Requires torch.distributed.rpc (not supported on Neuron)|
|2|DistributedOptimizer.step|N|Requires torch.distributed.rpc (not supported on Neuron)|
|3|PostLocalSGDOptimizer|Y|Uses dist.all_reduce (implemented on Neuron)|
|4|PostLocalSGDOptimizer.load_state_dict|Y|Uses dist.all_reduce (implemented on Neuron)|
|5|PostLocalSGDOptimizer.state_dict|Y|Uses dist.all_reduce (implemented on Neuron)|
|6|PostLocalSGDOptimizer.step|Y|Uses dist.all_reduce (implemented on Neuron)|
|7|torch.distributed.optim.ZeroRedundancyOptimizer|Y|Partitions optimizer states across ranks. Uses dist.broadcast|
|8|torch.distributed.optim.ZeroRedundancyOptimizer.add_param_group|Y|Pure Python. Adds param group and partitions new params|
|9|torch.distributed.optim.ZeroRedundancyOptimizer.consolidate_state_dict|Y|Gathers all optimizer states to rank 0 via dist.broadcast|
|10|torch.distributed.optim.ZeroRedundancyOptimizer.join_hook|Y|Returns _ZeROJoinHook for Join context manager|
|11|torch.distributed.optim.ZeroRedundancyOptimizer.load_state_dict|Y|Loads optimizer state dict and repartitions across ranks|
|12|torch.distributed.optim.ZeroRedundancyOptimizer.state_dict|Y|Returns consolidated or local optimizer state dict|
|13|torch.distributed.optim.ZeroRedundancyOptimizer.step|Y|Runs local optimizer step, then broadcasts updated params|
|14|as_functional_optim|Y|Pure Python utility|

## torch.distributed.tensor.parallel

||PyTorch API|Compatibility|Limitations|
|---|---|---|---|
|1|parallelize_module|Y|Uses all_reduce/reduce_scatter via DTensor|
|2|ColwiseParallel|Y|Pure Python ParallelStyle subclasses|
|3|RowwiseParallel|Y|Pure Python ParallelStyle subclasses|
|4|SequenceParallel|Y|Pure Python ParallelStyle subclasses|
|5|ParallelStyle|Y|Pure Python|
|6|PrepareModuleInput|Y|Pure Python|
|7|PrepareModuleOutput|Y|Pure Python|
|8|PrepareModuleInputOutput|Y|Pure Python|
|9|loss_parallel|Y|Pure Python|
|10|PairwiseParallel|Removed|Removed in PyTorch v2.11|
|11|enable_2d_with_fsdp|Removed|Removed in PyTorch v2.11|
|12|pre_dp_module_transform|Removed|Removed in PyTorch v2.11|

## torch.distributed.checkpoint

||PyTorch API|Compatibility|Limitations|
|---|---|---|---|
|1|save|Y|Uses all_gather_object, barrier, broadcast_object_list (all implemented)|
|2|load|Y|Uses all_gather_object, barrier, broadcast_object_list (all implemented)|
|3|save_state_dict|Y|Uses all_gather_object, barrier, broadcast_object_list (all implemented)|
|4|load_state_dict|Y|Uses all_gather_object, barrier, broadcast_object_list (all implemented)|
|5|async_save|Y|Uses all_gather_object, barrier, broadcast_object_list (all implemented)|
|6|CheckpointException|Y|Pure Python exception|
|7|load_sharded_optimizer_state_dict|Y|Same collective pattern as load()|
|8|StorageWriter/Reader ABCs|Y|Pure Python ABCs and implementations|
|9|SavePlanner/LoadPlanner ABCs|Y|Pure Python ABCs and implementations|
|10|data classes|Y|Pure Python ABCs and implementations|
|11|DefaultSavePlanner|Y|Pure Python ABCs and implementations|
|12|DefaultLoadPlanner|Y|Pure Python ABCs and implementations|
|13|FileSystemReader/Writer|Y|Pure Python ABCs and implementations|
|14|HuggingFaceStorageReader/Writer|Y|Pure Python ABCs and implementations|

## torch.distributions

||PyTorch API|Compatibility|Limitations|
|---|---|---|---|
|1|Distribution|Y|Pure Python base classes|
|2|ExponentialFamily|Y|Pure Python base classes|
|3|torch.distributions.Bernoulli|CPU_FALLBACK|Most use torch.rand/randn/bernoulli/multinomial (not registered on Neuron)|
|4|torch.distributions.Beta|CPU_FALLBACK|Most use torch.rand/randn/bernoulli/multinomial (not registered on Neuron)|
|5|torch.distributions.Binomial|CPU_FALLBACK|Most use torch.rand/randn/bernoulli/multinomial (not registered on Neuron)|
|6|torch.distributions.Categorical|CPU_FALLBACK|Most use torch.rand/randn/bernoulli/multinomial (not registered on Neuron)|
|7|torch.distributions.Cauchy|CPU_FALLBACK|Most use torch.rand/randn/bernoulli/multinomial (not registered on Neuron)|
|8|torch.distributions.Chi2|CPU_FALLBACK|Most use torch.rand/randn/bernoulli/multinomial (not registered on Neuron)|
|9|torch.distributions.ContinuousBernoulli|CPU_FALLBACK|Most use torch.rand/randn/bernoulli/multinomial (not registered on Neuron)|
|10|torch.distributions.Dirichlet|CPU_FALLBACK|Most use torch.rand/randn/bernoulli/multinomial (not registered on Neuron)|
|11|torch.distributions.Exponential|CPU_FALLBACK|Most use torch.rand/randn/bernoulli/multinomial (not registered on Neuron)|
|12|torch.distributions.FisherSnedecor|CPU_FALLBACK|Most use torch.rand/randn/bernoulli/multinomial (not registered on Neuron)|
|13|torch.distributions.Gamma|CPU_FALLBACK|Most use torch.rand/randn/bernoulli/multinomial (not registered on Neuron)|
|14|torch.distributions.Geometric|CPU_FALLBACK|Most use torch.rand/randn/bernoulli/multinomial (not registered on Neuron)|
|15|torch.distributions.Gumbel|CPU_FALLBACK|Most use torch.rand/randn/bernoulli/multinomial (not registered on Neuron)|
|16|torch.distributions.HalfCauchy|CPU_FALLBACK|Most use torch.rand/randn/bernoulli/multinomial (not registered on Neuron)|
|17|torch.distributions.HalfNormal|CPU_FALLBACK|Most use torch.rand/randn/bernoulli/multinomial (not registered on Neuron)|
|18|torch.distributions.Independent|CPU_FALLBACK|Most use torch.rand/randn/bernoulli/multinomial (not registered on Neuron)|
|19|torch.distributions.InverseGamma|CPU_FALLBACK|Most use torch.rand/randn/bernoulli/multinomial (not registered on Neuron)|
|20|torch.distributions.Kumaraswamy|CPU_FALLBACK|Most use torch.rand/randn/bernoulli/multinomial (not registered on Neuron)|
|21|torch.distributions.Laplace|CPU_FALLBACK|Most use torch.rand/randn/bernoulli/multinomial (not registered on Neuron)|
|22|torch.distributions.LKJCholesky|CPU_FALLBACK|Most use torch.rand/randn/bernoulli/multinomial (not registered on Neuron)|
|23|torch.distributions.LogNormal|CPU_FALLBACK|Most use torch.rand/randn/bernoulli/multinomial (not registered on Neuron)|
|24|torch.distributions.LogisticNormal|CPU_FALLBACK|Most use torch.rand/randn/bernoulli/multinomial (not registered on Neuron)|
|25|torch.distributions.LowRankMultivariateNormal|CPU_FALLBACK|Most use torch.rand/randn/bernoulli/multinomial (not registered on Neuron)|
|26|torch.distributions.MixtureSameFamily|CPU_FALLBACK|Most use torch.rand/randn/bernoulli/multinomial (not registered on Neuron)|
|27|torch.distributions.Multinomial|CPU_FALLBACK|Most use torch.rand/randn/bernoulli/multinomial (not registered on Neuron)|
|28|torch.distributions.MultivariateNormal|CPU_FALLBACK|Most use torch.rand/randn/bernoulli/multinomial (not registered on Neuron)|
|29|torch.distributions.NegativeBinomial|CPU_FALLBACK|Most use torch.rand/randn/bernoulli/multinomial (not registered on Neuron)|
|30|torch.distributions.Normal|CPU_FALLBACK|Most use torch.rand/randn/bernoulli/multinomial (not registered on Neuron)|
|31|torch.distributions.OneHotCategorical|CPU_FALLBACK|Most use torch.rand/randn/bernoulli/multinomial (not registered on Neuron)|
|32|torch.distributions.OneHotCategoricalStraightThrough|CPU_FALLBACK|Most use torch.rand/randn/bernoulli/multinomial (not registered on Neuron)|
|33|torch.distributions.Pareto|CPU_FALLBACK|Most use torch.rand/randn/bernoulli/multinomial (not registered on Neuron)|
|34|torch.distributions.Poisson|CPU_FALLBACK|Most use torch.rand/randn/bernoulli/multinomial (not registered on Neuron)|
|35|torch.distributions.RelaxedBernoulli|CPU_FALLBACK|Most use torch.rand/randn/bernoulli/multinomial (not registered on Neuron)|
|36|torch.distributions.RelaxedOneHotCategorical|CPU_FALLBACK|Most use torch.rand/randn/bernoulli/multinomial (not registered on Neuron)|
|37|torch.distributions.StudentT|CPU_FALLBACK|Most use torch.rand/randn/bernoulli/multinomial (not registered on Neuron)|
|38|torch.distributions.TransformedDistribution|CPU_FALLBACK|Most use torch.rand/randn/bernoulli/multinomial (not registered on Neuron)|
|39|torch.distributions.Uniform|CPU_FALLBACK|Most use torch.rand/randn/bernoulli/multinomial (not registered on Neuron)|
|40|torch.distributions.VonMises|CPU_FALLBACK|Most use torch.rand/randn/bernoulli/multinomial (not registered on Neuron)|
|41|torch.distributions.Weibull|CPU_FALLBACK|Most use torch.rand/randn/bernoulli/multinomial (not registered on Neuron)|
|42|torch.distributions.Wishart|CPU_FALLBACK|Most use torch.rand/randn/bernoulli/multinomial (not registered on Neuron)|
|43|transforms.Transform|Y/CPU_FALLBACK|Transforms using native ops (abs,mul,exp,pow,softmax) work; those using unsupported ops (cholesky,log1p,diag_embed) fall back|
|44|transforms.AbsTransform|Y/CPU_FALLBACK|Transforms using native ops (abs,mul,exp,pow,softmax) work; those using unsupported ops (cholesky,log1p,diag_embed) fall back|
|45|transforms.AffineTransform|Y/CPU_FALLBACK|Transforms using native ops (abs,mul,exp,pow,softmax) work; those using unsupported ops (cholesky,log1p,diag_embed) fall back|
|46|transforms.CatTransform|Y/CPU_FALLBACK|Transforms using native ops (abs,mul,exp,pow,softmax) work; those using unsupported ops (cholesky,log1p,diag_embed) fall back|
|47|transforms.ComposeTransform|Y/CPU_FALLBACK|Transforms using native ops (abs,mul,exp,pow,softmax) work; those using unsupported ops (cholesky,log1p,diag_embed) fall back|
|48|transforms.ExpTransform|Y/CPU_FALLBACK|Transforms using native ops (abs,mul,exp,pow,softmax) work; those using unsupported ops (cholesky,log1p,diag_embed) fall back|
|49|transforms.PowerTransform|Y/CPU_FALLBACK|Transforms using native ops (abs,mul,exp,pow,softmax) work; those using unsupported ops (cholesky,log1p,diag_embed) fall back|
|50|transforms.ReshapeTransform|Y/CPU_FALLBACK|Transforms using native ops (abs,mul,exp,pow,softmax) work; those using unsupported ops (cholesky,log1p,diag_embed) fall back|
|51|transforms.SoftmaxTransform|Y/CPU_FALLBACK|Transforms using native ops (abs,mul,exp,pow,softmax) work; those using unsupported ops (cholesky,log1p,diag_embed) fall back|
|52|transforms.StackTransform|Y/CPU_FALLBACK|Transforms using native ops (abs,mul,exp,pow,softmax) work; those using unsupported ops (cholesky,log1p,diag_embed) fall back|
|53|transforms....|Y/CPU_FALLBACK|Transforms using native ops (abs,mul,exp,pow,softmax) work; those using unsupported ops (cholesky,log1p,diag_embed) fall back|
|63|constraints|Y|Pure Python|
|64|constraint_registry.biject_to|Y|Pure Python|
|65|constraint_registry.transform_to|Y|Pure Python|
|66|kl_divergence|CPU_FALLBACK|Most KL functions use lgamma, digamma, log1p (not registered)|
|67|register_kl|Y|Pure Python decorator|

## torch.futures

||PyTorch API|Compatibility|Limitations|
|---|---|---|---|
|1|Future|Y|Pure Python|
|2|Future.done|Y|Pure Python|
|3|Future.wait|Y|Pure Python|
|4|Future.value|Y|Pure Python|
|5|Future.then|Y|Pure Python|
|6|Future.add_done_callback|Y|Pure Python|
|7|Future.set_result|Y|Pure Python|
|8|Future.set_exception|Y|Pure Python|
|9|collect_all|Y|Pure Python|
|10|wait_all|Y|Pure Python|

## torch.linalg

||pytorch_api|neuron_compatible|limitations|
|---|---|---|---|
|1|torch.linalg.norm|CPU_FALLBACK|Vector norm works natively; matrix norm may fall back|
|2|torch.linalg.vector_norm|Y|Native MLIR impl|
|3|torch.linalg.matrix_norm|CPU_FALLBACK|Frobenius norm works; spectral norms require SVD|
|4|torch.linalg.diagonal|Y|View operation|
|5|torch.linalg.det|CPU_FALLBACK|Backend dispatch (CPU/CUDA)|
|6|torch.linalg.slogdet|CPU_FALLBACK|Backend dispatch (CPU/CUDA)|
|7|torch.linalg.cond|CPU_FALLBACK|Backend dispatch (CPU/CUDA)|
|8|torch.linalg.matrix_rank|CPU_FALLBACK|Backend dispatch (CPU/CUDA)|
|9|torch.linalg.cholesky|CPU_FALLBACK|Backend dispatch (CPU/CUDA)|
|10|torch.linalg.qr|CPU_FALLBACK|Backend dispatch (CPU/CUDA)|
|11|torch.linalg.lu|CPU_FALLBACK|Backend dispatch (CPU/CUDA)|
|12|torch.linalg.lu_factor|CPU_FALLBACK|Backend dispatch (CPU/CUDA)|
|13|torch.linalg.eig|CPU_FALLBACK|Backend dispatch (CPU/CUDA)|
|14|torch.linalg.eigvals|CPU_FALLBACK|Backend dispatch (CPU/CUDA)|
|15|torch.linalg.eigh|CPU_FALLBACK|Backend dispatch (CPU/CUDA)|
|16|torch.linalg.eigvalsh|CPU_FALLBACK|Backend dispatch (CPU/CUDA)|
|17|torch.linalg.svd|CPU_FALLBACK|Backend dispatch (CPU/CUDA)|
|18|torch.linalg.svdvals|CPU_FALLBACK|Backend dispatch (CPU/CUDA)|
|19|torch.linalg.solve|CPU_FALLBACK|Backend dispatch (CPU/CUDA)|
|20|torch.linalg.solve_triangular|CPU_FALLBACK|Backend dispatch (CPU/CUDA)|
|21|torch.linalg.lu_solve|CPU_FALLBACK|Backend dispatch (CPU/CUDA)|
|22|torch.linalg.lstsq|CPU_FALLBACK|Backend dispatch (CPU/CUDA)|
|23|torch.linalg.inv|CPU_FALLBACK|Backend dispatch (CPU/CUDA)|
|24|torch.linalg.pinv|CPU_FALLBACK|Backend dispatch (CPU/CUDA)|
|25|torch.linalg.matrix_exp|CPU_FALLBACK|Backend dispatch (CPU/CUDA)|
|26|torch.linalg.matrix_power|CPU_FALLBACK|Backend dispatch (CPU/CUDA)|
|27|torch.linalg.cross|CPU_FALLBACK|Backend dispatch (CPU/CUDA)|
|28|torch.linalg.matmul|Y|Decomposes to mm/bmm (registered)|
|29|torch.linalg.vecdot|Y|Decomposes to supported ops|
|30|torch.linalg.multi_dot|Y|Decomposes to supported ops|
|31|torch.linalg.householder_product|CPU_FALLBACK/Y|vander is Y; rest CPU_FALLBACK|
|32|torch.linalg.tensorinv|CPU_FALLBACK/Y|vander is Y; rest CPU_FALLBACK|
|33|torch.linalg.tensorsolve|CPU_FALLBACK/Y|vander is Y; rest CPU_FALLBACK|
|34|torch.linalg.vander|CPU_FALLBACK/Y|vander is Y; rest CPU_FALLBACK|
|35|torch.linalg.cholesky_ex|CPU_FALLBACK/Y|vander is Y; rest CPU_FALLBACK|
|36|torch.linalg.inv_ex|CPU_FALLBACK/Y|vander is Y; rest CPU_FALLBACK|
|37|torch.linalg.solve_ex|CPU_FALLBACK/Y|vander is Y; rest CPU_FALLBACK|
|38|torch.linalg.lu_factor_ex|CPU_FALLBACK/Y|vander is Y; rest CPU_FALLBACK|
|39|torch.linalg.ldl_factor|CPU_FALLBACK/Y|vander is Y; rest CPU_FALLBACK|
|40|torch.linalg.ldl_factor_ex|CPU_FALLBACK/Y|vander is Y; rest CPU_FALLBACK|
|41|torch.linalg.ldl_solve|CPU_FALLBACK/Y|vander is Y; rest CPU_FALLBACK|

## torch.signal

||pytorch_api|neuron_compatible|limitations|
|---|---|---|---|
|1|torch.signal.windows.bartlett|CPU_FALLBACK|Uses torch.linspace (CPU fallback)|
|2|torch.signal.windows.blackman|CPU_FALLBACK|Uses torch.linspace (CPU fallback)|
|3|torch.signal.windows.cosine|CPU_FALLBACK|Uses torch.linspace (CPU fallback)|
|4|torch.signal.windows.exponential|CPU_FALLBACK|Uses torch.linspace (CPU fallback)|
|5|torch.signal.windows.gaussian|CPU_FALLBACK|Uses torch.linspace (CPU fallback)|
|6|torch.signal.windows.general_cosine|CPU_FALLBACK|Uses torch.linspace (CPU fallback)|
|7|torch.signal.windows.general_hamming|CPU_FALLBACK|Uses torch.linspace (CPU fallback)|
|8|torch.signal.windows.hamming|CPU_FALLBACK|Uses torch.linspace (CPU fallback)|
|9|torch.signal.windows.hann|CPU_FALLBACK|Uses torch.linspace (CPU fallback)|
|10|torch.signal.windows.kaiser|CPU_FALLBACK|Uses torch.linspace (CPU fallback)|
|11|torch.signal.windows.nuttall|CPU_FALLBACK|Uses torch.linspace (CPU fallback)|

## torch.special

||pytorch_api|neuron_compatible|limitations|
|---|---|---|---|
|1|torch.special.airy_ai|CPU_FALLBACK|Backend dispatch not registered|
|2|torch.special.bessel_*|CPU_FALLBACK|Backend dispatch not registered|
|3|torch.special.chebyshev_*|CPU_FALLBACK|Backend dispatch not registered|
|12|torch.special.erf|Y|Alias for torch.erf (native)|
|15|torch.special.erfinv|Y|Alias for torch.erfinv (native)|
|17|torch.special.expit|Y|Alias for torch.sigmoid (native)|
|32|torch.special.log_softmax|Y|Alias for F.log_softmax (native)|
|52|torch.special.softmax|Y|Alias for F.softmax (native)|
|All others|CPU_FALLBACK|Backend dispatch not registered on Neuron|

## torch.nn.init

||pytorch_api|neuron_compatible|limitations|
|---|---|---|---|
|1|calculate_gain|Y|Pure Python|
|2|uniform_|CPU_FALLBACK|aten::uniform_/normal_ falls back to CPU|
|3|normal_|CPU_FALLBACK|aten::uniform_/normal_ falls back to CPU|
|4|constant_|Y|Uses fill_/zero_/eye_/copy_ (supported)|
|5|ones_|Y|Uses fill_/zero_/eye_/copy_ (supported)|
|6|zeros_|Y|Uses fill_/zero_/eye_/copy_ (supported)|
|7|eye_|Y|Uses fill_/zero_/eye_/copy_ (supported)|
|8|dirac_|Y|Uses fill_/zero_/eye_/copy_ (supported)|
|9|xavier_uniform_|CPU_FALLBACK|Uses uniform_/normal_ internally|
|10|xavier_normal_|CPU_FALLBACK|Uses uniform_/normal_ internally|
|11|kaiming_uniform_|CPU_FALLBACK|Uses uniform_/normal_ internally|
|12|kaiming_normal_|CPU_FALLBACK|Uses uniform_/normal_ internally|
|13|trunc_normal_|CPU_FALLBACK|Uses uniform_/normal_ internally|
|14|orthogonal_|CPU_FALLBACK|Uses uniform_/normal_ internally|
|15|sparse_|CPU_FALLBACK|Uses uniform_/normal_ internally|

## torch.optim

||pytorch_api|neuron_compatible|limitations|
|---|---|---|---|
|1|torch.optim.Optimizer|Y|Constructor; pure Python|
|2|Optimizer.add_param_group|Y|Pure Python; no device ops|
|3|Optimizer.load_state_dict|Y|Pure Python; no device ops|
|4|Optimizer.state_dict|Y|Pure Python; no device ops|
|5|Optimizer.step|Y|Runs on Neuron; see specific optimizer|
|6|Optimizer.zero_grad|Y|Uses zero_ or fill_ (on Neuron)|
|7|torch.optim.Adadelta|Y|Uses _foreach_mul_, addcmul, sqrt, _foreach_div_ (all on Neuron)|
|8|torch.optim.Adadelta.step|Y|Uses _foreach_mul_, addcmul, sqrt, _foreach_div_ (all on Neuron)|
|9|torch.optim.Adagrad|Y|Uses addcmul, addcdiv, sqrt, _foreach_mul (all on Neuron)|
|10|torch.optim.Adagrad.step|Y|Uses addcmul, addcdiv, sqrt, _foreach_mul (all on Neuron)|
|11|torch.optim.Adam|Y|Uses _foreach_mul_, addcmul, addcdiv, lerp_, sqrt, div_ (all on Neuron)|
|12|torch.optim.Adam.step|Y|Uses _foreach_mul_, addcmul, addcdiv, lerp_, sqrt, div_ (all on Neuron)|
|13|torch.optim.AdamW|Y|Uses _foreach_mul_, addcmul, addcdiv, lerp_, sqrt, div_ (all on Neuron)|
|14|torch.optim.AdamW.step|Y|Uses _foreach_mul_, addcmul, addcdiv, lerp_, sqrt, div_ (all on Neuron)|
|15|torch.optim.SparseAdam|Y|Constructor; pure Python|
|16|torch.optim.SparseAdam.step|N|Sparse tensors not supported on Neuron|
|17|torch.optim.Adamax|Y|Uses _foreach_mul_, lerp_, addcdiv, abs, clamp_min (all on Neuron)|
|18|torch.optim.Adamax.step|Y|Uses _foreach_mul_, lerp_, addcdiv, abs, clamp_min (all on Neuron)|
|19|torch.optim.ASGD|Y|Uses _foreach_sub, _foreach_add, addcmul, pow, reciprocal (all on Neuron)|
|20|torch.optim.ASGD.step|Y|Uses _foreach_sub, _foreach_add, addcmul, pow, reciprocal (all on Neuron)|
|21|torch.optim.LBFGS|Y|Uses dot, linear, cat, sum, neg (all on Neuron)|
|22|torch.optim.LBFGS.step|Y|Uses dot, linear, cat, sum, neg (all on Neuron)|
|23|torch.optim.NAdam|Y|Uses _foreach_mul_, addcmul, addcdiv, lerp_, sqrt, div_ (all on Neuron)|
|24|torch.optim.NAdam.step|Y|Uses _foreach_mul_, addcmul, addcdiv, lerp_, sqrt, div_ (all on Neuron)|
|25|torch.optim.RAdam|Y|Uses _foreach_mul_, addcmul, addcdiv, lerp_, sqrt, div_ (all on Neuron)|
|26|torch.optim.RAdam.step|Y|Uses _foreach_mul_, addcmul, addcdiv, lerp_, sqrt, div_ (all on Neuron)|
|27|torch.optim.RMSprop|Y|Uses _foreach_mul_, addcmul, addcdiv, sqrt (all on Neuron)|
|28|torch.optim.RMSprop.step|Y|Uses _foreach_mul_, addcmul, addcdiv, sqrt (all on Neuron)|
|29|torch.optim.Rprop|Y|Uses _foreach_mul, sign, where, clamp (all on Neuron)|
|30|torch.optim.Rprop.step|Y|Uses _foreach_mul, sign, where, clamp (all on Neuron)|
|31|torch.optim.SGD|Y|Uses _foreach_add_ (on Neuron)|
|32|torch.optim.SGD.step|Y|Uses _foreach_add_ (on Neuron)|
|33|All optimizers: .add_param_group, .load_state_dict, .state_dict, .zero_grad, .register_*_hook|Y|Pure Python; no device ops|
|10|AdamW|Y|All use _foreach_add/mul/sub/div, addcmul, addcdiv, lerp_, sqrt, etc. (all on Neuron). Exception: SparseAdam.step is N (sparse tensors not supported)|
|11|SparseAdam|Y|All use _foreach_add/mul/sub/div, addcmul, addcdiv, lerp_, sqrt, etc. (all on Neuron). Exception: SparseAdam.step is N (sparse tensors not supported)|
|12|Adamax|Y|All use _foreach_add/mul/sub/div, addcmul, addcdiv, lerp_, sqrt, etc. (all on Neuron). Exception: SparseAdam.step is N (sparse tensors not supported)|
|13|ASGD|Y|All use _foreach_add/mul/sub/div, addcmul, addcdiv, lerp_, sqrt, etc. (all on Neuron). Exception: SparseAdam.step is N (sparse tensors not supported)|
|14|LBFGS|Y|All use _foreach_add/mul/sub/div, addcmul, addcdiv, lerp_, sqrt, etc. (all on Neuron). Exception: SparseAdam.step is N (sparse tensors not supported)|
|15|NAdam|Y|All use _foreach_add/mul/sub/div, addcmul, addcdiv, lerp_, sqrt, etc. (all on Neuron). Exception: SparseAdam.step is N (sparse tensors not supported)|
|16|RAdam|Y|All use _foreach_add/mul/sub/div, addcmul, addcdiv, lerp_, sqrt, etc. (all on Neuron). Exception: SparseAdam.step is N (sparse tensors not supported)|
|17|RMSprop|Y|All use _foreach_add/mul/sub/div, addcmul, addcdiv, lerp_, sqrt, etc. (all on Neuron). Exception: SparseAdam.step is N (sparse tensors not supported)|
|18|Rprop|Y|All use _foreach_add/mul/sub/div, addcmul, addcdiv, lerp_, sqrt, etc. (all on Neuron). Exception: SparseAdam.step is N (sparse tensors not supported)|
|163|All lr_schedulers + all methods|Y|Pure Python; adjusts lr scalar, no device ops|

## DDP Communication Hooks

||PyTorch API|Compatibility|Limitations|
|---|---|---|---|
|1|DDPCommHookType|Y||
|2|register_ddp_comm_hook|Y||
|3|default.allreduce_hook|Y||
|4|default.fp16_compress_hook|Y||
|5|default.bf16_compress_hook|N|DDP C++ rejects non-CUDA/XPU for bf16|
|6|default.fp16_compress_wrapper|Y||
|7|default.bf16_compress_wrapper|N||
|8|powerSGD.PowerSGDState|Y||
|9|powerSGD.powerSGD_hook|Y||
|10|powerSGD.batched_powerSGD_hook|N|Runtime failure in batched mode|
|11|quantization.quantization_pertensor_hook|Y||
|12|quantization.quantization_perchannel_hook|N|Per-channel quantization not supported on Neuron|
|13|post_localSGD|Y||
|14|debugging.noop_hook|Y||
|15|ddp_zero_hook.*|Y||
|16|optimizer_overlap.*|Y||
|17|mixed_precision.*|Y||

## Pipeline Parallelism

||PyTorch API|Compatibility|Limitations|
|---|---|---|---|
|1|All pipelining APIs (Pipe|N|Requires P2P (send/recv) which is not supported on Neuron|

## Quantization

||pytorch_api|neuron_compatible|limitations|
|---|---|---|---|
|1|torch.ao.quantization.quantize|Y|Pure Python graph transformation; no device ops|
|2|torch.ao.quantization.quantize_dynamic|Y|Pure Python graph transformation; no device ops|
|3|torch.ao.quantization.quantize_qat|Y|Pure Python graph transformation; no device ops|
|4|torch.ao.quantization.prepare|Y|Pure Python graph transformation; no device ops|
|5|torch.ao.quantization.prepare_qat|Y|Pure Python graph transformation; no device ops|
|6|torch.ao.quantization.convert|Y|Pure Python graph transformation; no device ops|
|7|torch.ao.quantization.fuse_modules|Y|Pure Python graph transformation; no device ops|
|8|torch.ao.quantization.QuantStub|Y|Pure Python graph transformation; no device ops|
|9|torch.ao.quantization.DeQuantStub|Y|Pure Python graph transformation; no device ops|
|10|torch.ao.quantization.QuantWrapper|Y|Pure Python graph transformation; no device ops|
|11|torch.ao.quantization.add_quant_dequant|Y|Pure Python graph transformation; no device ops|
|12|torch.ao.quantization.swap_module|Y|Pure Python graph transformation; no device ops|
|13|torch.ao.quantization.propagate_qconfig_|Y|Pure Python graph transformation; no device ops|
|14|torch.ao.quantization.default_eval_fn|Y|Pure Python graph transformation; no device ops|
|15|torch.ao.quantization.quantize_fx.*|Y|Pure Python graph transformation; no device ops|
|19|QConfigMapping|Y|Pure Python configuration|
|20|BackendConfig|Y|Pure Python configuration|
|21|BackendPatternConfig|Y|Pure Python configuration|
|22|DTypeConfig|Y|Pure Python configuration|
|23|ObservationType|Y|Pure Python configuration|
|24|FuseCustomConfig|Y|Pure Python configuration|
|25|PrepareCustomConfig|Y|Pure Python configuration|
|26|ConvertCustomConfig|Y|Pure Python configuration|
|27|StandaloneModuleConfigEntry|Y|Pure Python configuration|
|77|torch.ao.quantization.observer.ObserverBase|Y|Pure Python; observers run on CPU during calibration|
|78|torch.ao.quantization.observer.ObserverBase.with_args|Y|Pure Python|
|79|torch.ao.quantization.observer.ObserverBase.with_callable_args|Y|Pure Python|
|80|torch.ao.quantization.observer.MinMaxObserver|Y|Pure Python; observers run on CPU during calibration|
|81|torch.ao.quantization.observer.MinMaxObserver.calculate_qparams|Y|Pure Python|
|82|torch.ao.quantization.observer.MinMaxObserver.forward|Y|Pure Python|
|83|torch.ao.quantization.observer.MinMaxObserver.reset_min_max_vals|Y|Pure Python|
|84|torch.ao.quantization.observer.MovingAverageMinMaxObserver|Y|Pure Python|
|85|torch.ao.quantization.observer.PerChannelMinMaxObserver|Y|Pure Python|
|86|torch.ao.quantization.observer.PerChannelMinMaxObserver.reset_min_max_vals|Y|Pure Python|
|87|torch.ao.quantization.observer.MovingAveragePerChannelMinMaxObserver|Y|Pure Python|
|88|torch.ao.quantization.observer.HistogramObserver|Y|Pure Python|
|89|torch.ao.quantization.observer.PlaceholderObserver|Y|Pure Python|
|90|torch.ao.quantization.observer.RecordingObserver|Y|Pure Python|
|91|torch.ao.quantization.observer.NoopObserver|Y|Pure Python|
|92|torch.ao.quantization.observer.get_observer_state_dict|Y|Pure Python|
|93|torch.ao.quantization.observer.load_observer_state_dict|Y|Pure Python|
|94|torch.ao.quantization.observer.default_observer|Y|Pure Python|
|95|torch.ao.quantization.observer.default_placeholder_observer|Y|Pure Python|
|96|torch.ao.quantization.observer.default_debug_observer|Y|Pure Python|
|97|torch.ao.quantization.observer.default_weight_observer|Y|Pure Python|
|98|torch.ao.quantization.observer.default_histogram_observer|Y|Pure Python|
|99|torch.ao.quantization.observer.default_per_channel_weight_observer|Y|Pure Python|
|100|torch.ao.quantization.observer.default_dynamic_quant_observer|Y|Pure Python|
|101|torch.ao.quantization.observer.default_float_qparams_observer|Y|Pure Python|
|102|torch.ao.quantization.fake_quantize.FakeQuantizeBase|Y|Pure Python|
|103|torch.ao.quantization.fake_quantize.FakeQuantize|Y|Pure Python|
|104|torch.ao.quantization.fake_quantize.FixedQParamsFakeQuantize|Y|Pure Python|
|105|torch.ao.quantization.fake_quantize.FusedMovingAvgObsFakeQuantize|Y|Pure Python|
|106|torch.ao.quantization.fake_quantize.default_fake_quant|Y|Pure Python|
|107|torch.ao.quantization.fake_quantize.default_weight_fake_quant|Y|Pure Python|
|108|torch.ao.quantization.fake_quantize.default_per_channel_weight_fake_quant|Y|Pure Python|
|109|torch.ao.quantization.fake_quantize.default_histogram_fake_quant|Y|Pure Python|
|110|torch.ao.quantization.fake_quantize.default_fused_act_fake_quant|Y|Pure Python|
|111|torch.ao.quantization.fake_quantize.default_fused_wt_fake_quant|Y|Pure Python|
|112|torch.ao.quantization.fake_quantize.default_fused_per_channel_wt_fake_quant|Y|Pure Python|
|113|torch.ao.quantization.fake_quantize.disable_fake_quant|Y|Pure Python|
|114|torch.ao.quantization.fake_quantize.enable_fake_quant|Y|Pure Python|
|115|torch.ao.quantization.fake_quantize.disable_observer|Y|Pure Python|
|116|torch.ao.quantization.fake_quantize.enable_observer|Y|Pure Python|
|117|torch.ao.quantization.qconfig.QConfig|Y|Pure Python configuration|
|118|torch.ao.quantization.qconfig.default_qconfig|Y|Pure Python configuration|
|119|torch.ao.quantization.qconfig.default_debug_qconfig|Y|Pure Python configuration|
|120|torch.ao.quantization.qconfig.default_per_channel_qconfig|Y|Pure Python configuration|
|121|torch.ao.quantization.qconfig.default_dynamic_qconfig|Y|Pure Python configuration|
|122|torch.ao.quantization.qconfig.float16_dynamic_qconfig|Y|Pure Python configuration|
|123|torch.ao.quantization.qconfig.float16_static_qconfig|Y|Pure Python configuration|
|124|torch.ao.quantization.qconfig.per_channel_dynamic_qconfig|Y|Pure Python configuration|
|125|torch.ao.quantization.qconfig.float_qparams_weight_only_qconfig|Y|Pure Python configuration|
|126|torch.ao.quantization.qconfig.default_qat_qconfig|Y|Pure Python configuration|
|127|torch.ao.quantization.qconfig.default_weight_only_qconfig|Y|Pure Python configuration|
|128|torch.ao.quantization.qconfig.default_activation_only_qconfig|Y|Pure Python configuration|
|129|torch.ao.quantization.qconfig.default_qat_qconfig_v2|Y|Pure Python configuration|
|130|torch.ao.nn.intrinsic.ConvReLU1d|Y|Decomposes to convolution + relu (both on Neuron)|
|131|torch.ao.nn.intrinsic.ConvReLU2d|Y|Decomposes to convolution + relu (both on Neuron)|
|132|torch.ao.nn.intrinsic.ConvReLU3d|Y|Decomposes to convolution + relu (both on Neuron)|
|133|torch.ao.nn.intrinsic.LinearReLU|Y|Decomposes to convolution + relu (both on Neuron)|
|134|torch.ao.nn.intrinsic.ConvBn*|CPU_FALLBACK|native_batch_norm falls back to CPU|
|135|torch.ao.nn.intrinsic.BNReLU*|CPU_FALLBACK|native_batch_norm falls back to CPU|
|142|torch.ao.nn.intrinsic.qat.*|CPU_FALLBACK|QAT forward partially falls back|
|143|torch.ao.nn.qat.*|CPU_FALLBACK|QAT forward partially falls back|
|165|torch.ao.nn.quantized.ReLU6|CPU_FALLBACK|Quantized tensor ops not supported on Neuron; runs on CPU|
|166|torch.ao.nn.quantized.Hardswish|CPU_FALLBACK|Quantized tensor ops not supported on Neuron; runs on CPU|
|167|torch.ao.nn.quantized.ELU|CPU_FALLBACK|Quantized tensor ops not supported on Neuron; runs on CPU|
|168|torch.ao.nn.quantized.LeakyReLU|CPU_FALLBACK|Quantized tensor ops not supported on Neuron; runs on CPU|
|169|torch.ao.nn.quantized.Sigmoid|CPU_FALLBACK|Quantized tensor ops not supported on Neuron; runs on CPU|
|170|torch.ao.nn.quantized.BatchNorm2d|CPU_FALLBACK|Quantized tensor ops not supported on Neuron; runs on CPU|
|171|torch.ao.nn.quantized.BatchNorm3d|CPU_FALLBACK|Quantized tensor ops not supported on Neuron; runs on CPU|
|172|torch.ao.nn.quantized.Conv1d|CPU_FALLBACK|Quantized tensor ops not supported on Neuron; runs on CPU|
|173|torch.ao.nn.quantized.Conv1d.from_float|CPU_FALLBACK|Quantized tensor ops not supported on Neuron; runs on CPU|
|174|torch.ao.nn.quantized.Conv2d|CPU_FALLBACK|Quantized tensor ops not supported on Neuron; runs on CPU|
|175|torch.ao.nn.quantized.Conv2d.from_float|CPU_FALLBACK|Quantized tensor ops not supported on Neuron; runs on CPU|
|176|torch.ao.nn.quantized.Conv3d|CPU_FALLBACK|Quantized tensor ops not supported on Neuron; runs on CPU|
|177|torch.ao.nn.quantized.Conv3d.from_float|CPU_FALLBACK|Quantized tensor ops not supported on Neuron; runs on CPU|
|178|torch.ao.nn.quantized.ConvTranspose1d|CPU_FALLBACK|Quantized tensor ops not supported on Neuron; runs on CPU|
|179|torch.ao.nn.quantized.ConvTranspose2d|CPU_FALLBACK|Quantized tensor ops not supported on Neuron; runs on CPU|
|180|torch.ao.nn.quantized.ConvTranspose3d|CPU_FALLBACK|Quantized tensor ops not supported on Neuron; runs on CPU|
|181|torch.ao.nn.quantized.Embedding|CPU_FALLBACK|Quantized tensor ops not supported on Neuron; runs on CPU|
|182|torch.ao.nn.quantized.Embedding.from_float|CPU_FALLBACK|Quantized tensor ops not supported on Neuron; runs on CPU|
|183|torch.ao.nn.quantized.EmbeddingBag|CPU_FALLBACK|Quantized tensor ops not supported on Neuron; runs on CPU|
|184|torch.ao.nn.quantized.EmbeddingBag.from_float|CPU_FALLBACK|Quantized tensor ops not supported on Neuron; runs on CPU|
|185|torch.ao.nn.quantized.FloatFunctional|CPU_FALLBACK|Quantized tensor ops not supported on Neuron; runs on CPU|
|186|torch.ao.nn.quantized.FXFloatFunctional|CPU_FALLBACK|Quantized tensor ops not supported on Neuron; runs on CPU|
|187|torch.ao.nn.quantized.QFunctional|CPU_FALLBACK|Quantized tensor ops not supported on Neuron; runs on CPU|
|188|torch.ao.nn.quantized.Linear|CPU_FALLBACK|Quantized tensor ops not supported on Neuron; runs on CPU|
|189|torch.ao.nn.quantized.Linear.from_float|CPU_FALLBACK|Quantized tensor ops not supported on Neuron; runs on CPU|
|190|torch.ao.nn.quantized.Linear.from_reference|CPU_FALLBACK|Quantized tensor ops not supported on Neuron; runs on CPU|
|191|torch.ao.nn.quantized.LayerNorm|CPU_FALLBACK|Quantized tensor ops not supported on Neuron; runs on CPU|
|192|torch.ao.nn.quantized.GroupNorm|CPU_FALLBACK|Quantized tensor ops not supported on Neuron; runs on CPU|
|193|torch.ao.nn.quantized.InstanceNorm1d|CPU_FALLBACK|Quantized tensor ops not supported on Neuron; runs on CPU|
|194|torch.ao.nn.quantized.InstanceNorm2d|CPU_FALLBACK|Quantized tensor ops not supported on Neuron; runs on CPU|
|195|torch.ao.nn.quantized.InstanceNorm3d|CPU_FALLBACK|Quantized tensor ops not supported on Neuron; runs on CPU|
|196|torch.ao.nn.quantized.functional.avg_pool2d|CPU_FALLBACK|Quantized tensor ops not supported on Neuron; runs on CPU|
|197|torch.ao.nn.quantized.functional.avg_pool3d|CPU_FALLBACK|Quantized tensor ops not supported on Neuron; runs on CPU|
|198|torch.ao.nn.quantized.functional.adaptive_avg_pool2d|CPU_FALLBACK|Quantized tensor ops not supported on Neuron; runs on CPU|
|199|torch.ao.nn.quantized.functional.adaptive_avg_pool3d|CPU_FALLBACK|Quantized tensor ops not supported on Neuron; runs on CPU|
|200|torch.ao.nn.quantized.functional.conv1d|CPU_FALLBACK|Quantized tensor ops not supported on Neuron; runs on CPU|
|201|torch.ao.nn.quantized.functional.conv2d|CPU_FALLBACK|Quantized tensor ops not supported on Neuron; runs on CPU|
|202|torch.ao.nn.quantized.functional.conv3d|CPU_FALLBACK|Quantized tensor ops not supported on Neuron; runs on CPU|
|203|torch.ao.nn.quantized.functional.interpolate|CPU_FALLBACK|Quantized tensor ops not supported on Neuron; runs on CPU|
|204|torch.ao.nn.quantized.functional.linear|CPU_FALLBACK|Quantized tensor ops not supported on Neuron; runs on CPU|
|205|torch.ao.nn.quantized.functional.max_pool1d|CPU_FALLBACK|Quantized tensor ops not supported on Neuron; runs on CPU|
|206|torch.ao.nn.quantized.functional.max_pool2d|CPU_FALLBACK|Quantized tensor ops not supported on Neuron; runs on CPU|
|207|torch.ao.nn.quantized.functional.celu|CPU_FALLBACK|Quantized tensor ops not supported on Neuron; runs on CPU|
|208|torch.ao.nn.quantized.functional.leaky_relu|CPU_FALLBACK|Quantized tensor ops not supported on Neuron; runs on CPU|
|209|torch.ao.nn.quantized.functional.hardtanh|CPU_FALLBACK|Quantized tensor ops not supported on Neuron; runs on CPU|
|210|torch.ao.nn.quantized.functional.hardswish|CPU_FALLBACK|Quantized tensor ops not supported on Neuron; runs on CPU|
|211|torch.ao.nn.quantized.functional.threshold|CPU_FALLBACK|Quantized tensor ops not supported on Neuron; runs on CPU|
|212|torch.ao.nn.quantized.functional.elu|CPU_FALLBACK|Quantized tensor ops not supported on Neuron; runs on CPU|
|213|torch.ao.nn.quantized.functional.hardsigmoid|CPU_FALLBACK|Quantized tensor ops not supported on Neuron; runs on CPU|
|214|torch.ao.nn.quantized.functional.clamp|CPU_FALLBACK|Quantized tensor ops not supported on Neuron; runs on CPU|
|215|torch.ao.nn.quantized.functional.upsample|CPU_FALLBACK|Quantized tensor ops not supported on Neuron; runs on CPU|
|216|torch.ao.nn.quantized.functional.upsample_bilinear|CPU_FALLBACK|Quantized tensor ops not supported on Neuron; runs on CPU|
|217|torch.ao.nn.quantized.functional.upsample_nearest|CPU_FALLBACK|Quantized tensor ops not supported on Neuron; runs on CPU|
|218|torch.ao.nn.quantizable.LSTM|CPU_FALLBACK|Quantized tensor ops not supported on Neuron; runs on CPU|
|219|torch.ao.nn.quantizable.MultiheadAttention|CPU_FALLBACK|Quantized tensor ops not supported on Neuron; runs on CPU|
|220|torch.ao.nn.quantizable.MultiheadAttention.dequantize|CPU_FALLBACK|Quantized tensor ops not supported on Neuron; runs on CPU|
|221|torch.ao.nn.quantizable.MultiheadAttention.forward|CPU_FALLBACK|Quantized tensor ops not supported on Neuron; runs on CPU|
|222|torch.ao.nn.quantized.dynamic.Linear|CPU_FALLBACK|Quantized tensor ops not supported on Neuron; runs on CPU|
|223|torch.ao.nn.quantized.dynamic.Linear.from_float|CPU_FALLBACK|Quantized tensor ops not supported on Neuron; runs on CPU|
|224|torch.ao.nn.quantized.dynamic.Linear.from_reference|CPU_FALLBACK|Quantized tensor ops not supported on Neuron; runs on CPU|
|225|torch.ao.nn.quantized.dynamic.LSTM|CPU_FALLBACK|Quantized tensor ops not supported on Neuron; runs on CPU|
|226|torch.ao.nn.quantized.dynamic.GRU|CPU_FALLBACK|Quantized tensor ops not supported on Neuron; runs on CPU|
|227|torch.ao.nn.quantized.dynamic.RNNCell|CPU_FALLBACK|Quantized tensor ops not supported on Neuron; runs on CPU|
|228|torch.ao.nn.quantized.dynamic.LSTMCell|CPU_FALLBACK|Quantized tensor ops not supported on Neuron; runs on CPU|
|229|torch.ao.nn.quantized.dynamic.GRUCell|CPU_FALLBACK|Quantized tensor ops not supported on Neuron; runs on CPU|
|230|torch.ao.ns._numeric_suite.compare_weights|Y|Pure Python; compares float vs quantized model weights|
|231|torch.ao.ns._numeric_suite.get_logger_dict|Y|Pure Python; extracts logged tensor data|
|232|torch.ao.ns._numeric_suite.Logger|Y|Pure Python nn.Module; records tensor values|
|233|torch.ao.ns._numeric_suite.Logger.forward|Y|Pure Python nn.Module; records tensor values|
|234|torch.ao.ns._numeric_suite.ShadowLogger|Y|Pure Python nn.Module; records tensor values|
|235|torch.ao.ns._numeric_suite.ShadowLogger.forward|Y|Pure Python nn.Module; records tensor values|
|236|torch.ao.ns._numeric_suite.OutputLogger|Y|Pure Python nn.Module; records tensor values|
|237|torch.ao.ns._numeric_suite.OutputLogger.forward|Y|Pure Python nn.Module; records tensor values|
|238|torch.ao.ns._numeric_suite.Shadow|Y|Pure Python nn.Module; wraps module for comparison|
|239|torch.ao.ns._numeric_suite.Shadow.forward|CPU_FALLBACK|Forwards to quantized op; falls back to CPU|
|240|torch.ao.ns._numeric_suite.Shadow.add|CPU_FALLBACK|Forwards to quantized op; falls back to CPU|
|241|torch.ao.ns._numeric_suite.Shadow.add_scalar|CPU_FALLBACK|Forwards to quantized op; falls back to CPU|
|242|torch.ao.ns._numeric_suite.Shadow.mul|CPU_FALLBACK|Forwards to quantized op; falls back to CPU|
|243|torch.ao.ns._numeric_suite.Shadow.mul_scalar|CPU_FALLBACK|Forwards to quantized op; falls back to CPU|
|244|torch.ao.ns._numeric_suite.Shadow.cat|CPU_FALLBACK|Forwards to quantized op; falls back to CPU|
|245|torch.ao.ns._numeric_suite.Shadow.add_relu|CPU_FALLBACK|Forwards to quantized op; falls back to CPU|
|246|torch.ao.ns._numeric_suite.prepare_model_with_stubs|Y|Pure Python|
|247|torch.ao.ns._numeric_suite.compare_model_stub|Y|Pure Python|
|248|torch.ao.ns._numeric_suite.get_matching_activations|Y|Pure Python|
|249|torch.ao.ns._numeric_suite.prepare_model_outputs|Y|Pure Python|
|250|torch.ao.ns._numeric_suite.compare_model_outputs|Y|Pure Python|
|251|torch.ao.ns._numeric_suite_fx.OutputLogger|Y|Pure Python|
|252|torch.ao.ns._numeric_suite_fx.OutputLogger.forward|Y|Pure Python|
|253|torch.ao.ns._numeric_suite_fx.OutputComparisonLogger|Y|Pure Python|
|254|torch.ao.ns._numeric_suite_fx.OutputComparisonLogger.forward|Y|Pure Python|
|255|torch.ao.ns._numeric_suite_fx.NSTracer|Y|Pure Python|
|256|torch.ao.ns._numeric_suite_fx.NSTracer.is_leaf_module|Y|Pure Python|
|257|torch.ao.ns._numeric_suite_fx.extract_weights|Y|Pure Python|
|258|torch.ao.ns._numeric_suite_fx.add_loggers|Y|Pure Python|
|259|torch.ao.ns._numeric_suite_fx.extract_logger_info|Y|Pure Python|
|260|torch.ao.ns._numeric_suite_fx.add_shadow_loggers|Y|Pure Python|
|261|torch.ao.ns._numeric_suite_fx.extract_shadow_logger_info|Y|Pure Python|
|262|torch.ao.ns._numeric_suite_fx.extend_logger_results_with_comparison|Y|Pure Python|
|263|torch.ao.ns._numeric_suite_fx.prepare_n_shadows_model|Y|Pure Python|
|264|torch.ao.ns._numeric_suite_fx.loggers_set_enabled|Y|Pure Python|
|265|torch.ao.ns._numeric_suite_fx.loggers_set_save_activations|Y|Pure Python|
|266|torch.ao.ns._numeric_suite_fx.convert_n_shadows_model|Y|Pure Python|
|267|torch.ao.ns._numeric_suite_fx.extract_results_n_shadows_model|Y|Pure Python|
|268|torch.ao.ns._numeric_suite_fx.print_comparisons_n_shadows_model|Y|Pure Python|
|269|torch.ao.ns.fx.utils.compute_sqnr|Y|Pure Python|
|270|torch.ao.ns.fx.utils.compute_normalized_l2_error|Y|Pure Python|
|271|torch.ao.ns.fx.utils.compute_cosine_similarity|Y|Pure Python|

## torch.random

||pytorch_api|neuron_compatible|limitations|
|---|---|---|---|
|1|fork_rng|Y|Pure Python; manages CPU RNG state|
|2|get_rng_state|Y|Pure Python; manages CPU RNG state|
|3|initial_seed|Y|Pure Python; manages CPU RNG state|
|4|manual_seed|Y|Pure Python; manages CPU RNG state|
|5|seed|Y|Pure Python; manages CPU RNG state|
|6|set_rng_state|Y|Pure Python; manages CPU RNG state|

## torch.Storage

||pytorch_api|neuron_compatible|limitations|
|---|---|---|---|
|1|torch.TypedStorage|Y|Legacy typed storage class|
|2|torch.TypedStorage.bfloat16|Y|Returns dtype-cast view of storage|
|3|torch.TypedStorage.bool|Y|Returns dtype-cast view of storage|
|4|torch.TypedStorage.byte|Y|Returns dtype-cast view of storage|
|5|torch.TypedStorage.char|Y|Returns dtype-cast view of storage|
|6|torch.TypedStorage.clone|Y|Uses copy_ internally; supported via NeuronStorageImpl|
|7|torch.TypedStorage.complex_double|CPU_FALLBACK|Complex dtypes not supported on Neuron|
|8|torch.TypedStorage.complex_float|CPU_FALLBACK|Complex dtypes not supported on Neuron|
|9|torch.TypedStorage.copy_|Y|Supported via Neuron copy infrastructure|
|10|torch.TypedStorage.cpu|Y|Copies storage to CPU|
|11|torch.TypedStorage.cuda|N/A|Other backend specific|
|12|torch.TypedStorage.data_ptr|Y|Metadata query; no device compute|
|13|torch.TypedStorage.device|Y|Metadata query; no device compute|
|14|torch.TypedStorage.double|Y|Metadata query; no device compute|
|15|torch.TypedStorage.dtype|Y|Metadata query; no device compute|
|16|torch.TypedStorage.element_size|Y|Metadata query; no device compute|
|17|torch.TypedStorage.fill_|CPU_FALLBACK|Storage-level fill; falls back to CPU|
|18|torch.TypedStorage.float|Y|Returns dtype-cast view of storage|
|19|torch.TypedStorage.from_buffer|Y|CPU-side construction / metadata|
|20|torch.TypedStorage.from_file|Y|CPU-side construction / metadata|
|21|torch.TypedStorage.get_device|Y|CPU-side construction / metadata|
|22|torch.TypedStorage.half|Y|Returns dtype-cast view of storage|
|23|torch.TypedStorage.hpu|N/A|Other backend specific|
|24|torch.TypedStorage.int|Y|Returns dtype-cast view of storage|
|25|torch.TypedStorage.is_cuda|Y|Metadata query|
|26|torch.TypedStorage.is_hpu|Y|Metadata query|
|27|torch.TypedStorage.is_pinned|Y|Metadata query|
|28|torch.TypedStorage.is_shared|Y|Metadata query|
|29|torch.TypedStorage.is_sparse|Y|Metadata query|
|30|torch.TypedStorage.long|Y|Metadata query|
|31|torch.TypedStorage.nbytes|Y|Metadata query|
|32|torch.TypedStorage.pickle_storage_type|Y|Metadata query|
|33|torch.TypedStorage.pin_memory|CPU_FALLBACK|Not applicable to Neuron|
|34|torch.TypedStorage.resize_|Y|Supported via NeuronStorageImpl resize|
|35|torch.TypedStorage.share_memory_|CPU_FALLBACK|Not applicable to Neuron|
|36|torch.TypedStorage.short|Y|Metadata / copies to CPU|
|37|torch.TypedStorage.size|Y|Metadata / copies to CPU|
|38|torch.TypedStorage.tolist|Y|Metadata / copies to CPU|
|39|torch.TypedStorage.type|Y|Metadata / copies to CPU|
|40|torch.TypedStorage.untyped|Y|Metadata / copies to CPU|
|41|torch.UntypedStorage|Y|Base storage class|
|42|torch.UntypedStorage.bfloat16|Y/CPU_FALLBACK|dtype views (Y); byteswap (CPU_FALLBACK)|
|43|torch.UntypedStorage.bool|Y/CPU_FALLBACK|dtype views (Y); byteswap (CPU_FALLBACK)|
|44|torch.UntypedStorage.byte|Y/CPU_FALLBACK|dtype views (Y); byteswap (CPU_FALLBACK)|
|45|torch.UntypedStorage.byteswap|Y/CPU_FALLBACK|dtype views (Y); byteswap (CPU_FALLBACK)|
|46|torch.UntypedStorage.char|Y/CPU_FALLBACK|dtype views (Y); byteswap (CPU_FALLBACK)|
|47|torch.UntypedStorage.clone|Y|Uses copy_ internally|
|48|torch.UntypedStorage.complex_double|CPU_FALLBACK|Complex dtypes not supported|
|49|torch.UntypedStorage.complex_float|CPU_FALLBACK|Complex dtypes not supported|
|50|torch.UntypedStorage.copy_|Y|Supported via Neuron copy infrastructure|
|51|torch.UntypedStorage.cpu|Y|Copies storage to CPU|
|52|torch.UntypedStorage.cuda|N/A|Other backend specific|
|53|torch.UntypedStorage.data_ptr|Y|Metadata query|
|54|torch.UntypedStorage.device|Y|Metadata query|
|55|torch.UntypedStorage.double|Y|Metadata query|
|56|torch.UntypedStorage.element_size|Y|Metadata query|
|57|torch.UntypedStorage.fill_|CPU_FALLBACK|Storage-level fill; falls back to CPU|
|58|torch.UntypedStorage.float|Y|CPU-side construction / metadata|
|59|torch.UntypedStorage.from_buffer|Y|CPU-side construction / metadata|
|60|torch.UntypedStorage.from_file|Y|CPU-side construction / metadata|
|61|torch.UntypedStorage.get_device|Y|CPU-side construction / metadata|
|62|torch.UntypedStorage.half|Y|CPU-side construction / metadata|
|63|torch.UntypedStorage.hpu|N/A|Other backend specific|
|64|torch.UntypedStorage.int|Y|Metadata query|
|65|torch.UntypedStorage.is_cuda|Y|Metadata query|
|66|torch.UntypedStorage.is_hpu|Y|Metadata query|
|67|torch.UntypedStorage.is_pinned|Y|Metadata query|
|68|torch.UntypedStorage.is_shared|Y|Metadata query|
|69|torch.UntypedStorage.is_sparse|Y|Metadata query|
|70|torch.UntypedStorage.is_sparse_csr|Y|Metadata query|
|71|torch.UntypedStorage.long|Y|Returns dtype-cast view|
|72|torch.UntypedStorage.mps|N/A|Other backend specific|
|73|torch.UntypedStorage.nbytes|Y|Metadata / creates new storage|
|74|torch.UntypedStorage.new|Y|Metadata / creates new storage|
|75|torch.UntypedStorage.pin_memory|CPU_FALLBACK|Not applicable to Neuron|
|76|torch.UntypedStorage.resize_|Y|Supported via NeuronStorageImpl resize|
|77|torch.UntypedStorage.share_memory_|CPU_FALLBACK|Not applicable to Neuron|
|78|torch.UntypedStorage.short|Y|Metadata / copies to CPU|
|79|torch.UntypedStorage.size|Y|Metadata / copies to CPU|
|80|torch.UntypedStorage.tolist|Y|Metadata / copies to CPU|
|81|torch.UntypedStorage.type|Y|Metadata / copies to CPU|
|82|torch.UntypedStorage.untyped|Y|Metadata / copies to CPU|
|83|Legacy storage classes (DoubleStorage|Y|Legacy typed storage classes; metadata query|
|84|FloatStorage|Y|Legacy typed storage classes; metadata query|
|85|HalfStorage|Y|Legacy typed storage classes; metadata query|
|86|LongStorage|Y|Legacy typed storage classes; metadata query|
|87|IntStorage|Y|Legacy typed storage classes; metadata query|
|88|ShortStorage|Y|Legacy typed storage classes; metadata query|
|89|CharStorage|Y|Legacy typed storage classes; metadata query|
|90|ByteStorage|Y|Legacy typed storage classes; metadata query|
|91|BoolStorage|Y|Legacy typed storage classes; metadata query|
|92|BFloat16Storage|Y|Legacy typed storage classes; metadata query|
|93|ComplexDoubleStorage|Y|Legacy typed storage classes; metadata query|
|94|ComplexFloatStorage|Y|Legacy typed storage classes; metadata query|
|95|QUInt8Storage|Y|Legacy typed storage classes; metadata query|
|96|QInt8Storage|Y|Legacy typed storage classes; metadata query|
|97|QInt32Storage|Y|Legacy typed storage classes; metadata query|
|98|QUInt4x2Storage|Y|Legacy typed storage classes; metadata query|
|99|QUInt2x4Storage) + .dtype|Y|Legacy typed storage classes; metadata query|

## torch.utils.data

||pytorch_api|neuron_compatible|limitations|
|---|---|---|---|
|1|DataLoader|Y|Pure Python; loads data on CPU, user moves to device|
|2|Dataset|Y|Pure Python; loads data on CPU, user moves to device|
|3|IterableDataset|Y|Pure Python; loads data on CPU, user moves to device|
|4|TensorDataset|Y|Pure Python; loads data on CPU, user moves to device|
|5|StackDataset|Y|Pure Python; loads data on CPU, user moves to device|
|6|ConcatDataset|Y|Pure Python; loads data on CPU, user moves to device|
|7|ChainDataset|Y|Pure Python; loads data on CPU, user moves to device|
|8|Subset|Y|Pure Python; loads data on CPU, user moves to device|
|9|collate|Y|Pure Python; loads data on CPU, user moves to device|
|10|default_collate|Y|Pure Python; loads data on CPU, user moves to device|
|11|default_convert|Y|Pure Python; loads data on CPU, user moves to device|
|12|get_worker_info|Y|Pure Python; loads data on CPU, user moves to device|
|13|random_split|Y|Pure Python; loads data on CPU, user moves to device|
|14|Sampler|Y|Pure Python; loads data on CPU, user moves to device|
|15|SequentialSampler|Y|Pure Python; loads data on CPU, user moves to device|
|16|RandomSampler|Y|Pure Python; loads data on CPU, user moves to device|
|17|SubsetRandomSampler|Y|Pure Python; loads data on CPU, user moves to device|
|18|WeightedRandomSampler|Y|Pure Python; loads data on CPU, user moves to device|
|19|BatchSampler|Y|Pure Python; loads data on CPU, user moves to device|
|20|DistributedSampler|Y|Pure Python; loads data on CPU, user moves to device|

## Type Info

||pytorch_api|neuron_compatible|limitations|
|---|---|---|---|
|1|torch.finfo|Y|Pure Python; returns floating point dtype properties|
|2|torch.iinfo|Y|Pure Python; returns integer dtype properties|

## Named Tensors

||pytorch_api|neuron_compatible|limitations|
|---|---|---|---|
|1|Tensor.names|Y|Pure Python metadata; dimension name annotations|
|2|Tensor.rename|Y|Pure Python metadata; dimension name annotations|
|3|Tensor.rename_|Y|Pure Python metadata; dimension name annotations|
|4|Tensor.refine_names|Y|Pure Python metadata; dimension name annotations|
|5|Tensor.align_as|Y|Pure Python metadata; dimension name annotations|
|6|Tensor.align_to|Y|Pure Python metadata; dimension name annotations|
|7|Tensor.flatten|Y|Pure Python metadata; dimension name annotations|
