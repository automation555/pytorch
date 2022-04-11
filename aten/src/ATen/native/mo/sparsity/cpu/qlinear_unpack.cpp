#include <ATen/ATen.h>
#include <torch/custom_class.h>

#include <ATen/cpp_custom_type_hack.h>
#include <ATen/native/mo/sparsity/cpu/fbgemm_utils.h>
#include <ATen/native/mo/sparsity/cpu/packed_params.h>
#include <ATen/native/mo/sparsity/cpu/qnnpack_utils.h>

torch::class_<SparseLinearPackedParamsBase> register_sparse_linear_params();

#ifdef USE_FBGEMM

SerializationTypeSparseLinearPacked SparsePackedLinearWeight::unpack() {
  auto packW = w.get();

  int64_t N = static_cast<int64_t>(packW->R);
  int64_t K = static_cast<int64_t>(packW->C);

  at::Tensor weight_origin;
  if (q_scheme == c10::kPerTensorAffine) {
    weight_origin = at::_empty_affine_quantized(
        {N, K}, at::device(c10::kCPU).dtype(c10::kQInt8), w_scale[0], w_zp[0]);
  } else if (q_scheme == c10::kPerChannelAffine) {
    auto scales = at::from_blob(
        w_scale.data(), w_scale.size(), device(c10::kCPU).dtype(c10::kFloat));
    auto zero_points = at::from_blob(
        w_zp.data(), w_zp.size(), device(c10::kCPU).dtype(c10::kInt));

    weight_origin = at::_empty_per_channel_affine_quantized(
        {N, K},
        scales.toType(c10::kDouble),
        zero_points.toType(c10::kLong),
        0, // The output channel axis is 0
        device(c10::kCPU).dtype(c10::kQInt8));
  }

  // TODO: uncomment once unpack is implemented for BCSRMatrix
  // int8_t* weight_ptr_int8 =
  //     reinterpret_cast<int8_t*>(weight_origin.data_ptr<c10::qint8>());
  // packW->unpack(weight_ptr_int8);
  std::vector<int64_t> block_pattern({out_features_block_size_, in_features_block_size_});

  return std::make_tuple(weight_origin, bias_, std::move(block_pattern));
}

#endif // USE_FBGEMM

#ifdef USE_PYTORCH_QNNPACK

SerializationTypeSparseLinearPacked SparsePackedLinearWeightQnnp::unpack() {
  std::vector<int64_t> block_pattern({out_features_block_size_, in_features_block_size_});
  return std::make_tuple(orig_weight_, orig_bias_, std::move(block_pattern));
}

#endif // USE_FBGEMM

namespace {

class SparseQLinearUnpackWeightInt8 final {
 public:
  static SerializationTypeSparseLinearPacked run(
      const c10::intrusive_ptr<SparseLinearPackedParamsBase>& packed_weight) {
    return packed_weight->unpack();
  }
};

TORCH_LIBRARY_IMPL(sparsity, QuantizedCPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("sparsity::sparse_qlinear_unpack"),
      TORCH_FN(SparseQLinearUnpackWeightInt8::run));
}
} // namespace
