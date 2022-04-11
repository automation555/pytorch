#include <ATen/ATen.h>
#include <ATen/CPUApplyUtils.h>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <c10/util/Exception.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorCompare.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/TensorIndexing.h>

namespace at { namespace native {

DEFINE_DISPATCH(where_kernel); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(max_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(min_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(_aminmax_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(isposinf_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(isneginf_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(mode_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(isin_default_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

bool allclose(const Tensor& self, const Tensor& other, double rtol, double atol, bool equal_nan) {
  return at::isclose(self, other, rtol, atol, equal_nan).all().item<uint8_t>();
}

// Note [closeness]
// A number A is close to B when either:
//
// (1) A is equal to B, with NaNs comparing equal when equal_nan is true.
// (2) The error abs(A - B) is finite and less than the max error
//      (atol + abs(rtol * B)).
//
// Note that this is consistent with NumPy's isclose but divergent from
// Python's isclose, which computes the max error symmetrically as
// max(rtol * max(abs(A), abs(B)), atol).
// TODO: use bitwise operator overloads once we add them
// TODO: revisit complex inputs and equal_nan=true after
//  https://github.com/numpy/numpy/issues/15959 is resolved
Tensor isclose(const Tensor& self, const Tensor& other, double rtol, double atol, bool equal_nan) {
  TORCH_CHECK(self.scalar_type() == other.scalar_type(), self.scalar_type(), " did not match ", other.scalar_type());
  TORCH_CHECK(!(self.is_complex() && equal_nan),
    "isclose with equal_nan=True is not supported for complex inputs.");
  TORCH_CHECK(!(self.is_quantized() || other.is_quantized()),
    "isclose is not supported for quantized inputs.");

  // Checks that rtol and atol are non-negative
  // Note: consistent with Python's isclose but divergent from NumPy's, which
  //  allows negative atol and rtol.
  TORCH_CHECK(rtol >= 0, "rtol must be greater than or equal to zero, but got ", rtol);
  TORCH_CHECK(atol >= 0, "atol must be greater than or equal to zero, but got ", atol);

  // Computes equality closeness
  Tensor close = self == other;
  if (equal_nan && self.is_floating_point()) {
      close.__ior__((self != self).__iand__(other != other));
  }

  // Note [closeness error computation]
  // atol and rtol are provided as doubles, so the computation
  // rtol * other will produce a float or complex tensor.
  // When the difference (self - other) is compared to it then the
  // tensor representing the difference will also be cast to float or complex.
  // However, since (self - other) in uint8 is very likely to produce a
  // negative value, this moves the cast forward so the difference is
  // always computed in a float or complex type.
  // If the values of the integer tensors cannot be exactly represented
  // by the default scalar type then this may cause an incorrect result.

  // Computes allowed and actual error
  Tensor cast_other;
  if (c10::isIntegralType(self.scalar_type(), /*includeBool=*/true)) {
    cast_other = other.to(at::get_default_dtype());
  } else {
    cast_other = other;
  }
  Tensor allowed_error = atol + (rtol * cast_other).abs();
  Tensor actual_error = (self - cast_other).abs();

  // Computes finite closeness
  close.__ior__(at::isfinite(actual_error).__iand__(actual_error <= allowed_error));

  return close;
}

Tensor isnan(const Tensor& self) {
  return self != self;
}

Tensor isreal(const Tensor& self) {
  // Note: Integral and Floating tensor values are always real
  if (c10::isIntegralType(self.scalar_type(), /*includeBool=*/true) ||
      c10::isFloatingType(self.scalar_type())) {
    return at::ones_like(self, at::kBool, at::MemoryFormat::Preserve);
  }

  return at::imag(self) == 0;
}

Tensor isinf(const Tensor &self) {
  // Note: Integral tensor values are never infinite
  if (c10::isIntegralType(self.scalar_type(), /*includeBool=*/true)) {
    return at::zeros_like(self, at::kBool, at::MemoryFormat::Preserve);
  }

  // Note: a complex value is infinite when either part is infinite
  if (self.is_complex()) {
    return at::isinf(at::real(self)).__ior__
          (at::isinf(at::imag(self)));
  }

  return AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, self.scalar_type(), "isinf", [&]() {
    return self.abs() == std::numeric_limits<scalar_t>::infinity();
  });
}

Tensor isposinf(const Tensor &self) {
  Tensor result = at::empty_like(self, at::kBool, at::MemoryFormat::Preserve);
  at::isposinf_out(result, self);
  return result;
}

Tensor& isposinf_out(const Tensor& self, Tensor& result) {
  TORCH_CHECK(!self.is_complex(), "isposinf does not support complex inputs.");
  TORCH_CHECK(result.scalar_type() == at::kBool, "isposinf does not support non-boolean outputs.");
  result.resize_(self.sizes());

  if (c10::isIntegralType(self.scalar_type(), /*includeBool=*/true)) {
    result.fill_(false);
  } else {
    auto iter = TensorIteratorConfig()
      .check_all_same_dtype(false)
      .add_output(result)
      .add_input(self)
      .build();
    isposinf_stub(iter.device_type(), iter);
  }
  return result;
}

Tensor isneginf(const Tensor &self) {
  Tensor result = at::empty_like(self, at::kBool, at::MemoryFormat::Preserve);
  at::isneginf_out(result, self);
  return result;
}

Tensor& isneginf_out(const Tensor& self, Tensor& result) {
  TORCH_CHECK(!self.is_complex(), "isneginf does not support complex inputs.");
  TORCH_CHECK(result.scalar_type() == at::kBool, "isneginf does not support non-boolean outputs.");
  result.resize_(self.sizes());

  if (c10::isIntegralType(self.scalar_type(), /*includeBool=*/true)) {
    result.fill_(false);
  } else {
    auto iter = TensorIteratorConfig()
      .check_all_same_dtype(false)
      .add_output(result)
      .add_input(self)
      .build();
    isneginf_stub(iter.device_type(), iter);
  }
  return result;
}

Tensor isfinite(const Tensor& self) {
  // Note: Integral tensor values are always finite
  if (c10::isIntegralType(self.scalar_type(), /*includeBool=*/true)) {
    return at::ones_like(self, at::kBool, at::MemoryFormat::Preserve);
  }

  // Note: a complex value is finite iff both parts are finite
  if (self.is_complex()) {
    return at::isfinite(self.abs());
  }

  return AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, self.scalar_type(), "isfinite", [&]() {
    return (self == self) * (self.abs() != std::numeric_limits<scalar_t>::infinity());
  });
}

bool is_nonzero(const Tensor& self) {
  auto n = self.numel();
  TORCH_CHECK(n != 0, "Boolean value of Tensor with no values is ambiguous");
  TORCH_CHECK(n < 2, "Boolean value of Tensor with more than one value is ambiguous");

  Scalar localScalar = self.item();
  if (localScalar.isFloatingPoint()) {
    return localScalar.to<double>() != 0;
  } else if (localScalar.isComplex()) {
     return localScalar.to<c10::complex<double>>() != c10::complex<double>(0.0, 0.0);
  } else if (localScalar.isIntegral(false)){
    return localScalar.to<int64_t>() != 0;
  } else if (localScalar.isBoolean()) {
    return localScalar.to<bool>();
  }
  TORCH_INTERNAL_ASSERT(false, "Expected non-Tensor backend scalar");
}

void _assert_async_cpu(const Tensor& self) {
  TORCH_CHECK(native::is_nonzero(self), "Expected Tensor with single nonzero value, but got zero");
}

namespace {

// DO NOT USE THIS -- it's just an implementation detail of wrapped_scalar tensor below.
at::Tensor scalar_to_tensor_default_dtype(
    const Scalar& s,
    const Device device = at::kCPU) {
  if (s.isFloatingPoint()) {
    return at::scalar_tensor(
        s, at::device(device).dtype(at::get_default_dtype()));
  } else if (s.isBoolean()) {
    return at::scalar_tensor(s, at::device(device).dtype(at::kBool));
  } else if (s.isComplex()) {
    return at::scalar_tensor(
        s, at::device(device).dtype(at::get_default_complex_dtype()));
  } else {
    TORCH_INTERNAL_ASSERT(s.isIntegral(false));
    return at::scalar_tensor(s, at::device(device).dtype(at::kLong));
  }
}

// TLDR: Don't call with `use_default_dtype` true -- this is only necessary to support the partial
// type-promotion that torch.where supports.  Once torch.where fully supports type promotion, we
// won't need this function.
//
// Longer explanation:
// `use_default_dtype` is a bit of a hack because torch.where doesn't support type promotion, but
// does support `torch.where(tensor, scalar1, scalar2)` with default scalar types.  The trickiness is we
// usually convert double scalars to doubles, and `set_wrapped_number` defines type promotion priority
// as being below tensor types rather than as the default dtype (perhaps we should?).  This wouldn't matter
// if we just supported type normal type promotion on torch.where, however.
Tensor wrapped_scalar_tensor(
    const Scalar& scalar,
    Device device,
    bool use_default_dtype = false) {
  at::Tensor tensor;
  if (use_default_dtype) {
    tensor = scalar_to_tensor_default_dtype(scalar, device);
  } else {
    tensor = scalar_to_tensor(scalar, device);
  }
  tensor.unsafeGetTensorImpl()->set_wrapped_number(true);
  return tensor;
}

} // anonymous namespace

// Sorting-based algorithm for isin(); used when the number of test elements is large.
// 1. Concatenate unique elements with unique test elements in 1D form. If
//    assume_unique is true, skip calls to unique().
// 2. Stable sort all elements, maintaining order indices to reverse the
//    operation. Stable sort is necessary to keep elements before test
//    elements within the sorted list.
// 3. Create a mask for locations of adjacent duplicate values within the
//    sorted list. Duplicate values are in both elements and test elements.
// 4. Reorder the mask to match the pre-sorted element order.
// 5. Index the mask to match the pre-unique element order. If
//    assume_unique is true, just take the first N items of the mask,
//    where N is the original number of elements.
static void isin_sorting(const Tensor& elements, const Tensor& test_elements,
    bool assume_unique, bool invert, Tensor& out) {
  Tensor elements_flat, test_elements_flat, unique_order;
  if (assume_unique) {
    elements_flat = elements.ravel();
    test_elements_flat = test_elements.ravel();
  } else {
    std::tie (elements_flat, unique_order) = at::_unique(
        elements, /*sorted=*/ false, /*return_inverse=*/ true);
    std::tie (test_elements_flat, std::ignore) = at::_unique(test_elements, /*sorted=*/ false);
  }

  Tensor all_elements = at::_cat({elements_flat, test_elements_flat});
  Tensor sorted_elements, sorted_order;
  std::tie (sorted_elements, sorted_order) = all_elements.sort(
      /*stable=*/ true, /*dim=*/ 0, /*descending=*/ false);

  Tensor duplicate_mask = at::empty(all_elements.sizes(),
    TensorOptions(elements.device()).dtype(ScalarType::Bool));
  auto iter = TensorIteratorConfig()
    .add_output(duplicate_mask)
    .add_input(sorted_elements)
    .check_all_same_dtype(false)
    .build();
  AT_DISPATCH_ALL_TYPES(iter.dtype(1), "isin_sorting", [&]() {
    iter.for_each([&](char** data, const int64_t* strides, int64_t n) {
      auto* mask_bytes = data[0];
      const auto* element_bytes = data[1];

      for (int i = 0; i < n - 1; ++i) {
        auto* mask_data = reinterpret_cast<bool*>(mask_bytes);
        const scalar_t element_val = *(reinterpret_cast<const scalar_t*>(element_bytes));
        const scalar_t next_element_val = *(reinterpret_cast<const scalar_t*>(element_bytes + strides[1]));
        *mask_data = invert ? (element_val != next_element_val) : (element_val == next_element_val);

        mask_bytes += strides[0];
        element_bytes += strides[1];
      }

      // Last duplicate mask item is unconditionally true or false based on invert flag.
      auto* mask_data = reinterpret_cast<bool*>(mask_bytes);
      *mask_data = invert;
    });
  });

  Tensor mask = at::empty(all_elements.sizes(), TensorOptions(elements.device()).dtype(ScalarType::Bool));
  mask.index_copy_(0, sorted_order, duplicate_mask);

  if (assume_unique) {
    out.copy_(mask.slice(0, 0, elements.numel()).view_as(out));
  } else {
    out.copy_(at::index(mask, {c10::optional<Tensor>(unique_order)}));
  }
}

Tensor& isin_out(const Tensor& elements, const Tensor& test_elements, bool assume_unique, bool invert, Tensor& out) {
  TORCH_CHECK(out.device() == elements.device(),
      "Expected output tensor and elements to be on the same device, but found different devices: ",
      out.device(), " and ", elements.device());
  resize_output(out, elements.sizes());
  if (elements.numel() == 0) {
    return out;
  }

  // Heuristic taken from numpy's implementation.
  // See https://github.com/numpy/numpy/blob/fb215c76967739268de71aa4bda55dd1b062bc2e/numpy/lib/arraysetops.py#L575
  // TODO: Remove the is_cuda() check when stable sort is supported for CUDA.
  if (elements.device().is_cuda() || test_elements.numel() < static_cast<int64_t>(
        10.0f * std::pow(static_cast<double>(elements.numel()), 0.145))) {
    out.fill_(invert);
    isin_default_stub(elements.device().type(), out, elements, test_elements, invert);
  } else {
    isin_sorting(elements, test_elements, assume_unique, invert, out);
  }

  return out;
}

Tensor& isin_out(const Tensor& elements, const c10::Scalar& test_element, bool assume_unique, bool invert, Tensor& out) {
  Tensor test_elements = wrapped_scalar_tensor(test_element, elements.device());
  at::isin_out(out, elements, test_elements, assume_unique, invert);
  return out;
}

Tensor& isin_out(const c10::Scalar& element, const Tensor& test_elements, bool assume_unique, bool invert, Tensor& out) {
  Tensor elements = wrapped_scalar_tensor(element, test_elements.device());
  at::isin_out(out, elements, test_elements, assume_unique, invert);
  return out;
}

Tensor isin(const Tensor& elements, const Tensor& test_elements, bool assume_unique, bool invert) {
  Tensor out = at::empty({0}, TensorOptions(elements.device()).dtype(ScalarType::Bool));
  at::isin_out(out, elements, test_elements, assume_unique, invert);
  return out;
}

Tensor isin(const Tensor& elements, const c10::Scalar& test_element, bool assume_unique, bool invert) {
  Tensor out = at::empty({0}, TensorOptions(elements.device()).dtype(ScalarType::Bool));
  at::isin_out(out, elements, test_element, assume_unique, invert);
  return out;
}

Tensor isin(const c10::Scalar& element, const Tensor& test_elements, bool assume_unique, bool invert) {
  Tensor out = at::empty({0}, TensorOptions(test_elements.device()).dtype(ScalarType::Bool));
  at::isin_out(out, element, test_elements, assume_unique, invert);
  return out;
}

Tensor where(const Tensor& condition, const Tensor& self, const Tensor& other) {
  TORCH_CHECK(condition.device() == self.device() && self.device() == other.device(),
              "Expected condition, x and y to be on the same device, but condition is on ",
              condition.device(), " and x and y are on ", self.device(), " and ", other.device(),
              " respectively");
  TORCH_CHECK(condition.scalar_type() == ScalarType::Byte || condition.scalar_type() == ScalarType::Bool,
              "Expected condition to have ScalarType Byte, but got ScalarType ",
              toString(condition.scalar_type()));
  Tensor b_condition, b_self, b_other;
  std::tie(b_condition, b_self, b_other) = expand_outplace(condition, self, other, "where");
  return at::_s_where(b_condition, b_self, b_other);
}

Tensor where(const Tensor& condition, const Scalar& self, const Tensor& other) {
  return at::where(condition, wrapped_scalar_tensor(self, other.device()), other);
}

Tensor where(const Tensor& condition, const Tensor& self, const Scalar& other) {
  return at::where(condition, self, wrapped_scalar_tensor(other, self.device()));
}

Tensor where(const Tensor& condition, const Scalar& self, const Scalar& other) {
  const auto device = condition.device();
  const Tensor& other_t = wrapped_scalar_tensor(other, device, /*use_default_dtype=*/true);
  const Tensor& self_t = wrapped_scalar_tensor(self, device, /*use_default_dtype=*/true);
  return at::where(condition, self_t, other_t);
}

std::vector<Tensor> where(const Tensor& condition) {
  return condition.nonzero_numpy();
}

Tensor _s_where(const Tensor& condition, const Tensor& self, const Tensor& other) {
  TORCH_CHECK(self.dtype() == other.dtype(), "expected scalar type ", self.dtype(), " but found ", other.dtype());
  Tensor ret = at::empty(self.sizes(), self.options());
  auto iter = at::TensorIteratorConfig()
    .check_all_same_dtype(false)
    .add_output(ret)
    .add_input(condition)
    .add_input(self)
    .add_input(other)
    .build();
  where_kernel(iter.device_type(), iter, condition.scalar_type());
  return ret;
}

std::tuple<Tensor, Tensor> mode(const Tensor& self, int64_t dim, bool keepdim) {
  Tensor values = at::empty({0}, self.options());
  Tensor indices = at::empty({0}, self.options().dtype(kLong));
  return at::native::mode_out(self, dim, keepdim, values, indices);
}

std::tuple<Tensor &,Tensor &> mode_out(const Tensor& self, int64_t dim, bool keepdim,
                                       Tensor& values, Tensor& indices) {
  TORCH_CHECK(self.device().is_cpu() || self.is_cuda(),
              "mode only supports CPU AND CUDA device type, got: ", self.device().type());
  TORCH_CHECK(self.layout() == Layout::Strided,
              "mode only supports strided layout, got: ", self.layout());
  TORCH_CHECK(self.device() == values.device(),
              "expected device '", self.device(), "' but got '",
              values.device(), "' for values output");
  TORCH_CHECK(self.device() == indices.device(),
              "expected device '", self.device(), "' but got '",
              indices.device(), "' for indices output");
  TORCH_CHECK(self.scalar_type() == values.scalar_type(),
              "expected scalar type '", self.scalar_type(), "' but got '",
              values.scalar_type(), "' for values output");
  TORCH_CHECK(indices.scalar_type() == ScalarType::Long,
              "expected scalar type '", ScalarType::Long, "' but got '",
              indices.scalar_type(), "' for indices output");
  dim = maybe_wrap_dim(dim, self.dim());
  if (_dimreduce_return_trivial_no_ident(values, self, dim, keepdim, "mode")) {
    AT_ASSERT(values.dim() == 0);
    indices.resize_({}).fill_(0);
    return std::forward_as_tuple(values, indices);
  } else {
    auto result = [&]() {
      NoNamesGuard guard;
      mode_stub(self.device().type(), values, indices, self, dim, keepdim);
      return std::tuple<Tensor &,Tensor &>{values, indices};
    }();
    namedinference::propagate_names_for_reduction(std::get<0>(result), self, dim, keepdim);
    namedinference::propagate_names_for_reduction(std::get<1>(result), self, dim, keepdim);
    return result;
  }
}

std::tuple<Tensor, Tensor> max(const Tensor& self, int64_t dim, bool keepdim) {
  Tensor max_indices = at::empty({0}, self.options().dtype(kLong));
  if (self.is_quantized()) {
    Tensor max = at::empty({0}, self.options().dtype(toUnderlying(self.scalar_type())));
    at::native::max_out(self.int_repr() , dim, keepdim, max, max_indices);
    // TODO: qscheme
    return std::tuple<Tensor, Tensor>(at::_make_per_tensor_quantized_tensor(max, self.q_scale(), self.q_zero_point()), max_indices);
  } else {
    Tensor max = at::empty({0}, self.options());
    return at::native::max_out(self, dim, keepdim, max, max_indices);
  }
}

static std::tuple<Tensor &,Tensor &> max_out_impl(Tensor& max, Tensor& max_indices,
                                                  const Tensor& self, int64_t dim, bool keepdim) {
  TORCH_CHECK(self.device().is_cpu() || self.is_cuda(),
              "max only supports CPU AND CUDA device type, got: ", self.device().type());
  TORCH_CHECK(self.layout() == Layout::Strided,
              "max only supports strided layout, got: ", self.layout());
  TORCH_CHECK(self.device() == max.device(),
              "expected device ", self.device(), " but got ",
              max.device(), " for max values output");
  TORCH_CHECK(self.device() == max_indices.device(),
              "expected device ", self.device(), " but got ",
              max_indices.device(), " for indices output");
  dim = maybe_wrap_dim(dim, self.dim());
  if (_dimreduce_return_trivial_no_ident(max, self, dim, keepdim, "max")) {
    TORCH_CHECK(!self.is_complex(), "max does not support complex inputs.");
    AT_ASSERT(max.dim() == 0);
    max_indices.resize_({}).fill_(0);
    return std::forward_as_tuple(max, max_indices);
  } else {
    max_stub(self.device().type(), max, max_indices, self, dim, keepdim);
    return std::tuple<Tensor &,Tensor &>{max, max_indices};
  }
}

std::tuple<Tensor&,Tensor&> max_out(const Tensor& self, int64_t dim, bool keepdim, Tensor& max, Tensor& max_indices) {
  auto result = [&]() {
    NoNamesGuard guard;
    return max_out_impl(max, max_indices, self, dim, keepdim);
  }();
  namedinference::propagate_names_for_reduction(max, self, dim, keepdim);
  namedinference::propagate_names_for_reduction(max_indices, self, dim, keepdim);
  return result;
}

std::tuple<Tensor, Tensor> min(const Tensor& self, int64_t dim, bool keepdim) {
  Tensor min_indices = at::empty({0}, self.options().dtype(kLong));
  if (self.is_quantized()) {
    Tensor min = at::empty({0}, self.options().dtype(toUnderlying(self.scalar_type())));
    at::native::min_out(self.int_repr(), dim, keepdim, min, min_indices);
    return std::tuple<Tensor, Tensor>(at::_make_per_tensor_quantized_tensor(min, self.q_scale(), self.q_zero_point()), min_indices);
  } else {
    Tensor min = at::empty({0}, self.options());
    return at::native::min_out(self, dim, keepdim, min, min_indices);
  }
}

static std::tuple<Tensor &, Tensor &> _aminmax_out_impl(Tensor& min, Tensor& max,
                                                  const Tensor& self, int64_t dim, bool keepdim) {
  TORCH_CHECK(self.device().is_cpu() || self.is_cuda(),
              "min_max_val only supports CPU AND CUDA device type, got: ", self.device().type());
  TORCH_CHECK(self.layout() == Layout::Strided,
              "min_max only supports strided layout, got: ", self.layout());
  TORCH_CHECK(self.device() == min.device(),
              "expected device ", self.device(), " but got ",
              min.device(), " for min values output");
  TORCH_CHECK(self.device() == max.device(),
              "expected device ", self.device(), " but got ",
              max.device(), " for max values output");
  dim = maybe_wrap_dim(dim, self.dim());
  if (_dimreduce_return_trivial_no_ident(min, self, dim, keepdim, "min") &&
      _dimreduce_return_trivial_no_ident(max, self, dim, keepdim, "max")) {
    TORCH_CHECK(!self.is_complex(), "min_max does not support complex inputs.");
    return std::forward_as_tuple(min, max);
  } else {
    _aminmax_stub(self.device().type(), min, max, self, dim, keepdim);
    return std::tuple<Tensor &, Tensor &>{min, max};
  }
}

std::tuple<Tensor, Tensor> _aminmax(const Tensor& self, int64_t dim, bool keepdim) {
  TORCH_CHECK(!self.is_quantized(), "min is not yet implemented for quantized tensors.");

  Tensor min = at::empty({0}, self.options());
  Tensor max = at::empty({0}, self.options());

  auto result = _aminmax_out_impl(min, max, self, dim, keepdim);
  return result;
}

static std::tuple<Tensor &,Tensor &> min_out_impl(Tensor& min, Tensor& min_indices,
                                                  const Tensor& self, int64_t dim, bool keepdim) {
  TORCH_CHECK(self.device().is_cpu() || self.is_cuda(),
              "min only supports CPU AND CUDA device type, got: ", self.device().type());
  TORCH_CHECK(self.layout() == Layout::Strided,
              "min only supports strided layout, got: ", self.layout());
  TORCH_CHECK(self.device() == min.device(),
              "expected device ", self.device(), " but got ",
              min.device(), " for min values output");
  TORCH_CHECK(self.device() == min_indices.device(),
              "expected device ", self.device(), " but got ",
              min_indices.device(), " for indices output");
  dim = maybe_wrap_dim(dim, self.dim());
  if (_dimreduce_return_trivial_no_ident(min, self, dim, keepdim, "min")) {
    TORCH_CHECK(!self.is_complex(), "min does not support complex inputs.");
    AT_ASSERT(min.dim() == 0);
    min_indices.resize_({}).fill_(0);
    return std::forward_as_tuple(min, min_indices);
  } else {
    min_stub(self.device().type(), min, min_indices, self, dim, keepdim);
    return std::tuple<Tensor &,Tensor &>{min, min_indices};
  }
}

std::tuple<Tensor&, Tensor&> min_out(
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    Tensor& min,
    Tensor& min_indices) {
  auto result = [&]() {
    NoNamesGuard guard;
    return min_out_impl(min, min_indices, self, dim, keepdim);
  }();
  namedinference::propagate_names_for_reduction(min, self, dim, keepdim);
  namedinference::propagate_names_for_reduction(min_indices, self, dim, keepdim);
  return result;
}

// Named tensor overloads

std::tuple<Tensor, Tensor> min(const Tensor& self, Dimname dim, bool keepdim) {
  return at::min(self, dimname_to_position(self, dim), keepdim);
}
std::tuple<Tensor &,Tensor &> min_out(const Tensor& self, Dimname dim, bool keepdim, Tensor& min, Tensor& min_indices) {
  return at::min_out(min, min_indices, self, dimname_to_position(self, dim), keepdim);
}
std::tuple<Tensor, Tensor> max(const Tensor& self, Dimname dim, bool keepdim) {
  return at::max(self, dimname_to_position(self, dim), keepdim);
}
std::tuple<Tensor&, Tensor&> max_out(const Tensor& self, Dimname dim, bool keepdim, Tensor& max, Tensor& max_indices) {
  return at::max_out(max, max_indices, self, dimname_to_position(self, dim), keepdim);
}
Tensor argmax(const Tensor& self, Dimname dim, bool keepdim) {
  reportNYIDimnameOverload("argmax");
}
Tensor argmin(const Tensor& self, Dimname dim, bool keepdim) {
  reportNYIDimnameOverload("argmin");
}
Tensor argsort(const Tensor& self, Dimname dim, bool keepdim) {
  reportNYIDimnameOverload("argsort");
}
std::tuple<Tensor, Tensor> mode(const Tensor& self, Dimname dim, bool keepdim) {
  return at::mode(self, dimname_to_position(self, dim), keepdim);
}
std::tuple<Tensor &,Tensor &> mode_out(const Tensor& self, Dimname dim, bool keepdim, Tensor& values, Tensor& indices) {
  return at::mode_out(values, indices, self, dimname_to_position(self, dim), keepdim);
}

}} // namespace at::native
