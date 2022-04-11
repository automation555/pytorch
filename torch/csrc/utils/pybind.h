#pragma once

#include <torch/csrc/python_headers.h>

#include <c10/util/string_view.h>
#include <ATen/ATen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/utils/python_tuples.h>
#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/Generator.h>
#include <torch/csrc/Dtype.h>

#include <stdexcept>
#include <utility>

namespace py = pybind11;

// This makes intrusive_ptr to be available as a custom pybind11 holder type,
// see
// https://pybind11.readthedocs.io/en/stable/advanced/smart_ptrs.html#custom-smart-pointers
PYBIND11_DECLARE_HOLDER_TYPE(T, c10::intrusive_ptr<T>, true);

namespace pybind11 { namespace detail {

// torch.autograd.Variable <-> at::Tensor conversions (without unwrapping)
template <>
struct type_caster<at::Tensor> {
 public:
  PYBIND11_TYPE_CASTER(at::Tensor, _("at::Tensor"));

  bool load(handle src, bool) {
    PyObject* obj = src.ptr();
    if (THPVariable_Check(obj)) {
      value = reinterpret_cast<THPVariable*>(obj)->cdata;
      return true;
    }
    return false;
  }

  static handle
  cast(const at::Tensor& src, return_value_policy /* policy */, handle /* parent */) {
    return handle(THPVariable_Wrap(torch::autograd::Variable(src)));
  }
};

template <>
struct type_caster<at::Generator> {
 public:
  PYBIND11_TYPE_CASTER(at::Generator, _("at::Generator"));

  bool load(handle src, bool) {
    PyObject* obj = src.ptr();
    if (THPGenerator_Check(obj)) {
      value = reinterpret_cast<THPGenerator*>(obj)->cdata;
      return true;
    }
    return false;
  }

  static handle
  cast(const at::Generator& src, return_value_policy /* policy */, handle /* parent */) {
    return handle(THPGenerator_Wrap(src));
  }
};

template<> struct type_caster<at::IntArrayRef> {
public:
  PYBIND11_TYPE_CASTER(at::IntArrayRef, _("at::IntArrayRef"));

  bool load(handle src, bool) {
    PyObject *source = src.ptr();
    auto tuple = PyTuple_Check(source);
    if (tuple || PyList_Check(source)) {
      auto size = tuple ? PyTuple_GET_SIZE(source) : PyList_GET_SIZE(source);
      v_value.resize(size);
      for (int idx = 0; idx < size; idx++) {
        PyObject* obj = tuple ? PyTuple_GET_ITEM(source, idx) : PyList_GET_ITEM(source, idx);
        if (THPVariable_Check(obj)) {
          v_value[idx] = THPVariable_Unpack(obj).item<int64_t>();
        } else if (PyLong_Check(obj)) {
          // use THPUtils_unpackLong after it is safe to include python_numbers.h
          v_value[idx] = THPUtils_unpackLong(obj);
        } else {
          return false;
        }
      }
      value = v_value;
      return true;
    }
    return false;
  }
  static handle cast(at::IntArrayRef src, return_value_policy /* policy */, handle /* parent */) {
    return handle(THPUtils_packInt64Array(src.size(), src.data()));
  }
private:
  std::vector<int64_t> v_value;
};

// torch.dtype <-> at::ScalarType conversions
template<> struct type_caster<at::ScalarType> {
public:
  PYBIND11_TYPE_CASTER(at::ScalarType, _("at::ScalarType"));

  bool load(handle src, bool) {
    value = src.cast<const torch::PyDtype&>().scalar_type;
    return true;
  }

  static handle cast(at::ScalarType scalar_type, return_value_policy /* policy */, handle /* parent */) {
    return py::cast(torch::getPyDtype(scalar_type), return_value_policy::reference).release();
  }
};

// Pybind11 bindings for our optional type.
// http://pybind11.readthedocs.io/en/stable/advanced/cast/stl.html#c-17-library-containers
template <typename T>
struct type_caster<c10::optional<T>> : optional_caster<c10::optional<T>> {};

template <>
struct type_caster<c10::nullopt_t> : void_caster<c10::nullopt_t> {};

// Pybind11 bindings for our string_view type.
// https://pybind11.readthedocs.io/en/stable/advanced/cast/strings.html
template <typename CharT>
struct type_caster<c10::basic_string_view<CharT>, enable_if_t<is_std_char_type<CharT>::value>>
    : string_caster<c10::basic_string_view<CharT>, true> {};

}} // namespace pybind11::detail
