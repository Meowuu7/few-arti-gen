#pragma once
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
// C++ interface
#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)
#define CHECK_IS_INT(x)                              \
  do {                                               \
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, \
             #x " must be an int tensor");           \
  } while (0)

#define CHECK_IS_FLOAT(x)                              \
  do {                                                 \
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, \
             #x " must be a float tensor");            \
  } while (0)
