#include "utils.h"
#include <torch/extension.h>
#include <vector>
#include "interpolate_gpu.h"

// CUDA forward declarations


void furthest_sampling_cuda_forward(const int m, const int seedIdx,
  at::Tensor& input, at::Tensor& temp, at::Tensor& idx);

void gather_points_kernel_launcher_fast(int b, int c, int n, int npoints,
    const float *points, const int *idx, float *out, at::cuda::CUDAStream stream);

void gather_points_grad_kernel_launcher_fast(int b, int c, int n, int npoints,
    const float *grad_out, const int *idx, float *grad_points, at::cuda::CUDAStream stream);


int gather_points_wrapper_fast(int b, int c, int n, int npoints,
    at::Tensor& points_tensor, at::Tensor& idx_tensor, at::Tensor& out_tensor){
    const float *points = points_tensor.data_ptr<float>();
    const int *idx = idx_tensor.data_ptr<int>();
    float *out = out_tensor.data_ptr<float>();

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    gather_points_kernel_launcher_fast(b, c, n, npoints, points, idx, out, stream);
    return 1;
}


int gather_points_grad_wrapper_fast(int b, int c, int n, int npoints,
    at::Tensor& grad_out_tensor, at::Tensor& idx_tensor, at::Tensor& grad_points_tensor) {

    const float *grad_out = grad_out_tensor.data_ptr<float>();
    const int *idx = idx_tensor.data_ptr<int>();
    float *grad_points = grad_points_tensor.data_ptr<float>();

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    gather_points_grad_kernel_launcher_fast(b, c, n, npoints, grad_out, idx, grad_points, stream);
    return 1;
}

// at::Tensor gather_points_cuda_forward(int b, int c, int n, int npoints,
//                                       at::Tensor points, at::Tensor idx,
//                                       at::Tensor out);

// at::Tensor gather_points_cuda_backward(int b, int c, int n, int npoints,
//                                        at::Tensor grad_out, at::Tensor idx, at::Tensor grad_points);

// at::Tensor gather_points_forward(int b, int c, int n, int npoints,
//                                  at::Tensor points_tensor,
//                                  at::Tensor idx_tensor,
//                                  at::Tensor out_tensor)
// {
//   CHECK_INPUT(points_tensor);
//   CHECK_INPUT(idx_tensor);
//   return gather_points_cuda_forward(b, c, n, npoints, points_tensor, idx_tensor, out_tensor);
// }

// at::Tensor gather_points_backward(int b, int c, int n, int npoints,
//                                   at::Tensor grad_out_tensor,
//                                   at::Tensor idx_tensor,
//                                   at::Tensor grad_points_tensor)
// {
//   return gather_points_cuda_backward(b, c, n, npoints, grad_out_tensor, idx_tensor, grad_points_tensor);
// }

at::Tensor furthest_sampling_forward(
  const int m,
  const int seedIdx,
  at::Tensor& input,
  at::Tensor& temp,
  at::Tensor& idx
)
{
  CHECK_INPUT(input);
  CHECK_INPUT(temp);
  furthest_sampling_cuda_forward(m, seedIdx, input, temp, idx);
  return idx;
}

void ball_query_kernel_launcher_fast(int b, int n, int m, float radius, int nsample,
	const float *xyz, const float *new_xyz, int *idx, at::cuda::CUDAStream stream);

at::Tensor ball_query_wrapper_fast(at::Tensor& new_xyz_tensor, at::Tensor& xyz_tensor,
      const float radius, const int nsample) {
    CHECK_INPUT(new_xyz_tensor);
    CHECK_INPUT(xyz_tensor);
    CHECK_CUDA(new_xyz_tensor);
    CHECK_CUDA(xyz_tensor);
    const float *new_xyz = new_xyz_tensor.data_ptr<float>();
    const float *xyz = xyz_tensor.data_ptr<float>();
    at::Tensor idx_tensor = torch::zeros({new_xyz_tensor.size(0), new_xyz_tensor.size(1), nsample},
                                  at::device(new_xyz_tensor.device()).dtype(at::ScalarType::Int));
    int *idx = idx_tensor.data_ptr<int>();

    const int b = new_xyz_tensor.size(0);
    const int m = new_xyz_tensor.size(1);
    const int n = xyz_tensor.size(1);

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    ball_query_kernel_launcher_fast(b, n, m, radius, nsample, new_xyz, xyz, idx, stream);
    return idx_tensor;
}
void group_points_kernel_wrapper(int b, int c, int n, int npoints, int nsample,
                                 const float *points, const int *idx,
                                 float *out);

void group_points_grad_kernel_wrapper(int b, int c, int n, int npoints,
                                      int nsample, const float *grad_out,
                                      const int *idx, float *grad_points);

at::Tensor group_points(at::Tensor points, at::Tensor idx) {
  CHECK_CONTIGUOUS(points);
  CHECK_CONTIGUOUS(idx);
  CHECK_IS_FLOAT(points);
  CHECK_IS_INT(idx);

  if (points.is_cuda()) {
    CHECK_CUDA(idx);
  }

  at::Tensor output =
      torch::zeros({points.size(0), points.size(1), idx.size(1), idx.size(2)},
                   at::device(points.device()).dtype(at::ScalarType::Float));

  if (points.is_cuda()) {
    group_points_kernel_wrapper(points.size(0), points.size(1), points.size(2),
                                idx.size(1), idx.size(2), points.data_ptr<float>(),
                                idx.data_ptr<int>(), output.data_ptr<float>());
  } else {
    TORCH_CHECK(false, "CPU not supported");
  }

  return output;
}

at::Tensor group_points_grad(at::Tensor grad_out, at::Tensor idx, const int n) {
  CHECK_CONTIGUOUS(grad_out);
  CHECK_CONTIGUOUS(idx);
  CHECK_IS_FLOAT(grad_out);
  CHECK_IS_INT(idx);

  if (grad_out.is_cuda()) {
    CHECK_CUDA(idx);
  }

  at::Tensor output =
      torch::zeros({grad_out.size(0), grad_out.size(1), n},
                   at::device(grad_out.device()).dtype(at::ScalarType::Float));

  if (grad_out.is_cuda()) {
    group_points_grad_kernel_wrapper(
        grad_out.size(0), grad_out.size(1), n, idx.size(1), idx.size(2),
        grad_out.data_ptr<float>(), idx.data_ptr<int>(), output.data_ptr<float>());
  } else {
    TORCH_CHECK(false, "CPU not supported");
  }

  return output;
}

void three_nn_wrapper_fast(int b, int n, int m, at::Tensor unknown_tensor,
    at::Tensor known_tensor, at::Tensor dist2_tensor, at::Tensor idx_tensor) {
    const float *unknown = unknown_tensor.data_ptr<float>();
    const float *known = known_tensor.data_ptr<float>();
    float *dist2 = dist2_tensor.data_ptr<float>();
    int *idx = idx_tensor.data_ptr<int>();

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    three_nn_kernel_launcher_fast(b, n, m, unknown, known, dist2, idx, stream);
}


void three_interpolate_wrapper_fast(int b, int c, int m, int n,
                         at::Tensor points_tensor,
                         at::Tensor idx_tensor,
                         at::Tensor weight_tensor,
                         at::Tensor out_tensor) {

    const float *points = points_tensor.data_ptr<float>();
    const float *weight = weight_tensor.data_ptr<float>();
    float *out = out_tensor.data_ptr<float>();
    const int *idx = idx_tensor.data_ptr<int>();

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    three_interpolate_kernel_launcher_fast(b, c, m, n, points, idx, weight, out, stream);
}

void three_interpolate_grad_wrapper_fast(int b, int c, int n, int m,
                            at::Tensor grad_out_tensor,
                            at::Tensor idx_tensor,
                            at::Tensor weight_tensor,
                            at::Tensor grad_points_tensor) {

    const float *grad_out = grad_out_tensor.data_ptr<float>();
    const float *weight = weight_tensor.data_ptr<float>();
    float *grad_points = grad_points_tensor.data_ptr<float>();
    const int *idx = idx_tensor.data_ptr<int>();

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    three_interpolate_grad_kernel_launcher_fast(b, c, n, m, grad_out, idx, weight, grad_points, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("furthest_sampling", &furthest_sampling_forward, "furthest point sampling (no gradient)");
  m.def("gather_forward", &gather_points_wrapper_fast, "gather npoints points along an axis");
  m.def("gather_backward", &gather_points_grad_wrapper_fast, "gather npoints points along an axis backward");
  m.def("ball_query", &ball_query_wrapper_fast, "ball query");
  m.def("group_points", &group_points);
  m.def("group_points_grad", &group_points_grad);
  m.def("three_nn_wrapper", &three_nn_wrapper_fast, "three_nn_wrapper_fast");
  m.def("three_interpolate_wrapper", &three_interpolate_wrapper_fast, "three_interpolate_wrapper_fast");
  m.def("three_interpolate_grad_wrapper", &three_interpolate_grad_wrapper_fast, "three_interpolate_grad_wrapper_fast");
}