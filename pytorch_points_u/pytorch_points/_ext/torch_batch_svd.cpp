#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
//#undef NDEBUG

#include <cstdio>
#include <iostream>
#include <memory>
#include <algorithm>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolver_common.h>
#include <cusolverDn.h>


namespace torch_batch_svd
{

template<int success = CUSOLVER_STATUS_SUCCESS, class T, class Status> // , class A = Status(*)(P), class D = Status(*)(T)>
std::unique_ptr<T, Status(*)(T*)> unique_allocate(Status(allocator)(T**),  Status(deleter)(T*))
{
    T* ptr;
    auto stat = allocator(&ptr);
    TORCH_CHECK(stat == success);
    return {ptr, deleter};
}

template <class T>
std::unique_ptr<T, decltype(&cudaFree)> unique_cuda_ptr(size_t len) {
    T* ptr;
    auto stat = cudaMalloc(&ptr, sizeof(T) * len);
    TORCH_CHECK(stat == cudaSuccess);
    return {ptr, cudaFree};
}

// solve U S V = svd(A)  a.k.a. syevj, where A (b, m, n), U (b, m, m), S (b, min(m, n)), V (b, n, n)
// see also https://docs.nvidia.com/cuda/cusolver/index.html#batchgesvdj-example1
std::tuple<at::Tensor, at::Tensor, at::Tensor>
batch_svd_forward(at::Tensor a, bool is_sort, double tol=1e-7, int max_sweeps=100)
{
    TORCH_CHECK(a.is_cuda(), "only cuda tensor is supported");
    TORCH_CHECK(a.scalar_type() == at::kFloat, "only float is supported");

    auto handle_ptr = unique_allocate(cusolverDnCreate, cusolverDnDestroy);
    const auto A = a.contiguous().clone().transpose(1, 2).contiguous().transpose(1, 2);
    // const auto A = a;
    const auto batch_size = A.size(0);
    const auto m = A.size(1);
    TORCH_CHECK(m <= 32, "matrix row should be <= 32");
    const auto n = A.size(2);
    TORCH_CHECK(n <= 32, "matrix col should be <= 32");
    const auto lda = m;
    const auto d_A = A.data_ptr<float>();
    const auto minmn = std::min(m, n);
    auto s = at::empty({batch_size, minmn}, a.scalar_type());
    auto d_s = s.data_ptr<float>();
    auto U = at::empty({batch_size, m, m}, a.scalar_type());
    const auto d_U = U.data_ptr<float>();
    const auto ldu = m;
    auto V = at::empty({batch_size, n, n}, a.scalar_type());
    const auto d_V = V.data_ptr<float>();
    const auto ldv = n;

    auto params = unique_allocate(cusolverDnCreateGesvdjInfo, cusolverDnDestroyGesvdjInfo);
    auto status = cusolverDnXgesvdjSetTolerance(params.get(), tol);
    TORCH_CHECK(CUSOLVER_STATUS_SUCCESS == status);
    status = cusolverDnXgesvdjSetMaxSweeps(params.get(), max_sweeps);
    TORCH_CHECK(CUSOLVER_STATUS_SUCCESS == status);
    status = cusolverDnXgesvdjSetSortEig(params.get(), is_sort);
    TORCH_CHECK(CUSOLVER_STATUS_SUCCESS == status);

    auto jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and eigenvectors
    int lwork;
    auto status_buffer = cusolverDnSgesvdjBatched_bufferSize(
        handle_ptr.get(),
        jobz,
        m,
        n,
        d_A,
        lda,
        d_s,
        d_U,
        ldu,
        d_V,
        ldv,
        &lwork,
        params.get(),
        batch_size);
    TORCH_CHECK(CUSOLVER_STATUS_SUCCESS == status_buffer);
    auto work_ptr = unique_cuda_ptr<float>(lwork);
    auto info_ptr = unique_cuda_ptr<int>(batch_size);
    status = cusolverDnSgesvdjBatched(
        handle_ptr.get(),
        jobz,
        m,
        n,
        d_A,
        lda,
        d_s,
        d_U,
        ldu,
        d_V,
        ldv,
        work_ptr.get(),
        lwork,
        info_ptr.get(),
        params.get(),
        batch_size
        );
    TORCH_CHECK(CUSOLVER_STATUS_SUCCESS == status);

    std::vector<int> hinfo(batch_size);
    auto status_memcpy = cudaMemcpy(hinfo.data(), info_ptr.get(), sizeof(int) * batch_size, cudaMemcpyDeviceToHost);
    TORCH_CHECK(cudaSuccess == status_memcpy);

    for(int i = 0 ; i < batch_size; ++i)
    {
        if ( 0 == hinfo[i] )
        {
            continue;
        }
        else if ( 0 > hinfo[i] )
        {
            printf("Error: %d-th parameter is wrong \n", -hinfo[i]);
            TORCH_CHECK(false);
        }
        else
        {
            printf("WARNING: matrix %d, info = %d : Jacobi method does not converge \n", i, hinfo[i] );
        }
    }

    U = U.contiguous().transpose(1, 2).contiguous();
    s = s.contiguous();
    V = V.contiguous().transpose(1, 2).contiguous();
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      printf("batch_svd_forward kernel failed: %s\n", cudaGetErrorString(err));
      exit(-1);
    }
    return std::make_tuple(U, s, V);
}



// https://j-towns.github.io/papers/svd-derivative.pdf
//
// This makes no assumption on the signs of sigma.
at::Tensor batch_svd_backward(const std::vector<at::Tensor> &grads, const at::Tensor& self,
          bool some, bool compute_uv, const at::Tensor& raw_u, const at::Tensor& sigma, const at::Tensor& raw_v) {
  TORCH_CHECK(compute_uv,
           "svd_backward: Setting compute_uv to false in torch.svd doesn't compute singular matrices, ",
           "and hence we cannot compute backward. Please use torch.svd(compute_uv=True)");

  // A [b, m, n]
  // auto b = self.size(0);
  auto m = self.size(1);
  auto n = self.size(2);
  auto k = sigma.size(1);
  auto gsigma = grads[1];

  auto u = raw_u;
  auto v = raw_v;
  auto gu = grads[0];
  auto gv = grads[2];

  if (!some) {
    // We ignore the free subspace here because possible base vectors cancel
    // each other, e.g., both -v and +v are valid base for a dimension.
    // Don't assume behavior of any particular implementation of svd.
    u = raw_u.narrow(2, 0, k);
    v = raw_v.narrow(2, 0, k);
    if (gu.defined()) {
      gu = gu.narrow(2, 0, k);
    }
    if (gv.defined()) {
      gv = gv.narrow(2, 0, k);
    }
  }
  auto vt = v.transpose(1, 2);

  at::Tensor sigma_term;
  if (gsigma.defined()) {
    sigma_term = u.bmm(gsigma.diag_embed()).bmm(vt);
  } else {
    sigma_term = at::zeros({1}, self.options()).expand_as(self);
  }
  // in case that there are no gu and gv, we can avoid the series of kernel
  // calls below
  if (!gv.defined() && !gu.defined()) {
    return sigma_term;
  }

  auto ut = u.transpose(1, 2);
  auto im = at::eye(m, self.options());  // work if broadcast
  auto in = at::eye(n, self.options());
  auto sigma_mat = sigma.diag_embed();
  auto sigma_mat_inv = sigma.pow(-1).diag_embed();
  auto sigma_expanded_sq = sigma.pow(2).unsqueeze(1).expand_as(sigma_mat);
  auto F = sigma_expanded_sq - sigma_expanded_sq.transpose(1, 2);
  // The following two lines invert values of F, and fills the diagonal with 0s.
  // Notice that F currently has 0s on diagonal. So we fill diagonal with +inf
  // first to prevent nan from appearing in backward of this function.
  F.diagonal(0, -2, -1).fill_(INFINITY);
  F = F.pow(-1);

  at::Tensor u_term, v_term;

  if (gu.defined()) {
    u_term = u.bmm(F.mul(ut.bmm(gu) - gu.transpose(1, 2).bmm(u))).bmm(sigma_mat);
    if (m > k) {
      u_term = u_term + (im - u.bmm(ut)).bmm(gu).bmm(sigma_mat_inv);
    }
    u_term = u_term.bmm(vt);
  } else {
    u_term = at::zeros({1}, self.options()).expand_as(self);
  }

  if (gv.defined()) {
    auto gvt = gv.transpose(1, 2);
    v_term = sigma_mat.bmm(F.mul(vt.bmm(gv) - gvt.bmm(v))).bmm(vt);
    if (n > k) {
      v_term = v_term + sigma_mat_inv.bmm(gvt.bmm(in - v.bmm(vt)));
    }
    v_term = u.bmm(v_term);
  } else {
    v_term = at::zeros({1}, self.options()).expand_as(self);
  }

  return u_term + sigma_term + v_term;
}


} // namespace torch_batch_svd


// generate wrappers
// FIXME do not use legacy preprocessor macro
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batch_svd_forward", &torch_batch_svd::batch_svd_forward,
          "cusolver based batch svd implementation");
    m.def("batch_svd_backward", &torch_batch_svd::batch_svd_backward,
          "batch svd backward");
}
