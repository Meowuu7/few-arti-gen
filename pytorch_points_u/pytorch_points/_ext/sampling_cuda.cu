#include <THC/THCAtomics.cuh>
#include <stdlib.h>
#include <vector>

#include "cuda_utils.h"



__global__ void gather_points_kernel_fast(int b, int c, int n, int m,
    const float *__restrict__ points, const int *__restrict__ idx, float *__restrict__ out) {
    // points: (B, C, N)
    // idx: (B, M)
    // output:
    //      out: (B, C, M)

    int bs_idx = blockIdx.z;
    int c_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs_idx >= b || c_idx >= c || pt_idx >= m) return;

    out += bs_idx * c * m + c_idx * m + pt_idx;
    idx += bs_idx * m + pt_idx;
    points += bs_idx * c * n + c_idx * n;
    out[0] = points[idx[0]];
}

void gather_points_kernel_launcher_fast(int b, int c, int n, int npoints,
    const float *points, const int *idx, float *out, at::cuda::CUDAStream stream) {
    // points: (B, C, N)
    // idx: (B, npoints)
    // output:
    //      out: (B, C, npoints)

    cudaError_t err;
    dim3 blocks(DIVUP(npoints, THREADS_PER_BLOCK), c, b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    gather_points_kernel_fast<<<blocks, threads, 0, stream>>>(b, c, n, npoints, points, idx, out);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

__global__ void gather_points_grad_kernel_fast(int b, int c, int n, int m, const float *__restrict__ grad_out,
    const int *__restrict__ idx, float *__restrict__ grad_points) {
    // grad_out: (B, C, M)
    // idx: (B, M)
    // output:
    //      grad_points: (B, C, N)

    int bs_idx = blockIdx.z;
    int c_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs_idx >= b || c_idx >= c || pt_idx >= m) return;

    grad_out += bs_idx * c * m + c_idx * m + pt_idx;
    idx += bs_idx * m + pt_idx;
    grad_points += bs_idx * c * n + c_idx * n;

    atomicAdd(grad_points + idx[0], grad_out[0]);
}

void gather_points_grad_kernel_launcher_fast(int b, int c, int n, int npoints,
    const float *grad_out, const int *idx, float *grad_points, at::cuda::CUDAStream stream) {
    // grad_out: (B, C, npoints)
    // idx: (B, npoints)
    // output:
    //      grad_points: (B, C, N)

    cudaError_t err;
    dim3 blocks(DIVUP(npoints, THREADS_PER_BLOCK), c, b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    gather_points_grad_kernel_fast<<<blocks, threads, 0, stream>>>(b, c, n, npoints, grad_out, idx, grad_points);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

// // input: points(b, c, n) idx(b, m)
// // output: out(b, c, m)
// template <typename scalar_t>
// __global__ void gather_points_forward_kernel(int b, int c, int n, int m,
//                      const scalar_t *__restrict__ points,
//                      const int *__restrict__ idx,
//                      scalar_t *__restrict__ out) {
//     for (int i = blockIdx.x; i < b; i += gridDim.x) {
//         for (int l = blockIdx.y; l < c; l += gridDim.y) {
//             for (int j = threadIdx.x; j < m; j += blockDim.x) {
//             int a = idx[i * m + j];
//             out[(i * c + l) * m + j] = points[(i * c + l) * n + a];
//             }
//         }
//     }
// }

// at::Tensor gather_points_cuda_forward(int b, int c, int n, int npoints,
//                   at::Tensor points, at::Tensor idx,
//                   at::Tensor out) {

//     cudaError_t err;
//     AT_DISPATCH_FLOATING_TYPES_AND_HALF(points.scalar_type(), "gather_points_cuda_forward", ([&] {
//             gather_points_forward_kernel<scalar_t><<<dim3(b, c, 1), opt_n_threads(npoints)>>>(
//             b, c, n, npoints,
//             points.data_ptr<scalar_t>(),
//             idx.data_ptr<int32_t>(),
//             out.data_ptr<scalar_t>());
//         }));

//     err = cudaGetLastError();
//     if (cudaSuccess != err) {
//     fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
//     exit(-1);
//     }
//     return out;
// }

// // input: grad_out(b, c, m) idx(b, m)
// // output: grad_points(b, c, n)
// template <typename scalar_t>
// __global__ void gather_points_backward_kernel(int b, int c, int n, int m,
//                       scalar_t *__restrict__ grad_out,
//                       const int *__restrict__ idx,
//                       scalar_t *__restrict__ grad_points) {
//     for (int i = blockIdx.x; i < b; i += gridDim.x) {
//         for (int l = blockIdx.y; l < c; l += gridDim.y) {
//             for (int j = threadIdx.x; j < m; j += blockDim.x) {
//             int a = idx[i * m + j];
//             atomicAdd(grad_points + (i * c + l) * n + a,
//                   grad_out[(i * c + l) * m + j]);
//             }
//         }
//     }
// }


// at::Tensor gather_points_cuda_backward(int b, int c, int n, int npoints,
//                        at::Tensor grad_out, at::Tensor idx, at::Tensor grad_points) {
//     cudaError_t err;
//     AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_out.scalar_type(), "gather_points_cuda_backward", ([&] {
//         gather_points_backward_kernel<scalar_t><<<dim3(b, c, 1), opt_n_threads(npoints)>>>(
//             b, c, n, npoints,
//             grad_out.data_ptr<scalar_t>(),
//             idx.data_ptr<int32_t>(),
//             grad_points.data_ptr<scalar_t>());
//       }));

//     err = cudaGetLastError();
//     if (cudaSuccess != err) {
//     fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
//     exit(-1);
//     }
//     return grad_points;
// }

template <unsigned int block_size>
__global__ void furthest_point_sampling_forward_kernel(int b, int n, int m, const int first_idx,
    const float * __restrict__ input, float * __restrict__ temp, int * __restrict__ idx) {
    // temp: (nxb) the closest distance from each of the n points to the existing set
    if (m <= 0) return;
    __shared__ float dists[block_size];
    __shared__ int dists_i[block_size];
    const unsigned int buffer_size = block_size;
    __shared__ float buf[block_size*3];
    for (int i=blockIdx.x; i<b; i+=gridDim.x){
        int old=first_idx;
        // first out of sought m points is point0
        if (threadIdx.x==0) idx[i*m+0]=old;
        // fill buffer in the shared memory with input *once* for faster read
        for (int j=threadIdx.x;j<min(buffer_size,n)*3;j+=blockDim.x){
          buf[j]=input[i*n*3+j];
        }
        __syncthreads();
        // iteratively add m points
        for (int j=1; j<m; j++){
              int besti=0;
              float best=-1;
              // position of the last point
              float x1=input[i*n*3+old*3+0];
              float y1=input[i*n*3+old*3+1];
              float z1=input[i*n*3+old*3+2];
              // Neither do i understand this loop
              for (int k=threadIdx.x;k<n;k+=blockDim.x){
                float td=temp[blockIdx.x*n+k];
                float x2,y2,z2;
                // if buffer not filled, set new point an input point
                if (k<buffer_size){
                  x2=buf[k*3+0];
                  y2=buf[k*3+1];
                  z2=buf[k*3+2];
                }else{
                  x2=input[i*n*3+k*3+0];
                  y2=input[i*n*3+k*3+1];
                  z2=input[i*n*3+k*3+2];
                }
                float d=(x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1);
                float d2=min(d,td);
                if (d2!=td)
                  temp[blockIdx.x*n+k]=d2;
                if (d2>best){
                  best=d2;
                  besti=k;
                }
              }
              dists[threadIdx.x]=best;
              dists_i[threadIdx.x]=besti;
              // u from 0~log2(block_size)
              for (int u=0;(1<<u)<blockDim.x;u++){
                __syncthreads();
                // maximize pairwise between the current thread and
                // the sibling thread in the binary-tree
                if (threadIdx.x<(blockDim.x>>(u+1))){
                  int i1=(threadIdx.x*2)<<u;
                  int i2=(threadIdx.x*2+1)<<u;
                  if (dists[i1]<dists[i2]){
                    dists[i1]=dists[i2];
                    dists_i[i1]=dists_i[i2];
                  }
                }
              }
              __syncthreads();
              old=dists_i[0];
              if (threadIdx.x==0)
                idx[i*m+j]=old;
            }
          }
        }

void furthest_sampling_cuda_forward(const int m, const int first_idx,
    at::Tensor& input, at::Tensor& temp, at::Tensor& idx) {
    const int b = input.size(0);
    const int n = input.size(1);
    unsigned int n_threads = opt_n_threads(n);
    unsigned int n_blocks = min(32, (n*b + n_threads/2)/n_threads);
    switch (n_threads) {
      case 512:
      furthest_point_sampling_forward_kernel<512><<<b, n_threads>>>(
          b, n, m, first_idx, input.data_ptr<float>(),
          temp.data_ptr<float>(),
          idx.data_ptr<int32_t>());
      break;
      case 256:
      furthest_point_sampling_forward_kernel<256><<<b, n_threads>>>(
          b, n, m, first_idx,
          input.data_ptr<float>(),
          temp.data_ptr<float>(),
          idx.data_ptr<int32_t>());
      break;
      case 128:
      furthest_point_sampling_forward_kernel<128><<<b, n_threads>>>(
          b, n, m, first_idx,
          input.data_ptr<float>(),
          temp.data_ptr<float>(),
          idx.data_ptr<int32_t>());
      break;
      case 64:
      furthest_point_sampling_forward_kernel<64><<<b, n_threads>>>(
          b, n, m, first_idx,
          input.data_ptr<float>(),
          temp.data_ptr<float>(),
          idx.data_ptr<int32_t>());
      break;
      case 32:
      furthest_point_sampling_forward_kernel<32><<<b, n_threads>>>(
          b, n, m, first_idx,
          input.data_ptr<float>(),
          temp.data_ptr<float>(),
          idx.data_ptr<int32_t>());
      break;
      case 16:
      furthest_point_sampling_forward_kernel<16><<<b, n_threads>>>(
          b, n, m, first_idx,
          input.data_ptr<float>(),
          temp.data_ptr<float>(),
          idx.data_ptr<int32_t>());
      break;
      case 8:
      furthest_point_sampling_forward_kernel<8><<<b, n_threads>>>(
          b, n, m, first_idx,
          input.data_ptr<float>(),
          temp.data_ptr<float>(),
          idx.data_ptr<int32_t>());
      break;
      case 4:
      furthest_point_sampling_forward_kernel<4><<<b, n_threads>>>(
          b, n, m, first_idx,
          input.data_ptr<float>(),
          temp.data_ptr<float>(),
          idx.data_ptr<int32_t>());
      break;
      case 2:
      furthest_point_sampling_forward_kernel<2><<<b, n_threads>>>(
          b, n, m, first_idx,
          input.data_ptr<float>(),
          temp.data_ptr<float>(),
          idx.data_ptr<int32_t>());
      break;
      case 1:
      furthest_point_sampling_forward_kernel<1><<<b, n_threads>>>(
          b, n, m, first_idx,
          input.data_ptr<float>(),
          temp.data_ptr<float>(),
          idx.data_ptr<int32_t>());
      break;
      default:
      furthest_point_sampling_forward_kernel<512><<<b, n_threads>>>(
          b, n, m, first_idx,
          input.data_ptr<float>(),
          temp.data_ptr<float>(),
          idx.data_ptr<int32_t>());
      }

    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
    }
    return;
}


template <typename scalar_t>
__device__ scalar_t euDistance2(const scalar_t * x1, const scalar_t* x2, const int c) {
  scalar_t d = 0;
  for(size_t i = 0; i < c; i++)
  {
    d += (x1[i]-x2[i]) * (x1[i]-x2[i]);
  }
  return d;
}

// input: new_xyz(b, m, 3) xyz(b, n, 3)
// output: idx(b, m, nsample)
__global__ void ball_query_kernel_fast(int b, int n, int m, float radius, int nsample,
    const float *__restrict__ new_xyz, const float *__restrict__ xyz, int *__restrict__ idx) {
    // new_xyz: (B, M, 3)
    // xyz: (B, N, 3)
    // output:
    //      idx: (B, M, nsample)
    int bs_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs_idx >= b || pt_idx >= m) return;

    new_xyz += bs_idx * m * 3 + pt_idx * 3;
    xyz += bs_idx * n * 3;
    idx += bs_idx * m * nsample + pt_idx * nsample;

    float radius2 = radius * radius;
    float new_x = new_xyz[0];
    float new_y = new_xyz[1];
    float new_z = new_xyz[2];

    int cnt = 0;
    for (int k = 0; k < n; ++k) {
        float x = xyz[k * 3 + 0];
        float y = xyz[k * 3 + 1];
        float z = xyz[k * 3 + 2];
        float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) + (new_z - z) * (new_z - z);
        if (d2 < radius2){
            if (cnt == 0){
                for (int l = 0; l < nsample; ++l) {
                    idx[l] = k;
                }
            }
            idx[cnt] = k;
            ++cnt;
            if (cnt >= nsample) break;
        }
    }
}


void ball_query_kernel_launcher_fast(int b, int n, int m, float radius, int nsample, \
    const float *new_xyz, const float *xyz, int *idx, at::cuda::CUDAStream stream) {
    // new_xyz: (B, M, 3)
    // xyz: (B, N, 3)
    // output:
    //      idx: (B, M, nsample)

    cudaError_t err;

    dim3 blocks(DIVUP(m, THREADS_PER_BLOCK), b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    ball_query_kernel_fast<<<blocks, threads, 0, stream>>>(b, n, m, radius, nsample, new_xyz, xyz, idx);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}
// template <typename scalar_t>
// __global__ void query_ball_point_kernel(const int b, const int n, const int m, const int c,
//                                         float radius, int nsample,
//                                         const scalar_t * new_xyz,
//                                         const scalar_t * xyz,
//                                         int *__restrict__ idx) {
//   int batch_index = blockIdx.x;
//   xyz += batch_index * n * c;
//   new_xyz += batch_index * m * c;
//   idx += m * nsample * batch_index;

//   int index = threadIdx.x;
//   int stride = blockDim.x;

//   float radius2 = radius * radius;
//   for (int j = index; j < m; j += stride) {
//     for (int k = 0, cnt = 0; k < n; ++k) {
//       scalar_t d2 = euDistance2(new_xyz+c*j, xyz+c*k, c);
//       if (d2 < radius2) {
//         if (cnt == 0) {
//           for (int l = 0; l < nsample; ++l) {
//             idx[j * nsample + l] = k;
//           }
//         }
//         idx[j * nsample + cnt] = k;
//         ++cnt;
//       }
//     }
//   }
// }

// at::Tensor ball_query_cuda_forward(float radius, int nsample, at::Tensor query,
//                                   at::Tensor xyz, at::Tensor idx) {
//   at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
//   const int b = xyz.size(0);
//   const int n = xyz.size(1);
//   const int m = query.size(1);
//   const int c = query.size(-1);
//   AT_DISPATCH_FLOATING_TYPES(xyz.scalar_type(), "query_ball_point_kernel", ([&]() {
//     query_ball_point_kernel<<<b, opt_n_threads(m), 0, stream>>>(b, n, m, c, radius, nsample,
//       query.data_ptr<scalar_t>(), xyz.data_ptr<scalar_t>(), idx.data_ptr<int32_t>());
// 		  }));
//   CUDA_CHECK_ERRORS();
//   return idx;
// }

// input: points(b, c, n) idx(b, npoints, nsample)
// output: out(b, c, npoints, nsample)
__global__ void group_points_kernel(int b, int c, int n, int npoints,
                                    int nsample,
                                    const float *__restrict__ points,
                                    const int *__restrict__ idx,
                                    float *__restrict__ out) {
  int batch_index = blockIdx.x;
  points += batch_index * n * c;
  idx += batch_index * npoints * nsample;
  out += batch_index * npoints * nsample * c;

  const int index = threadIdx.y * blockDim.x + threadIdx.x;
  const int stride = blockDim.y * blockDim.x;
  for (int i = index; i < c * npoints; i += stride) {
    const int l = i / npoints;
    const int j = i % npoints;
    for (int k = 0; k < nsample; ++k) {
      int ii = idx[j * nsample + k];
      out[(l * npoints + j) * nsample + k] = points[l * n + ii];
    }
  }
}

void group_points_kernel_wrapper(int b, int c, int n, int npoints, int nsample,
                                 const float *points, const int *idx,
                                 float *out) {
  at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

  group_points_kernel<<<b, opt_block_config(npoints, c), 0, stream>>>(
      b, c, n, npoints, nsample, points, idx, out);

  CUDA_CHECK_ERRORS();
}

// input: grad_out(b, c, npoints, nsample), idx(b, npoints, nsample)
// output: grad_points(b, c, n)
__global__ void group_points_grad_kernel(int b, int c, int n, int npoints,
                                         int nsample,
                                         const float *__restrict__ grad_out,
                                         const int *__restrict__ idx,
                                         float *__restrict__ grad_points) {
  int batch_index = blockIdx.x;
  grad_out += batch_index * npoints * nsample * c;
  idx += batch_index * npoints * nsample;
  grad_points += batch_index * n * c;

  const int index = threadIdx.y * blockDim.x + threadIdx.x;
  const int stride = blockDim.y * blockDim.x;
  for (int i = index; i < c * npoints; i += stride) {
    const int l = i / npoints;
    const int j = i % npoints;
    for (int k = 0; k < nsample; ++k) {
      int ii = idx[j * nsample + k];
      atomicAdd(grad_points + l * n + ii,
                grad_out[(l * npoints + j) * nsample + k]);
    }
  }
}

void group_points_grad_kernel_wrapper(int b, int c, int n, int npoints,
                                      int nsample, const float *grad_out,
                                      const int *idx, float *grad_points) {
  at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

  group_points_grad_kernel<<<b, opt_block_config(npoints, c), 0, stream>>>(
      b, c, n, npoints, nsample, grad_out, idx, grad_points);

  CUDA_CHECK_ERRORS();
}