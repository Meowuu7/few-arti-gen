#include <stdio.h>
#include <torch/extension.h>
#include <THC/THCAtomics.cuh>

const int BATCH = 512;

template <typename scalar_t>
__global__ void NmDistanceKernel(int b,int n,int c,const scalar_t * xyz,int m,const scalar_t * xyz2,scalar_t * result,int * result_i){
	// const int batch=512;
	extern __shared__ __align__(sizeof(scalar_t)) unsigned char my_smem[];
	scalar_t *buf = reinterpret_cast<scalar_t *>(my_smem);

	// SharedMemory<scalar_t> smem;
	// scalar_t* buf = smem.getPointer();
	for (int i=blockIdx.x;i<b;i+=gridDim.x){
		// process xyz2 points by chunks
		// each chunk:
		// 	1. sequentially fill shared buffer with xyz2
		//  2. for each point in xyz1, find NN from the buffer
		for (int k2=0;k2<m;k2+=BATCH){
			int end_k=min(m,k2+BATCH)-k2;
			for (int j=threadIdx.x;j<end_k*c;j+=blockDim.x){
				buf[j]=xyz2[(i*m+k2)*c+j];
			}
			__syncthreads();
			for (int j=threadIdx.x+blockIdx.y*blockDim.x;j<n;j+=blockDim.x*gridDim.y){
				int best_i=0;
				scalar_t best=0;
				// int end_ka=end_k-(end_k&3);
				for (int k=0;k<end_k;k++){
					scalar_t d = 0;
					for (int _c = 0; _c < c; _c++){
						scalar_t tmp = buf[k*c+_c]-xyz[(i*n+j)*c+_c];
						d += (tmp*tmp);
					}
					if (k==0 || d<best){
						best=d;
						best_i=k+k2;
					}
				}
				if (k2==0 || result[(i*n+j)]>best){
					result[(i*n+j)]=best;
					result_i[(i*n+j)]=best_i;
				}
			}
			__syncthreads();
		}
	}
}

// xyz    (b,n,3)
// xyz2   (b,m,3)
// label  (b,n,1)
// label2 (b,m,1)
template <typename scalar_t>
__global__ void LabeledNmDistanceKernel(int b,int n,int c, const scalar_t * xyz, const scalar_t *label,
										int m, const scalar_t * xyz2, const scalar_t *label2,
								 		scalar_t * result, int * result_i){
	// const int BATCH=512;
	extern __shared__ __align__(sizeof(scalar_t)) unsigned char my_smem[];
	scalar_t *buf = reinterpret_cast<scalar_t *>(my_smem);
	// loop over BATCH of xyz1
	for (int i=blockIdx.x;i<b;i+=gridDim.x){
		// process xyz2 points by chunks
		// each chunk:
		// 	1. sequentially fill shared buffer with xyz2 (into first BATCH*c) and label2 (into BATCH*1)
		//  2. for each point in xyz1, find NN from the buffer
		for (int k2=0;k2<m;k2+=BATCH){
			// idx of the last point in xyz2 to fill the batch
			int end_k=min(m,k2+BATCH)-k2;
			int begin_of_label = BATCH*c;
			for (int j=threadIdx.x;j<end_k*c;j+=blockDim.x){
				buf[j]=xyz2[(i*m+k2)*c+j];
			}
			for (int j=threadIdx.x;j<end_k;j+=blockDim.x){
				buf[j+begin_of_label]=label2[(i*m+k2)+j];
			}
			__syncthreads();
			// loop over current BATCH in xyz1
			for (int j=threadIdx.x+blockIdx.y*blockDim.x;j<n;j+=blockDim.x*gridDim.y){
				const scalar_t l1 = label[(i*n+j)];
				int best_i=-1;
				// TODO initialize with max scalar_t
				scalar_t best=scalar_t(1e10);
				// int end_ka=end_k-(end_k&3);
				for (int k=0;k<end_k;k++){
					const scalar_t l2 = buf[begin_of_label+k];
					// const scalar_t l2 = label2[(i*m+k+k2)];
					if (l1 == l2) {
						scalar_t d = 0;
						for (int _c = 0; _c < c; _c++){
							scalar_t tmp = buf[k*c+_c]-xyz[(i*n+j)*c+_c];
							d += (tmp*tmp);
						}
						if (k==0 || d<best){
							best=d;
							best_i=k+k2;
						}
					}
				}
				if (k2==0 || result[(i*n+j)]>best){
					result[(i*n+j)]=best;
					result_i[(i*n+j)]=best_i;
				}
			}
			__syncthreads();
		}
		// after processing all xyz2 and xyz1 for this batch
		// go over result_i of xyz1, check if index < 0, change distance to 0
		for (int j=threadIdx.x+blockIdx.y*blockDim.x;j<n;j+=blockDim.x*gridDim.y){
			if (result_i[((i*n+j))] < 0)
				result[(i*n+j)] = 0;
		}
	}
}

// int chamfer_cuda_forward(int b,int n,const float * xyz,int m,const float * xyz2,float * result,int * result_i,float * result2,int * result2_i, cudaStream_t stream){
int chamfer_cuda_forward(at::Tensor& xyz1, at::Tensor& xyz2, at::Tensor& dist1, at::Tensor& dist2, at::Tensor& idx1, at::Tensor& idx2){

	const auto batch_size = xyz1.size(0);
	const auto n = xyz1.size(1); //num_points point cloud A
	const auto m = xyz2.size(1); //num_points point cloud B
	const auto c = xyz1.size(2); //point dimension
	CHECK_EQ(xyz2.size(2), c);
	AT_DISPATCH_FLOATING_TYPES_AND_HALF(
		xyz1.scalar_type(), "NmDistanceKernel", ([&] {
			NmDistanceKernel<scalar_t><<<dim3(batch_size,16,1),BATCH,BATCH*c*sizeof(scalar_t)>>>(batch_size, n, c, xyz1.data_ptr<scalar_t>(), m, xyz2.data_ptr<scalar_t>(), dist1.data_ptr<scalar_t>(), idx1.data_ptr<int>());
			NmDistanceKernel<scalar_t><<<dim3(batch_size,16,1),BATCH,BATCH*c*sizeof(scalar_t)>>>(batch_size, m, c, xyz2.data_ptr<scalar_t>(), n, xyz1.data_ptr<scalar_t>(), dist2.data_ptr<scalar_t>(), idx2.data_ptr<int>());
			})
	);

	cudaError_t err = cudaGetLastError();
	  if (err != cudaSuccess) {
	    printf("error in nnd updateOutput: %s\n", cudaGetErrorString(err));
	    return 0;
	  }
	  return 1;


}

int labeled_chamfer_cuda_forward(const at::Tensor& xyz1, const at::Tensor& xyz2, const at::Tensor& label1, const at::Tensor& label2,
						 		 at::Tensor& dist1, at::Tensor& dist2, at::Tensor& idx1, at::Tensor& idx2){

	const int batch_size = xyz1.size(0);
	const int n = xyz1.size(1); //num_points point cloud A
	const int m = xyz2.size(1); //num_points point cloud B
	const int c = xyz1.size(2); //point dimension

	CHECK_EQ(xyz2.size(2), c);
	AT_DISPATCH_FLOATING_TYPES_AND_HALF(
		xyz1.scalar_type(), "NmDistanceKernel", ([&] {
			LabeledNmDistanceKernel<scalar_t><<<dim3(batch_size,16,1),BATCH,BATCH*(c+1)*sizeof(scalar_t)>>>(batch_size, n, c, xyz1.data_ptr<scalar_t>(), label1.toType(xyz1.scalar_type()).data_ptr<scalar_t>(), m,
																								 xyz2.data_ptr<scalar_t>(), label2.toType(xyz1.scalar_type()).data_ptr<scalar_t>(), dist1.data_ptr<scalar_t>(), idx1.data_ptr<int>());
			LabeledNmDistanceKernel<scalar_t><<<dim3(batch_size,16,1),BATCH,BATCH*(c+1)*sizeof(scalar_t)>>>(batch_size, m, c, xyz2.data_ptr<scalar_t>(), label2.toType(xyz1.scalar_type()).data_ptr<scalar_t>(), n,
																								 xyz1.data_ptr<scalar_t>(), label1.toType(xyz1.scalar_type()).data_ptr<scalar_t>(), dist2.data_ptr<scalar_t>(), idx2.data_ptr<int>());
			})
	);

	cudaError_t err = cudaGetLastError();
	  if (err != cudaSuccess) {
	    printf("error in nnd updateOutput: %s\n", cudaGetErrorString(err));
	    return 0;
	  }
	  return 1;
}

template <typename scalar_t>
__global__ void NmDistanceGradKernel(int b,int n,int c,const scalar_t * xyz1,int m,const scalar_t * xyz2,
									 const scalar_t * grad_dist1, const int * idx1, scalar_t * grad_xyz1, scalar_t * grad_xyz2){
	for (int i=blockIdx.x;i<b;i+=gridDim.x){
		for (int j=threadIdx.x+blockIdx.y*blockDim.x;j<n;j+=blockDim.x*gridDim.y){
			int j2=idx1[i*n+j];
			// ignore negative indices (for labeled_nmdistance points can have no closest neighbors!)
			if (j2 < 0) continue;
			scalar_t g=grad_dist1[i*n+j]*2;
			for (int _c = 0; _c < c; _c++)
			{
				scalar_t xyz_g = g*(xyz1[(i*n+j)*c+_c]-xyz2[(i*m+j2)*c+_c]);
				atomicAdd(&(grad_xyz1[(i*n+j)*c+_c]),  xyz_g);
				atomicAdd(&(grad_xyz2[(i*m+j2)*c+_c]),-xyz_g);
			}
		}
	}
}

// xyz1        (b,n,c)
// xyz2        (b,m,c)
// gradxyz1    (b,n,3)
// gradxyz2    (b,m,3)
// graddist1   (b,n)
// graddist2   (b,m)
// idx1        (b,n)            idx for the closest neighbor of each point in xyz1
// idx2        (b,m)    		...			    							  xyz2
int chamfer_cuda_backward(at::Tensor& xyz1, at::Tensor& xyz2, at::Tensor& gradxyz1, at::Tensor& gradxyz2, at::Tensor& graddist1, at::Tensor& graddist2, at::Tensor& idx1, at::Tensor& idx2){
	// cudaMemset(grad_xyz1,0,b*n*3*4);
	// cudaMemset(grad_xyz2,0,b*m*3*4);

	const auto batch_size = xyz1.size(0);
	const auto n = xyz1.size(1); //num_points point cloud A
	const auto m = xyz2.size(1); //num_points point cloud B
	const auto c = xyz1.size(2); //point dimension
	// set to zero
	gradxyz1.zero_();
	gradxyz2.zero_();
	CHECK_EQ(xyz2.size(2), c);
	AT_DISPATCH_FLOATING_TYPES_AND_HALF(
		xyz1.scalar_type(), "NmDistanceGradKernel", ([&] {
			NmDistanceGradKernel<scalar_t><<<dim3(batch_size,16,1),256>>>(batch_size,n,c,xyz1.data_ptr<scalar_t>(),m,xyz2.data_ptr<scalar_t>(),graddist1.data_ptr<scalar_t>(),idx1.data_ptr<int>(),gradxyz1.data_ptr<scalar_t>(),gradxyz2.data_ptr<scalar_t>());
			NmDistanceGradKernel<scalar_t><<<dim3(batch_size,16,1),256>>>(batch_size,m,c,xyz2.data_ptr<scalar_t>(),n,xyz1.data_ptr<scalar_t>(),graddist2.data_ptr<scalar_t>(),idx2.data_ptr<int>(),gradxyz2.data_ptr<scalar_t>(),gradxyz1.data_ptr<scalar_t>());
		})
	);

	cudaError_t err = cudaGetLastError();
	  if (err != cudaSuccess) {
	    printf("error in nnd get grad: %s\n", cudaGetErrorString(err));
	    return 0;
	  }
	  return 1;

}