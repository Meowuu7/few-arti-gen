#include <torch/extension.h>
#include <vector>


int chamfer_cuda_forward(at::Tensor& xyz1, at::Tensor& xyz2, at::Tensor& dist1, at::Tensor& dist2, at::Tensor& idx1, at::Tensor& idx2);


int chamfer_cuda_backward(at::Tensor& xyz1, at::Tensor& xyz2, at::Tensor& gradxyz1, at::Tensor& gradxyz2, at::Tensor& graddist1, at::Tensor& graddist2, at::Tensor& idx1, at::Tensor& idx2);

int labeled_chamfer_cuda_forward(const at::Tensor& xyz1, const at::Tensor& xyz2, const at::Tensor& label1, const at::Tensor& label2,
						 		                 at::Tensor& dist1, at::Tensor& dist2, at::Tensor& idx1, at::Tensor& idx2);

int chamfer_forward(at::Tensor& xyz1, at::Tensor& xyz2, at::Tensor& dist1, at::Tensor& dist2, at::Tensor& idx1, at::Tensor& idx2) {
    return chamfer_cuda_forward(xyz1, xyz2, dist1, dist2, idx1, idx2);
}

int labeled_chamfer_forward(const at::Tensor& xyz1, const at::Tensor& xyz2, const at::Tensor& label1, const at::Tensor& label2,
						 		            at::Tensor& dist1, at::Tensor& dist2, at::Tensor& idx1, at::Tensor& idx2) {
    return labeled_chamfer_cuda_forward(xyz1, xyz2, label1, label2, dist1, dist2, idx1, idx2);
}


int chamfer_backward(at::Tensor& xyz1, at::Tensor& xyz2, at::Tensor& gradxyz1, at::Tensor& gradxyz2, at::Tensor& graddist1,
					  at::Tensor& graddist2, at::Tensor& idx1, at::Tensor& idx2) {

    return chamfer_cuda_backward(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nmdistance_forward", &chamfer_forward, "chamfer forward (CUDA)");
  m.def("labeled_nmdistance_forward", &labeled_chamfer_forward, "labeled chamfer forward (CUDA)");
  m.def("nmdistance_backward", &chamfer_backward, "chamfer backward (CUDA)");
}