#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

using namespace kittens;

// template <int Mb_, int Nb_, int Dqk_, int Dvo_>
// struct config {
    
// }



void dispatch(at::Tensor Q, at::Tensor K, at::Tensor V,
    at::Tensor O, at::Tensor LSE) {
CHECK_INPUT(Q); CHECK_INPUT(K); CHECK_INPUT(V); CHECK_INPUT(O); CHECK_INPUT(LSE);

using C = config<128, 128, 192, 128>;
using G = globals<C>;

G g{
kittens::py::tensor_to_gl<typename G::q_gl>(Q),
kittens::py::tensor_to_gl<typename G::k_gl>(K),
kittens::py::tensor_to_gl<typename G::v_gl>(V),
kittens::py::tensor_to_gl<typename G::o_gl>(O),
kittens::py::tensor_to_gl<typename G::lse_gl>(LSE)
};

CUDACHECK(cudaFuncSetAttribute(kernel<C>, cudaFuncAttributeMaxDynamicSharedMemorySize, g.dynamic_shared_memory()));
LaunchConfig<true, false> launch_config(g.grid(), g.block(), g.dynamic_shared_memory(), 0, C::CLUSTER_SIZE);
CUDACHECK(cudaLaunchKernelEx(launch_config, kernel<C>, g));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("forward", &dispatch, "MHA forward (bf16, B300, non-causal)");
}