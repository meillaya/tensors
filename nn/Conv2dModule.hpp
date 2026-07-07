// TensorForge - Conv2dModule (Wave 7 / T45)
//
// 2D convolution as an nn::Module: holds learnable weight
// [Cout, Cin, kH, kW] and bias [Cout] parameters, with Kaiming-uniform
// weight initialisation. forward() performs a CPU reference convolution
// (sufficient for shape/initialisation smoke tests; production GEMM
// goes through cuda/kernels for the GPU path).

#pragma once

#include "nn/Module.hpp"
#include "tensor/Tensor.hpp"

namespace tensorforge::nn {

class Conv2dModule : public Module {
public:
    Conv2dModule(int in_channels, int out_channels, int kernel_size,
                 int stride = 1, int padding = 0, int dilation = 1,
                 bool bias = true);

    [[nodiscard]] Tensor forward(Tensor x) override;

private:
    int in_channels_;
    int out_channels_;
    int kernel_size_;
    int stride_;
    int padding_;
    int dilation_;
    bool bias_enabled_;
};

}  // namespace tensorforge::nn
