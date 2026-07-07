#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "autograd/ops/AddBackward.hpp"
#include "autograd/Node.hpp"
#include "autograd/SavedTensor.hpp"
#include "tensor/Tensor.hpp"

#include <cstdint>
#include <memory>

using tensorforge::AddBackward;
using tensorforge::Device;
using tensorforge::Dtype;
using tensorforge::Edge;
using tensorforge::Node;
using tensorforge::NodePtr;
using tensorforge::Shape;
using tensorforge::Tensor;

namespace {

// Helper: create a Float32 tensor filled with a scalar value.
// Avoids dependency on //tensor:factory to prevent duplicate-symbol
// conflicts with //tensor:tensor_core at link time.
Tensor make_filled(Shape shape, float value) {
    Tensor t = Tensor::empty(shape, Dtype::Float32, Device::cpu());
    float* ptr = static_cast<float*>(t.data());
    for (int64_t i = 0; i < t.numel(); ++i) {
        ptr[i] = value;
    }
    return t;
}

} // namespace

TEST_CASE("Node creation and edge wiring") {
    // Create two leaf nodes and wire an edge from a parent to each.
    auto leaf_a = std::make_shared<AddBackward>();
    auto leaf_b = std::make_shared<AddBackward>();

    auto parent = std::make_shared<AddBackward>();
    parent->next_edges_ = {Edge(leaf_a, 0), Edge(leaf_b, 0)};

    CHECK(parent->next_edges_.size() == 2);
    CHECK(parent->next_edges_[0].function == leaf_a);
    CHECK(parent->next_edges_[0].input_nr == 0);
    CHECK(parent->next_edges_[1].function == leaf_b);
    CHECK(parent->next_edges_[1].input_nr == 0);
    CHECK(parent->sequence_nr_ == 0);
}

TEST_CASE("AddBackward::apply returns 2 pass-through gradients from 1 output grad") {
    AddBackward node;
    Tensor grad = make_filled({2, 3}, 5.0f);

    std::vector<Tensor> result = node.apply(std::vector<Tensor>{std::move(grad)});

    CHECK(result.size() == 2);
    CHECK(result[0].shape() == Shape{2, 3});
    CHECK(result[1].shape() == Shape{2, 3});

    const float* p0 = static_cast<const float*>(result[0].data());
    const float* p1 = static_cast<const float*>(result[1].data());
    for (int64_t i = 0; i < result[0].numel(); ++i) {
        CHECK(p0[i] == 5.0f);
        CHECK(p1[i] == 5.0f);
    }
}

TEST_CASE("SavedTensor version check passes when unmodified") {
    Tensor t = make_filled({2, 2}, 3.0f);
    tensorforge::SavedTensor saved(t);

    const Tensor& unpacked = saved.unpack();
    CHECK(unpacked.version() == 0);

    const float* ptr = static_cast<const float*>(unpacked.data());
    CHECK(ptr[0] == 3.0f);
}

TEST_CASE("SavedTensor version check errors on mismatch") {
    Tensor t = make_filled({2, 2}, 3.0f);
    tensorforge::SavedTensor saved(t);
    CHECK(saved.version_at_save_ == 0);

    // Mutate the underlying storage — SavedTensor shares storage, so
    // its version counter also reflects the bump.
    (void)t.bump_version();
    CHECK(t.version() == 1);

    CHECK_THROWS_AS(saved.unpack(), std::runtime_error);
}
