#pragma once

#include "autograd/InputBuffer.hpp"
#include "autograd/Node.hpp"
#include "tensor/Tensor.hpp"

#include <unordered_map>
#include <queue>

namespace tensorforge {

// GraphTask — per-backward-pass state: dependency counts and accumulated
// input buffers for each node in the subgraph reachable from the root.
class GraphTask {
public:
    std::unordered_map<Node*, int> dependencies_;
    std::unordered_map<Node*, InputBuffer> input_buffers_;
    bool keep_graph_;

    explicit GraphTask(bool keep_graph) : keep_graph_(keep_graph) {}
};

// Engine — single-threaded topological execution of the backward graph.
// 1. compute_dependencies: BFS from root, counting inbound edges per node.
// 2. execute: process nodes whose dependency count has reached zero,
//    distributing apply() outputs to downstream InputBuffers.
class Engine {
public:
    void execute(NodePtr<Node> root, const Tensor& initial_grad, bool keep_graph);

private:
    void compute_dependencies(Node* root, GraphTask& task);
};

} // namespace tensorforge
