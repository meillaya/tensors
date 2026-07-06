#pragma once

#include "autograd/IntrusivePtr.hpp"

#include <cstdint>
#include <vector>

namespace tensorforge {

class Tensor;

class Node;

// Edge connects a Node to one of its input sources in the backward graph.
//   function  — the Node that produced this input
//   input_nr  — which output of `function` was used (0 for single-output nodes)
struct Edge {
    NodePtr<Node> function;
    uint32_t input_nr = 0;

    Edge() = default;
    Edge(NodePtr<Node> fn, uint32_t nr) : function(std::move(fn)), input_nr(nr) {}
};

// Node — abstract autograd vertex.
// Each Node represents a backward function: given gradients w.r.t. its
// output, it computes gradients w.r.t. its inputs via apply().
class Node : public std::enable_shared_from_this<Node> {
public:
    // Edges to the nodes that produced this node's inputs (backward direction).
    std::vector<Edge> next_edges_;

    // Ordering hint for the engine. Lower sequence_nr runs earlier.
    // AccumulateGrad sets this to UINT64_MAX so it sorts last (leaf sink).
    uint64_t sequence_nr_ = 0;

    Node() = default;
    explicit Node(std::vector<Edge> next_edges) : next_edges_(std::move(next_edges)) {}

    // Apply backward pass: takes output gradients, returns input gradients.
    virtual std::vector<Tensor> apply(std::vector<Tensor>&& grads) = 0;

    // Release saved tensors (called after apply if !keep_graph).
    virtual void release_saved() {}

    virtual ~Node() = default;
};

} // namespace tensorforge
