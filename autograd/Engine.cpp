#include "autograd/Engine.hpp"

namespace tensorforge {

void Engine::compute_dependencies(Node* root, GraphTask& task) {
    std::queue<Node*> to_visit;
    to_visit.push(root);
    task.dependencies_[root] = 0;

    while (!to_visit.empty()) {
        Node* node = to_visit.front();
        to_visit.pop();

        for (const Edge& edge : node->next_edges_) {
            if (!edge.function) {
                continue;
            }
            Node* next = edge.function.get();
            if (task.dependencies_.find(next) == task.dependencies_.end()) {
                task.dependencies_[next] = 0;
                to_visit.push(next);
            }
            task.dependencies_[next]++;
        }
    }
}

void Engine::execute(NodePtr<Node> root, const Tensor& initial_grad, bool keep_graph) {
    GraphTask task(keep_graph);
    compute_dependencies(root.get(), task);

    task.input_buffers_[root.get()] = InputBuffer(1);
    task.input_buffers_[root.get()].add(0, initial_grad);

    std::queue<Node*> ready_queue;
    ready_queue.push(root.get());

    while (!ready_queue.empty()) {
        Node* node = ready_queue.front();
        ready_queue.pop();

        auto it = task.input_buffers_.find(node);
        if (it == task.input_buffers_.end()) {
            continue;
        }

        std::vector<Tensor> grads = it->second.take();
        std::vector<Tensor> input_grads = node->apply(std::move(grads));

        for (size_t i = 0; i < node->next_edges_.size() && i < input_grads.size(); ++i) {
            const Edge& edge = node->next_edges_[i];
            if (!edge.function) {
                continue;
            }
            Node* next = edge.function.get();

            if (task.input_buffers_.find(next) == task.input_buffers_.end()) {
                task.input_buffers_[next] = InputBuffer(1);
            }
            task.input_buffers_[next].add(edge.input_nr, input_grads[i]);

            if (--task.dependencies_[next] == 0) {
                ready_queue.push(next);
            }
        }

        if (!keep_graph) {
            node->release_saved();
        }
    }
}

} // namespace tensorforge
