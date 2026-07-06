#include "nn/Module.hpp"

#include <string>
#include <utility>
#include <vector>

namespace tensorforge::nn {

void Module::register_parameter(const std::string& name, Parameter p) {
    parameters_[name] = std::move(p);
}

void Module::register_module(const std::string& name, std::shared_ptr<Module> m) {
    submodules_[name] = std::move(m);
}

std::vector<Parameter*> Module::parameters() {
    std::vector<Parameter*> out;
    out.reserve(parameters_.size());

    // Direct parameters of this module.
    for (auto& kv : parameters_) {
        out.push_back(&kv.second);
    }

    // Recurse into submodules. We use recursion (not BFS) so the
    // ordering is deterministic and matches PyTorch's named_parameters()
    // depth-first layout — easier for snapshot testing later.
    for (auto& kv : submodules_) {
        if (kv.second) {
            auto sub = kv.second->parameters();
            for (Parameter* p : sub) {
                out.push_back(p);
            }
        }
    }

    return out;
}

std::vector<std::pair<std::string, Parameter*>>
Module::named_parameters(const std::string& prefix) {
    std::vector<std::pair<std::string, Parameter*>> out;
    out.reserve(parameters_.size());

    // Direct parameters — name is prefix + local_key. If prefix is
    // empty, the bare name is used (matches PyTorch at the top level).
    for (auto& kv : parameters_) {
        std::string full = prefix.empty() ? kv.first : prefix + "." + kv.first;
        out.emplace_back(std::move(full), &kv.second);
    }

    // Submodule recursion. Module name acts as the next path segment.
    for (auto& kv : submodules_) {
        if (!kv.second) continue;
        std::string child_prefix = prefix.empty() ? kv.first : prefix + "." + kv.first;
        auto sub = kv.second->named_parameters(child_prefix);
        for (auto& pair : sub) {
            out.push_back(std::move(pair));
        }
    }

    return out;
}

} // namespace tensorforge::nn