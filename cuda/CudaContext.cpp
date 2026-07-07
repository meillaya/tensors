// TensorForge — CudaContext implementation (Wave 3 / T14)

#include "cuda/CudaContext.hpp"

#include <cstdio>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>

namespace tensorforge {

namespace {

// Per-thread cache of DeviceContext keyed by device_index. We use a
// map<thread::id, map<int32_t, DeviceContext>> so different threads get
// independent streams (helps avoid cross-thread sync accidents).
std::unordered_map<int32_t, DeviceContext>& per_thread_context_map() {
    // thread_local so each thread has its own map.
    static thread_local std::unordered_map<int32_t, DeviceContext> map;
    return map;
}

// Helper: get-or-create the DeviceContext for (this_thread, device_index).
DeviceContext& get_or_make(int32_t device_index) {
    auto& map = per_thread_context_map();
    auto it = map.find(device_index);
    if (it != map.end()) {
        return it->second;
    }

    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaGetDeviceCount failed: ") +
                                  cudaGetErrorString(err));
    }
    if (device_count <= 0) {
        throw std::runtime_error("No CUDA devices found");
    }
    if (device_index < 0 || device_index >= device_count) {
        throw std::invalid_argument("Invalid device_index");
    }

    DeviceContext ctx;
    ctx.device_index = device_index;
    ctx.owned_stream = CudaStream(device_index);
    ctx.current_stream = ctx.owned_stream.get();

    auto [inserted_it, ok] = map.emplace(device_index, std::move(ctx));
    (void)ok;
    return inserted_it->second;
}

} // namespace

DeviceContext& DeviceContext::current() {
    // Default device_index is 0; get it lazily.
    return get_or_make(0);
}

} // namespace tensorforge