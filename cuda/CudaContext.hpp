// TensorForge — CudaContext (Wave 3 / T14)
//
// RAII wrappers around cudaStream_t and cudaEvent_t plus a per-device
// DeviceContext. The current_device_context() thread-local singleton gives
// kernel-launch code a uniform way to fetch "the stream I should launch on".
//
// IMPORTANT: include <cstdint> / <cstddef> BEFORE <cuda_runtime.h> to dodge
// the libstdc++12 + nvcc12 __noinline__ macro collision.

#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace tensorforge {

// RAII cudaStream_t wrapper. Creates a non-blocking stream by default so the
// default stream is not the legacy null stream (which serializes everything
// across the whole device — kills overlap). Non-blocking means: ops on this
// stream run concurrently with the legacy default stream.
class CudaStream {
public:
    CudaStream() noexcept = default;

    explicit CudaStream(int device_index, bool high_priority = false) {
        cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
        (void)device_index;
        (void)high_priority;
    }

    ~CudaStream() {
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
    }

    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    CudaStream(CudaStream&& other) noexcept : stream_(other.stream_) { other.stream_ = nullptr; }
    CudaStream& operator=(CudaStream&& other) noexcept {
        if (this != &other) {
            if (stream_ != nullptr) {
                cudaStreamDestroy(stream_);
            }
            stream_ = other.stream_;
            other.stream_ = nullptr;
        }
        return *this;
    }

    [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }
    [[nodiscard]] explicit operator cudaStream_t() const noexcept { return stream_; }

    void synchronize() const {
        if (stream_ != nullptr) {
            cudaStreamSynchronize(stream_);
        }
    }

private:
    cudaStream_t stream_ = nullptr;
};

// RAII cudaEvent_t wrapper. Uses cudaEventDisableTiming by default since we
// only use events for cross-stream synchronization (not timing measurements —
// those use cudaEventElapsedTime via a different code path if ever needed).
class CudaEvent {
public:
    enum UnallocatedTag { Unallocated };

    CudaEvent() noexcept : event_(nullptr) {}

    explicit CudaEvent(unsigned flags) {
        cudaEventCreateWithFlags(&event_, flags);
    }

    explicit CudaEvent(UnallocatedTag /*tag*/) noexcept : event_(nullptr) {}

    ~CudaEvent() {
        if (event_ != nullptr) {
            cudaEventDestroy(event_);
            event_ = nullptr;
        }
    }

    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;
    CudaEvent(CudaEvent&& other) noexcept : event_(other.event_) { other.event_ = nullptr; }
    CudaEvent& operator=(CudaEvent&& other) noexcept {
        if (this != &other) {
            if (event_ != nullptr) {
                cudaEventDestroy(event_);
            }
            event_ = other.event_;
            other.event_ = nullptr;
        }
        return *this;
    }

    [[nodiscard]] cudaEvent_t get() const noexcept { return event_; }
    [[nodiscard]] bool is_valid() const noexcept { return event_ != nullptr; }

    void record(cudaStream_t stream) const {
        if (event_ != nullptr) {
            cudaEventRecord(event_, stream);
        }
    }

    void synchronize() const {
        if (event_ != nullptr) {
            cudaEventSynchronize(event_);
        }
    }

private:
    cudaEvent_t event_ = nullptr;
};

// Per-device context: device_index + the current stream we launch kernels on.
// All kernel launchers take a cudaStream_t (not a DeviceContext) for ABI
// simplicity, but DeviceContext bundles "which device + which stream" so a
// caller can pin a stream to a device.
//
// There is exactly one DeviceContext per (thread, device_index). Lazily
// created on first access via current_device_context().
//
// NOTE: cudaSetDevice is process-global, NOT thread-local. We honor the
// thread-local default-stream convention but the actual current device is
// shared across threads. This matches what PyTorch does (their at::cuda::
// CurrentDevice is also process-global via cudaSetDevice).
struct DeviceContext {
    int32_t device_index = 0;
    cudaStream_t current_stream = nullptr;
    CudaStream owned_stream;  // Stream we own so its lifetime matches this ctx.

    [[nodiscard]] static DeviceContext& current();
};

} // namespace tensorforge