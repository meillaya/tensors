#include "data/CifarLoader.hpp"

#include "tensor/Device.hpp"
#include "tensor/Dtype.hpp"
#include "tensor/Shape.hpp"
#include "tensor/Tensor.hpp"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <ios>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tensorforge::data {

namespace {

constexpr int64_t kImageBytes = 32 * 32; // 1024 pixels per channel
constexpr int64_t kRecordBytes = 1 + 3 * kImageBytes; // label + 3*1024

// CIFAR-10 per-channel normalization (TensorForge image stats).
constexpr float kMeanR = 0.4914f;
constexpr float kMeanG = 0.4822f;
constexpr float kMeanB = 0.4465f;
constexpr float kStdR = 0.2470f;
constexpr float kStdG = 0.2435f;
constexpr float kStdB = 0.2616f;
constexpr float kInv255 = 1.0f / 255.0f;

struct ReadResult {
    std::vector<float> images; // size = N * 3 * 32 * 32, NCHW layout
    std::vector<int64_t> labels; // size = N
    int64_t count = 0;
};

void parse_record(const uint8_t* record,
                  int64_t n_within_batch,
                  int64_t global_index,
                  std::vector<float>& images,
                  std::vector<int64_t>& labels) {
    const uint8_t label_byte = record[0];
    if (label_byte > 9) {
        throw std::runtime_error("CIFAR-10: label byte out of range (got " +
                                 std::to_string(label_byte) + ")");
    }
    labels[static_cast<size_t>(global_index)] = static_cast<int64_t>(label_byte);

    // Disk order per image: all R pixels (1024), then all G pixels, then all B.
    // We output NCHW [N, 3, 32, 32]: identical layout, no transpose needed.
    const int64_t base = global_index * 3 * kImageBytes;
    const int64_t r_offset = base;
    const int64_t g_offset = base + kImageBytes;
    const int64_t b_offset = base + 2 * kImageBytes;

    const uint8_t* r_src = record + 1;
    const uint8_t* g_src = r_src + kImageBytes;
    const uint8_t* b_src = g_src + kImageBytes;

    for (int64_t p = 0; p < kImageBytes; ++p) {
        images[static_cast<size_t>(r_offset + p)] =
            (static_cast<float>(r_src[p]) * kInv255 - kMeanR) / kStdR;
        images[static_cast<size_t>(g_offset + p)] =
            (static_cast<float>(g_src[p]) * kInv255 - kMeanG) / kStdG;
        images[static_cast<size_t>(b_offset + p)] =
            (static_cast<float>(b_src[p]) * kInv255 - kMeanB) / kStdB;
    }

    (void)n_within_batch; // currently unused; records are independent.
}

ReadResult read_batch_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        throw std::runtime_error("CIFAR-10: cannot open batch file: " + path.string());
    }
    const std::streamsize size = in.tellg();
    if (size % kRecordBytes != 0) {
        throw std::runtime_error("CIFAR-10: batch file size (" + std::to_string(size) +
                                 ") is not a multiple of record size (" +
                                 std::to_string(kRecordBytes) + ")");
    }
    const int64_t n = size / kRecordBytes;
    in.seekg(0, std::ios::beg);

    std::vector<uint8_t> raw(static_cast<size_t>(size));
    in.read(reinterpret_cast<char*>(raw.data()), size);
    if (in.gcount() != size) {
        throw std::runtime_error("CIFAR-10: short read on batch file: " + path.string());
    }

    ReadResult out;
    out.count = n;
    out.images.assign(static_cast<size_t>(n) * 3 * 32 * 32, 0.0f);
    out.labels.assign(static_cast<size_t>(n), -1);

    for (int64_t i = 0; i < n; ++i) {
        const uint8_t* record = raw.data() + i * kRecordBytes;
        parse_record(record, 0, i, out.images, out.labels);
    }
    return out;
}

Tensor images_to_tensor(const std::vector<float>& images, int64_t count) {
    Tensor t = Tensor::empty(Shape{count, 3, 32, 32}, Dtype::Float32, Device::cpu());
    std::memcpy(t.data(), images.data(),
                static_cast<size_t>(count) * 3 * 32 * 32 * sizeof(float));
    return t;
}

Tensor labels_to_tensor(const std::vector<int64_t>& labels, int64_t count) {
    Tensor t = Tensor::empty(Shape{count}, Dtype::Int64, Device::cpu());
    std::memcpy(t.data(), labels.data(),
                static_cast<size_t>(count) * sizeof(int64_t));
    return t;
}

std::pair<Tensor, Tensor> concat_into(const std::vector<ReadResult>& batches) {
    int64_t total = 0;
    for (const auto& b : batches) {
        total += b.count;
    }
    std::vector<float> all_images(static_cast<size_t>(total) * 3 * 32 * 32);
    std::vector<int64_t> all_labels(static_cast<size_t>(total));
    int64_t offset = 0;
    for (const auto& b : batches) {
        std::memcpy(all_images.data() + offset * 3 * 32 * 32,
                    b.images.data(),
                    static_cast<size_t>(b.count) * 3 * 32 * 32 * sizeof(float));
        std::memcpy(all_labels.data() + offset,
                    b.labels.data(),
                    static_cast<size_t>(b.count) * sizeof(int64_t));
        offset += b.count;
    }
    return {images_to_tensor(all_images, total),
            labels_to_tensor(all_labels, total)};
}

std::pair<Tensor, Tensor> load_split(const std::string& data_dir,
                                     const std::vector<std::string>& files) {
    if (files.empty()) {
        throw std::runtime_error("CIFAR-10: empty file list");
    }
    std::vector<ReadResult> batches;
    batches.reserve(files.size());
    for (const auto& fname : files) {
        const std::string full = data_dir + "/" + fname;
        batches.push_back(read_batch_file(full));
    }
    return concat_into(batches);
}

} // namespace

std::pair<Tensor, Tensor> load_cifar10_train(const std::string& data_dir) {
    return load_split(data_dir,
                      {"data_batch_1.bin", "data_batch_2.bin", "data_batch_3.bin",
                       "data_batch_4.bin", "data_batch_5.bin"});
}

std::pair<Tensor, Tensor> load_cifar10_test(const std::string& data_dir) {
    return load_split(data_dir, {"test_batch.bin"});
}

} // namespace tensorforge::data
