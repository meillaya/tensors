#include "data/MnistLoader.hpp"

#include "tensor/Device.hpp"
#include "tensor/Dtype.hpp"
#include "tensor/Shape.hpp"
#include "tensor/Tensor.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <ios>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tensorforge::data {

namespace {

// IDX magic numbers (big-endian on disk).
constexpr uint32_t kMagicImage = 0x00000803;
constexpr uint32_t kMagicLabel = 0x00000801;

// MNIST normalization constants (TensorForge image stats).
constexpr float kMean = 0.1307f;
constexpr float kStd = 0.3081f;

inline uint32_t read_be_u32(std::istream& in) {
    uint8_t buf[4];
    in.read(reinterpret_cast<char*>(buf), 4);
    if (in.gcount() != 4) {
        throw std::runtime_error("MNIST IDX: unexpected EOF while reading 4-byte header");
    }
    return (static_cast<uint32_t>(buf[0]) << 24) |
           (static_cast<uint32_t>(buf[1]) << 16) |
           (static_cast<uint32_t>(buf[2]) << 8) |
           (static_cast<uint32_t>(buf[3]));
}

void read_idx_images(const std::string& path, std::vector<float>& out_data, int64_t& out_count) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("MNIST IDX: cannot open image file: " + path);
    }

    const uint32_t magic = read_be_u32(in);
    if (magic != kMagicImage) {
        throw std::runtime_error("MNIST IDX: bad magic for image file: 0x" + std::to_string(magic));
    }

    // The MNIST IDX image file layout (per yann.lecun.com/exdb/mnist) is
    // magic | count | rows | cols | pixels — no separate `num_dims` field.
    const uint32_t count = read_be_u32(in);
    const uint32_t rows = read_be_u32(in);
    const uint32_t cols = read_be_u32(in);

    if (rows != 28 || cols != 28) {
        throw std::runtime_error("MNIST IDX: image is not 28x28 (got " +
                                 std::to_string(rows) + "x" + std::to_string(cols) + ")");
    }

    const int64_t n = static_cast<int64_t>(count);
    out_data.resize(static_cast<size_t>(n) * 28 * 28);

    std::vector<uint8_t> raw(static_cast<size_t>(n) * 28 * 28);
    in.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(raw.size()));
    if (static_cast<uint64_t>(in.gcount()) != raw.size()) {
        throw std::runtime_error("MNIST IDX: short read on image data");
    }

    // Normalize: (pixel / 255.0 - mean) / std.
    constexpr float kInv255 = 1.0f / 255.0f;
    const float inv_std = 1.0f / kStd;
    for (size_t i = 0; i < raw.size(); ++i) {
        const float px = static_cast<float>(raw[i]) * kInv255;
        out_data[i] = (px - kMean) * inv_std;
    }

    out_count = n;
}

void read_idx_labels(const std::string& path, std::vector<int64_t>& out_labels, int64_t& out_count) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("MNIST IDX: cannot open label file: " + path);
    }

    const uint32_t magic = read_be_u32(in);
    if (magic != kMagicLabel) {
        throw std::runtime_error("MNIST IDX: bad magic for label file: 0x" + std::to_string(magic));
    }

    // Label IDX layout: magic | count | labels — same as images, no
    // separate num_dims field.
    const uint32_t count = read_be_u32(in);
    std::vector<uint8_t> raw(count);
    in.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(count));
    if (in.gcount() != static_cast<std::streamsize>(count)) {
        throw std::runtime_error("MNIST IDX: short read on label data");
    }

    out_labels.resize(count);
    for (uint32_t i = 0; i < count; ++i) {
        if (raw[i] > 9) {
            throw std::runtime_error("MNIST IDX: label out of range (got " + std::to_string(raw[i]) + ")");
        }
        out_labels[i] = static_cast<int64_t>(raw[i]);
    }
    out_count = static_cast<int64_t>(count);
}

Tensor images_to_tensor(const std::vector<float>& data, int64_t count) {
    // Storage convention: row-major NCHW — output [N, 1, 28, 28]. Our IDX reader
    // already produced pixels in [N, rows, cols] order, so for a single channel
    // [N, H, W] and [N, 1, H, W] share the same flat layout.
    Tensor t = Tensor::empty(
        Shape{count, 1, 28, 28}, Dtype::Float32, Device::cpu());
    std::memcpy(t.data(), data.data(), data.size() * sizeof(float));
    return t;
}

Tensor labels_to_tensor(const std::vector<int64_t>& labels, int64_t count) {
    Tensor t = Tensor::empty(Shape{count}, Dtype::Int64, Device::cpu());
    std::memcpy(t.data(), labels.data(), labels.size() * sizeof(int64_t));
    return t;
}

std::pair<Tensor, Tensor> load_split(const std::string& data_dir,
                                     const std::string& image_name,
                                     const std::string& label_name) {
    const std::string image_path = data_dir + "/" + image_name;
    const std::string label_path = data_dir + "/" + label_name;

    std::vector<float> images;
    std::vector<int64_t> labels;
    int64_t n_images = 0;
    int64_t n_labels = 0;

    read_idx_images(image_path, images, n_images);
    read_idx_labels(label_path, labels, n_labels);

    if (n_images != n_labels) {
        throw std::runtime_error("MNIST IDX: image/label count mismatch (" +
                                 std::to_string(n_images) + " vs " + std::to_string(n_labels) + ")");
    }

    return {images_to_tensor(images, n_images), labels_to_tensor(labels, n_labels)};
}

} // namespace

std::pair<Tensor, Tensor> load_mnist_train(const std::string& data_dir) {
    return load_split(data_dir, "train-images-idx3-ubyte", "train-labels-idx1-ubyte");
}

std::pair<Tensor, Tensor> load_mnist_test(const std::string& data_dir) {
    return load_split(data_dir, "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte");
}

} // namespace tensorforge::data
