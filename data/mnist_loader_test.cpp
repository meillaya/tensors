#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "data/MnistLoader.hpp"
#include "tensor/Device.hpp"
#include "tensor/Dtype.hpp"
#include "tensor/Shape.hpp"
#include "tensor/Tensor.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <ios>
#include <string>
#include <vector>
#include <unistd.h>

namespace {

constexpr uint32_t kMagicImage = 0x00000803;
constexpr uint32_t kMagicLabel = 0x00000801;
constexpr float kMean = 0.1307f;
constexpr float kStd = 0.3081f;
constexpr float kInv255 = 1.0f / 255.0f;

void write_be_u32(std::ostream& out, uint32_t v) {
    uint8_t buf[4] = {
        static_cast<uint8_t>((v >> 24) & 0xFF),
        static_cast<uint8_t>((v >> 16) & 0xFF),
        static_cast<uint8_t>((v >> 8) & 0xFF),
        static_cast<uint8_t>(v & 0xFF),
    };
    out.write(reinterpret_cast<const char*>(buf), 4);
}

void write_synthetic_idx_dir(const std::filesystem::path& dir, int64_t n) {
    std::filesystem::create_directories(dir);

    // Build a deterministic pixel pattern: image i, pixel p has value
    // (i + p) mod 256. This lets us verify every pixel after normalization.
    std::vector<uint8_t> image_bytes(static_cast<size_t>(n) * 28 * 28);
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t p = 0; p < 28 * 28; ++p) {
            image_bytes[static_cast<size_t>(i) * 28 * 28 + static_cast<size_t>(p)] =
                static_cast<uint8_t>((i + p) & 0xFF);
        }
    }

    // Labels: image i gets label i % 10.
    std::vector<uint8_t> label_bytes(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        label_bytes[static_cast<size_t>(i)] = static_cast<uint8_t>(i % 10);
    }

    const std::filesystem::path image_path = dir / "train-images-idx3-ubyte";
    const std::filesystem::path label_path = dir / "train-labels-idx1-ubyte";

    {
        std::ofstream out(image_path, std::ios::binary);
        write_be_u32(out, kMagicImage);
        write_be_u32(out, static_cast<uint32_t>(n));
        write_be_u32(out, 28);
        write_be_u32(out, 28);
        out.write(reinterpret_cast<const char*>(image_bytes.data()),
                  static_cast<std::streamsize>(image_bytes.size()));
    }
    {
        std::ofstream out(label_path, std::ios::binary);
        write_be_u32(out, kMagicLabel);
        write_be_u32(out, static_cast<uint32_t>(n));
        out.write(reinterpret_cast<const char*>(label_bytes.data()),
                  static_cast<std::streamsize>(label_bytes.size()));
    }
}

} // namespace

TEST_CASE("MnistLoader parses synthetic IDX and returns normalized (N,1,28,28)") {
    const std::filesystem::path tmp = std::filesystem::temp_directory_path() /
                                      ("mini-torch-mnist-" + std::to_string(::getpid()));
    std::filesystem::remove_all(tmp);

    constexpr int64_t kN = 5;
    write_synthetic_idx_dir(tmp, kN);

    auto [images, labels] = tensorforge::data::load_mnist_train(tmp.string());

    CHECK(images.shape() == tensorforge::Shape{kN, 1, 28, 28});
    CHECK(images.dtype() == tensorforge::Dtype::Float32);
    CHECK(images.device() == tensorforge::Device::cpu());
    CHECK(images.numel() == kN * 1 * 28 * 28);

    CHECK(labels.shape() == tensorforge::Shape{kN});
    CHECK(labels.dtype() == tensorforge::Dtype::Int64);
    CHECK(labels.numel() == kN);

    // Verify every pixel has been normalized: ((raw / 255) - mean) / std.
    const float* px = static_cast<const float*>(images.data());
    const int64_t* lbl = static_cast<const int64_t*>(labels.data());
    for (int64_t i = 0; i < kN; ++i) {
        for (int64_t p = 0; p < 28 * 28; ++p) {
            const float raw = static_cast<float>((i + p) & 0xFF);
            const float expected = ((raw * kInv255) - kMean) / kStd;
            const size_t idx = static_cast<size_t>(i) * 28 * 28 + static_cast<size_t>(p);
            CHECK(px[idx] == doctest::Approx(expected).epsilon(1e-6));
        }
        CHECK(lbl[i] == i % 10);
    }

    std::filesystem::remove_all(tmp);
}

TEST_CASE("MnistLoader errors on image/label count mismatch") {
    const std::filesystem::path tmp = std::filesystem::temp_directory_path() /
                                      ("mini-torch-mnist-mis-" + std::to_string(::getpid()));
    std::filesystem::remove_all(tmp);
    std::filesystem::create_directories(tmp);

    constexpr int64_t kN = 3;
    write_synthetic_idx_dir(tmp, kN);

    // Overwrite the label file with a different count to force a mismatch.
    {
        std::ofstream out(tmp / "train-labels-idx1-ubyte", std::ios::binary);
        write_be_u32(out, kMagicLabel);
        write_be_u32(out, static_cast<uint32_t>(kN + 1));
        std::vector<uint8_t> wrong(static_cast<size_t>(kN + 1), 0);
        out.write(reinterpret_cast<const char*>(wrong.data()),
                  static_cast<std::streamsize>(wrong.size()));
    }

    CHECK_THROWS_AS(tensorforge::data::load_mnist_train(tmp.string()), std::runtime_error);

    std::filesystem::remove_all(tmp);
}

TEST_CASE("MnistLoader errors on bad magic number") {
    const std::filesystem::path tmp = std::filesystem::temp_directory_path() /
                                      ("mini-torch-mnist-bad-" + std::to_string(::getpid()));
    std::filesystem::remove_all(tmp);
    std::filesystem::create_directories(tmp);

    {
        std::ofstream out(tmp / "train-images-idx3-ubyte", std::ios::binary);
        // Garbage magic.
        write_be_u32(out, 0xDEADBEEF);
        write_be_u32(out, 3);
        write_be_u32(out, 1);
        write_be_u32(out, 28);
        write_be_u32(out, 28);
        char dummy = 0;
        out.write(&dummy, 1);
    }
    {
        std::ofstream out(tmp / "train-labels-idx1-ubyte", std::ios::binary);
        write_be_u32(out, kMagicLabel);
        write_be_u32(out, 1);
        char label = 0;
        out.write(&label, 1);
    }

    CHECK_THROWS_AS(tensorforge::data::load_mnist_train(tmp.string()), std::runtime_error);

    std::filesystem::remove_all(tmp);
}
