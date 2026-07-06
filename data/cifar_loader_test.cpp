#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "data/CifarLoader.hpp"
#include "tensor/Device.hpp"
#include "tensor/Dtype.hpp"
#include "tensor/Shape.hpp"
#include "tensor/Tensor.hpp"

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <ios>
#include <string>
#include <vector>
#include <unistd.h>

namespace {

constexpr int64_t kRecordBytes = 1 + 3 * 32 * 32; // label + RGB

constexpr float kMeanR = 0.4914f;
constexpr float kMeanG = 0.4822f;
constexpr float kMeanB = 0.4465f;
constexpr float kStdR = 0.2470f;
constexpr float kStdG = 0.2435f;
constexpr float kStdB = 0.2616f;
constexpr float kInv255 = 1.0f / 255.0f;

void write_synthetic_cifar_batch(const std::filesystem::path& path,
                                  int64_t n,
                                  int64_t global_offset) {
    std::ofstream out(path, std::ios::binary);
    std::vector<uint8_t> record(static_cast<size_t>(kRecordBytes));

    for (int64_t local = 0; local < n; ++local) {
        const int64_t i = global_offset + local;
        // Label = i % 10.
        record[0] = static_cast<uint8_t>(i % 10);
        // Pixel pattern uses the global image index so it matches the
        // verification loop below.
        for (int64_t p = 0; p < 32 * 32; ++p) {
            record[1 + p] = static_cast<uint8_t>((i + p) & 0xFF);
            record[1 + 1024 + p] = static_cast<uint8_t>((i + p + 7) & 0xFF);
            record[1 + 2048 + p] = static_cast<uint8_t>((i + p + 14) & 0xFF);
        }
        out.write(reinterpret_cast<const char*>(record.data()),
                  static_cast<std::streamsize>(record.size()));
    }
}

void write_synthetic_cifar_dir(const std::filesystem::path& dir,
                               const std::vector<std::string>& files,
                               const std::vector<int64_t>& per_batch_counts) {
    std::filesystem::create_directories(dir);
    int64_t global_offset = 0;
    for (size_t i = 0; i < files.size(); ++i) {
        write_synthetic_cifar_batch(dir / files[i], per_batch_counts[i], global_offset);
        global_offset += per_batch_counts[i];
    }
}

} // namespace

TEST_CASE("CifarLoader parses synthetic binary and returns normalized (N,3,32,32)") {
    const std::filesystem::path tmp = std::filesystem::temp_directory_path() /
                                      ("mini-torch-cifar-" + std::to_string(::getpid()));
    std::filesystem::remove_all(tmp);
    std::filesystem::create_directories(tmp);

    // 5 train batches + 1 test batch.
    constexpr int64_t kPerBatch = 4;
    write_synthetic_cifar_dir(
        tmp,
        {"data_batch_1.bin", "data_batch_2.bin", "data_batch_3.bin",
         "data_batch_4.bin", "data_batch_5.bin", "test_batch.bin"},
        {kPerBatch, kPerBatch, kPerBatch, kPerBatch, kPerBatch, kPerBatch});

    auto [images, labels] = tensorforge::data::load_cifar10_train(tmp.string());

    constexpr int64_t kN = 5 * kPerBatch;
    CHECK(images.shape() == tensorforge::Shape{kN, 3, 32, 32});
    CHECK(images.dtype() == tensorforge::Dtype::Float32);
    CHECK(images.device() == tensorforge::Device::cpu());
    CHECK(images.numel() == kN * 3 * 32 * 32);

    CHECK(labels.shape() == tensorforge::Shape{kN});
    CHECK(labels.dtype() == tensorforge::Dtype::Int64);
    CHECK(labels.numel() == kN);

    // Verify every pixel normalized with the per-channel mean/std.
    const float* px = static_cast<const float*>(images.data());
    const int64_t* lbl = static_cast<const int64_t*>(labels.data());
    for (int64_t i = 0; i < kN; ++i) {
        const int64_t base_n = i * 3 * 32 * 32;
        for (int64_t p = 0; p < 32 * 32; ++p) {
            const float rawR = static_cast<float>((i + p) & 0xFF);
            const float rawG = static_cast<float>((i + p + 7) & 0xFF);
            const float rawB = static_cast<float>((i + p + 14) & 0xFF);
            const float expR = (rawR * kInv255 - kMeanR) / kStdR;
            const float expG = (rawG * kInv255 - kMeanG) / kStdG;
            const float expB = (rawB * kInv255 - kMeanB) / kStdB;
            CHECK(px[base_n + 0 * 1024 + p] == doctest::Approx(expR).epsilon(1e-6));
            CHECK(px[base_n + 1 * 1024 + p] == doctest::Approx(expG).epsilon(1e-6));
            CHECK(px[base_n + 2 * 1024 + p] == doctest::Approx(expB).epsilon(1e-6));
        }
        CHECK(lbl[i] == i % 10);
    }

    auto [t_images, t_labels] = tensorforge::data::load_cifar10_test(tmp.string());
    CHECK(t_images.shape() == tensorforge::Shape{kPerBatch, 3, 32, 32});
    CHECK(t_labels.shape() == tensorforge::Shape{kPerBatch});

    std::filesystem::remove_all(tmp);
}

TEST_CASE("CifarLoader errors on truncated batch file") {
    const std::filesystem::path tmp = std::filesystem::temp_directory_path() /
                                      ("mini-torch-cifar-trunc-" + std::to_string(::getpid()));
    std::filesystem::remove_all(tmp);
    std::filesystem::create_directories(tmp);

    {
        std::ofstream out(tmp / "data_batch_1.bin", std::ios::binary);
        // Skip a single record's label byte. The file won't be a multiple
        // of kRecordBytes, so parsing should reject it.
        std::vector<uint8_t> partial(static_cast<size_t>(kRecordBytes) - 1, 0);
        out.write(reinterpret_cast<const char*>(partial.data()),
                  static_cast<std::streamsize>(partial.size()));
    }
    // Create empty 4 remaining batch files so load_split can iterate.
    for (const char* fn : {"data_batch_2.bin", "data_batch_3.bin",
                           "data_batch_4.bin", "data_batch_5.bin"}) {
        std::ofstream f(tmp / fn, std::ios::binary);
    }

    CHECK_THROWS_AS(tensorforge::data::load_cifar10_train(tmp.string()),
                    std::runtime_error);

    std::filesystem::remove_all(tmp);
}

TEST_CASE("CifarLoader errors on out-of-range label") {
    const std::filesystem::path tmp = std::filesystem::temp_directory_path() /
                                      ("mini-torch-cifar-bad-label-" +
                                       std::to_string(::getpid()));
    std::filesystem::remove_all(tmp);
    std::filesystem::create_directories(tmp);

    // Single fully-formed record with a bogus label byte (e.g. 200).
    {
        std::ofstream out(tmp / "data_batch_1.bin", std::ios::binary);
        std::vector<uint8_t> record(static_cast<size_t>(kRecordBytes), 0);
        record[0] = 200;
        out.write(reinterpret_cast<const char*>(record.data()),
                  static_cast<std::streamsize>(record.size()));
    }
    for (const char* fn : {"data_batch_2.bin", "data_batch_3.bin",
                           "data_batch_4.bin", "data_batch_5.bin"}) {
        std::ofstream f(tmp / fn, std::ios::binary);
    }

    CHECK_THROWS_AS(tensorforge::data::load_cifar10_train(tmp.string()),
                    std::runtime_error);

    std::filesystem::remove_all(tmp);
}
