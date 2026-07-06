{ pkgs, lib, config, inputs, ... }:

let
  cudaPackages = pkgs.cudaPackages_12_6;
in
{
  nixpkgs.config.allowUnfree = true;
  nixpkgs.config.cudaSupport = true;
  nixpkgs.config.cudaCapabilities = [ "8.0" "8.6" "8.9" "9.0" ];

  packages = with pkgs; [
    # Build tools
    bazelisk
    bazel-buildtools  # buildifier
    cmake
    ninja
    bear
    gdb

    # C++ toolchain
    clang_18
    clang-tools_18  # clangd, clang-format

    # CUDA redistributables (preferred over monolithic cudatoolkit)
    cudaPackages.cuda_cccl
    cudaPackages.cuda_cudart
    cudaPackages.cuda_nvcc
    cudaPackages.cuda_nvrtc
    cudaPackages.cuda_nvtx
    cudaPackages.libcublas
    cudaPackages.libcurand
  ] ++ lib.optionals pkgs.stdenv.isLinux [
    # Profiling tools (Linux only)
    cudaPackages.cuda_nsys  # if available in Nixpkgs
  ];

  env.CC = lib.getExe pkgs.clang_18;
  env.CXX = lib.getExe pkgs.clang_18;

  env.CUDA_HOME = lib.makeSearchPathOutput "dev" "" (with cudaPackages; [
    cuda_nvcc
    cuda_cudart
    libcublas
    libcurand
  ]);

  env.LD_LIBRARY_PATH = lib.makeLibraryPath [
    cudaPackages.cuda_cudart
    cudaPackages.libcublas
    cudaPackages.libcurand
    "/run/opengl-driver"  # system NVIDIA driver
  ];

  # Convenience script for compile_commands.json
  scripts.refresh-compdb.exec = ''
    bazelisk run @hedron_compile_commands//:refresh_all -- --config=sm80_sm90
  '';

  languages.cplusplus.enable = true;
  languages.cplusplus.lsp.package = pkgs.clang-tools_18;

  enterShell = ''
    echo "TensorForge dev shell"
    echo "  CUDA: $(nvcc --version 2>&1 | head -1 || echo 'cuda not in PATH yet')"
    echo "  Bazel: $(bazelisk --version 2>&1 | head -2 || echo 'bazelisk not found')"
    echo "  Clang: $(clang++ --version 2>&1 | head -1)"
  '';
}
