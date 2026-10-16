# Multi-arch (linux/amd64, linux/arm64) image of the ArmoniK worker, running
# both the path-tracing and the denoising (Open Image Denoise) tasks.
# Build from the repository root (the worker needs the workspace and the lib):
#   docker buildx build --platform linux/amd64,linux/arm64 -f rust-tracer-worker/Dockerfile .
#
# Every build stage runs on the build machine and cross-compiles for the
# target, so no emulation is involved; the final stage only copies files.
#
# Denoising runs on an NVIDIA GPU when the container has one (DENOISE_DEVICE,
# `auto` by default), else on the CPU. The image holds no CUDA library: OIDN's
# CUDA device only needs the driver's libcuda.so.1, which the NVIDIA Container
# Toolkit mounts in the containers given a GPU (in Kubernetes, a
# `nvidia.com/gpu` limit). Its GPU code (sm_75, sm_80 and up, plus sm_87 on
# arm64, no PTX) runs on Turing and newer GPUs and on Jetson Orin, with a
# CUDA 12 driver: 525.60.13 or newer, or JetPack 6's (CUDA 12.2, which runs
# CUDA 12.9 code by minor version compatibility), not JetPack 5's.

# renovate: datasource=github-releases depName=RenderKit/oidn
ARG OIDN_VERSION=2.5.1

# Open Image Denoise for amd64: Intel's binary package.
FROM --platform=${BUILDPLATFORM:-linux} debian:trixie-slim AS oidn-amd64
ARG OIDN_VERSION
ARG OIDN_X86_64_LINUX_SHA256=743c3e2aff8c220d5d70fe6cb970fb3d36f2702d2693c61d1d148e404cf37cd6
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates
RUN curl -fsSL -o oidn.tar.gz "https://github.com/RenderKit/oidn/releases/download/v${OIDN_VERSION}/oidn-${OIDN_VERSION}.x86_64.linux.tar.gz" && \
    echo "${OIDN_X86_64_LINUX_SHA256}  oidn.tar.gz" | sha256sum -c && \
    mkdir /oidn && tar -xzf oidn.tar.gz --strip-components=1 -C /oidn

# Open Image Denoise for arm64: there is no binary package for Linux arm64,
# so it is cross-compiled from source (with Debian's TBB for arm64), in the
# layout of Intel's packages (OIDN_ZIP_MODE: `$ORIGIN/../lib` RPATHs, which
# OIDN computes itself, overriding CMAKE_INSTALL_RPATH), with the CPU and the
# CUDA devices. On bookworm: CUDA 12's math_functions.h conflicts with the
# glibc 2.41 of trixie (cospi and sinpi now noexcept), which nvcc rejects;
# libraries built against bookworm's glibc 2.36 run on the runtime image's.
FROM --platform=${BUILDPLATFORM:-linux} debian:bookworm-slim AS oidn-arm64
ARG OIDN_VERSION
ARG OIDN_SRC_SHA256=e71fd043a70f1cc80e301d1b90df6c1f536098c4dd94baa612742f6db3369c36
# renovate: datasource=github-releases depName=ispc/ispc
ARG ISPC_VERSION=1.31.0
ARG ISPC_LINUX_SHA256=d74089c835e10fd7e2c4b9225ced38b87d1fb53d35c7ceabd48cdf035da11b11
ARG ISPC_LINUX_AARCH64_SHA256=660ccac47ff7e0980b89b00a3ebd70201acf55f9e816c127fc28e868ab456193
# The CUDA toolkit, from NVIDIA's redistributable archives (versions and
# hashes from https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.9.2.json):
# nvcc for the build machine, and the headers and driver stub of arm64 servers
# (sbsa), linking to the same libcuda.so.1 as Jetson's. CUDA 12.9 is the newest
# that JetPack 6's driver runs (OIDN needs 12.8 or newer; 13.x needs JetPack 7).
ARG CUDA_NVCC_VERSION=12.9.86
ARG CUDA_NVCC_X86_64_SHA256=7a1a5b652e5ef85c82b721d10672fc9a2dbaab44e9bd3c65a69517bf53998c35
ARG CUDA_NVCC_SBSA_SHA256=0aa1fce92dbae76c059c27eefb9d0ffb58e1291151e44ff7c7f1fc2dd9376c0d
ARG CUDA_CUDART_VERSION=12.9.79
ARG CUDA_CUDART_SBSA_SHA256=8b422a3b2cb8452cb678181b0bf9d7aa7342df168b5319c5488ae3b8514101fc
ARG CUDA_CCCL_VERSION=12.9.27
ARG CUDA_CCCL_SBSA_SHA256=8c3da24801b500f1d9217d191bb4b63e5d2096c8e7d0b7695e876853180ba82f
ARG BUILDARCH
RUN dpkg --add-architecture arm64 && \
    apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates xz-utils cmake ninja-build \
        python3 g++-aarch64-linux-gnu libtbb-dev:arm64
# ISPC for the build machine; it targets aarch64 from any host
RUN case "${BUILDARCH:-$(dpkg --print-architecture)}" in \
        amd64) archive="ispc-v${ISPC_VERSION}-linux.tar.gz" sha="${ISPC_LINUX_SHA256}" ;; \
        arm64) archive="ispc-v${ISPC_VERSION}-linux.aarch64.tar.gz" sha="${ISPC_LINUX_AARCH64_SHA256}" ;; \
        *) exit 1 ;; \
    esac && \
    curl -fsSL -o ispc.tar.gz "https://github.com/ispc/ispc/releases/download/v${ISPC_VERSION}/${archive}" && \
    echo "${sha}  ispc.tar.gz" | sha256sum -c && \
    mkdir /opt/ispc && tar -xzf ispc.tar.gz --strip-components=1 -C /opt/ispc && rm ispc.tar.gz
# The toolkit's layout: the build machine's nvcc at the top, the target's
# files in targets/aarch64-linux, where nvcc looks for them (bin/nvcc.profile)
# with --target-directory; the name that CMake 3.25 expects for aarch64.
RUN base=https://developer.download.nvidia.com/compute/cuda/redist && \
    case "${BUILDARCH:-$(dpkg --print-architecture)}" in \
        amd64) host=x86_64 sha="${CUDA_NVCC_X86_64_SHA256}" ;; \
        arm64) host=sbsa sha="${CUDA_NVCC_SBSA_SHA256}" ;; \
        *) exit 1 ;; \
    esac && \
    fetch() { \
        curl -fsSL -o "$1.tar.xz" "$base/$1/linux-$2/$1-linux-$2-$3-archive.tar.xz" && \
        echo "$4  $1.tar.xz" | sha256sum -c ; \
    } && \
    target=/usr/local/cuda/targets/aarch64-linux && mkdir -p "$target" && \
    fetch cuda_nvcc "$host" "${CUDA_NVCC_VERSION}" "$sha" && \
    tar -xJf cuda_nvcc.tar.xz --strip-components=1 -C /usr/local/cuda && \
    fetch cuda_nvcc sbsa "${CUDA_NVCC_VERSION}" "${CUDA_NVCC_SBSA_SHA256}" && \
    tar -xJf cuda_nvcc.tar.xz --strip-components=1 -C "$target" --wildcards '*/include' && \
    fetch cuda_cudart sbsa "${CUDA_CUDART_VERSION}" "${CUDA_CUDART_SBSA_SHA256}" && \
    tar -xJf cuda_cudart.tar.xz --strip-components=1 -C "$target" && \
    fetch cuda_cccl sbsa "${CUDA_CCCL_VERSION}" "${CUDA_CCCL_SBSA_SHA256}" && \
    tar -xJf cuda_cccl.tar.xz --strip-components=1 -C "$target" && \
    rm ./*.tar.xz
RUN curl -fsSL -o oidn.tar.gz "https://github.com/RenderKit/oidn/releases/download/v${OIDN_VERSION}/oidn-${OIDN_VERSION}.src.tar.gz" && \
    echo "${OIDN_SRC_SHA256}  oidn.tar.gz" | sha256sum -c && \
    mkdir /src && tar -xzf oidn.tar.gz --strip-components=1 -C /src && rm oidn.tar.gz
# OIDN builds sm_75, 80, 90, 100 and 120 only, and Jetson Orin (sm_87) cannot
# load sm_80 code (CUDA_ERROR_INVALID_SOURCE: there is no binary compatibility
# between desktop and Tegra GPUs): add sm_87 to both architecture lists.
RUN sed -i -e 's/\(OIDN_CUDA_SM_FLAGS 75 80\) /\1 87 /' -e 's/\(OIDN_CUDA_SM80_FLAGS 80\) /\1 87 /' \
        /src/devices/cuda/CMakeLists.txt && \
    test "$(grep -c ' 80 87 ' /src/devices/cuda/CMakeLists.txt)" = 2
# A toolchain file: OIDN builds the CUDA device in a CMake project of its own,
# which only gets the toolchain file and CUDAToolkit_ROOT.
RUN printf '%s\n' \
        'set(CMAKE_SYSTEM_NAME Linux)' \
        'set(CMAKE_SYSTEM_PROCESSOR aarch64)' \
        'set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)' \
        'set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)' \
        'set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)' \
        'set(CMAKE_CUDA_HOST_COMPILER aarch64-linux-gnu-g++)' \
        'set(CMAKE_CUDA_FLAGS_INIT "--target-directory aarch64-linux")' \
        > /aarch64.cmake
RUN cmake -S /src -B /build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/oidn \
        -DOIDN_ZIP_MODE=ON \
        -DCMAKE_TOOLCHAIN_FILE=/aarch64.cmake \
        -DCUDAToolkit_ROOT=/usr/local/cuda \
        -DTBB_DIR=/usr/lib/aarch64-linux-gnu/cmake/TBB \
        -DISPC_EXECUTABLE=/opt/ispc/bin/ispc \
        -DOIDN_APPS=OFF \
        -DOIDN_DEVICE_CPU=ON \
        -DOIDN_DEVICE_SYCL=OFF \
        -DOIDN_DEVICE_CUDA=ON \
        -DOIDN_DEVICE_HIP=OFF \
        -DOIDN_DEVICE_METAL=OFF && \
    cmake --build /build && \
    cmake --install /build && \
    cp -a /usr/lib/aarch64-linux-gnu/libtbb.so.12* /oidn/lib/

# BuildKit only builds the stage of the target architecture
FROM oidn-${TARGETARCH:-amd64} AS oidn

# BUILDPLATFORM is unset without BuildKit: "linux" means the host platform
FROM --platform=${BUILDPLATFORM:-linux} rust AS build

# Embed dep tree in the binary so the SBOM scanner sees it in the image
# renovate: datasource=crate depName=cargo-auditable
ARG CARGO_AUDITABLE_VERSION=0.7.6
RUN cargo install cargo-auditable --locked --version "$CARGO_AUDITABLE_VERSION"

# Cross toolchains (Debian cross gcc, glibc): Open Image Denoise is a glibc
# C++ library, and the C parts of the dependencies (ring, mimalloc) need a C
# compiler for the target.
ENV CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=x86_64-linux-gnu-gcc \
    CC_x86_64_unknown_linux_gnu=x86_64-linux-gnu-gcc \
    AR_x86_64_unknown_linux_gnu=x86_64-linux-gnu-ar \
    CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
    CC_aarch64_unknown_linux_gnu=aarch64-linux-gnu-gcc \
    AR_aarch64_unknown_linux_gnu=aarch64-linux-gnu-ar
# Unset without BuildKit: build for the native arch
ARG TARGETARCH
RUN arch="${TARGETARCH:-$(dpkg --print-architecture)}" ; \
    case "$arch" in \
        amd64) cpu=x86_64 ;; \
        arm64) cpu=aarch64 ;; \
        *) exit 1 ;; \
    esac ; \
    apt-get update && \
    apt-get install -y "gcc-$(echo "$cpu" | tr _ -)-linux-gnu" && \
    rustup target add "$cpu-unknown-linux-gnu" && \
    echo "$cpu-unknown-linux-gnu" > /rust-target

# The `oidn` crate links against the library in OIDN_DIR. At runtime the
# binary is in bin/ and the libraries in lib/, the layout of Intel's package,
# whose libraries find each other through `$ORIGIN/../lib` (the arm64 build
# uses the same "zip mode" as Intel's packages).
COPY --from=oidn /oidn /oidn
ENV OIDN_DIR=/oidn \
    RUSTFLAGS="-C link-arg=-Wl,-rpath,\$ORIGIN/../lib -C link-arg=-Wl,-rpath-link,/oidn/lib"

WORKDIR /app

# Dependencies first, against stub sources, so they are cached until a manifest
# changes. Cargo loads the whole workspace, so every member needs its manifest
# and a target to discover, even those not built here.
COPY Cargo.toml Cargo.lock ./
COPY rust-tracer-lib/Cargo.toml rust-tracer-lib/
COPY rust-tracer-worker/Cargo.toml rust-tracer-worker/
COPY rust-tracer-client/Cargo.toml rust-tracer-client/
COPY rust-tracer-standalone/Cargo.toml rust-tracer-standalone/
RUN for member in rust-tracer-lib rust-tracer-worker rust-tracer-client rust-tracer-standalone; do \
        mkdir -p "$member/src" && touch "$member/src/lib.rs" && echo 'fn main() {}' > "$member/src/main.rs" ; \
    done && \
    rm rust-tracer-lib/src/main.rs && \
    cargo auditable build --release --locked -p rust-tracer-worker \
        --target "$(cat /rust-target)"

COPY rust-tracer-lib/src rust-tracer-lib/src
COPY rust-tracer-worker/src rust-tracer-worker/src
# Newer than the stubs' build, whatever the timestamps in the build context.
# Only the OIDN libraries the CPU and CUDA devices need (Intel's package also
# has the other GPU devices and their runtimes, some 60 MB), TBB included.
RUN find rust-tracer-lib/src rust-tracer-worker/src -name '*.rs' -exec touch {} + && \
    cargo auditable build --release --locked -p rust-tracer-worker \
        --target "$(cat /rust-target)" && \
    mkdir -p /out/bin /out/lib && \
    mv "target/$(cat /rust-target)/release/rust-tracer-worker" /out/bin/ && \
    cp -a /oidn/lib/libOpenImageDenoise.so.* /oidn/lib/libOpenImageDenoise_core.so.* \
        /oidn/lib/libOpenImageDenoise_device_cpu.so.* \
        /oidn/lib/libOpenImageDenoise_device_cuda.so.* /oidn/lib/libtbb.so.12* /out/lib/

# Like the ArmoniK.Core test workers: the base image's own non-root user, and
# no shared folder in the image. The deployment mounts the folders shared with
# the agent (the sockets' `/cache`), writable by any user (tmpfs mode=0777 in
# Core's docker deployment, emptyDir in ArmoniK.Infra, which also runs the pod
# as uid 5000 by default). The cc image adds the C/C++ runtime OIDN needs.
FROM gcr.io/distroless/cc-debian13:nonroot
COPY --from=build /out /app
# Distroless has no ld.so.conf, so ldconfig ignores the drop-in that the NVIDIA
# Container Toolkit (before 1.18.1) adds in ld.so.conf.d for the driver
# libraries it mounts outside the default paths (Jetson's
# /usr/lib/aarch64-linux-gnu/nvidia), and libcuda.so.1 is then not found.
COPY rust-tracer-worker/ld.so.conf /etc/ld.so.conf
# The worker logs nothing below `error` without a filter
ENV RUST_LOG=info
# Jetson's driver directory, found through the ld cache above in principle, but
# not yet on our Jetson (to investigate). Elsewhere it does not exist, and the
# loader skips it.
ENV LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/nvidia
ENTRYPOINT ["/app/bin/rust-tracer-worker"]
CMD []
