# Multi-arch (linux/amd64, linux/arm64) image of the ArmoniK worker, running
# both the path-tracing and the denoising (Open Image Denoise) tasks.
# Build from the repository root (the worker needs the workspace and the lib):
#   docker buildx build --platform linux/amd64,linux/arm64 -f rust-tracer-worker/Dockerfile .
#
# Every build stage runs on the build machine and cross-compiles for the
# target, so no emulation is involved; the final stage only copies files.

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
# OIDN computes itself, overriding CMAKE_INSTALL_RPATH).
FROM --platform=${BUILDPLATFORM:-linux} debian:trixie-slim AS oidn-arm64
ARG OIDN_VERSION
ARG OIDN_SRC_SHA256=e71fd043a70f1cc80e301d1b90df6c1f536098c4dd94baa612742f6db3369c36
# renovate: datasource=github-releases depName=ispc/ispc
ARG ISPC_VERSION=1.31.0
ARG ISPC_LINUX_SHA256=d74089c835e10fd7e2c4b9225ced38b87d1fb53d35c7ceabd48cdf035da11b11
ARG ISPC_LINUX_AARCH64_SHA256=660ccac47ff7e0980b89b00a3ebd70201acf55f9e816c127fc28e868ab456193
ARG BUILDARCH
RUN dpkg --add-architecture arm64 && \
    apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates cmake ninja-build python3 \
        g++-aarch64-linux-gnu libtbb-dev:arm64
# ISPC for the build machine; it targets aarch64 from any host
RUN case "${BUILDARCH:-$(dpkg --print-architecture)}" in \
        amd64) archive="ispc-v${ISPC_VERSION}-linux.tar.gz" sha="${ISPC_LINUX_SHA256}" ;; \
        arm64) archive="ispc-v${ISPC_VERSION}-linux.aarch64.tar.gz" sha="${ISPC_LINUX_AARCH64_SHA256}" ;; \
        *) exit 1 ;; \
    esac && \
    curl -fsSL -o ispc.tar.gz "https://github.com/ispc/ispc/releases/download/v${ISPC_VERSION}/${archive}" && \
    echo "${sha}  ispc.tar.gz" | sha256sum -c && \
    mkdir /opt/ispc && tar -xzf ispc.tar.gz --strip-components=1 -C /opt/ispc && rm ispc.tar.gz
RUN curl -fsSL -o oidn.tar.gz "https://github.com/RenderKit/oidn/releases/download/v${OIDN_VERSION}/oidn-${OIDN_VERSION}.src.tar.gz" && \
    echo "${OIDN_SRC_SHA256}  oidn.tar.gz" | sha256sum -c && \
    mkdir /src && tar -xzf oidn.tar.gz --strip-components=1 -C /src && rm oidn.tar.gz
RUN cmake -S /src -B /build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/oidn \
        -DOIDN_ZIP_MODE=ON \
        -DCMAKE_SYSTEM_NAME=Linux \
        -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
        -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc \
        -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ \
        -DTBB_DIR=/usr/lib/aarch64-linux-gnu/cmake/TBB \
        -DISPC_EXECUTABLE=/opt/ispc/bin/ispc \
        -DOIDN_APPS=OFF \
        -DOIDN_DEVICE_CPU=ON \
        -DOIDN_DEVICE_SYCL=OFF \
        -DOIDN_DEVICE_CUDA=OFF \
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
# Only the OIDN libraries the CPU device needs (Intel's package also has the
# GPU devices and their runtimes, some 60 MB), TBB included.
RUN find rust-tracer-lib/src rust-tracer-worker/src -name '*.rs' -exec touch {} + && \
    cargo auditable build --release --locked -p rust-tracer-worker \
        --target "$(cat /rust-target)" && \
    mkdir -p /out/bin /out/lib && \
    mv "target/$(cat /rust-target)/release/rust-tracer-worker" /out/bin/ && \
    cp -a /oidn/lib/libOpenImageDenoise.so.* /oidn/lib/libOpenImageDenoise_core.so.* \
        /oidn/lib/libOpenImageDenoise_device_cpu.so.* /oidn/lib/libtbb.so.12* /out/lib/

# Like the ArmoniK.Core test workers: the base image's own non-root user, and
# no shared folder in the image. The deployment mounts the folders shared with
# the agent (the sockets' `/cache`), writable by any user (tmpfs mode=0777 in
# Core's docker deployment, emptyDir in ArmoniK.Infra, which also runs the pod
# as uid 5000 by default). The cc image adds the C/C++ runtime OIDN needs.
FROM gcr.io/distroless/cc-debian13:nonroot
COPY --from=build /out /app
# The worker logs nothing below `error` without a filter
ENV RUST_LOG=info
ENTRYPOINT ["/app/bin/rust-tracer-worker"]
CMD []
