# Multi-arch (linux/amd64, linux/arm64) image of the ArmoniK worker.
# Build from the repository root (the worker needs the workspace and the lib):
#   docker buildx build --platform linux/amd64,linux/arm64 -f rust-tracer-worker/Dockerfile .
#
# The build stage always runs on the build machine and cross-compiles a static
# musl binary for the target, so no emulation is involved; the final stage only
# copies files.

# BUILDPLATFORM is unset without BuildKit: "linux" means the host platform
FROM --platform=${BUILDPLATFORM:-linux} rust AS build

# Embed dep tree in the binary so the SBOM scanner sees it in the scratch image
# renovate: datasource=crate depName=cargo-auditable
ARG CARGO_AUDITABLE_VERSION=0.7.6
RUN cargo install cargo-auditable --locked --version "$CARGO_AUDITABLE_VERSION"

# Cross toolchains: Debian cross gcc + target musl (musl-dev ships <cpu>-linux-musl-gcc),
# for the C parts of the dependencies (ring, mimalloc)
ENV CARGO_TARGET_X86_64_UNKNOWN_LINUX_MUSL_LINKER=x86_64-linux-musl-gcc \
    CC_x86_64_unknown_linux_musl=x86_64-linux-musl-gcc \
    AR_x86_64_unknown_linux_musl=x86_64-linux-gnu-ar \
    CARGO_TARGET_AARCH64_UNKNOWN_LINUX_MUSL_LINKER=aarch64-linux-musl-gcc \
    CC_aarch64_unknown_linux_musl=aarch64-linux-musl-gcc \
    AR_aarch64_unknown_linux_musl=aarch64-linux-gnu-ar
# Unset without BuildKit: build for the native arch
ARG TARGETARCH
RUN arch="${TARGETARCH:-$(dpkg --print-architecture)}" ; \
    case "$arch" in \
        amd64) cpu=x86_64 ;; \
        arm64) cpu=aarch64 ;; \
        *) exit 1 ;; \
    esac ; \
    dpkg --add-architecture "$arch" && \
    apt-get update && \
    apt-get install -y "gcc-$(echo "$cpu" | tr _ -)-linux-gnu" "musl-dev:$arch" && \
    rustup target add "$cpu-unknown-linux-musl"

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
        --target "$(rustup target list --installed | grep musl)"

COPY rust-tracer-lib/src rust-tracer-lib/src
COPY rust-tracer-worker/src rust-tracer-worker/src
# Newer than the stubs' build, whatever the timestamps in the build context
RUN find rust-tracer-lib/src rust-tracer-worker/src -name '*.rs' -exec touch {} + && \
    cargo auditable build --release --locked -p rust-tracer-worker \
        --target "$(rustup target list --installed | grep musl)" && \
    mkdir /out && \
    mv target/*/release/rust-tracer-worker /out/

# Like the ArmoniK.Core test workers: the base image's own non-root user, and
# no shared folder in the image. The deployment mounts the folders shared with
# the agent (the sockets' `/cache`), writable by any user (tmpfs mode=0777 in
# Core's docker deployment, emptyDir in ArmoniK.Infra, which also runs the pod
# as uid 5000 by default).
FROM gcr.io/distroless/static-debian13:nonroot
COPY --from=build /out/rust-tracer-worker /rust-tracer-worker
# The worker logs nothing below `error` without a filter
ENV RUST_LOG=info
ENTRYPOINT ["/rust-tracer-worker"]
CMD []
