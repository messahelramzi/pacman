# ─────────────────────────────────────────────────────────────────────────────
# PACMAN – CI image (Serial + OpenMP execution spaces)
#
# Provides all build/test/doc dependencies for the CI pipeline:
#   cmake >= 3.31, ninja, gcc >= 13 (C++20)
#   kokkos 5.1, kokkos-kernels 5.1, arborx 2.0.1
#   pybind11, numpy
#   gcovr (coverage), doxygen (docs)
#
# Build & push:
#   docker build -t registry.gitlab.com/drti/pacman:latest .
#   docker push registry.gitlab.com/drti/pacman:latest
#
# For GPU variants pass --build-arg SPACK_ENV=spack_cuda  (or spack_hip)
# and ensure the base image carries the matching GPU runtime.
# ─────────────────────────────────────────────────────────────────────────────

ARG BASE_IMAGE=ubuntu:24.04
ARG PACMAN_VERSION=v0.1.0
ARG SPACK_VERSION=develop
ARG SPACK_ENV=spack_openmp

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 – Bootstrap Spack and concretize + install the environment
# ─────────────────────────────────────────────────────────────────────────────
FROM ${BASE_IMAGE} AS spack-builder

ARG SPACK_VERSION
ARG SPACK_ENV

ENV DEBIAN_FRONTEND=noninteractive \
    SPACK_ROOT=/opt/spack \
    SPACK_ENV_ROOT=/opt/spack-env

# System prerequisites required by Spack and its package builds
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        file \
        gcc \
        g++\
        gfortran \
        git \
        gnupg \
        libssl-dev \
        patchelf \
        python3 \
        python3-pip \
        unzip \
        xz-utils \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone Spack at a pinned release tag for reproducible builds
RUN git clone --depth=1 --branch ${SPACK_VERSION} \
        https://github.com/spack/spack.git ${SPACK_ROOT}

RUN . ${SPACK_ROOT}/share/spack/setup-env.sh 

RUN unset SPACK_ENV && . ${SPACK_ROOT}/share/spack/setup-env.sh && spack compiler find

# Register the apt-installed cmake as a Spack external (never build from source)
RUN unset SPACK_ENV && . ${SPACK_ROOT}/share/spack/setup-env.sh

# Copy the chosen spack environment spec into the image
COPY spack_envs/${SPACK_ENV}.yaml /tmp/spack.yaml

# Create and activate the Spack environment, then install
RUN unset SPACK_ENV && . ${SPACK_ROOT}/share/spack/setup-env.sh && \
    mkdir -p ${SPACK_ENV_ROOT} && \
    cp /tmp/spack.yaml ${SPACK_ENV_ROOT}/spack.yaml && \
    spack -e ${SPACK_ENV_ROOT} concretize -f && \
    spack -e ${SPACK_ENV_ROOT} install \
          --fail-fast \
          --no-check-signature \
          -j$(nproc) && \
    spack -e ${SPACK_ENV_ROOT} gc -y && \
    spack clean --all

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 – Lean runtime image
# Copy only the Spack-installed view (resolved symlink tree) to reduce size.
# ─────────────────────────────────────────────────────────────────────────────
FROM ${BASE_IMAGE} AS ci

ARG SPACK_VERSION
ARG SPACK_ENV

ENV DEBIAN_FRONTEND=noninteractive

# System prerequisites required by Spack and its package builds
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        doxygen \
        file \
        gcc \
        g++\
        gfortran \
        git \
        gnupg \
        libssl-dev \
        patchelf \
        python3 \
        python3-pip \
        unzip \
        xz-utils \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

ENV SPACK_ROOT=/opt/spack \
    SPACK_ENV_ROOT=/opt/spack-env

# Bring over Spack itself (needed to regenerate the view / module files)
COPY --from=spack-builder ${SPACK_ROOT} ${SPACK_ROOT}
COPY --from=spack-builder ${SPACK_ENV_ROOT} ${SPACK_ENV_ROOT}
# Bring over the Spack package store
COPY --from=spack-builder /opt/spack/opt /opt/spack/opt

ENV PATH="${SPACK_ROOT}/bin:${PATH}"

# Re-generate the unified view so all binaries/headers/libs are in one tree
RUN spack -e ${SPACK_ENV_ROOT} env activate --sh > /opt/spack-env/activate.sh && \
    echo "source /opt/spack-env/activate.sh" >> /etc/profile.d/spack.sh

# Make the spack env available by default for non-login shells (used by CI runners)
ENV PATH="${SPACK_ENV_ROOT}/.spack-env/view/bin:${PATH}" \
    CMAKE_PREFIX_PATH="${SPACK_ENV_ROOT}/.spack-env/view" \
    LD_LIBRARY_PATH="${SPACK_ENV_ROOT}/.spack-env/view/lib:${SPACK_ENV_ROOT}/.spack-env/view/lib64"

RUN pip3 install --break-system-packages gcovr

    # Smoke-test: verify the key tools are reachable
RUN cmake --version && \
    python3 -c "import pybind11; print('pybind11', pybind11.__version__)" && \
    python3 -c "import numpy; print('numpy', numpy.__version__)" && \
    gcovr --version && \
    doxygen --version

WORKDIR /workspace

# Default to an interactive bash shell; CI jobs override this via `script:`
CMD ["/bin/bash"]
