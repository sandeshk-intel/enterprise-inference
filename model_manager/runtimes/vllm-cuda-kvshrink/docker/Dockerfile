# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# vLLM with KVShrink (IAXL, intel/intel-accel-for-llm) built in, for the vllm-cuda-kvshrink runtime.
# IAXL and vLLM must be a matched pair: IAXL main (1ddbe90 and later) targets vLLM 0.29.0.
#
#   docker build -f Dockerfile --build-arg KERNEL_HEADERS_PACKAGE=linux-headers-<GPU node's uname -r> \
#     -t <registry>/vllm-kvshrink:v0.29.0-556c832 .
#
# KERNEL_HEADERS_PACKAGE is required: IAXL's idxd-config build needs headers of the node's running kernel.
# QAT uses qatlib in standalone mode (IAXL_QATLIB=intree): the in-tree qat_4xxx driver's user space, no
# qatmgr on the host, each process opens the vfio VFs given to the pod.
ARG BASE_IMAGE=vllm/vllm-openai:v0.29.0
FROM ${BASE_IMAGE}
ENV DEBIAN_FRONTEND=noninteractive

ARG KERNEL_HEADERS_PACKAGE
RUN test -n "$KERNEL_HEADERS_PACKAGE" || { echo "ERROR: pass --build-arg KERNEL_HEADERS_PACKAGE=linux-headers-<uname -r>" >&2; exit 1; } \
    && CUDA_DIR=$(ls -d /usr/local/cuda-[0-9]*.[0-9]* 2>/dev/null | sort -V | tail -n 1) \
    && CUDA_VERSION=$(echo "$CUDA_DIR" | sed -E 's/.*cuda-([0-9]+)\.([0-9]+).*/\1-\2/') \
    && apt-get update -y \
    && apt-get install -y git iproute2 wget curl python3-pip pkg-config \
        libboost-all-dev libudev-dev nasm yasm liblz4-dev libsqlite3-dev \
        zlib1g-dev libtcmalloc-minimal4 uuid-dev libjson-c-dev \
        build-essential cmake autoconf automake autoconf-archive libtool libssl-dev \
        rdma-core libibverbs-dev librdmacm-dev libnuma-dev \
        "$KERNEL_HEADERS_PACKAGE" ${CUDA_VERSION:+cuda-nvml-dev-$CUDA_VERSION} \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Own prefix: Ubuntu's libucx0 (pulled in by libboost-mpi) would shadow this UCX at runtime.
RUN git clone https://github.com/openucx/ucx.git /opt/ucx && \
    cd /opt/ucx && git checkout v1.21.0 && \
    ./autogen.sh && \
    ./contrib/configure-release-mt --prefix=/usr/local/ucx --enable-shared --disable-static \
        --disable-doxygen-doc --enable-optimizations --enable-cma --enable-devel-headers \
        --with-cuda=/usr/local/cuda --with-verbs --with-dm && \
    make -j$(nproc) && make -j$(nproc) install-strip && \
    echo /usr/local/ucx/lib > /etc/ld.so.conf.d/ucx.conf && ldconfig && \
    rm -rf /opt/ucx
ENV PATH=/usr/local/ucx/bin:$PATH

RUN git clone https://github.com/ai-dynamo/nixl.git /opt/nixl && \
    cd /opt/nixl && git checkout v1.2.0 && \
    meson setup build -Ducx_path=/usr/local/ucx -Dinstall_headers=true -Denable_plugins=UCX \
        -Ddisable_gds_backend=true -Dbuild_docs=false --prefix=/usr/local && \
    cd build && ninja && ninja install && ldconfig && \
    rm -rf /opt/nixl

RUN git clone --depth 1 --branch stable https://github.com/intel/idxd-config.git /idxd-config \
    && cd /idxd-config && ./autogen.sh \
    && ./configure CFLAGS='-g -O2' --prefix=/usr --sysconfdir=/etc --libdir=/usr/lib --disable-docs \
    && make install && rm -rf /idxd-config

ARG QATLIB_REF="26.08.0"
RUN git clone --depth 1 --branch "$QATLIB_REF" https://github.com/intel/qatlib.git /opt/qatlib \
    && cd /opt/qatlib && ./autogen.sh && ./configure --prefix=/usr --enable-systemd=no \
    && make -j$(nproc) && make install && ldconfig && rm -rf /opt/qatlib \
    && pkg-config --exists qatlib

# Editable install from a tree kept in the image: IAXL's CMake step writes iaxl/torch_ext*.so into the source
# tree after setuptools has collected the package files, so a regular wheel lacks iaxl.torch_ext.
ARG KVSHRINK_GIT_URL="https://github.com/intel/intel-accel-for-llm"
ARG KVSHRINK_GIT_REF="556c83247815cb49abddf25ec93db99ba303d4ea"
ARG IAXL_QATLIB="intree"
RUN git clone "$KVSHRINK_GIT_URL" /opt/kvshrink-src \
    && cd /opt/kvshrink-src && git checkout "$KVSHRINK_GIT_REF" \
    && IAXL_VERSION=$(sed -n 's/^export IAXL_VERSION=\${IAXL_VERSION:-\(.*\)}$/\1/p' setvars.sh) \
    && test -n "$IAXL_VERSION" \
    && IAXL_VERSION="$IAXL_VERSION" IAXL_QATLIB="$IAXL_QATLIB" pip install --no-cache-dir -e . --no-build-isolation \
    && chmod -R a+rX /opt/kvshrink-src

RUN ls /opt/kvshrink-src/iaxl/torch_ext*.so >/dev/null \
    || { echo "ERROR: iaxl/torch_ext*.so was not built" >&2; exit 1; }; \
    if [ "$IAXL_QATLIB" = intree ]; then \
        ldd /opt/kvshrink-src/iaxl/torch_ext*.so | grep -q 'libqat\.so' \
        || { echo "ERROR: torch_ext is not linked against the system qatlib" >&2; exit 1; }; \
    fi

# IAXL's examples preload /usr/local/lib/libiomp5.so; older intel-openmp wheels install it elsewhere.
RUN _iomp5=$(find /usr/local/lib /usr/lib -name libiomp5.so 2>/dev/null | head -n 1) \
    && test -n "$_iomp5" \
    && { [ "$_iomp5" = /usr/local/lib/libiomp5.so ] || ln -sf "$_iomp5" /usr/local/lib/libiomp5.so; } \
    && ldconfig

COPY kvshrink-entrypoint.sh /usr/local/bin/kvshrink-entrypoint.sh
RUN chmod 0755 /usr/local/bin/kvshrink-entrypoint.sh
WORKDIR /workspace
