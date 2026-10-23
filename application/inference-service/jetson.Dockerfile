# Jetson runtime: Ubuntu, Python and only the CUDA, cuDNN and TensorRT libraries ONNX Runtime loads.

FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0 AS jetpack
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /usr/local/bin/uv
ENV UV_PYTHON_INSTALL_DIR=/opt/python
RUN uv python install 3.12.15 \
    && ln -s "$(uv python find 3.12.15)" /usr/local/bin/python

FROM jetpack AS libraries
RUN mkdir -p /keep/cuda /keep/system \
    && cd /usr/local/cuda/lib64 \
    && cp -P libcudart.so.12* libcublas.so.12* libcublasLt.so.12* libcufft.so.11* \
        libnvrtc.so.12* libnvrtc-builtins.so.12* libcudla.so.1* /keep/cuda/ \
    && cd /usr/lib/aarch64-linux-gnu \
    && cp -P libcudnn.so.9* libcudnn_graph.so.9* libcudnn_ops.so.9* libcudnn_cnn.so.9* \
        libcudnn_heuristic.so.9* libcudnn_engines_precompiled.so.9* \
        libcudnn_engines_runtime_compiled.so.9* libnvinfer.so.10* libnvinfer_plugin.so.10* \
        libnvonnxparser.so.10* libnvinfer_builder_resource.so.10* /keep/system/

FROM ubuntu:22.04
COPY --from=libraries /keep/cuda/ /usr/local/cuda/lib64/
COPY --from=libraries /keep/system/ /usr/lib/aarch64-linux-gnu/
COPY --from=jetpack /opt/python /opt/python
RUN echo /usr/local/cuda/lib64 > /etc/ld.so.conf.d/000_cuda.conf \
    && ldconfig \
    && ln -s "$(ls -d /opt/python/cpython-3.12.*)/bin/python3.12" /usr/local/bin/python
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all
