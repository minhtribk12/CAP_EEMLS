# Builds the ONNX Runtime wheel with CUDA and TensorRT for the Jetson boards.
#
#   docker buildx bake onnxruntime-jetson
FROM jetson AS build

RUN apt-get update \
    && apt-get install -y --no-install-recommends git build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/build \
    && /opt/build/bin/pip install --no-cache-dir cmake==3.31.6 numpy==2.4.6 packaging wheel setuptools

ARG ORT_VERSION
WORKDIR /src
RUN git clone --depth 1 --branch "v${ORT_VERSION}" --recursive https://github.com/microsoft/onnxruntime.git .

ARG PARALLEL=2
ENV PATH=/opt/build/bin:$PATH
RUN python tools/ci_build/build.py --build_dir build --config Release --update --build \
    --build_wheel --parallel ${PARALLEL} --skip_tests --allow_running_as_root \
    --compile_no_warning_as_error \
    --use_cuda --cuda_home /usr/local/cuda --cudnn_home /usr/lib/aarch64-linux-gnu \
    --use_tensorrt --tensorrt_home /usr/lib/aarch64-linux-gnu \
    --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=87 onnxruntime_BUILD_UNIT_TESTS=OFF

FROM scratch
COPY --from=build /src/build/Release/dist/ /
