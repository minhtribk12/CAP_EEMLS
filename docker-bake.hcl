# Every service image.
#
#   docker buildx bake --push                 every image, both architectures
#   docker buildx bake --load local           the x86 images, into the local daemon

variable "REGISTRY" {
  default = "localhost:5000"
}

variable "VERSION" {
  default = "latest"
}

# ONNX Runtime version on the Jetson boards.
variable "ORT_JETSON" {
  default = "1.24.4"
}

# Framework source of each app: the root, or a checkout of the tag it pins.
variable "FRAMEWORK_CONTROL_PLANE" {
  default = "."
}

variable "FRAMEWORK_OBSERVATION_SERVICE" {
  default = "."
}

variable "FRAMEWORK_DATA_HUB" {
  default = "."
}

variable "FRAMEWORK_CONSUMER" {
  default = "."
}

variable "FRAMEWORK_INFERENCE_SERVICE" {
  default = "."
}

group "default" {
  targets = ["control-plane", "observation-service", "data-hub", "consumer", "inference-service-cpu-amd64", "inference-service-cpu-arm64", "inference-service-cuda-amd64"]
}

group "local" {
  targets = ["control-plane-local", "observation-service-local", "data-hub-local", "consumer-local", "inference-service-cpu-amd64"]
}

target "_both" {
  platforms = ["linux/amd64", "linux/arm64"]
}

target "_amd64" {
  platforms = ["linux/amd64"]
}

target "control-plane" {
  inherits   = ["_both"]
  context    = "application/control-plane"
  dockerfile = "Dockerfile"
  contexts   = { framework = FRAMEWORK_CONTROL_PLANE }
  tags       = ["${REGISTRY}/eemls-control-plane:${VERSION}"]
}

target "observation-service" {
  inherits   = ["_both"]
  context    = "application/observation-service"
  dockerfile = "Dockerfile"
  contexts   = { framework = FRAMEWORK_OBSERVATION_SERVICE }
  tags       = ["${REGISTRY}/eemls-observation-service:${VERSION}"]
}

target "data-hub" {
  inherits   = ["_both"]
  context    = "application/data-hub"
  dockerfile = "Dockerfile"
  contexts   = { framework = FRAMEWORK_DATA_HUB }
  tags       = ["${REGISTRY}/eemls-data-hub:${VERSION}"]
}

target "consumer" {
  inherits   = ["_both"]
  context    = "application/consumer"
  dockerfile = "Dockerfile"
  contexts   = { framework = FRAMEWORK_CONSUMER }
  tags       = ["${REGISTRY}/eemls-consumer:${VERSION}"]
}

target "_inference-service" {
  context    = "application/inference-service"
  dockerfile = "Dockerfile"
  contexts   = { framework = FRAMEWORK_INFERENCE_SERVICE }
}

target "inference-service-cpu-amd64" {
  inherits = ["_inference-service", "_amd64"]
  tags     = ["${REGISTRY}/eemls-inference-service:cpu-amd64", "${REGISTRY}/eemls-inference-service:cpu-amd64-${VERSION}"]
}

target "inference-service-cpu-arm64" {
  inherits  = ["_inference-service"]
  platforms = ["linux/arm64"]
  tags      = ["${REGISTRY}/eemls-inference-service:cpu-arm64", "${REGISTRY}/eemls-inference-service:cpu-arm64-${VERSION}"]
}

target "inference-service-cuda-amd64" {
  inherits = ["_inference-service", "_amd64"]
  args = {
    ONNX_RUNTIME = "--require-hashes --no-deps --requirement cuda-requirements.txt"
  }
  tags = ["${REGISTRY}/eemls-inference-service:cuda-amd64", "${REGISTRY}/eemls-inference-service:cuda-amd64-${VERSION}"]
}

target "control-plane-local" {
  inherits = ["control-plane", "_amd64"]
}

target "observation-service-local" {
  inherits = ["observation-service", "_amd64"]
}

target "data-hub-local" {
  inherits = ["data-hub", "_amd64"]
}

target "consumer-local" {
  inherits = ["consumer", "_amd64"]
}

target "jetpack" {
  context    = "application/inference-service"
  dockerfile = "jetson.Dockerfile"
  target     = "jetpack"
  platforms  = ["linux/arm64"]
}

target "jetson-runtime" {
  context    = "application/inference-service"
  dockerfile = "jetson.Dockerfile"
  platforms  = ["linux/arm64"]
}

target "onnxruntime-jetson" {
  context    = "application/inference-service"
  dockerfile = "onnxruntime.Dockerfile"
  contexts   = { jetson = "target:jetpack" }
  platforms  = ["linux/arm64"]
  output     = ["type=local,dest=application/inference-service/onnxruntime-wheels"]
  args       = { ORT_VERSION = ORT_JETSON }
}

target "inference-service-jetson" {
  inherits  = ["_inference-service"]
  platforms = ["linux/arm64"]
  contexts  = { runtime = "target:jetson-runtime", wheels = "application/inference-service/onnxruntime-wheels" }
  args = {
    RUNTIME      = "runtime"
    ONNX_RUNTIME = "/opt/wheels/onnxruntime_gpu-${ORT_JETSON}-cp312-cp312-linux_aarch64.whl"
  }
  tags = ["${REGISTRY}/eemls-inference-service:jetson", "${REGISTRY}/eemls-inference-service:jetson-${VERSION}"]
}
