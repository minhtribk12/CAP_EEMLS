#!/bin/sh
# Label one node with its device type, and as core when asked.
#
#   sh deploy/kube/label.sh <node> <device type> [core]
set -eu
NODE="${1:?is the node, as kubectl get nodes names it}"
DEVICE_TYPE="${2:?is the device type, as config/cost/prices.yaml names it}"
core=""
[ "${3:-}" != core ] || core=--core
labels="$(poetry run eemls-deploy labels --config config --device-type "$DEVICE_TYPE" $core)"
kubectl label node "$NODE" $labels --overwrite
