#!/bin/sh
# Build every image and push it to the registry config/local/images.yaml names.
#
#   make images                      every image of the default group
#   sh deploy/images.sh control-plane       one target
set -eu
[ -f config/local/images.yaml ] || { echo "no config/local/images.yaml: run tools/new_local.py and fill it in"; exit 1; }
registry_and_version="$(poetry run python3 -c "
from eemls_tools.services import load_images_config
images = load_images_config('config')
print(images.registry, images.version)
")"
REGISTRY="${registry_and_version% *}" VERSION="${registry_and_version#* }" poetry run python tools/pin.py --bake --push "$@"
