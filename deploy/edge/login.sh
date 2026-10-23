#!/bin/sh
# Log one device in to the image registry.
#
#   sh deploy/edge/login.sh <username>@<host> [.env]
#
# Needs DOCKERHUB_READ_ONLY_TOKEN in the env file; sent on stdin only.
set -eu
HOST="${1:?is the device to log in, as user@host}"
ENV_FILE="${2:-.env}"
[ -f config/local/images.yaml ] || { echo "no config/local/images.yaml: run tools/new_local.py and fill it in"; exit 1; }
grep -q '^DOCKERHUB_READ_ONLY_TOKEN=.' "$ENV_FILE" || { echo "no DOCKERHUB_READ_ONLY_TOKEN in $ENV_FILE"; exit 1; }
registry="$(poetry run python3 -c "
from eemls_tools.services import load_images_config
print(load_images_config('config').registry)
")"
server="${registry%%/*}"
owner="${registry#*/}"
sed -n 's/^DOCKERHUB_READ_ONLY_TOKEN=//p' "$ENV_FILE" \
  | ssh "$HOST" "docker login -u '$owner' --password-stdin '$server'" >/dev/null
echo "$HOST: logged in to $registry"
