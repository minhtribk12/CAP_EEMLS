#!/bin/sh
# Join one machine to the k3s server as an agent.
#
#   sh deploy/kube/k3s/agent.sh <username>@<agent> <username>@<server> <server address>
set -eu
HOST="${1:?is the machine to join, as user@host}"
SERVER="${2:?is the server, as user@host}"
ADDRESS="${3:?is the address the agent reaches the server at}"
K3S_VERSION="${K3S_VERSION:-v1.33.5+k3s1}"
ssh "$SERVER" "sudo cat /var/lib/rancher/k3s/server/node-token" </dev/null \
  | ssh "$HOST" "read -r token; curl -sfL https://get.k3s.io | K3S_URL='https://$ADDRESS:6443' K3S_TOKEN=\"\$token\" INSTALL_K3S_VERSION='$K3S_VERSION' sh -s - agent"
echo "$HOST: k3s $K3S_VERSION agent of $ADDRESS"
