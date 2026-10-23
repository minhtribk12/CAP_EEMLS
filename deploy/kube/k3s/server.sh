#!/bin/sh
# Install the k3s server and write its kubeconfig here.
#
#   sh deploy/kube/k3s/server.sh <username>@<host> <server address> [~/.kube/eemls.yaml]
set -eu
HOST="${1:?is the machine to run the server on, as user@host}"
ADDRESS="${2:?is the address this machine and the agents reach the server at}"
OUT="${3:-$HOME/.kube/eemls.yaml}"
K3S_VERSION="${K3S_VERSION:-v1.33.5+k3s1}"
ssh "$HOST" "curl -sfL https://get.k3s.io | INSTALL_K3S_VERSION='$K3S_VERSION' sh -s - server --disable traefik --tls-san $ADDRESS" </dev/null
mkdir -p "$(dirname "$OUT")"
umask 077
temp_kubeconfig="$(mktemp "$OUT.XXXXXX")"
ssh "$HOST" "sudo cat /etc/rancher/k3s/k3s.yaml" </dev/null | sed "s/127.0.0.1/$ADDRESS/" > "$temp_kubeconfig"
if [ ! -s "$temp_kubeconfig" ]; then
  rm -f "$temp_kubeconfig"
  echo "$HOST: no kubeconfig came back, so $OUT is left as it was"
  exit 1
fi
mv "$temp_kubeconfig" "$OUT"
echo "$HOST: k3s $K3S_VERSION server at $ADDRESS; KUBECONFIG=$OUT"
