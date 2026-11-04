#!/usr/bin/env bash
# Императивное развертывание тем же набором ресурсов через openstack CLI
# (удобно для демонстрации на защите; идемпотентность — у Terraform/Heat).
#   source openrc.sh && REPO_URL=https://github.com/<you>/automl-cloud.git ./openstack/cli/deploy.sh
set -euo pipefail
P=${PREFIX:-automl}
EXT=${EXTERNAL_NETWORK:-public}
IMAGE=${IMAGE:-Ubuntu-24.04}
FLAVOR=${FLAVOR:-m1.large}
KEY=${KEY:-$HOME/.ssh/id_ed25519.pub}
: "${REPO_URL:?set REPO_URL}"

openstack network create "$P-net"
openstack subnet create "$P-subnet" --network "$P-net" --subnet-range 10.40.0.0/24 --dns-nameserver 8.8.8.8
openstack router create "$P-router"
openstack router set "$P-router" --external-gateway "$EXT"
openstack router add subnet "$P-router" "$P-subnet"

openstack security group create "$P-sg"
openstack security group rule create "$P-sg" --protocol tcp --dst-port 22
openstack security group rule create "$P-sg" --protocol tcp --dst-port 8000
openstack keypair create --public-key "$KEY" "$P-key"

openstack volume create --size 20 "$P-models"
openstack container create "$P-artifacts"

TMP=$(mktemp)
sed -e "s#\${repo_url}#$REPO_URL#; s#\${repo_ref}#main#; s#\${max_budget}#1800#; s#\${swift_bucket}#$P-artifacts#" \
  "$(dirname "$0")/../terraform/cloud-init.yaml.tftpl" > "$TMP"

openstack server create "$P-vm" --image "$IMAGE" --flavor "$FLAVOR" --network "$P-net" \
  --security-group "$P-sg" --key-name "$P-key" --user-data "$TMP" --wait
openstack server add volume "$P-vm" "$P-models"

FIP=$(openstack floating ip create "$EXT" -f value -c floating_ip_address)
openstack server add floating ip "$P-vm" "$FIP"
echo "API: http://$FIP:8000  (cloud-init собирает образ ~3-5 мин)"
echo "Проверка: curl http://$FIP:8000/health"
