#!/usr/bin/env bash
# Удаление ресурсов, созданных deploy.sh
set -uo pipefail
P=${PREFIX:-automl}
for fip in $(openstack server show "$P-vm" -f json -c addresses 2>/dev/null | grep -oE '([0-9]+\.){3}[0-9]+' | tail -n +2); do
  openstack floating ip delete "$fip"
done
openstack server delete "$P-vm" --wait
openstack volume delete "$P-models"
openstack container delete --recursive "$P-artifacts"
openstack keypair delete "$P-key"
openstack security group delete "$P-sg"
openstack router remove subnet "$P-router" "$P-subnet"
openstack router delete "$P-router"
openstack subnet delete "$P-subnet"
openstack network delete "$P-net"
