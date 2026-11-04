output "floating_ip" {
  value = openstack_networking_floatingip_v2.automl.address
}

output "api_url" {
  value = "http://${openstack_networking_floatingip_v2.automl.address}:8000"
}

output "ssh" {
  value = "ssh ubuntu@${openstack_networking_floatingip_v2.automl.address}"
}

output "swift_container" {
  value = openstack_objectstorage_container_v1.models.name
}
