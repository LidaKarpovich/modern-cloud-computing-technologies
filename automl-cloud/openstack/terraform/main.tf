# AutoML-as-a-Service в OpenStack: сеть, ВМ (Nova), том для моделей (Cinder),
# контейнер Swift для артефактов, floating IP. Провайдер terraform-provider-openstack 3.x.
#
#   source openrc.sh                       # переменные OS_* вашего стенда
#   terraform init && terraform apply -var="external_network=public" -var="image=Ubuntu-24.04"

terraform {
  required_version = ">= 1.5"
  required_providers {
    openstack = {
      source  = "terraform-provider-openstack/openstack"
      version = "~> 3.0"
    }
  }
}

provider "openstack" {} # берёт OS_AUTH_URL, OS_PROJECT_NAME, ... из окружения

# ---------------------------------------------------------------- network (Neutron)
data "openstack_networking_network_v2" "ext" {
  name = var.external_network
}

resource "openstack_networking_network_v2" "automl" {
  name           = "${var.prefix}-net"
  admin_state_up = true
}

resource "openstack_networking_subnet_v2" "automl" {
  name            = "${var.prefix}-subnet"
  network_id      = openstack_networking_network_v2.automl.id
  cidr            = var.cidr
  ip_version      = 4
  dns_nameservers = var.dns
}

resource "openstack_networking_router_v2" "automl" {
  name                = "${var.prefix}-router"
  external_network_id = data.openstack_networking_network_v2.ext.id
}

resource "openstack_networking_router_interface_v2" "automl" {
  router_id = openstack_networking_router_v2.automl.id
  subnet_id = openstack_networking_subnet_v2.automl.id
}

resource "openstack_networking_secgroup_v2" "automl" {
  name        = "${var.prefix}-sg"
  description = "SSH + AutoML API"
}

resource "openstack_networking_secgroup_rule_v2" "ssh" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 22
  port_range_max    = 22
  remote_ip_prefix  = var.admin_cidr
  security_group_id = openstack_networking_secgroup_v2.automl.id
}

resource "openstack_networking_secgroup_rule_v2" "api" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 8000
  port_range_max    = 8000
  remote_ip_prefix  = var.api_cidr
  security_group_id = openstack_networking_secgroup_v2.automl.id
}

resource "openstack_networking_port_v2" "automl" {
  name               = "${var.prefix}-port"
  network_id         = openstack_networking_network_v2.automl.id
  security_group_ids = [openstack_networking_secgroup_v2.automl.id]
  fixed_ip {
    subnet_id = openstack_networking_subnet_v2.automl.id
  }
}

# ---------------------------------------------------------------- compute (Nova)
resource "openstack_compute_keypair_v2" "automl" {
  name       = "${var.prefix}-key"
  public_key = file(var.ssh_public_key)
}

resource "openstack_compute_instance_v2" "automl" {
  name        = "${var.prefix}-vm"
  image_name  = var.image
  flavor_name = var.flavor
  key_pair    = openstack_compute_keypair_v2.automl.name
  user_data = templatefile("${path.module}/cloud-init.yaml.tftpl", {
    repo_url     = var.repo_url
    repo_ref     = var.repo_ref
    max_budget   = var.max_time_budget
    swift_bucket = openstack_objectstorage_container_v1.models.name
  })

  network {
    port = openstack_networking_port_v2.automl.id
  }

  metadata = {
    role    = "automl-service"
    project = "cloud-tech-2026-var20"
  }

  depends_on = [openstack_networking_router_interface_v2.automl]
}

# ---------------------------------------------------------------- storage (Cinder + Swift)
resource "openstack_blockstorage_volume_v3" "models" {
  name = "${var.prefix}-models"
  size = var.models_volume_gb
}

resource "openstack_compute_volume_attach_v2" "models" {
  instance_id = openstack_compute_instance_v2.automl.id
  volume_id   = openstack_blockstorage_volume_v3.models.id
}

resource "openstack_objectstorage_container_v1" "models" {
  name = "${var.prefix}-artifacts"
}

# ---------------------------------------------------------------- floating IP
resource "openstack_networking_floatingip_v2" "automl" {
  pool = var.external_network
}

resource "openstack_networking_floatingip_associate_v2" "automl" {
  floating_ip = openstack_networking_floatingip_v2.automl.address
  port_id     = openstack_networking_port_v2.automl.id
}
