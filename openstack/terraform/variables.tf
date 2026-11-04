variable "prefix" {
  type    = string
  default = "automl"
}

variable "external_network" {
  description = "Имя внешней (provider) сети для floating IP"
  type        = string
  default     = "public"
}

variable "image" {
  type    = string
  default = "Ubuntu-24.04"
}

variable "flavor" {
  description = "Для AutoML на CPU разумно 4-8 vCPU / 8-16 GB RAM"
  type        = string
  default     = "m1.large"
}

variable "cidr" {
  type    = string
  default = "10.20.0.0/24"
}

variable "dns" {
  type    = list(string)
  default = ["8.8.8.8", "1.1.1.1"]
}

variable "admin_cidr" {
  description = "Откуда разрешён SSH"
  type        = string
  default     = "0.0.0.0/0"
}

variable "api_cidr" {
  description = "Откуда разрешён доступ к API :8000"
  type        = string
  default     = "0.0.0.0/0"
}

variable "ssh_public_key" {
  type    = string
  default = "~/.ssh/id_ed25519.pub"
}

variable "models_volume_gb" {
  type    = number
  default = 20
}

variable "repo_url" {
  description = "Git-репозиторий с этим проектом (форк на GitHub/GitLab)"
  type        = string
}

variable "repo_ref" {
  type    = string
  default = "main"
}

variable "max_time_budget" {
  type    = number
  default = 1800
}
