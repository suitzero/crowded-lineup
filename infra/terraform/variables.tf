variable "project_id" {
  description = "GCP Project ID hosting Crowded Lineup Data & Intelligence Platform"
  type        = string
  default     = "crowded-lineup-prod"
}

variable "project_number" {
  description = "Numeric GCP Project ID for VPC Service Controls perimeter"
  type        = string
  default     = "123456789012"
}

variable "access_policy_id" {
  description = "Access Context Manager Policy ID for VPC-SC"
  type        = string
  default     = "987654321"
}

variable "region" {
  description = "Primary GCP region"
  type        = string
  default     = "us-west1"
}

variable "environment" {
  description = "Deployment environment label"
  type        = string
  default     = "prod"
}

variable "db_tier" {
  description = "Cloud SQL Postgres machine tier"
  type        = string
  default     = "db-custom-2-7680"
}
