resource "google_compute_network" "arena_vpc" {
  name                    = "crowded-lineup-vpc-${var.environment}"
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "arena_subnet" {
  name                     = "crowded-lineup-subnet-${var.environment}"
  ip_cidr_range            = "10.20.0.0/20"
  region                   = var.region
  network                  = google_compute_network.arena_vpc.id
  private_ip_google_access = true
}

resource "google_compute_global_address" "private_service_range" {
  name          = "crowded-lineup-psa-${var.environment}"
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  prefix_length = 16
  network       = google_compute_network.arena_vpc.id
}

resource "google_service_networking_connection" "private_vpc_connection" {
  network                 = google_compute_network.arena_vpc.id
  service                 = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.private_service_range.name]
}

# VPC Service Controls perimeter isolating private corpus GCS & Cloud SQL from exfiltration
resource "google_access_context_manager_service_perimeter" "corpus_perimeter" {
  parent = "accessPolicies/${var.access_policy_id}"
  name   = "accessPolicies/${var.access_policy_id}/servicePerimeters/crowded_lineup_corpus_${var.environment}"
  title  = "crowded-lineup-corpus-${var.environment}"

  status {
    resources = ["projects/${var.project_number}"]
    restricted_services = [
      "storage.googleapis.com",
      "sqladmin.googleapis.com",
      "secretmanager.googleapis.com",
      "dataflow.googleapis.com",
    ]
  }
}
