resource "google_sql_database_instance" "arena_pg" {
  name             = "crowded-lineup-pg-${var.environment}"
  database_version = "POSTGRES_16"
  region           = var.region
  depends_on       = [google_service_networking_connection.private_vpc_connection]

  settings {
    tier = var.db_tier
    ip_configuration {
      ipv4_enabled                                  = false
      private_network                               = google_compute_network.arena_vpc.id
      enable_private_path_for_google_cloud_services = true
    }
    backup_configuration {
      enabled                        = true
      point_in_time_recovery_enabled = true
    }
  }

  deletion_protection = true
}

resource "google_sql_database" "arena_db" {
  name     = "crowded_lineup"
  instance = google_sql_database_instance.arena_pg.name
}

resource "google_storage_bucket" "raw_telemetry_and_corpus" {
  name                        = "${var.project_id}-corpus-telemetry-${var.environment}"
  location                    = var.region
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }
}
