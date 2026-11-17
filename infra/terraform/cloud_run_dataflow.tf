resource "google_pubsub_topic" "telemetry_events" {
  name = "crowded-lineup-telemetry-v2-${var.environment}"
}

resource "google_pubsub_subscription" "telemetry_stream_sub" {
  name  = "crowded-lineup-stream-sub-${var.environment}"
  topic = google_pubsub_topic.telemetry_events.name

  ack_deadline_seconds       = 20
  retain_acked_messages      = false
  message_retention_duration = "604800s"
}

resource "google_secret_manager_secret" "db_dsn" {
  secret_id = "crowded-lineup-db-dsn-${var.environment}"
  replication {
    auto {}
  }
}

resource "google_cloud_run_v2_service" "api_service" {
  name     = "crowded-lineup-api-${var.environment}"
  location = var.region

  template {
    service_account = google_service_account.api_sa.email
    scaling {
      min_instance_count = 1
      max_instance_count = 10
    }
    vpc_access {
      network_interfaces {
        network    = google_compute_network.arena_vpc.name
        subnetwork = google_compute_subnetwork.arena_subnet.name
      }
      egress = "PRIVATE_RANGES_ONLY"
    }
    containers {
      image = "us-west1-docker.pkg.dev/${var.project_id}/crowded-lineup/api:latest"
      ports {
        container_port = 8000
      }
      env {
        name  = "CROWDED_ENVIRONMENT"
        value = var.environment
      }
      env {
        name = "CROWDED_DATABASE_URL"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.db_dsn.secret_id
            version = "latest"
          }
        }
      }
    }
  }
}
