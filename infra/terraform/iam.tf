resource "google_service_account" "api_sa" {
  account_id   = "crowded-api-${var.environment}"
  display_name = "Crowded Lineup FastAPI Control Plane"
}

resource "google_service_account" "beam_worker_sa" {
  account_id   = "crowded-beam-${var.environment}"
  display_name = "Crowded Lineup Beam/Dataflow Pipeline Worker"
}

resource "google_storage_bucket_iam_member" "api_corpus_reader" {
  bucket = google_storage_bucket.raw_telemetry_and_corpus.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.api_sa.email}"
}

resource "google_storage_bucket_iam_member" "beam_corpus_rw" {
  bucket = google_storage_bucket.raw_telemetry_and_corpus.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.beam_worker_sa.email}"
}

resource "google_project_iam_member" "api_cloudsql_client" {
  project = var.project_id
  role    = "roles/cloudsql.client"
  member  = "serviceAccount:${google_service_account.api_sa.email}"
}

resource "google_project_iam_member" "beam_dataflow_worker" {
  project = var.project_id
  role    = "roles/dataflow.worker"
  member  = "serviceAccount:${google_service_account.beam_worker_sa.email}"
}
