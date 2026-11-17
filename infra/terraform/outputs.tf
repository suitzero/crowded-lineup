output "api_service_uri" {
  value = google_cloud_run_v2_service.api_service.uri
}

output "corpus_bucket_name" {
  value = google_storage_bucket.raw_telemetry_and_corpus.name
}

output "telemetry_pubsub_topic" {
  value = google_pubsub_topic.telemetry_events.id
}

output "postgres_private_ip" {
  value = google_sql_database_instance.arena_pg.private_ip_address
}
