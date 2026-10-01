output "bucket_names" {
  description = "S3 Standard bucket names. Only model-artifacts has versioning enabled."
  value       = { for name, bucket in aws_s3_bucket.artifacts : name => bucket.bucket }
}

output "pipeline_names" {
  description = "SageMaker pipeline name for each configured brand."
  value       = { for brand, pipeline in aws_sagemaker_pipeline.brand : brand => pipeline.pipeline_name }
}

output "model_package_groups" {
  description = "Brand-specific SageMaker Model Registry package groups."
  value = {
    for brand, group in aws_sagemaker_model_package_group.brand : brand => group.model_package_group_name
  }
}

output "sagemaker_role_arn" {
  description = "Execution role used by the SageMaker pipelines (created here or provided)."
  value       = local.effective_role_arn
}
