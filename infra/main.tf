locals {
  pipeline_config = yamldecode(file("${path.module}/../config.yaml"))
  brands          = { for brand in local.pipeline_config.brands : brand.id => brand }

  buckets = {
    training           = "${var.bucket_prefix}-training"
    model-artifacts    = "${var.bucket_prefix}-model-artifacts"
    pipeline-artifacts = "${var.bucket_prefix}-pipeline-artifacts"
    raw-archive        = "${var.bucket_prefix}-raw-archive"
  }

  # Each brand ships its own fully-rendered definition file. Terraform selects
  # the per-brand definition by filename, so brand-specific tokens such as
  # BrandNamingPattern are never cloned from a single-brand baseline.
  per_brand_pipeline_definitions = {
    for brand_id in keys(local.brands) :
    brand_id => file("${path.module}/pipeline_definition.${brand_id}.json")
  }
}

resource "aws_s3_bucket" "artifacts" {
  for_each = local.buckets

  bucket        = each.value
  force_destroy = false

  # Object Lock protects the published code bundles against overwrite/delete
  # by any principal (including the CI publisher) once a retention is applied
  # (CWE-494). It is enabled at bucket-creation time; retention policy for the
  # ``code@sha256=*`` prefixes should be configured out-of-band per bundle.
  object_lock_enabled = each.key == "pipeline-artifacts"

  tags = {
    Name    = each.value
    Purpose = each.key
  }
}

resource "aws_s3_bucket_public_access_block" "artifacts" {
  for_each = aws_s3_bucket.artifacts

  bucket                  = each.value.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_ownership_controls" "artifacts" {
  for_each = aws_s3_bucket.artifacts

  bucket = each.value.id
  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  for_each = aws_s3_bucket.artifacts

  bucket = each.value.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# No lifecycle transition rules are defined: all buckets retain the S3 Standard
# storage class. Versioning is enabled for the model-artifacts bucket (model
# traceability) and the pipeline-artifacts bucket (code-bundle immutability --
# CWE-494: an overwrite of a ``code@sha256=*`` prefix is otherwise
# unrecoverable and untraceable).
resource "aws_s3_bucket_versioning" "model_artifacts" {
  bucket = aws_s3_bucket.artifacts["model-artifacts"].id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_versioning" "pipeline_artifacts" {
  bucket = aws_s3_bucket.artifacts["pipeline-artifacts"].id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_sagemaker_model_package_group" "brand" {
  for_each = local.brands

  model_package_group_name        = each.value.model_package_group
  model_package_group_description = "Miso food-classifier models for ${each.key}."

  tags = {
    Brand = each.key
  }
}

resource "aws_sagemaker_pipeline" "brand" {
  for_each = local.brands

  pipeline_name         = "${var.pipeline_name_prefix}-${each.key}"
  pipeline_display_name = "${var.pipeline_name_prefix}-${each.key}"
  role_arn              = local.effective_role_arn
  pipeline_definition   = local.per_brand_pipeline_definitions[each.key]

  depends_on = [
    aws_sagemaker_model_package_group.brand,
    aws_s3_bucket_versioning.model_artifacts,
    aws_s3_bucket_versioning.pipeline_artifacts,
  ]

  tags = {
    Brand = each.key
  }
}
