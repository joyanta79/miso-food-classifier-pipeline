locals {
  pipeline_config = yamldecode(file("${path.module}/../config.yaml"))
  brands          = { for brand in local.pipeline_config.brands : brand.id => brand }

  buckets = {
    training           = "${var.bucket_prefix}-training"
    model-artifacts    = "${var.bucket_prefix}-model-artifacts"
    pipeline-artifacts = "${var.bucket_prefix}-pipeline-artifacts"
    raw-archive        = "${var.bucket_prefix}-raw-archive"
  }

  # The committed definition is a white-castle baseline. Terraform replaces only
  # the declared brand and registry group while creating each per-brand pipeline.
  base_pipeline_definition = file("${path.module}/pipeline_definition.json")
}

resource "aws_s3_bucket" "artifacts" {
  for_each = local.buckets

  bucket        = each.value
  force_destroy = false

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
# storage class. Versioning is intentionally enabled for this bucket only.
resource "aws_s3_bucket_versioning" "model_artifacts" {
  bucket = aws_s3_bucket.artifacts["model-artifacts"].id

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
  pipeline_definition = replace(
    replace(
      local.base_pipeline_definition,
      "white-castle",
      each.key,
    ),
    "FoodClassifier-WhiteCastle",
    each.value.model_package_group,
  )

  depends_on = [
    aws_sagemaker_model_package_group.brand,
    aws_s3_bucket_versioning.model_artifacts,
  ]

  tags = {
    Brand = each.key
  }
}
