# SageMaker pipeline execution role.
#
# Created here so a real deployment has an execution identity. It is scoped to the
# four project buckets, the brand Model Package Groups, and the SageMaker/logging
# actions pipeline jobs need -- not AdministratorAccess. Toggle creation off with
# create_sagemaker_role=false to instead pass a pre-reviewed role via
# sagemaker_role_arn (variables.tf).

data "aws_caller_identity" "current" {}

locals {
  create_role        = var.create_sagemaker_role
  effective_role_arn = local.create_role ? aws_iam_role.sagemaker[0].arn : var.sagemaker_role_arn
  bucket_arns = concat(
    [for b in aws_s3_bucket.artifacts : b.arn],
    [for b in aws_s3_bucket.artifacts : "${b.arn}/*"],
  )
}

data "aws_iam_policy_document" "sagemaker_assume" {
  count = local.create_role ? 1 : 0

  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["sagemaker.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "sagemaker" {
  count = local.create_role ? 1 : 0

  name               = "${var.pipeline_name_prefix}-sagemaker-exec"
  assume_role_policy = data.aws_iam_policy_document.sagemaker_assume[0].json

  tags = {
    Purpose = "sagemaker-pipeline-execution"
  }
}

data "aws_iam_policy_document" "sagemaker_permissions" {
  count = local.create_role ? 1 : 0

  # Bucket-scoped object access for the four project buckets only.
  statement {
    sid = "ProjectBucketAccess"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject",
      "s3:ListBucket",
      "s3:GetBucketLocation",
    ]
    resources = local.bucket_arns
  }

  # SageMaker jobs, pipelines, and registry within this account/region.
  statement {
    sid = "SageMakerJobs"
    actions = [
      "sagemaker:CreateProcessingJob",
      "sagemaker:CreateTrainingJob",
      "sagemaker:CreateModel",
      "sagemaker:CreateModelPackage",
      "sagemaker:DescribeProcessingJob",
      "sagemaker:DescribeTrainingJob",
      "sagemaker:DescribeModelPackage",
      "sagemaker:DescribeModelPackageGroup",
      "sagemaker:ListModelPackages",
      "sagemaker:AddTags",
      "sagemaker:StartPipelineExecution",
      "sagemaker:DescribePipelineExecution",
    ]
    resources = ["arn:aws:sagemaker:${var.aws_region}:${data.aws_caller_identity.current.account_id}:*"]
  }

  # CloudWatch Logs for job output.
  statement {
    sid = "Logs"
    actions = [
      "logs:CreateLogGroup",
      "logs:CreateLogStream",
      "logs:PutLogEvents",
      "logs:DescribeLogStreams",
    ]
    resources = ["arn:aws:logs:${var.aws_region}:${data.aws_caller_identity.current.account_id}:log-group:/aws/sagemaker/*"]
  }

  # ECR pull for the AWS-managed Deep Learning Containers.
  statement {
    sid = "EcrPull"
    actions = [
      "ecr:GetAuthorizationToken",
      "ecr:BatchCheckLayerAvailability",
      "ecr:GetDownloadUrlForLayer",
      "ecr:BatchGetImage",
    ]
    resources = ["*"]
  }

  # Publish drift/notification events.
  statement {
    sid       = "PublishNotifications"
    actions   = ["sns:Publish"]
    resources = ["arn:aws:sns:${var.aws_region}:${data.aws_caller_identity.current.account_id}:*"]
  }
}

resource "aws_iam_role_policy" "sagemaker" {
  count = local.create_role ? 1 : 0

  name   = "${var.pipeline_name_prefix}-sagemaker-exec"
  role   = aws_iam_role.sagemaker[0].id
  policy = data.aws_iam_policy_document.sagemaker_permissions[0].json
}
