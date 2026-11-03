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

  # Buckets whose object content the role may freely create or delete. The
  # model-artifacts bucket is intentionally excluded: its writes are scoped
  # below so sibling jobs cannot plant objects at a brand prefix root and
  # decide what the EvaluateModel step loads.
  bucket_arns_general_write = [
    for key, bucket in aws_s3_bucket.artifacts :
    "${bucket.arn}/*" if key != "model-artifacts"
  ]

  # The model-artifacts bucket is written to by SageMaker Training jobs under
  # ``<brand>/<job-name>/output/*`` (training artifacts) and
  # ``<brand>/checkpoints/*`` (managed-spot checkpoints). Both patterns have at
  # least three ``/``-separated components after the bucket, i.e. they match
  # ``<brand>/*/*``. Restricting the role to those shapes stops a sibling job
  # from writing a file directly at ``<brand>/model.keras`` (two components)
  # that the EvaluateModel step would otherwise load. Per-brand prefixes are
  # enumerated from config.yaml so adding a brand automatically grants writes
  # only under its own namespace.
  model_artifacts_bucket_arn = aws_s3_bucket.artifacts["model-artifacts"].arn
  model_artifacts_write_arns = [
    for brand_id in keys(local.brands) :
    "${local.model_artifacts_bucket_arn}/${brand_id}/*/*"
  ]
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

  # Bucket-scoped object access for the four project buckets only. Read and
  # list operations stay bucket-wide; writes to the model-artifacts bucket are
  # scoped below so a compromised sibling job cannot plant a loadable
  # ``<brand>/model.keras`` at the brand prefix root for EvaluateModel.
  statement {
    sid = "ProjectBucketRead"
    actions = [
      "s3:GetObject",
      "s3:ListBucket",
      "s3:GetBucketLocation",
    ]
    resources = local.bucket_arns
  }

  statement {
    sid = "ProjectBucketWrite"
    actions = [
      "s3:PutObject",
      "s3:DeleteObject",
    ]
    resources = concat(
      local.bucket_arns_general_write,
      local.model_artifacts_write_arns,
    )
  }

  # SageMaker jobs and registry within this account/region. ``StartPipelineExecution``
  # is split out below and scoped to the specific per-brand pipeline ARNs so a
  # misbehaving caller cannot start arbitrary pipelines in the account (confused
  # deputy).
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
      "sagemaker:DescribePipelineExecution",
    ]
    resources = ["arn:aws:sagemaker:${var.aws_region}:${data.aws_caller_identity.current.account_id}:*"]
  }

  # Start-pipeline is restricted to the per-brand pipelines Terraform creates in
  # this stack. Any other pipeline in the account is out of reach even if a
  # caller manages to influence the ``PipelineName`` passed by the handler.
  statement {
    sid       = "StartBrandPipelines"
    actions   = ["sagemaker:StartPipelineExecution"]
    resources = [for p in aws_sagemaker_pipeline.brand : p.arn]
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
