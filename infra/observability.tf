# Audit / observability resources that back the structured logs emitted by
# ``lambdas/handlers.py``.
#
# The handlers emit one JSON record per security decision through a logger
# named ``lambdas.handlers``. In production that stream lands in the Lambda
# function CloudWatch Log Group. Because the Lambda functions themselves are
# declared outside this stack today, the log group this module provisions is
# a dedicated per-pipeline audit stream the Lambda deployment is expected to
# write to via the ``MISO_AUDIT_LOG_GROUP`` environment variable, or that an
# operator wires in as the log destination for the trigger Lambdas.
#
# Two controls back the records:
#   1. A CloudWatch metric filter + alarm on repeated authorization denials
#      per principal, so cross-tenant enumeration against
#      ``manual_trigger_handler`` paints a signal instead of being silent.
#   2. A CloudTrail trail that records management + S3 data events for the
#      four project buckets, so S3 uploads driving ``s3_trigger_handler`` are
#      attributable to a caller identity (the handler fnmatch-only brand
#      attribution never left a durable trace).
#
# S3 server-access logging is enabled on the four project buckets, writing
# into a dedicated access-logs bucket.

variable "audit_log_retention_days" {
  description = "CloudWatch log group retention for the lambdas/handlers audit stream."
  type        = number
  default     = 365
}

variable "audit_authorization_denied_alarm_threshold" {
  description = "Number of authorization denials in a 5-minute window that triggers the alarm."
  type        = number
  default     = 5
}

variable "audit_notification_topic_arn" {
  description = "Optional SNS topic to notify on the audit alarm. Leave empty to disable notifications."
  type        = string
  default     = ""

  validation {
    condition     = var.audit_notification_topic_arn == "" || can(regex("^arn:[^:]+:sns:", var.audit_notification_topic_arn))
    error_message = "audit_notification_topic_arn must be empty or an SNS topic ARN."
  }
}

locals {
  audit_log_group_name = "/miso/lambdas/handlers-audit"
}

resource "aws_cloudwatch_log_group" "lambda_audit" {
  name              = local.audit_log_group_name
  retention_in_days = var.audit_log_retention_days

  tags = {
    Purpose = "lambda-handlers-audit"
  }
}

resource "aws_cloudwatch_log_metric_filter" "authorization_denied" {
  name           = "miso-lambda-authorization-denied"
  log_group_name = aws_cloudwatch_log_group.lambda_audit.name

  pattern = "{ ($.event = \"miso.audit.authorization\") && ($.outcome = \"denied\") }"

  metric_transformation {
    name          = "AuthorizationDenied"
    namespace     = "Miso/Lambdas"
    value         = "1"
    default_value = "0"
    unit          = "Count"
  }
}

resource "aws_cloudwatch_log_metric_filter" "authentication_denied" {
  name           = "miso-lambda-authentication-denied"
  log_group_name = aws_cloudwatch_log_group.lambda_audit.name

  pattern = "{ ($.event = \"miso.audit.authentication\") && ($.outcome = \"denied\") }"

  metric_transformation {
    name          = "AuthenticationDenied"
    namespace     = "Miso/Lambdas"
    value         = "1"
    default_value = "0"
    unit          = "Count"
  }
}

resource "aws_cloudwatch_metric_alarm" "authorization_denied_burst" {
  alarm_name          = "miso-lambda-authorization-denied-burst"
  alarm_description   = "Repeated authorization denials in manual_trigger_handler. Possible cross-tenant enumeration."
  comparison_operator = "GreaterThanOrEqualToThreshold"
  evaluation_periods  = 1
  threshold           = var.audit_authorization_denied_alarm_threshold
  period              = 300
  statistic           = "Sum"
  metric_name         = aws_cloudwatch_log_metric_filter.authorization_denied.metric_transformation[0].name
  namespace           = aws_cloudwatch_log_metric_filter.authorization_denied.metric_transformation[0].namespace
  treat_missing_data  = "notBreaching"

  alarm_actions = var.audit_notification_topic_arn == "" ? [] : [var.audit_notification_topic_arn]
  ok_actions    = var.audit_notification_topic_arn == "" ? [] : [var.audit_notification_topic_arn]
}

resource "aws_s3_bucket" "access_logs" {
  bucket        = "${var.bucket_prefix}-access-logs"
  force_destroy = false

  tags = {
    Name    = "${var.bucket_prefix}-access-logs"
    Purpose = "s3-server-access-logs"
  }
}

resource "aws_s3_bucket_public_access_block" "access_logs" {
  bucket                  = aws_s3_bucket.access_logs.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_ownership_controls" "access_logs" {
  bucket = aws_s3_bucket.access_logs.id
  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "access_logs" {
  bucket = aws_s3_bucket.access_logs.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_logging" "artifacts" {
  for_each = aws_s3_bucket.artifacts

  bucket        = each.value.id
  target_bucket = aws_s3_bucket.access_logs.id
  target_prefix = "s3-access/${each.key}/"
}

resource "aws_s3_bucket" "cloudtrail" {
  bucket        = "${var.bucket_prefix}-cloudtrail"
  force_destroy = false

  tags = {
    Name    = "${var.bucket_prefix}-cloudtrail"
    Purpose = "cloudtrail-trail"
  }
}

resource "aws_s3_bucket_public_access_block" "cloudtrail" {
  bucket                  = aws_s3_bucket.cloudtrail.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_ownership_controls" "cloudtrail" {
  bucket = aws_s3_bucket.cloudtrail.id
  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "cloudtrail" {
  bucket = aws_s3_bucket.cloudtrail.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

data "aws_iam_policy_document" "cloudtrail_bucket" {
  statement {
    sid       = "AWSCloudTrailAclCheck"
    effect    = "Allow"
    actions   = ["s3:GetBucketAcl"]
    resources = [aws_s3_bucket.cloudtrail.arn]
    principals {
      type        = "Service"
      identifiers = ["cloudtrail.amazonaws.com"]
    }
    condition {
      test     = "StringEquals"
      variable = "aws:SourceArn"
      values   = ["arn:aws:cloudtrail:${var.aws_region}:${data.aws_caller_identity.current.account_id}:trail/${var.pipeline_name_prefix}-trail"]
    }
  }

  statement {
    sid       = "AWSCloudTrailWrite"
    effect    = "Allow"
    actions   = ["s3:PutObject"]
    resources = ["${aws_s3_bucket.cloudtrail.arn}/AWSLogs/${data.aws_caller_identity.current.account_id}/*"]
    principals {
      type        = "Service"
      identifiers = ["cloudtrail.amazonaws.com"]
    }
    condition {
      test     = "StringEquals"
      variable = "s3:x-amz-acl"
      values   = ["bucket-owner-full-control"]
    }
    condition {
      test     = "StringEquals"
      variable = "aws:SourceArn"
      values   = ["arn:aws:cloudtrail:${var.aws_region}:${data.aws_caller_identity.current.account_id}:trail/${var.pipeline_name_prefix}-trail"]
    }
  }
}

resource "aws_s3_bucket_policy" "cloudtrail" {
  bucket = aws_s3_bucket.cloudtrail.id
  policy = data.aws_iam_policy_document.cloudtrail_bucket.json
}

resource "aws_cloudtrail" "main" {
  name                          = "${var.pipeline_name_prefix}-trail"
  s3_bucket_name                = aws_s3_bucket.cloudtrail.id
  include_global_service_events = true
  is_multi_region_trail         = true
  enable_log_file_validation    = true

  event_selector {
    read_write_type           = "All"
    include_management_events = true

    data_resource {
      type   = "AWS::S3::Object"
      values = [for b in aws_s3_bucket.artifacts : "${b.arn}/"]
    }
  }

  depends_on = [aws_s3_bucket_policy.cloudtrail]

  tags = {
    Purpose = "audit-trail"
  }
}
