variable "aws_region" {
  description = "AWS Region in which to create the Miso ML resources."
  type        = string
  default     = "us-west-2"
}

variable "bucket_prefix" {
  description = "Globally unique, lowercase prefix for the four S3 buckets."
  type        = string

  validation {
    condition     = can(regex("^[a-z0-9](?:[a-z0-9-]{1,48}[a-z0-9])$", var.bucket_prefix))
    error_message = "bucket_prefix must be 3-50 lowercase letters, digits, or hyphens and cannot end in a hyphen."
  }
}

variable "sagemaker_role_arn" {
  description = "Existing least-privilege IAM role ARN used by SageMaker pipeline executions. Ignored when create_sagemaker_role=true."
  type        = string
  default     = ""

  validation {
    condition     = var.sagemaker_role_arn == "" || can(regex("^arn:[^:]+:iam::[0-9]{12}:role/.+", var.sagemaker_role_arn))
    error_message = "sagemaker_role_arn must be empty or an IAM role ARN."
  }
}

variable "create_sagemaker_role" {
  description = "When true, create a least-privilege SageMaker execution role in this stack instead of requiring sagemaker_role_arn."
  type        = bool
  default     = true
}

variable "pipeline_name_prefix" {
  description = "Prefix used for one SageMaker pipeline per configured brand."
  type        = string
  default     = "miso-food-classifier"
}
