"""Export a SageMaker pipeline definition from the shared configuration.

By default, the exporter writes one definition file per brand declared in
``config.yaml`` to ``infra/pipeline_definition.<brand>.json``. Terraform
consumes the matching per-brand file directly, so no textual rewriting of a
single-brand baseline is required. This avoids leaking per-brand tokens
(for example ``BrandNamingPattern``) from the baseline into other brands.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.pipeline import (
    DEFAULT_CONFIG_PATH,
    build_sagemaker_pipeline,
    get_brand,
    load_config,
    render_dry_run_json,
    validate_sdk_pipeline_definition,
)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "infra"
DEFAULT_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / "pipeline_definition.json"


def per_brand_definition_path(output_dir: Path, brand_id: str) -> Path:
    """Return the per-brand definition path consumed by Terraform."""
    return output_dir / f"pipeline_definition.{brand_id}.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the Miso Robotics SageMaker pipeline definition.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the pipeline YAML configuration.",
    )
    parser.add_argument(
        "--brand",
        default=None,
        help=(
            "Export a single declared brand. When omitted, the exporter writes "
            "one definition file per brand declared in config.yaml."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Destination JSON path for single-brand exports. Ignored when "
            "exporting all brands; in that mode, --output-dir controls the "
            "destination directory (default: infra/)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Destination directory for per-brand definition files "
            "(default: infra/)."
        ),
    )
    parser.add_argument(
        "--sdk",
        action="store_true",
        help=(
            "Use the optional SageMaker SDK to emit its native definition "
            "instead of local dry-run JSON."
        ),
    )
    return parser.parse_args()


def _render(config_path: Path, brand_id: str, *, use_sdk: bool) -> str:
    if use_sdk:
        definition = build_sagemaker_pipeline(config_path, brand_id).definition()
        # Fail loudly if the SDK graph has drifted from the validated local DAG
        # before the operator writes the file / deploys it.
        validate_sdk_pipeline_definition(definition)
        return definition if definition.endswith("\n") else f"{definition}\n"
    return render_dry_run_json(config_path, brand_id)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    if args.brand is not None:
        # Single-brand export. Validate the brand before writing anything.
        get_brand(config, args.brand)
        content = _render(args.config, args.brand, use_sdk=args.sdk)
        output = args.output or per_brand_definition_path(args.output_dir, args.brand)
        _write(output, content)
        return 0

    # Multi-brand export: one definition file per brand declared in config.yaml.
    for brand in config["brands"]:
        brand_id = brand["id"]
        get_brand(config, brand_id)
        content = _render(args.config, brand_id, use_sdk=args.sdk)
        output = per_brand_definition_path(args.output_dir, brand_id)
        _write(output, content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
