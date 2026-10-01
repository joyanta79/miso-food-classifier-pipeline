"""Export a SageMaker pipeline definition from the shared configuration."""

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
)

DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "infra" / "pipeline_definition.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the Miso Robotics SageMaker pipeline definition."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the pipeline YAML configuration.",
    )
    parser.add_argument(
        "--brand",
        default="white-castle",
        help="Declared brand to export (default: white-castle).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Destination JSON definition path.",
    )
    parser.add_argument(
        "--sdk",
        action="store_true",
        help="Use the optional SageMaker SDK to emit its native definition instead of local dry-run JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    get_brand(config, args.brand)  # Produce an actionable error before writing anything.

    if args.sdk:
        definition = build_sagemaker_pipeline(args.config, args.brand).definition()
        content = definition if definition.endswith("\n") else f"{definition}\n"
    else:
        content = render_dry_run_json(args.config, args.brand)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
