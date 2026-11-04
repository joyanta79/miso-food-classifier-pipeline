"""Shared containment helper for manifest-supplied image paths.

Several steps consume per-record image paths from JSON manifests that
transit untrusted S3 artifacts between pipeline steps. The legitimate
producer (``pipeline/steps/data_selection.py``) only emits keys
``relative_to(image_root)``, so a well-formed manifest never carries an
absolute value, a ``..`` segment or a symlink. Any such value must be
rejected before the path is opened or read.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Any

__all__ = ["ManifestPathError", "resolve_manifest_image"]


class ManifestPathError(ValueError):
    """Raised when a manifest record's image path fails the containment check."""


def _record_value(record: dict[str, Any]) -> str | None:
    """Return the first non-empty path-like field from ``record``.

    Keeps the historical ``image_path`` -> ``key`` -> ``id`` fallback order
    so existing manifests continue to resolve.
    """
    for field in ("image_path", "key", "id"):
        value = record.get(field)
        if value is not None and str(value) != "":
            return str(value)
    return None


def resolve_manifest_image(
    image_root: Path | str | None,
    record: dict[str, Any],
    *,
    require_exists: bool = True,
) -> Path | None:
    """Resolve ``record``'s image path safely under ``image_root``.

    Returns the resolved :class:`~pathlib.Path` on success. Returns ``None``
    when the record carries no path-like field, or when ``require_exists``
    is true and the file is absent. Raises :class:`ManifestPathError` on
    any containment violation: absolute values, ``..`` segments, symlinks,
    non-regular files, or targets that resolve outside ``image_root``.

    When ``image_root`` is ``None`` the helper preserves the previous
    permissive behaviour used by unit tests that pass fully-qualified
    ``tmp_path`` values and no root -- in that mode only the symlink /
    non-regular-file check is enforced and no containment is applied,
    because there is no root to contain against. Production callers
    (``embedding_dedup``, ``train_step``, ``evaluation``) always pass a
    concrete ``image_root``.
    """
    raw = _record_value(record)
    if raw is None:
        return None

    normalised = raw.replace("\\", "/")

    if image_root is not None:
        if normalised.startswith("/") or os.path.isabs(normalised):
            raise ManifestPathError(
                f"manifest image path {raw!r} is absolute; refusing to bypass image_root"
            )
        if any(part == ".." for part in normalised.split("/")):
            raise ManifestPathError(
                f"manifest image path {raw!r} contains '..' segments; refusing traversal"
            )
        root = Path(image_root)
        root_real = Path(os.path.realpath(root))
        candidate = root / normalised
        candidate_real = Path(os.path.realpath(candidate))
        try:
            candidate_real.relative_to(root_real)
        except ValueError as exc:
            raise ManifestPathError(
                f"manifest image path {raw!r} resolves outside image_root "
                f"{str(root_real)!r} (resolved to {str(candidate_real)!r})"
            ) from exc
    else:
        candidate = Path(raw)
        candidate_real = Path(os.path.realpath(candidate))

    if require_exists and not candidate_real.exists():
        return None

    if candidate_real.exists():
        if candidate.is_symlink() or candidate_real.is_symlink():
            raise ManifestPathError(
                f"manifest image path {raw!r} is a symlink; refusing to follow"
            )
        try:
            st = candidate_real.lstat()
        except OSError as exc:
            raise ManifestPathError(
                f"manifest image path {raw!r} could not be stat'd: {exc}"
            ) from exc
        if not stat.S_ISREG(st.st_mode):
            raise ManifestPathError(
                f"manifest image path {raw!r} is not a regular file (mode={st.st_mode:o})"
            )

    return candidate_real
