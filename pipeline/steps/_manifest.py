"""In-container integrity check for the mounted step-code bundle.

Security contract (CWE-494: Download of Code Without Integrity Check):
    * Every step entrypoint calls :func:`verify_mounted_code` BEFORE adding
      the mounted directory to ``sys.path``. The function fails closed: a
      missing manifest, a wrong hash, an unexpected file, or a sentinel /
      malformed digest raises and the step exits non-zero.
    * The expected top-level content digest is read from the environment
      variable ``MISO_CODE_MANIFEST_DIGEST`` (baked into the pipeline step
      Environment by ``pipeline.pipeline.build_dry_run_definition``).
    * The manifest is ``MANIFEST.sha256`` -- a text file with one
      ``<64hex>  <relative-path>`` entry per bundled file. Its own sha256
      must equal the top-level digest, so a single compare pins both the
      manifest and (transitively) every file it enumerates.
    * The all-zero sentinel digest is explicitly rejected. It is the
      placeholder shipped in ``config.yaml`` so dry-run rendering is
      deterministic; a real deployment publishes a bundle under a real
      digest-addressed prefix and records that digest in config.

Python 3.9+ compatible (SageMaker DLC images ship 3.9 or newer).
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Iterable


_SENTINEL_DIGEST = "0" * 64
_MANIFEST_FILENAME = "MANIFEST.sha256"
_FRAMING_FILES = frozenset({_MANIFEST_FILENAME, _MANIFEST_FILENAME + ".sig"})


class CodeIntegrityError(RuntimeError):
    """Raised when the mounted code bundle fails its integrity check."""


def _is_valid_digest(digest):
    if not isinstance(digest, str) or len(digest) != 64:
        return False
    try:
        int(digest, 16)
    except ValueError:
        return False
    return digest == digest.lower()


def _sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_manifest(text):
    entries = {}
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            raise CodeIntegrityError(
                "malformed manifest entry on line %d: %r" % (lineno, raw)
            )
        hex_digest, rel = parts[0], parts[1].lstrip("*").strip()
        if not _is_valid_digest(hex_digest):
            raise CodeIntegrityError(
                "manifest entry on line %d has invalid digest: %r" % (lineno, hex_digest)
            )
        if (
            not rel
            or rel.startswith("/")
            or ".." in Path(rel).parts
            or rel in _FRAMING_FILES
            or "__pycache__" in Path(rel).parts
            or rel.endswith(".pyc")
        ):
            raise CodeIntegrityError(
                "manifest entry on line %d has unsafe path: %r" % (lineno, rel)
            )
        if rel in entries:
            raise CodeIntegrityError(
                "manifest entry on line %d duplicates path %r" % (lineno, rel)
            )
        entries[rel] = hex_digest
    if not entries:
        raise CodeIntegrityError("manifest lists no files")
    return entries


def _iter_bundle_files(root):
    for path in root.rglob("*"):
        if path.is_symlink():
            raise CodeIntegrityError(
                "bundle contains a symlink which cannot be integrity-checked: %s" % path
            )
        if path.is_file():
            # __pycache__ byte-compile artifacts are produced by the Python
            # runtime itself during import; they are never part of a
            # published bundle, so ignore them when enumerating the mount.
            if "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            yield path


def _expected_digest_from_env():
    digest = os.environ.get("MISO_CODE_MANIFEST_DIGEST", "")
    if not _is_valid_digest(digest):
        raise CodeIntegrityError(
            "MISO_CODE_MANIFEST_DIGEST is not a 64-char lowercase sha256 hex "
            "digest; refusing to execute mounted code (CWE-494)."
        )
    if digest == _SENTINEL_DIGEST:
        raise CodeIntegrityError(
            "MISO_CODE_MANIFEST_DIGEST is the all-zero template sentinel. "
            "Publish the step-code bundle to its digest-addressed prefix and "
            "set s3.code_manifest_digest in config.yaml before executing jobs."
        )
    return digest


def verify_mounted_code(mount):
    """Verify the mounted code bundle against MISO_CODE_MANIFEST_DIGEST.

    Raises CodeIntegrityError and does NOT add anything to sys.path on
    failure. The caller is responsible for exiting the process (fail-closed)
    when this raises.
    """
    expected_digest = _expected_digest_from_env()
    root = Path(mount).resolve()
    if not root.is_dir():
        raise CodeIntegrityError(
            "code mount %s is not a directory; refusing to import from it" % root
        )
    manifest_path = root / _MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise CodeIntegrityError(
            "code mount %s is missing %s; the CI publisher must emit it "
            "alongside the bundled .py files." % (root, _MANIFEST_FILENAME)
        )
    manifest_bytes = manifest_path.read_bytes()
    manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_digest != expected_digest:
        raise CodeIntegrityError(
            "%s sha256 %r does not match expected code-channel digest %r; "
            "the mounted bundle is not the one pinned in config.yaml."
            % (_MANIFEST_FILENAME, manifest_digest, expected_digest)
        )
    expected_entries = _parse_manifest(manifest_bytes.decode("utf-8"))

    actual = {}
    for file_path in _iter_bundle_files(root):
        rel = file_path.relative_to(root).as_posix()
        if rel in _FRAMING_FILES:
            continue
        actual[rel] = _sha256_file(file_path)

    missing = sorted(set(expected_entries) - set(actual))
    if missing:
        raise CodeIntegrityError(
            "manifest references files that are not mounted: %s" % missing
        )
    extra = sorted(set(actual) - set(expected_entries))
    if extra:
        raise CodeIntegrityError(
            "mounted bundle contains files not in the manifest: %s; refuse to "
            "import an un-pinned module (CWE-494)." % extra
        )
    bad = sorted(
        rel for rel, hexd in actual.items() if expected_entries[rel] != hexd
    )
    if bad:
        raise CodeIntegrityError(
            "mounted files whose sha256 does not match the manifest: %s" % bad
        )


def verify_and_insert_path(mount):
    """Verify ``mount`` then prepend it to sys.path.

    On failure the function raises CodeIntegrityError; the caller must let
    the exception propagate so the step exits non-zero without importing
    anything from the un-trusted mount.
    """
    verify_mounted_code(mount)
    resolved = str(Path(mount).resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
