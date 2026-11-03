#!/usr/bin/env bash
# CI drift check: fail if any pip manifest and its committed .lock disagree.
#
# How it works:
#   * For every <manifest>.txt in requirements.txt, training/requirements.txt
#     and processing/requirements/step*.txt, require a sibling <manifest>.lock
#     that was produced by `scripts/compile_locks.sh`.
#   * Re-run `pip-compile --generate-hashes` into a scratch file and diff it
#     against the committed .lock. Any difference (new transitive, moved
#     version, missing hash, dropped artifact) fails the job.
#   * `--index-url` MUST point at the private mirror (PIP_INDEX_URL). The
#     check cannot "pass" by silently resolving against public PyPI.
#
# Usage:
#   PIP_INDEX_URL=https://pypi.example.internal/simple/ scripts/check_lock_drift.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -z "${PIP_INDEX_URL:-}" ]]; then
    echo "ERROR: PIP_INDEX_URL must be set to a private, write-restricted mirror." >&2
    exit 2
fi

PYTHON="${PYTHON:-python3}"
VENV_DIR="${REPO_ROOT}/.venv-piptools"
PIP_TOOLS_PIN="pip-tools==7.4.1"

if [[ ! -x "${VENV_DIR}/bin/pip-compile" ]]; then
    "${PYTHON}" -m venv "${VENV_DIR}"
    "${VENV_DIR}/bin/pip" install --quiet --upgrade pip
    "${VENV_DIR}/bin/pip" install --quiet --index-url "${PIP_INDEX_URL}" "${PIP_TOOLS_PIN}"
fi

PIP_COMPILE="${VENV_DIR}/bin/pip-compile"
SCRATCH="$(mktemp -d)"
trap 'rm -rf "${SCRATCH}"' EXIT

MANIFESTS=(
    "requirements.txt"
    "build-requirements.txt"
    "training/requirements.txt"
)
while IFS= read -r -d '' f; do
    MANIFESTS+=("${f#./}")
done < <(find ./processing/requirements -maxdepth 1 -name 'step*.txt' -print0 | sort -z)

STATUS=0
for src in "${MANIFESTS[@]}"; do
    lock="${src%.txt}.lock"
    if [[ ! -f "${lock}" ]]; then
        echo "DRIFT: ${lock} is missing. Run scripts/compile_locks.sh and commit it." >&2
        STATUS=1
        continue
    fi
    regen="${SCRATCH}/$(echo "${src}" | tr '/' '_').lock"
    "${PIP_COMPILE}" \
        --quiet \
        --generate-hashes \
        --allow-unsafe \
        --no-header \
        --index-url "${PIP_INDEX_URL}" \
        --output-file "${regen}" \
        "${src}"
    if ! diff -u "${lock}" "${regen}"; then
        echo "DRIFT: ${src} and ${lock} disagree." >&2
        echo "       Re-run scripts/compile_locks.sh and commit the updated ${lock}." >&2
        STATUS=1
    fi
done

if [[ "${STATUS}" -ne 0 ]]; then
    echo "" >&2
    echo "Dependency lockfile drift detected. See messages above." >&2
    exit 1
fi
echo "All manifests match their committed lockfiles."
