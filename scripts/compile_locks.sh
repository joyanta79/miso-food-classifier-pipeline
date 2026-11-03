#!/usr/bin/env bash
# Generate hash-locked lockfiles for every pip manifest in the repository.
#
# Security contract (CWE-494: Download of Code Without Integrity Check):
#   * Every manifest in the repo (requirements.txt, training/requirements.txt,
#     processing/requirements/step*.txt) is compiled with
#     `pip-compile --generate-hashes --allow-unsafe` so the resulting .lock
#     files pin BOTH the exact version AND the sha256 of every direct and
#     transitive wheel/sdist. `pip install --require-hashes -r <lock>` then
#     refuses to install any artifact the committed digests do not cover.
#   * `--index-url` is required and must point at a private, write-restricted
#     mirror (set PIP_INDEX_URL in the environment). This disables the pip
#     default of silently falling back to the public PyPI index.
#   * The output path for each manifest `<name>.txt` is `<name>.lock` next to
#     it, so a CI drift check can compare manifest/lockfile pairs 1:1.
#
# Usage:
#   PIP_INDEX_URL=https://pypi.example.internal/simple/ scripts/compile_locks.sh
#   PIP_INDEX_URL=https://pypi.example.internal/simple/ scripts/compile_locks.sh --upgrade

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -z "${PIP_INDEX_URL:-}" ]]; then
    echo "ERROR: PIP_INDEX_URL must be set to a private, write-restricted mirror." >&2
    echo "       Falling back to the default public PyPI index is forbidden by the" >&2
    echo "       repository CWE-494 security contract (see pipeline/steps/_bootstrap.py)." >&2
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

EXTRA_ARGS=()
if [[ "${1:-}" == "--upgrade" ]]; then
    EXTRA_ARGS+=(--upgrade)
fi

compile_one() {
    local src="$1"
    local out="${src%.txt}.lock"
    echo ">> compiling ${src} -> ${out}"
    "${PIP_COMPILE}" \
        --quiet \
        --generate-hashes \
        --allow-unsafe \
        --no-header \
        --index-url "${PIP_INDEX_URL}" \
        --output-file "${out}" \
        "${EXTRA_ARGS[@]}" \
        "${src}"
}

MANIFESTS=(
    "requirements.txt"
    "build-requirements.txt"
    "training/requirements.txt"
)
while IFS= read -r -d '' f; do
    MANIFESTS+=("${f#./}")
done < <(find ./processing/requirements -maxdepth 1 -name 'step*.txt' -print0 | sort -z)

for m in "${MANIFESTS[@]}"; do
    compile_one "${m}"
done

echo ""
echo "Done. Commit every *.lock alongside its *.txt. CI (dependency-lock-drift"
echo "job) will fail if the two drift."
