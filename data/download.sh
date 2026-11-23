#!/usr/bin/env bash
# Download the InquireMobile dataset (JSONL + screenshots) from Hugging Face
# into data/InquireMobile/.
#
# Prerequisites:  pip install -U huggingface_hub
# Usage:          bash data/download.sh            # everything (~6k screenshots)
#                 bash data/download.sh --no-images # JSONL files only

set -euo pipefail

REPO_ID="PG23/InquireMobile"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${HERE}/InquireMobile"

ARGS=()
if [[ "${1:-}" == "--no-images" ]]; then
    ARGS+=(--include "data/*" --include "README.md")
fi

if command -v hf >/dev/null 2>&1; then
    hf download "${REPO_ID}" --repo-type dataset --local-dir "${TARGET}" ${ARGS[@]+"${ARGS[@]}"}
else
    huggingface-cli download "${REPO_ID}" --repo-type dataset --local-dir "${TARGET}" ${ARGS[@]+"${ARGS[@]}"}
fi

echo "Dataset downloaded to ${TARGET}"
