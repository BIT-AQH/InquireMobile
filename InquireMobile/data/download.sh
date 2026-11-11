#!/usr/bin/env bash
# Download the InquireMobile dataset from Hugging Face into ./data.
#
# Prerequisites:  pip install "huggingface_hub[cli]"

set -euo pipefail

REPO_ID="PG23/InquireMobile"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

huggingface-cli download "${REPO_ID}" --repo-type dataset \
    --include "data/*" \
    --local-dir "${HERE}"

echo "Dataset downloaded to ${HERE}/data/"
