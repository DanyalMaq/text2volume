#!/usr/bin/env bash
set -euo pipefail

FINDINGS_FILE="eval/valid/overall_findings.txt"
export MONAI_DATA_DIRECTORY="./temp_work_dir"

BASE_INFER_JSON="configs/config_infer_chest_text.json"
TMP_DIR="configs/tmp_infer_prompts"

mkdir -p "$TMP_DIR"

i=1

while IFS= read -r finding || [[ -n "$finding" ]]; do
  # Skip empty lines
  [[ -z "$finding" ]] && continue

  tmp_json="$TMP_DIR/config_infer_chest_text_${i}.json"

  jq --arg prompt "$finding" \
    '.text_prompt = $prompt' \
    "$BASE_INFER_JSON" > "$tmp_json"

  echo "Running inference $i with prompt:"
  echo "$finding"

  python -m scripts.inference \
    -t configs/config_network_rflow_text.json \
    -e configs/environment_maisi_infer_chest_text.json \
    -i "$tmp_json" \
    --version rflow-ct

  rm -rf /tmp

  i=$((i + 1))

done < "$FINDINGS_FILE"