#!/usr/bin/env bash
set -euo pipefail

OMNI_DIR="${HOME}/.cache/omnivoice"
VOICE_SAMPLES_DIR="${OMNI_DIR}/voice_samples"
mkdir -p "${OMNI_DIR}" "${VOICE_SAMPLES_DIR}"

# 1. Seed repository voice assets if present
if [ -f "assets/voice_preview_mark - cartoonish, funny and cheerful.mp3" ]; then
  cp -f "assets/voice_preview_mark - cartoonish, funny and cheerful.mp3" "${VOICE_SAMPLES_DIR}/voice_preview_mark - cartoonish, funny and cheerful.mp3"
  cp -f "assets/voice_preview_mark - cartoonish, funny and cheerful.mp3" "${OMNI_DIR}/voice_preview_mark - cartoonish, funny and cheerful.mp3"
  echo "✓ Seeded voice_preview_mark - cartoonish, funny and cheerful.mp3 into cache."
fi
if [ -f "assets/reference.wav" ]; then
  cp -f "assets/reference.wav" "${VOICE_SAMPLES_DIR}/reference.wav"
  cp -f "assets/reference.txt" "${VOICE_SAMPLES_DIR}/reference.txt"
  echo "✓ Pre-seeded reference sample from repository assets."
fi

# 2. Ensure reference voice sample is pinned in cache via ModelCacheManager
python3 -c "
import sys; sys.path.insert(0, 'src')
from cache_manager import ModelCacheManager
mgr = ModelCacheManager()
p = mgr.ensure_voice_sample_cached()
print(f'Pinned voice sample confirmed: {p}')
"

# 3. Strict verification of reference audio file size
REF_FILE="${VOICE_SAMPLES_DIR}/reference.wav"
if [ ! -f "${REF_FILE}" ] || [ $(stat -c%s "${REF_FILE}" 2>/dev/null || stat -f%z "${REF_FILE}") -lt 50000 ]; then
  echo "❌ INTEGRITY ERROR: Reference voice sample missing or too small at ${REF_FILE}"
  exit 1
fi

echo "✅ k2-fsa/OmniVoice pinned voice sample seeded and verified successfully."
