#!/usr/bin/env bash
set -euo pipefail

K2FSA_DIR="${HOME}/.cache/k2-fsa"
OMNI_DIR="${HOME}/.cache/omnivoice"
VOICE_SAMPLES_DIR="${OMNI_DIR}/voice_samples"
mkdir -p "${K2FSA_DIR}" "${VOICE_SAMPLES_DIR}"

VOCODER_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos_24khz.onnx"
ZIPVOICE_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-zipvoice-distill-int8-zh-en-emilia.tar.bz2"

# 1. Download Vocos 24kHz vocoder if missing or small
VOCODER_FILE="${K2FSA_DIR}/vocos_24khz.onnx"
if [ ! -f "${VOCODER_FILE}" ] || [ $(stat -c%s "${VOCODER_FILE}" 2>/dev/null || stat -f%z "${VOCODER_FILE}") -lt 50000000 ]; then
  echo "📥 Downloading Vocos 24kHz vocoder (~54MB)..."
  curl -fL --retry 3 --retry-delay 2 -o "${VOCODER_FILE}" "${VOCODER_URL}"
fi

# Compatibility vocoder alias
cp -f "${VOCODER_FILE}" "${K2FSA_DIR}/vocoder.onnx"

# 2. Download and Extract ZipVoice int8 model if missing or small
DECODER_FILE="${K2FSA_DIR}/zipvoice/decoder.int8.onnx"
if [ ! -f "${DECODER_FILE}" ] || [ $(stat -c%s "${DECODER_FILE}" 2>/dev/null || stat -f%z "${DECODER_FILE}") -lt 100000000 ]; then
  echo "📥 Downloading ZipVoice neural model archive (~109MB)..."
  TMP_TAR="/tmp/sherpa_zipvoice.tar.bz2"
  curl -fL --retry 3 --retry-delay 2 -o "${TMP_TAR}" "${ZIPVOICE_URL}"
  echo "📦 Extracting ZipVoice archive via native tar..."
  tar -xjf "${TMP_TAR}" -C "${K2FSA_DIR}"
  rm -f "${TMP_TAR}"
  if [ -d "${K2FSA_DIR}/sherpa-onnx-zipvoice-distill-int8-zh-en-emilia" ]; then
    rm -rf "${K2FSA_DIR}/zipvoice"
    mv "${K2FSA_DIR}/sherpa-onnx-zipvoice-distill-int8-zh-en-emilia" "${K2FSA_DIR}/zipvoice"
  fi
fi

# Compatibility aliases for legacy tests
cp -f "${K2FSA_DIR}/zipvoice/decoder.int8.onnx" "${OMNI_DIR}/omnivoice_weights.bin"
cp -f "${K2FSA_DIR}/zipvoice/encoder.int8.onnx" "${OMNI_DIR}/reference_encoder.onnx"
cp -f "${K2FSA_DIR}/zipvoice/tokens.txt" "${K2FSA_DIR}/tokens.txt"

# 3. Seed repository assets first if present
if [ -f "assets/ManVoice.mp3" ]; then
  cp -f "assets/ManVoice.mp3" "${VOICE_SAMPLES_DIR}/ManVoice.mp3"
  cp -f "assets/ManVoice.mp3" "${OMNI_DIR}/ManVoice.mp3"
  echo "✓ Seeded ManVoice.mp3 into cache."
fi
if [ -f "assets/reference.wav" ]; then
  cp -f "assets/reference.wav" "${VOICE_SAMPLES_DIR}/reference.wav"
  cp -f "assets/reference.txt" "${VOICE_SAMPLES_DIR}/reference.txt"
  echo "✓ Pre-seeded reference sample from repository assets."
fi

# 4. Ensure reference voice sample is pinned in cache via ModelCacheManager
python3 -c "
import sys; sys.path.insert(0, 'src')
from cache_manager import ModelCacheManager
mgr = ModelCacheManager()
p = mgr.ensure_voice_sample_cached()
print(f'Pinned voice sample confirmed: {p}')
"

# 5. Strict verification of reference audio file size
REF_FILE="${VOICE_SAMPLES_DIR}/reference.wav"
if [ ! -f "${REF_FILE}" ] || [ $(stat -c%s "${REF_FILE}" 2>/dev/null || stat -f%z "${REF_FILE}") -lt 50000 ]; then
  echo "❌ INTEGRITY ERROR: Reference voice sample missing or too small at ${REF_FILE}"
  exit 1
fi

echo "✅ All neural models and pinned voice sample seeded successfully."
