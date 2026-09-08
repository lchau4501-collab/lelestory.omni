# 🎙️ lelestory.omni

**Component 3 of 4: OmniVoice k2-fsa TTS, 7 Parallel Workflows & Gatekeeper 3** for LeLe Storybook Video Engine (`@lelehoctiengtrung`).

## Overview
- **OmniVoice k2-fsa Neural TTS**: Synthesizes Chinese & Vietnamese voice narration normalized to LUFS -14 EBU R128 with synchronized 0.75s intro chime WAV.
- **7 Parallel Workflows**:
  1. `wf_tts_audio.yml`: Master TTS voice synthesis.
  2. `wf_hanzi_story.yml`: Asset compilation for Hanzi Stories.
  3. `wf_idioms.yml`: Asset compilation for Idioms.
  4. `wf_slangs.yml`: Asset compilation for Slangs.
  5. `wf_vs_series.yml`: Asset compilation for VS Series.
  6. `wf_dialogues.yml`: Asset compilation for Dialogues.
  7. `wf_social_carousels.yml`: Social media image prompt binding.
- **GitHub Actions Model Checkpoint Caching**: Caches model files in `~/.cache/omni_k2fsa` using `actions/cache@v4` with cache key `omni-k2fsa-models-${{ runner.os }}-${{ hashFiles('models.json') }}`.
- **Gatekeeper 3 (GK3)**: Validates WAV audio properties, EBU R128 volume levels, intro chime sync, and asset manifest integrity.
- Runs 100% on **GitHub Actions Cloud Runners** (`ubuntu-22.04`) - strictly enforcing the **Zero-VPS Ban**.

## Local Development & Testing
```bash
pip install -r requirements.txt
python src/cache_manager.py
pytest tests/
python src/gatekeeper3.py --script-input path/to/script_gk2.json --output artifacts/omni_gk3.json
```
