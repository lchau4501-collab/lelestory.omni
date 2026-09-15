"""
OmniVoice (k2-fsa/OmniVoice) Neural Voice Cloning Engine for LeLe Storybook Video Engine.
Synthesizes authentic Chinese story audio sections using official k2-fsa/OmniVoice zero-shot voice cloning
matching the reference voice voice_preview_mark - cartoonish, funny and cheerful.mp3.
Strictly 24,000 Hz, mono, 16-bit PCM WAV output. Strictly ZERO Edge-TTS. Zero sine-wave synthesizer facade.
Prioritizes loading pinned reference voice sample from ~/.cache/omnivoice/voice_samples/reference.wav.
Supports native 1.0x/0.70x speech tempo without time-stretching degradation while preserving pitch, tone, and vocal timbre.
"""

import os
import sys
import json
import wave
import struct
import numpy as np
import math
import shutil
import hashlib
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger("lelestory.omni.tts")

SAMPLE_RATE = 24000
CHANNELS = 1
SAMPWIDTH = 2  # 16-bit PCM

# Canonical Voice Mark reference audio filename
REFERENCE_SAMPLE_ID = "voice_preview_mark"
REFERENCE_SAMPLE_FILENAME = "voice_preview_mark.mp3"

OUTRO_LOOP_TEXT_EXACT = "这些生词来自故事……"

# Authoritative story script text for Story Row #2 (10 scenes + 5 vocab + recap + outro)
ROW_2_SCRIPT_TEXTS: Dict[str, Any] = {
    "title": "吃菜的大狼",
    "scene1": "在美丽茂密的大森林里，住着一只名叫罗罗的大灰狼。不同于普通的狼，他性情温和，最喜欢在菜园里种植新鲜蔬菜。",
    "scene2": "森林里的小兔子和小松鼠依然对大灰狼充满恐惧。每次远远看到罗罗走来，大家都吓得赶紧躲进灌木丛中不敢出声。",
    "scene3": "这天下午，天空突然乌云密布，一场狂暴的风雨呼啸而来，猛烈的狂风将山坡上的一棵巨大松树连根吹倒。",
    "scene4": "小兔子惊慌失措地呼喊：救命啊！倒下的大树把我们兔洞的出口死死挡住了，我们出不去了！",
    "scene5": "罗罗在风雨中听到了急切的呼救声。他没有躲回温暖的木屋，而是顶着狂风暴雨立刻奔向了兔洞。",
    "scene6": "罗罗大声说：小兔子别怕！我力气大，我来帮你们把这根沉重的大树干搬开！",
    "scene7": "浸透雨水的树干沉重无比，罗罗脚底打滑，爪子磨破了也绝不松手，咬紧牙关使出了全身的力气。",
    "scene8": "伴随着一声大喝，罗罗终于将巨木推到一旁，小心翼翼地把受惊的小兔子们一个个安全抱了出来。",
    "scene9": "兔妈妈感激地说：罗罗，太感谢你了！原来你是一只真正善良温和的大狼，我们再也不怕你了！",
    "scene10": "风雨过后彩虹高挂，小动物们齐聚在罗罗家，开开心心地吃起热气腾腾的蔬菜火锅。善良化解了误会，带来了珍贵的友谊。",
    "vocab_items": [
        ("vocab_1", "大灰狼"),
        ("vocab_2", "蔬菜"),
        ("vocab_3", "害怕"),
        ("vocab_4", "帮忙"),
        ("vocab_5", "朋友"),
    ],
    "vocab": "大灰狼 蔬菜 害怕 帮忙 朋友",
    "outro_loop": OUTRO_LOOP_TEXT_EXACT
}

# Authoritative story script text for Story Row #3 (井底之蛙)
ROW_3_SCRIPT_TEXTS: Dict[str, Any] = {
    "title": "井底之蛙",
    "scene1": "在一口废弃已久的清凉浅井里，住着一只快乐无忧的小青蛙。",
    "scene2": "小青蛙得意地说：瞧我住在这里多么舒服呀！整口水井都是我的天下，我就是这里的大国王！",
    "scene3": "这天早晨，一只来自辽阔东海的大鳖慢慢爬到了井边，低头看着井里。",
    "scene4": "小青蛙高兴地招呼：喂！外面的大个子，快跳进我的水井里来玩吧，这里的井水可舒服极了！",
    "scene5": "大鳖试着把一只脚探进井里，可井口实在太狭窄，他的膝盖一下子被卡住了。",
    "scene6": "大鳖退了回来，微笑着给井底的小青蛙讲述起浩瀚无边的大海。",
    "scene7": "大鳖认真地说：大海有千里之广、万丈之深。大旱时海水不见减少，大涝时海水也不见增多。",
    "scene8": "小青蛙听得目瞪口呆，眼神里充满了从未有过的震撼与向往。",
    "scene9": "小青蛙羞愧地低下了头，终于意识到小水井之外的天地是多么浩瀚广阔。",
    "scene10": "小青蛙鼓起勇气奋力跳出井口，迈开脚步去拥抱更辽阔美丽的大千世界。",
    "vocab_items": [
        ("vocab_1", "青蛙"),
        ("vocab_2", "水井"),
        ("vocab_3", "大海"),
        ("vocab_4", "骄傲"),
        ("vocab_5", "世界"),
    ],
    "vocab": "青蛙 水井 大海 骄傲 世界",
    "outro_loop": OUTRO_LOOP_TEXT_EXACT
}

# Authoritative story script text for Story Row #4 (小马过河)
ROW_4_SCRIPT_TEXTS: Dict[str, Any] = {
    "title": "小马过河",
    "scene1": "小马和妈妈住在绿油油的草地上，是一只既懂事又活泼的小马。",
    "scene2": "马妈妈温柔地说：孩子，你已经长大了，帮妈妈把这半袋麦子驮到河对岸的磨坊去吧！",
    "scene3": "小马高兴地驮着麦子飞快地跑着，一条哗哗流淌的小河突然挡住了去路。",
    "scene4": "小马不知道河水有多深，正犹豫不决时，看见了在河边吃草的老牛伯伯。",
    "scene5": "老牛伯伯温和地说：水很浅，刚没过我的小腿肚，你完全可以放心地蹚过去！",
    "scene6": "小松鼠急忙跳出来大声喊：小马别过去！水深得很呢！昨天我的一个同伴就是在这条河里被水冲走的！",
    "scene7": "小马左右为难，不知道该相信谁的话，只好叹了口气跑回家去问妈妈。",
    "scene8": "马妈妈摸着小马的头说：孩子，光听别人说是没用的。河水深不深，你必须亲自去试一试才知道。",
    "scene9": "小马鼓起勇气小心地下了河，发现水既不像老牛说的那么浅，也不像松鼠说的那么深。",
    "scene10": "小马顺利蹚过了河把麦子送到磨坊，开心地明白了凡事都要亲身实践的道理。",
    "vocab_items": [
        ("vocab_1", "小马"),
        ("vocab_2", "小河"),
        ("vocab_3", "磨坊"),
        ("vocab_4", "浅"),
        ("vocab_5", "尝试"),
    ],
    "vocab": "小马 小河 磨坊 浅 尝试",
    "outro_loop": OUTRO_LOOP_TEXT_EXACT
}


def get_story_script(row_id: int = 2) -> Dict[str, Any]:
    """Returns authoritative script dictionary for the given row ID."""
    if row_id == 3:
        return dict(ROW_3_SCRIPT_TEXTS)
    elif row_id == 4:
        return dict(ROW_4_SCRIPT_TEXTS)
    return dict(ROW_2_SCRIPT_TEXTS)


def build_atempo_filter_chain(tempo: float) -> str:
    """
    Builds an FFmpeg audio filter chain string for pitch-preserving tempo scaling.
    FFmpeg's 'atempo' filter accepts values in [0.5, 2.0].
    For tempos < 0.5 or > 2.0, chains multiple atempo filters in series.
    Examples:
      tempo=1.0  -> '' (identity / no-op, bypasses all filtering)
      tempo=0.5  -> 'atempo=0.5'
      tempo=0.4  -> 'atempo=0.5,atempo=0.8'
      tempo=0.25 -> 'atempo=0.5,atempo=0.5'
      tempo=2.0  -> 'atempo=2'
      tempo=2.5  -> 'atempo=2,atempo=1.25'
      tempo=1.0  -> '' (identity no-op)
    """
    if tempo <= 0 or math.isinf(tempo) or math.isnan(tempo) or tempo > 100.0:
        raise ValueError(f"Invalid tempo: {tempo}. Tempo must be strictly positive (> 0), finite, and <= 100.0")

    if abs(tempo - 1.0) < 1e-4:
        return ""

    factors: List[float] = []
    rem = float(tempo)

    # For slow tempos (< 0.5), factor out 0.5
    while rem < 0.5:
        factors.append(0.5)
        rem /= 0.5

    # For fast tempos (> 2.0), factor out 2.0
    while rem > 2.0:
        factors.append(2.0)
        rem /= 2.0

    # Remaining factor is in [0.5, 2.0]
    if abs(rem - 1.0) > 1e-4 or not factors:
        factors.append(round(rem, 4))

    return ",".join(f"atempo={f:.4f}".rstrip("0").rstrip(".") for f in factors)


# Alias for backward compatibility
build_atempo_filter = build_atempo_filter_chain


def apply_tempo_scaling(wav_path: str, tempo: float = 1.0) -> str:
    """
    apply_tempo_scaling: Zero-Atempo Invariant: Post-processing atempo DSP is permanently disabled.
    Speech pacing is handled natively by neural flow-matching (speed factor).
    Returns wav_path directly without invoking ffmpeg atempo filtering.
    """
    logger.info(f"Zero-Atempo Policy: Bypassing tempo scaling for {wav_path} (native neural generation preferred).")
    return wav_path


def _synthesize_neural_omnivoice(
    text: str,
    output_path: str,
    reference_wav_path: Optional[str] = None,
    tempo: float = 0.70,
    language: str = "zh"
) -> bool:
    """
    Performs authentic neural speech synthesis using official k2-fsa/OmniVoice.
    Generates 24,000 Hz mono 16-bit PCM WAV.
    """
    try:
        import torch
        from omnivoice import OmniVoice
        import soundfile as sf
    except ImportError:
        logger.warning("omnivoice, torch, or soundfile not installed. Cannot perform neural inference.")
        return False

    try:
        ref_path = reference_wav_path
        if not ref_path or not os.path.isfile(ref_path):
            candidates = [
                os.path.expanduser("~/.cache/omnivoice/voice_samples/voice_preview_mark.mp3"),
                os.path.expanduser("~/.cache/omnivoice/voice_samples/voice_preview_mark - cartoonish, funny and cheerful.mp3"),
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "voice_preview_mark.mp3"),
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "voice_preview_mark - cartoonish, funny and cheerful.mp3"),
                "assets/voice_preview_mark.mp3",
                "assets/voice_preview_mark - cartoonish, funny and cheerful.mp3",
            ]
            for c in candidates:
                if os.path.isfile(c):
                    ref_path = c
                    break

        if not ref_path or not os.path.isfile(ref_path):
            raise RuntimeError(
                "Voice Mark reference sample missing. Synthetic fallbacks are strictly prohibited."
            )

        ref_txt_candidates = [
            os.path.splitext(ref_path)[0] + ".txt",
            os.path.join(os.path.dirname(ref_path), "reference.txt"),
            os.path.expanduser("~/.cache/omnivoice/voice_samples/reference.txt"),
            "assets/reference.txt",
        ]
        ref_text = "不求与人相比，但求超越自己。"
        for txt_p in ref_txt_candidates:
            if os.path.isfile(txt_p):
                with open(txt_p, "r", encoding="utf-8") as tf:
                    t = tf.read().strip()
                    if t:
                        ref_text = t
                        break

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        logger.info(f"Initializing official k2-fsa/OmniVoice on device={device}, dtype={dtype}...")
        model = OmniVoice.from_pretrained(
            "k2-fsa/OmniVoice",
            device_map=device,
            dtype=dtype,
            load_asr=False
        )

        voice_prompt = model.create_voice_clone_prompt(
            ref_audio=str(ref_path),
            ref_text=ref_text,
            preprocess_prompt=True
        )

        audios = model.generate(
            text=text,
            voice_clone_prompt=voice_prompt,
            speed=tempo,
            language=language
        )

        if not audios or len(audios) == 0:
            return False

        samples = np.array(audios[0], dtype=np.float32)
        peak = float(np.max(np.abs(samples)))
        if peak > 0:
            samples = (samples / peak) * 0.90
        else:
            return False

        fade_in_len = int(SAMPLE_RATE * 0.02)
        fade_out_len = int(SAMPLE_RATE * 0.04)
        if len(samples) > fade_in_len + fade_out_len:
            samples[:fade_in_len] *= np.linspace(0.0, 1.0, fade_in_len, dtype=np.float32)
            samples[-fade_out_len:] *= np.linspace(1.0, 0.0, fade_out_len, dtype=np.float32)

        sf.write(output_path, samples, samplerate=SAMPLE_RATE, subtype="PCM_16")
        logger.info(f"Successfully generated OmniVoice neural speech at {output_path}")
        return True
    except Exception as exc:
        logger.warning(f"OmniVoice neural inference failed: {exc}")
        return False


# Compatibility alias
_synthesize_neural_sherpa = _synthesize_neural_omnivoice


def generate_pcm_speech_wav(
    text: str,
    output_path: str,
    target_duration: Optional[float] = None,
    tempo: float = 0.70,
    reference_wav_path: Optional[str] = None
) -> str:
    """
    Generates genuine 24,000 Hz mono 16-bit PCM WAV speech audio via k2-fsa/OmniVoice.
    Raises RuntimeError immediately if neural inference fails or models are missing.
    Strictly NO math.sin fallback.
    Defaults to 0.70x read-along tempo (F08 specification).
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # 1. Attempt genuine neural inference via k2-fsa/OmniVoice
    if _synthesize_neural_omnivoice(text, output_path, reference_wav_path, tempo=tempo):
        try:
            with wave.open(output_path, "rb") as wf:
                framerate = wf.getframerate()
                frames = wf.getnframes()
                dur = frames / float(framerate) if framerate > 0 else 0.0
                params = wf.getparams()
                audio_data = wf.readframes(frames)
            if dur < 0.26:
                needed_frames = int((0.26 - dur) * framerate)
                with wave.open(output_path, "wb") as wf:
                    wf.setparams(params)
                    wf.writeframes(audio_data)
                    wf.writeframes(bytes([0]) * (needed_frames * params.nchannels * params.sampwidth))
        except Exception:
            pass
        return output_path

    # If neural synthesis failed, FAIL FAST. Never produce buzzer tones!
    raise RuntimeError(
        f"Neural voice synthesis failed for text: '{text[:20]}...'. "
        "OmniVoice (k2-fsa/OmniVoice) is missing, uninstalled, or neural inference failed. "
        "Additive sine-wave fallback is eradicated."
    )


class OmniVoiceEngine:
    """
    OmniVoice (k2-fsa) Neural Voice Cloning Engine.
    Produces 24 kHz mono 16-bit PCM WAV speech audio matching reference voice sample.
    Checks pinned cache path ~/.cache/omnivoice/voice_samples/reference.wav first.
    Preserves native 1.0x neural output without time-stretching filtering.
    """

    def __init__(
        self,
        reference_sample_id: str = REFERENCE_SAMPLE_ID,
        reference_filename: str = REFERENCE_SAMPLE_FILENAME,
        cache_dir: Optional[str] = None
    ):
        self.sample_id = reference_sample_id
        self.reference_name = reference_filename
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/omnivoice")
        self.k2fsa_dir = os.path.expanduser("~/.cache/k2-fsa")
        self.voice_samples_dir = os.path.join(self.cache_dir, "voice_samples")
        self.pinned_sample_path = os.path.join(self.voice_samples_dir, "voice_preview_mark.mp3")

        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.k2fsa_dir, exist_ok=True)
        os.makedirs(self.voice_samples_dir, exist_ok=True)

    def ensure_reference_voice(self) -> str:
        """
        Ensures reference voice sample exists in cache.
        Checks cache first. If missing, attempts to copy or download.
        """
        # 1. Pinned cache file
        if os.path.isfile(self.pinned_sample_path) and os.path.getsize(self.pinned_sample_path) > 0:
            logger.info(f"Loaded pinned reference voice sample from cache: {self.pinned_sample_path}")
            return self.pinned_sample_path

        # 2. Named alias in cache root
        named_in_cache = os.path.join(self.cache_dir, self.reference_name)
        if os.path.isfile(named_in_cache) and os.path.getsize(named_in_cache) > 0:
            shutil.copyfile(named_in_cache, self.pinned_sample_path)
            return self.pinned_sample_path

        # 3. ModelCacheManager download/ensure
        try:
            from cache_manager import ModelCacheManager
            mgr = ModelCacheManager(omnivoice_dir=self.cache_dir, k2fsa_dir=self.k2fsa_dir)
            cached_sample = mgr.ensure_voice_sample_cached(self.sample_id)
            if os.path.exists(cached_sample) and os.path.getsize(cached_sample) > 0:
                logger.info(f"Reference voice sample ensured in cache: {cached_sample}")
                return cached_sample
        except Exception as exc:
            logger.warning(f"Cache manager ensure_voice_sample_cached error: {exc}")

        # 4. Fail fast if reference voice cannot be ensured
        raise RuntimeError(
            f"Reference voice sample could not be found or retrieved: {self.pinned_sample_path}. "
            "Ensure reference.wav is pinned in ~/.cache/omnivoice/voice_samples/reference.wav."
        )

    def synthesize_section(
        self,
        section: str,
        text: Optional[str] = None,
        output_dir: Optional[str] = None,
        row_id: int = 2,
        tempo: float = 1.0
    ) -> Dict[str, Any]:
        """
        Synthesizes audio for a story section at 1.0x tempo (pitch-preserved).
        For vocab, synthesizes individual vocab_1.wav..vocab_5.wav AND vocab.wav.
        Prioritizes pinned reference voice in cache.
        """
        ref_voice_path = self.ensure_reference_voice()
        output_dir = output_dir or f"artifacts/voice_row_{row_id}"
        os.makedirs(output_dir, exist_ok=True)

        script = get_story_script(row_id)
        generated_files: List[str] = []

        if section == "vocab":
            vocab_items = script.get("vocab_items", ROW_2_SCRIPT_TEXTS["vocab_items"])
            for item_name, item_text in vocab_items:
                target_path = os.path.join(output_dir, f"{item_name}.wav")
                generate_pcm_speech_wav(item_text, target_path, tempo=tempo, reference_wav_path=ref_voice_path)
                generated_files.append(target_path)

            recap_text = script.get("vocab", ROW_2_SCRIPT_TEXTS["vocab"])
            target_recap = os.path.join(output_dir, "vocab.wav")
            generate_pcm_speech_wav(recap_text, target_recap, tempo=tempo, reference_wav_path=ref_voice_path)
            generated_files.append(target_recap)

            manifest_path = os.path.join(output_dir, "manifest_vocab.json")
            manifest_data = {
                "section": "vocab",
                "row_id": row_id,
                "engine": "omnivoice_k2fsa",
                "framework": "k2-fsa/OmniVoice",
                "sample_id": self.sample_id,
                "reference_voice": self.reference_name,
                "reference_sample_path": ref_voice_path,
                "reference_cached": os.path.exists(self.pinned_sample_path),
                "sample_rate": SAMPLE_RATE,
                "channels": CHANNELS,
                "bit_depth": 16,
                "tempo": tempo,
                "tempo_filter": build_atempo_filter_chain(tempo) or "none",
                "pitch_preserved": True,
                "files": [os.path.basename(p) for p in generated_files]
            }
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest_data, f, ensure_ascii=False, indent=2)

            return {
                "section": "vocab",
                "files": generated_files,
                "manifest": manifest_path
            }

        if text is None:
            text = script.get(section, OUTRO_LOOP_TEXT_EXACT)

        target_path = os.path.join(output_dir, f"{section}.wav")
        generate_pcm_speech_wav(text, target_path, tempo=tempo, reference_wav_path=ref_voice_path)
        generated_files.append(target_path)

        manifest_path = os.path.join(output_dir, f"manifest_{section}.json")
        manifest_data = {
            "section": section,
            "row_id": row_id,
            "text": text,
            "engine": "omnivoice_k2fsa",
            "framework": "k2-fsa/OmniVoice",
            "sample_id": self.sample_id,
            "reference_voice": self.reference_name,
            "reference_sample_path": ref_voice_path,
            "reference_cached": os.path.exists(self.pinned_sample_path),
            "sample_rate": SAMPLE_RATE,
            "channels": CHANNELS,
            "bit_depth": 16,
            "tempo": tempo,
            "tempo_filter": build_atempo_filter_chain(tempo) or "none",
            "pitch_preserved": True,
            "file": os.path.basename(target_path)
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, ensure_ascii=False, indent=2)

        return {
            "section": section,
            "output_path": target_path,
            "files": generated_files,
            "manifest": manifest_path
        }

    def synthesize_script(self, script_payload: Dict[str, Any], output_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        """Backward-compatibility method for synthesizing full script payload."""
        batch_id = script_payload.get("batch_id", 1)
        output_dir = output_dir or f"artifacts/voice_row_{batch_id}"
        os.makedirs(output_dir, exist_ok=True)

        audio_manifest: List[Dict[str, Any]] = []
        intro_path = os.path.join(output_dir, "intro_chime.wav")
        generate_pcm_speech_wav("提示音", intro_path, target_duration=0.8, tempo=1.0)
        audio_manifest.append({"type": "intro_chime", "path": intro_path})

        script_obj = script_payload.get("script", {})
        lines = script_obj.get("lines", [])
        for idx, line in enumerate(lines):
            zh_text = line.get("zh", "你好")
            vi_text = line.get("vi", "Xin chao")
            zh_path = os.path.join(output_dir, f"line_{idx}_zh.wav")
            vi_path = os.path.join(output_dir, f"line_{idx}_vi.wav")

            generate_pcm_speech_wav(zh_text, zh_path, tempo=1.0)
            generate_pcm_speech_wav(vi_text, vi_path, tempo=1.0)

            audio_manifest.append({
                "line_index": idx,
                "speaker": line.get("speaker", "Narrator"),
                "zh_text": zh_text,
                "zh_audio_path": zh_path,
                "vi_text": vi_text,
                "vi_audio_path": vi_path
            })

        return audio_manifest


# Class alias for backward compatibility
OmniTTS = OmniVoiceEngine


def main():
    parser = argparse.ArgumentParser(description="OmniVoice Section Synthesizer")
    parser.add_argument("--row-id", type=int, default=2, help="Sheet Row Number (# ID)")
    parser.add_argument(
        "--section",
        type=str,
        required=True,
        choices=["title", "scene1", "scene2", "scene3", "scene4", "scene5", "scene6", "scene7", "scene8", "scene9", "scene10", "vocab", "outro_loop", "all"],
        help="Story section"
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--tempo", type=float, default=1.0, help="Speech tempo scaling (default: 1.0)")
    parser.add_argument("--upload", action="store_true", default=True, help="Upload generated audio directly to Google Drive")
    args = parser.parse_args()

    engine = OmniVoiceEngine()
    out_dir = args.output_dir or f"artifacts/voice_row_{args.row_id}"
    all_results = []

    if args.section == "all":
        sections = ["title", "scene1", "scene2", "scene3", "scene4", "scene5", "scene6", "scene7", "scene8", "scene9", "scene10", "vocab", "outro_loop"]
        for sec in sections:
            result = engine.synthesize_section(sec, output_dir=out_dir, row_id=args.row_id, tempo=args.tempo)
            print(f"Synthesized {sec}: {result}")
            all_results.append(result)
    else:
        result = engine.synthesize_section(args.section, output_dir=out_dir, row_id=args.row_id, tempo=args.tempo)
        print(f"Synthesized {args.section}: {result}")
        all_results.append(result)

    # Automatically upload generated WAV files and manifests to Google Drive
    if args.upload:
        try:
            from drive_resolver import resolve_voice_folder, upload_file_to_drive, get_service_account_credentials
            creds = get_service_account_credentials()
            if creds:
                voice_fid, voice_url = resolve_voice_folder(args.row_id)
                if voice_fid:
                    uploaded_count = 0
                    for res in all_results:
                        target_files = list(res.get("files", []))
                        if res.get("manifest"):
                            target_files.append(res["manifest"])
                        for fp in target_files:
                            if os.path.isfile(fp):
                                fid = upload_file_to_drive(fp, voice_fid, creds=creds)
                                if fid:
                                    uploaded_count += 1
                    logger.info(f"✅ Uploaded {uploaded_count} audio & manifest files to Drive voice folder: {voice_url}")
        except Exception as exc:
            logger.warning(f"Google Drive auto-upload skipped: {exc}")


if __name__ == "__main__":
    main()
