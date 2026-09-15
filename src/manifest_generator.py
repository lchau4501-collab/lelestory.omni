"""
Manifest Generator for LeLe Storybook OmniVoice Pipeline.
Calculates precise timing, durations, character bounds, and generates
job manifests for 10 scenes + 5 vocabulary items + outro loop (16–18 audio sections).
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from audio_qc import calculate_duration_bounds, strip_punctuation

logger = logging.getLogger("lelestory.omni.manifest")

DEFAULT_OUTRO_TEXT = "这些生词来自故事……"
DEFAULT_OUTRO_PY = "zhè xiē shēng cí lái zì gù shì ……"
DEFAULT_OUTRO_EN = "Those vocabulary comes from the story..."

class StoryManifestGenerator:
    """
    Generates structured, validated timing and audio manifests
    for 10 progressive scenes + 5 vocabulary flashcards + outro loop.
    """

    def __init__(self, row_id: int = 2, tempo: float = 1.0):
        self.row_id = row_id
        self.tempo = tempo

    def build_10_scenes_manifest(
        self,
        title: str,
        scenes: List[Dict[str, Any]],
        vocabulary: List[Dict[str, Any]],
        outro: Optional[Dict[str, Any]] = None,
        include_vocab_recap: bool = True
    ) -> Dict[str, Any]:
        """
        Builds canonical 10-scenes audio and timing manifest.
        Calculates speech durations, timeline start offsets, and validates 3-layer text.
        """
        outro = outro or {
            "zh": DEFAULT_OUTRO_TEXT,
            "pinyin": DEFAULT_OUTRO_PY,
            "en": DEFAULT_OUTRO_EN
        }

        timeline_sections: List[Dict[str, Any]] = []
        script_items: Dict[str, str] = {"title": title}
        current_offset = 0.0

        # Title intro
        t_min, t_max = calculate_duration_bounds(title)
        est_title_dur = round(max(1.5, (t_min + t_max) / 2.0 / self.tempo), 2)
        timeline_sections.append({
            "section": "title",
            "file": "title.wav",
            "text": title,
            "start_time": current_offset,
            "estimated_duration": est_title_dur,
            "duration_bounds": [t_min, t_max]
        })
        current_offset += est_title_dur

        # 10 Progressive Scenes
        formatted_scenes = []
        for idx, sc in enumerate(scenes, start=1):
            s_num = sc.get("scene_num", idx)
            zh = sc.get("zh", "").strip()
            py = sc.get("pinyin", sc.get("py", "")).strip()
            en = sc.get("en", "").strip()
            spk = sc.get("speaker", "旁白").strip()

            script_items[f"scene{s_num}"] = zh
            t_min, t_max = calculate_duration_bounds(zh)
            char_count = len(strip_punctuation(zh))
            est_dur = round(max(2.0, (char_count * 0.28) / self.tempo), 2)

            sc_entry = {
                "scene_num": s_num,
                "section": f"scene{s_num}",
                "file": f"scene{s_num}.wav",
                "speaker": spk,
                "zh": zh,
                "pinyin": py,
                "en": en,
                "start_time": round(current_offset, 2),
                "estimated_duration": est_dur,
                "char_count": char_count,
                "duration_bounds": [t_min, t_max]
            }
            formatted_scenes.append(sc_entry)
            timeline_sections.append(sc_entry)
            current_offset += est_dur

        # 5 Key Vocabulary Items
        formatted_vocab = []
        vocab_words = []
        for v_idx, v in enumerate(vocabulary[:5], start=1):
            w = v.get("word", "").strip()
            py = v.get("pinyin", v.get("py", "")).strip()
            en = v.get("en", "").strip()
            vocab_words.append(w)
            script_items[f"vocab_{v_idx}"] = w

            t_min, t_max = calculate_duration_bounds(w)
            est_dur = round(max(1.5, 1.8 / self.tempo), 2)

            v_entry = {
                "index": v_idx,
                "section": f"vocab_{v_idx}",
                "file": f"vocab_{v_idx}.wav",
                "word": w,
                "zh": w,
                "pinyin": py,
                "en": en,
                "start_time": round(current_offset, 2),
                "estimated_duration": est_dur,
                "duration_bounds": [t_min, t_max]
            }
            formatted_vocab.append(v_entry)
            timeline_sections.append(v_entry)
            current_offset += est_dur

        # Optional Vocab Recap
        if include_vocab_recap and vocab_words:
            recap_text = " ".join(vocab_words)
            script_items["vocab"] = recap_text
            t_min, t_max = calculate_duration_bounds(recap_text)
            est_dur = round(max(3.0, (len(vocab_words) * 1.2) / self.tempo), 2)
            timeline_sections.append({
                "section": "vocab",
                "file": "vocab.wav",
                "text": recap_text,
                "start_time": round(current_offset, 2),
                "estimated_duration": est_dur,
                "duration_bounds": [t_min, t_max]
            })
            current_offset += est_dur

        # Outro Loop
        outro_zh = outro.get("zh", DEFAULT_OUTRO_TEXT)
        script_items["outro_loop"] = outro_zh
        t_min, t_max = calculate_duration_bounds(outro_zh)
        est_outro_dur = round(max(2.0, 2.5 / self.tempo), 2)
        timeline_sections.append({
            "section": "outro_loop",
            "file": "outro_loop.wav",
            "zh": outro_zh,
            "pinyin": outro.get("pinyin", DEFAULT_OUTRO_PY),
            "en": outro.get("en", DEFAULT_OUTRO_EN),
            "start_time": round(current_offset, 2),
            "estimated_duration": est_outro_dur,
            "duration_bounds": [t_min, t_max]
        })
        current_offset += est_outro_dur

        manifest = {
            "row_id": self.row_id,
            "title": title,
            "tempo": self.tempo,
            "sample_rate": 24000,
            "channels": 1,
            "bit_depth": 16,
            "scenes_count": len(formatted_scenes),
            "vocab_count": len(formatted_vocab),
            "total_estimated_duration": round(current_offset, 2),
            "total_audio_sections": len(timeline_sections),
            "scenes": formatted_scenes,
            "vocabulary": formatted_vocab,
            "outro": outro,
            "timeline": timeline_sections,
            "script_items": script_items
        }

        return manifest
