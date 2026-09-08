import os
import json
import logging
from typing import Dict, Any

logger = logging.getLogger("lelestory.omni.parallel")

WORKFLOW_KEYS = [
    "tts_audio",
    "hanzi_story",
    "idioms",
    "slangs",
    "vs_series",
    "dialogues",
    "social_carousels"
]

class ParallelOrchestrator:
    """Coordinates asset compilation across the 7 parallel sub-workflows."""

    def __init__(self, script_gk2_data: Dict[str, Any]):
        self.data = script_gk2_data
        self.batch_id = script_gk2_data.get("batch_id", 1)
        self.theme = script_gk2_data.get("theme", "HANZIDEGUSHI")

    def compile_theme_assets(self, target_wf: str, audio_manifest: list) -> Dict[str, Any]:
        """Compiles theme-specific asset manifest for one of the 7 parallel workflows."""
        if target_wf not in WORKFLOW_KEYS:
            raise ValueError(f"Unknown workflow '{target_wf}'. Must be one of {WORKFLOW_KEYS}")

        logger.info(f"Compiling assets for parallel workflow '{target_wf}' (batch #{self.batch_id})")
        compiled_manifest = {
            "batch_id": self.batch_id,
            "target_workflow": target_wf,
            "theme": self.theme,
            "audio_manifest": audio_manifest,
            "prompts": self.data.get("prompts", {}),
            "metadata": self.data.get("metadata", {}),
            "script": self.data.get("script", {}),
            "status": f"{target_wf}_Compiled"
        }
        return compiled_manifest
