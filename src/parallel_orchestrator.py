"""
Parallel Sub-Workflow Orchestrator for LeLe Storybook OmniVoice Pipeline.
Dispatches the 7 parallel story audio generation sub-workflows (wfl1 to wfl7)
concurrently via GitHub Actions REST API or gh CLI.
"""

import os
import sys
import json
import logging
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Tuple, Optional
import requests

logger = logging.getLogger("lelestory.omni.parallel")

DEFAULT_REPO = "naadld/lelestory.omni"
DEFAULT_REF = "main"

# Mapping of section name to workflow file
SUB_WORKFLOWS = {
    "title": "wfl1_gen_title.yml",
    "scene1": "wfl2_gen_scene1.yml",
    "scene2": "wfl3_gen_scene2.yml",
    "scene3": "wfl4_gen_scene3.yml",
    "scene4": "wfl5_gen_scene4.yml",
    "vocab": "wfl6_gen_vocab.yml",
    "outro_loop": "wfl7_gen_outro_loop.yml",
}

# Legacy workflow keys for backward compatibility
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
    """
    Coordinates and dispatches the 7 parallel OmniVoice sub-workflows.
    """

    def __init__(
        self,
        script_gk2_data: Optional[Dict[str, Any]] = None,
        repo: str = DEFAULT_REPO,
        token: Optional[str] = None
    ):
        self.data = script_gk2_data or {}
        self.batch_id = self.data.get("batch_id", 2)
        self.theme = self.data.get("theme", "HANZIDEGUSHI")
        self.repo = repo or os.environ.get("GITHUB_REPOSITORY", DEFAULT_REPO)
        self.token = (
            token or
            os.environ.get("GITHUB_TOKEN") or
            os.environ.get("PIPELINE_PAT")
        )

    def _get_token(self) -> Optional[str]:
        if self.token:
            return self.token
        # Try fetching from gh CLI if available
        try:
            res = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, timeout=5)
            if res.returncode == 0 and res.stdout.strip():
                self.token = res.stdout.strip()
                return self.token
        except Exception:
            pass
        return None

    def dispatch_workflow(
        self,
        workflow_id: str,
        row_id: int = 2,
        ref: str = DEFAULT_REF
    ) -> Tuple[bool, str]:
        """
        Dispatches a single GitHub Actions workflow via REST API or gh CLI fallback.
        """
        token = self._get_token()
        # Clean workflow identifier (ensure .yml extension if not provided)
        if not workflow_id.endswith(".yml") and not workflow_id.isdigit():
            workflow_file = SUB_WORKFLOWS.get(workflow_id, f"{workflow_id}.yml")
        else:
            workflow_file = workflow_id

        # 1. Try REST API with token
        if token:
            url = f"https://api.github.com/repos/{self.repo}/actions/workflows/{workflow_file}/dispatches"
            headers = {
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {token}",
                "X-GitHub-Api-Version": "2022-11-28"
            }
            payload = {
                "ref": ref,
                "inputs": {
                    "row_id": str(row_id)
                }
            }
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=15)
                if resp.status_code in (200, 204):
                    msg = f"Successfully dispatched {workflow_file} for Row #{row_id} via REST API (HTTP {resp.status_code})"
                    logger.info(msg)
                    return True, msg
                else:
                    logger.warning(f"REST API dispatch for {workflow_file} returned HTTP {resp.status_code}: {resp.text}")
            except Exception as e:
                logger.warning(f"REST API dispatch error for {workflow_file}: {e}")

        # 2. Subprocess fallback using gh CLI
        try:
            cmd = ["gh", "workflow", "run", workflow_file, "-R", self.repo, "-r", ref, "-f", f"row_id={row_id}"]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            if res.returncode == 0:
                msg = f"Successfully dispatched {workflow_file} for Row #{row_id} via gh CLI"
                logger.info(msg)
                return True, msg
            else:
                msg = f"Failed to dispatch {workflow_file}: {res.stderr.strip()}"
                logger.error(msg)
                return False, msg
        except Exception as e:
            msg = f"Failed to invoke gh CLI for {workflow_file}: {str(e)}"
            logger.error(msg)
            return False, msg

    def dispatch_all_parallel(
        self,
        row_id: int = 2,
        ref: str = DEFAULT_REF,
        max_workers: int = 7
    ) -> Dict[str, Tuple[bool, str]]:
        """
        Dispatches all 7 sub-workflows concurrently using ThreadPoolExecutor.
        """
        results: Dict[str, Tuple[bool, str]] = {}
        logger.info(f"🚀 Dispatching {len(SUB_WORKFLOWS)} sub-workflows in parallel for Row #{row_id} on {self.repo}...")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_section = {
                executor.submit(self.dispatch_workflow, wf, row_id, ref): (section, wf)
                for section, wf in SUB_WORKFLOWS.items()
            }
            for future in as_completed(future_to_section):
                section, wf = future_to_section[future]
                try:
                    success, msg = future.result()
                    results[section] = (success, msg)
                except Exception as exc:
                    err_msg = f"Exception dispatching {wf} ({section}): {exc}"
                    logger.error(err_msg)
                    results[section] = (False, err_msg)

        all_ok = all(s for s, _ in results.values())
        logger.info(f"Parallel dispatch complete: {sum(1 for s, _ in results.values() if s)}/{len(results)} succeeded.")
        return results

    # Backward compatibility with legacy tests
    def compile_theme_assets(self, target_wf: str, audio_manifest: list) -> Dict[str, Any]:
        """Compiles theme-specific asset manifest for one of the workflows (backward compatibility)."""
        logger.info(f"Compiling assets for workflow '{target_wf}' (batch #{self.batch_id})")
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


def main():
    parser = argparse.ArgumentParser(description="Parallel OmniVoice Workflow Orchestrator")
    parser.add_argument("--row-id", type=int, default=2, help="Target Sheet Row Number (# ID)")
    parser.add_argument("--section", type=str, default=None, choices=list(SUB_WORKFLOWS.keys()) + ["all"], help="Specific section to trigger or 'all'")
    parser.add_argument("--repo", type=str, default=DEFAULT_REPO, help="GitHub repository (owner/repo)")
    parser.add_argument("--ref", type=str, default=DEFAULT_REF, help="Git branch or tag")
    args = parser.parse_args()

    orchestrator = ParallelOrchestrator(repo=args.repo)

    if args.section and args.section != "all":
        wf_file = SUB_WORKFLOWS[args.section]
        print(f"Triggering single sub-workflow '{wf_file}' for section '{args.section}' (Row #{args.row_id})...")
        success, msg = orchestrator.dispatch_workflow(wf_file, row_id=args.row_id, ref=args.ref)
        print(f"Result: {'SUCCESS' if success else 'FAILED'} - {msg}")
        sys.exit(0 if success else 1)
    else:
        print(f"Triggering 7 parallel sub-workflows for Row #{args.row_id}...")
        results = orchestrator.dispatch_all_parallel(row_id=args.row_id, ref=args.ref)
        failures = 0
        for sec, (ok, msg) in results.items():
            print(f"  [{'OK' if ok else 'FAIL'}] {sec} ({SUB_WORKFLOWS[sec]}): {msg}")
            if not ok:
                failures += 1
        sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()
