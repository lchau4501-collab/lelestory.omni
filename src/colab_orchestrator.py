#!/usr/bin/env python3
"""
Master Colab Voice Orchestrator for LeLe Storybook Video Engine.
Orchestrates neural speech synthesis on Google Colab GPU/CPU runtimes using google-colab-cli,
with automatic multi-account rotation across the Gmail pool, direct Google Drive storage,
and Gatekeeper 3 quality audit & Google Sheet synchronization.

STRICT POLICY:
The account "aleron.dt@gmail.com" is PERMANENTLY FORBIDDEN and BLACKLISTED.
"""

import os
import sys
import json
import time
import tarfile
import shutil
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from colab_rotator import ColabAccountManager, BLACKLISTED_EMAILS
from gatekeeper3 import Gatekeeper3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("lelestory.omni.colab_orchestrator")

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
SCRIPTS_DIR = REPO_ROOT / "scripts"
ASSETS_DIR = REPO_ROOT / "assets"

VOICE_FOLDER_ID_ROW_2 = "1AgtKSIhgRDW4NfMJi5I1X1l2sXYgFh1e"


class ColabVoiceOrchestrator:
    """Orchestrates end-to-end voice synthesis on Google Colab with account rotation."""

    def __init__(self, pool_manager: Optional[ColabAccountManager] = None):
        self.mgr = pool_manager or ColabAccountManager()

    def run(self, row_id: int = 2, max_rotations: int = 3, force_gpu: bool = False) -> bool:
        logger.info(f"=== Starting Colab Voice Pipeline for Story Row #{row_id} ===")
        ref_wav = ASSETS_DIR / "reference.wav"
        ref_txt = ASSETS_DIR / "reference.txt"

        if not ref_wav.exists():
            raise FileNotFoundError(f"Missing reference voice at {ref_wav}")

        rotation_attempt = 0
        success = False

        while rotation_attempt < max_rotations:
            active_alias = self.mgr.select_active_account()
            if not active_alias:
                logger.error("❌ No active accounts available in the Colab Account Pool! All cooling down or none configured.")
                logger.info("👉 Vui lòng chạy `python3 scripts/colab_auth_pool.py add <alias>` để thêm tài khoản.")
                return False

            rotation_attempt += 1
            logger.info(f"--- Attempt {rotation_attempt}/{max_rotations} using account: [{active_alias}] ---")

            try:
                success = self._execute_colab_synthesis(
                    account_alias=active_alias,
                    row_id=row_id,
                    ref_wav_path=ref_wav,
                    ref_txt_path=ref_txt,
                    force_gpu=force_gpu
                )
                if success:
                    self.mgr.mark_success(active_alias)
                    break
            except Exception as exc:
                err_msg = str(exc)
                logger.warning(f"Execution failed on account [{active_alias}]: {err_msg}")
                quota_indicators = ["quota", "429", "resourceexhausted", "limit", "rate", "busy"]
                if any(q in err_msg.lower() for q in quota_indicators):
                    self.mgr.mark_cooldown(active_alias, duration_seconds=14400, reason=err_msg)
                else:
                    self.mgr.mark_cooldown(active_alias, duration_seconds=600, reason=err_msg)

        if not success:
            logger.error("❌ Colab voice synthesis failed across all rotated accounts.")
            return False

        # Step 2: Upload WAV files to Google Drive
        local_voice_dir = ARTIFACTS_DIR / f"voice_row_{row_id}"
        logger.info(f"📤 Uploading verified WAV files to Google Drive voice folder: {VOICE_FOLDER_ID_ROW_2}...")
        rclone_cmd = [
            "rclone", "copy",
            str(local_voice_dir),
            "gdrive_rd:",
            "--drive-root-folder-id", VOICE_FOLDER_ID_ROW_2,
            "--include", "*.wav",
            "--retries", "3",
            "-v"
        ]
        rc = subprocess.run(rclone_cmd)
        if rc.returncode != 0:
            logger.warning("rclone upload returned non-zero.")

        # Step 3: Run Gatekeeper 3 Quality Audit & Sheet Synchronization
        logger.info("🛡️ Initiating Gatekeeper 3 Quality Audit & Sheet Sync...")
        auditor = Gatekeeper3()
        passed, summary, details = auditor.audit_row(
            row_id=row_id,
            voice_dir=str(local_voice_dir),
            sync_sheet=True
        )

        if passed:
            logger.info(f"🎉 SUCCESS: Row #{row_id} Gatekeeper 3 AUDIT PASS! All 12 files verified and synced.")
            return True
        else:
            logger.error(f"❌ GK3 Audit Failed: {summary}")
            return False

    def _execute_colab_synthesis(
        self,
        account_alias: str,
        row_id: int,
        ref_wav_path: Path,
        ref_txt_path: Path,
        force_gpu: bool = False
    ) -> bool:
        session_name = f"voice_worker_{int(time.time()) % 10000}"
        logger.info(f"[{account_alias}] Provisioning Colab session: {session_name}...")

        # 1. Provision VM: Try T4 GPU first, fallback to CPU
        logger.info(f"[{account_alias}] Attempting to provision Colab VM (session: {session_name}) with T4 GPU...")
        code, stdout, stderr = self.mgr.run_colab_command(account_alias, ["new", "-s", session_name, "--gpu", "T4"], timeout=90)
        if code != 0:
            if force_gpu:
                raise RuntimeError(f"Colab GPU provisioning failed (code {code}): {stderr or stdout}")
            logger.warning(f"GPU unavailable for {account_alias} ({stderr or stdout}), falling back to high-speed CPU...")
            code, stdout, stderr = self.mgr.run_colab_command(account_alias, ["new", "-s", session_name], timeout=90)
            if code != 0:
                raise RuntimeError(f"Colab provisioning failed (code {code}): {stderr or stdout}")

        logger.info(f"✓ Colab VM provisioned: {session_name}")

        try:
            # 2. Upload Reference audio and text to VM (upload to both content/ and root)
            logger.info(f"[{account_alias}] Uploading reference voice and transcript to Colab VM...")
            self.mgr.run_colab_command(account_alias, ["upload", "-s", session_name, str(ref_wav_path), "content/reference.wav"], timeout=45)
            self.mgr.run_colab_command(account_alias, ["upload", "-s", session_name, str(ref_wav_path), "reference.wav"], timeout=45)

            if ref_txt_path.exists():
                self.mgr.run_colab_command(account_alias, ["upload", "-s", session_name, str(ref_txt_path), "content/reference.txt"], timeout=45)
                self.mgr.run_colab_command(account_alias, ["upload", "-s", session_name, str(ref_txt_path), "reference.txt"], timeout=45)

            # 3. Execute Worker Script on Colab VM
            worker_script = SCRIPTS_DIR / "colab_worker_synth.py"
            logger.info(f"[{account_alias}] Executing neural synthesis worker on Colab VM...")
            code, stdout, stderr = self.mgr.run_colab_command(
                account_alias,
                ["exec", "-s", session_name, "-f", str(worker_script), "--timeout", "600"],
                timeout=660
            )

            print(stdout)
            if "[COLAB_SYNTH_COMPLETE]" not in stdout or code != 0:
                raise RuntimeError(f"Colab worker synthesis failed (code {code}): {stderr or stdout}")

            # 4. Download generated archive (try content/ first, then root)
            ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
            local_archive = ARTIFACTS_DIR / f"voice_row_{row_id}.tar.gz"
            logger.info(f"[{account_alias}] Downloading synthesized audio archive to {local_archive}...")

            code, _, _ = self.mgr.run_colab_command(
                account_alias,
                ["download", "-s", session_name, f"content/voice_row_{row_id}.tar.gz", str(local_archive)],
                timeout=90
            )
            if code != 0 or not local_archive.exists() or local_archive.stat().st_size == 0:
                code, _, err = self.mgr.run_colab_command(
                    account_alias,
                    ["download", "-s", session_name, f"voice_row_{row_id}.tar.gz", str(local_archive)],
                    timeout=90
                )
                if code != 0 or not local_archive.exists() or local_archive.stat().st_size == 0:
                    raise RuntimeError(f"Failed to download audio archive: {err}")

            # 5. Extract local archive
            target_extract_dir = ARTIFACTS_DIR / f"voice_row_{row_id}"
            target_extract_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"📦 Extracting {local_archive} to {target_extract_dir}...")
            with tarfile.open(str(local_archive), "r:gz") as tar:
                tar.extractall(path=str(ARTIFACTS_DIR))

            wavs = list(target_extract_dir.glob("*.wav"))
            logger.info(f"✓ Successfully extracted {len(wavs)} WAV files for Row #{row_id}.")
            return len(wavs) >= 12

        finally:
            # 6. Tear down VM immediately to preserve compute units
            logger.info(f"🛑 Releasing Colab VM: {session_name}...")
            self.mgr.run_colab_command(account_alias, ["stop", "-s", session_name], timeout=30)


def main():
    parser = argparse.ArgumentParser(description="Colab Voice Pipeline Orchestrator")
    parser.add_argument("--row-id", type=int, default=2, help="Story row ID to synthesize")
    parser.add_argument("--force-gpu", action="store_true", help="Force GPU allocation (default: try GPU, fallback CPU)")
    parser.add_argument("--max-rotations", type=int, default=3, help="Max account rotations on failure")
    args = parser.parse_args()

    orchestrator = ColabVoiceOrchestrator()
    ok = orchestrator.run(row_id=args.row_id, max_rotations=args.max_rotations, force_gpu=args.force_gpu)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
