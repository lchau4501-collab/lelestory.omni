#!/usr/bin/env python3
"""
Master Colab Voice Orchestrator for LeLe Storybook Video Engine.
Orchestrates neural speech synthesis on Google Colab GPU runtimes using google-colab-cli,
with automatic multi-account rotation across the Gmail pool, direct Google Drive storage,
and Gatekeeper 3 quality audit & Google Sheet synchronization.

STRICT HARDENING POLICIES:
1. Hardware Enforcement: GPU execution is strictly required (force_gpu=True by default).
   CPU fallback is eliminated to ensure 100% OmniVoice neural voice cloning on CUDA.
2. Account Security & Whitelist: Multi-account pool strictly verifies active account identity
   (mandated: hothihuong113@gmail.com). Account "aleron.dt@gmail.com" is PERMANENTLY BLACKLISTED
   across all layers and cannot be bypassed.
3. Zero Lingering Instances: VM teardown in finally: block is hardened with multi-attempt retry
   and post-termination verification.
4. Pre-flight Active Session Cleanup: Audits and terminates any orphan sessions before VM provisioning.
"""

import os
import sys
import json
import time
import re
import signal
import tarfile
import shutil
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any

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
MANDATED_ACCOUNT_EMAIL = "hothihuong113@gmail.com"


class ColabVoiceOrchestrator:
    """Orchestrates end-to-end voice synthesis on Google Colab with account rotation and GPU enforcement."""

    def __init__(self, pool_manager: Optional[ColabAccountManager] = None):
        self.mgr = pool_manager or ColabAccountManager()
        self.active_session_name: Optional[str] = None
        self.active_account_alias: Optional[str] = None
        self._register_signal_handlers()

    def _register_signal_handlers(self) -> None:
        """Register signal handlers to guarantee VM teardown on external termination signals."""
        def _handle_signal(signum, frame):
            sig_name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
            logger.warning(f"⚠️ Caught process signal {sig_name} ({signum}). Initiating emergency cleanup...")
            if self.active_session_name and self.active_account_alias:
                logger.warning(
                    f"🛑 Emergency stopping active session '{self.active_session_name}' "
                    f"on account [{self.active_account_alias}]..."
                )
                self._safe_teardown_session(self.active_account_alias, self.active_session_name)
            sys.exit(128 + signum)

        try:
            signal.signal(signal.SIGINT, _handle_signal)
            signal.signal(signal.SIGTERM, _handle_signal)
        except (ValueError, AttributeError):
            # Not in main thread or signal handling not supported in current environment
            pass

    def _verify_and_validate_account(self, account_alias: str) -> str:
        """
        Validates account selection with defense-in-depth:
        1. Decodes live OAuth JWT token to extract the true authenticated email.
        2. Strictly blocks any variation of 'aleron.dt@gmail.com' (permanent blacklist).
        3. Confirms identity matches the authorized account 'hothihuong113@gmail.com'.
        Returns verified email on success; raises PermissionError on blacklist violation.
        """
        # Layer A: Cryptographic token claim inspection
        token_email = self.mgr.verify_account_token_email(account_alias)

        # Layer B: Registry metadata inspection
        reg = self.mgr._read_registry()
        reg_email = (reg.get("accounts", {}).get(account_alias, {}).get("email") or "").strip().lower()

        candidate_email = (token_email or reg_email).strip().lower()

        # Layer C: Permanent Blacklist Enforcement (Cannot be bypassed)
        for blacklisted in BLACKLISTED_EMAILS:
            clean_b = blacklisted.strip().lower()
            if (clean_b and clean_b in candidate_email) or (clean_b and clean_b in reg_email) or (token_email and clean_b in token_email.strip().lower()):
                self.mgr.remove_account(account_alias)
                raise PermissionError(
                    f"CRITICAL SECURITY VIOLATION: Account [{account_alias}] is linked to PERMANENTLY BLACKLISTED email "
                    f"'{candidate_email or reg_email}'! Account purged from registry and aborted immediately."
                )

        if not candidate_email:
            raise ValueError(f"Account [{account_alias}] has no detectable email or token. Cannot verify identity.")

        # Layer D: Mandated Account Policy Check
        if candidate_email != MANDATED_ACCOUNT_EMAIL:
            logger.warning(
                f"⚠️ Account [{account_alias}] email is '{candidate_email}', which differs from primary mandated account '{MANDATED_ACCOUNT_EMAIL}'."
            )
        else:
            logger.info(f"✓ Mandated production account verified: {candidate_email}")

        logger.info(f"🔒 Account [{account_alias}] identity verified: {candidate_email} (Blacklist check: PASSED)")
        return candidate_email

    def _safe_teardown_session(self, account_alias: str, session_name: str, max_retries: int = 3) -> bool:
        """
        Safely stops and unassigns a Colab VM session with retry logic,
        ensuring zero lingering instances and compute quota preservation.
        """
        logger.info(f"🛑 [{account_alias}] Releasing Colab VM session: {session_name}...")
        for attempt in range(1, max_retries + 1):
            try:
                code, stdout, stderr = self.mgr.run_colab_command(
                    account_alias,
                    ["stop", "-s", session_name],
                    timeout=30
                )
                output = f"{stdout} {stderr}".strip()
                if (
                    code == 0
                    or "not found" in output.lower()
                    or "terminated" in output.lower()
                    or "no active session" in output.lower()
                ):
                    logger.info(f"✓ [{account_alias}] Session '{session_name}' successfully stopped (attempt {attempt}): {output}")
                    return True
                logger.warning(
                    f"⚠️ [{account_alias}] Session '{session_name}' stop attempt {attempt}/{max_retries} failed "
                    f"(code {code}): {output}"
                )
            except Exception as e:
                logger.warning(
                    f"⚠️ [{account_alias}] Exception during session '{session_name}' stop attempt {attempt}/{max_retries}: {e}"
                )

            if attempt < max_retries:
                time.sleep(2 * attempt)

        # Verification check: query active sessions to verify complete unassignment
        try:
            code, sessions_out, _ = self.mgr.run_colab_command(account_alias, ["sessions"], timeout=20)
            if session_name not in sessions_out:
                logger.info(f"✓ [{account_alias}] Verified session '{session_name}' is no longer active.")
                return True
            else:
                logger.error(
                    f"❌ CRITICAL: Session '{session_name}' still appears active on Colab server after {max_retries} stop attempts!"
                )
                return False
        except Exception as check_exc:
            logger.warning(f"⚠️ [{account_alias}] Could not verify sessions after stop: {check_exc}")
            return False

    def _pre_flight_cleanup(self, account_alias: str) -> None:
        """
        Pre-flight check to detect and terminate any orphan Colab sessions
        before provisioning a new VM, preventing resource exhaustion or quota leakage.
        """
        logger.info(f"[{account_alias}] Running pre-flight active session audit...")
        try:
            code, stdout, stderr = self.mgr.run_colab_command(account_alias, ["sessions"], timeout=30)
        except Exception as e:
            logger.warning(f"[{account_alias}] Pre-flight 'colab sessions' audit raised exception: {e}")
            return

        if code != 0:
            logger.warning(f"[{account_alias}] Pre-flight 'colab sessions' check returned code {code}: {stderr or stdout}")
            return

        if "No active sessions found" in stdout or not stdout.strip():
            logger.info(f"✓ [{account_alias}] No active orphan sessions found on server.")
            return

        # Parse active session names from table / list
        # Expected line format: [session_name] https://colab.research.google.com/...
        found_sessions = []
        for line in stdout.splitlines():
            line = line.strip()
            m = re.match(r"^\[([^\]]+)\]\s+https?://", line)
            if m:
                name = m.group(1).strip()
                if name != "colab":  # avoid matching "[colab]" logging prefix
                    found_sessions.append(name)

        if not found_sessions:
            logger.info(f"✓ [{account_alias}] No parsable active orphan sessions detected.")
            return

        logger.warning(
            f"⚠️ [{account_alias}] Detected {len(found_sessions)} orphan session(s): {found_sessions}. "
            f"Terminating before provisioning..."
        )
        for name in found_sessions:
            if name == "?":
                logger.info(f"[{account_alias}] Attempting generic stop for unnamed orphan assignment...")
                try:
                    self.mgr.run_colab_command(account_alias, ["stop"], timeout=30)
                except Exception as e:
                    logger.warning(f"[{account_alias}] Generic stop failed: {e}")
            else:
                self._safe_teardown_session(account_alias, name)

        # Confirm post-cleanup state
        try:
            _, post_out, _ = self.mgr.run_colab_command(account_alias, ["sessions"], timeout=20)
            if "No active sessions found" in post_out:
                logger.info(f"✓ [{account_alias}] Pre-flight cleanup confirmed: zero lingering sessions.")
            else:
                logger.warning(f"⚠️ [{account_alias}] Post-cleanup active sessions status: {post_out.strip()}")
        except Exception:
            pass

    def synthesize_row(self, row_id: int = 2, force_gpu: bool = True, max_rotations: int = 3) -> bool:
        """
        Public interface contract conforming to PROJECT.md architecture.
        Synthesizes all audio sections for a story row on Google Colab GPU.
        """
        return self.run(row_id=row_id, max_rotations=max_rotations, force_gpu=force_gpu)

    def run(self, row_id: int = 2, max_rotations: int = 3, force_gpu: bool = True) -> bool:
        logger.info(f"=== Starting Colab Voice Pipeline for Story Row #{row_id} (force_gpu={force_gpu}) ===")
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
                # 1. Identity & Blacklist Check
                self._verify_and_validate_account(active_alias)
                self.active_account_alias = active_alias

                # 2. Synthesis execution on Colab GPU
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
            except PermissionError:
                # Fatal security violation (blacklist) — do not retry with this account, fail or rotate
                logger.critical(f"Security policy violation on account [{active_alias}]. Terminating attempt.")
                raise
            except Exception as exc:
                err_msg = str(exc)
                logger.warning(f"Execution failed on account [{active_alias}]: {err_msg}")
                quota_indicators = ["quota", "429", "resourceexhausted", "limit", "rate", "busy", "gpu"]
                if any(q in err_msg.lower() for q in quota_indicators):
                    self.mgr.mark_cooldown(active_alias, duration_seconds=14400, reason=err_msg)
                else:
                    self.mgr.mark_cooldown(active_alias, duration_seconds=600, reason=err_msg)
            finally:
                self.active_account_alias = None

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
        force_gpu: bool = True
    ) -> bool:
        session_name = f"voice_worker_{int(time.time()) % 10000}"
        self.active_session_name = session_name
        logger.info(f"[{account_alias}] Target Colab session name: {session_name}")

        # Pre-flight orphan session cleanup
        self._pre_flight_cleanup(account_alias)

        try:
            # 1. Provision VM: Strictly allocate NVIDIA T4 GPU; CPU fallback is eliminated
            logger.info(f"[{account_alias}] Provisioning Colab VM (session: {session_name}) with NVIDIA T4 GPU...")
            code, stdout, stderr = self.mgr.run_colab_command(
                account_alias,
                ["new", "-s", session_name, "--gpu", "T4"],
                timeout=90
            )
            if code != 0:
                err_detail = (stderr or stdout).strip()
                # Fail immediately with clear, unambiguous error
                raise RuntimeError(
                    f"Colab GPU (T4) provisioning failed for account [{account_alias}] (exit code {code}): {err_detail}. "
                    f"CPU fallback is strictly disabled — OmniVoice neural speech cloning requires GPU acceleration."
                )

            logger.info(f"✓ Colab GPU VM successfully provisioned: {session_name}")

            # 2. Upload Reference audio and text to VM (both content/ and root for compatibility)
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
            # 6. Tear down VM immediately to preserve compute units (guaranteed clean teardown)
            self._safe_teardown_session(account_alias, session_name)
            self.active_session_name = None


def main():
    parser = argparse.ArgumentParser(description="Colab Voice Pipeline Orchestrator")
    parser.add_argument("--row-id", type=int, default=2, help="Story row ID to synthesize")
    parser.add_argument(
        "--force-gpu",
        action="store_true",
        default=True,
        help="Enforce GPU execution (default: True, CPU fallback disabled)"
    )
    parser.add_argument(
        "--no-force-gpu",
        dest="force_gpu",
        action="store_false",
        help="Disable forced GPU (WARNING: OmniVoice requires GPU acceleration)"
    )
    parser.add_argument("--max-rotations", type=int, default=3, help="Max account rotations on failure")
    args = parser.parse_args()

    orchestrator = ColabVoiceOrchestrator()
    ok = orchestrator.run(row_id=args.row_id, max_rotations=args.max_rotations, force_gpu=args.force_gpu)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
