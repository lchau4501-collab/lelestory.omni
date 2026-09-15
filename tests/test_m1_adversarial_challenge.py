#!/usr/bin/env python3
"""
Adversarial Stress Test Suite for Milestone M1 Hardening.
Challenger: Challenger M1-2

Areas Challenged:
1. GPU Enforcement & Impossibility of CPU Fallback
2. Account Security & Blacklist Inviolability (aleron.dt@gmail.com)
3. Session Cleanup Resilience & Post-Teardown Verification
4. Audio QC Calibration (DEFAULT_MIN_DURATION = 0.25s) & Signal Integrity
"""

import os
import sys
import json
import time
import base64
import wave
import struct
import math
import inspect
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import audio_qc
from audio_qc import (
    check_wav_file,
    check_duration_bounds,
    calculate_rms,
    check_spectral_distribution,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_CHANNELS,
    DEFAULT_BIT_DEPTH,
    DEFAULT_MIN_RMS,
    DEFAULT_MIN_DURATION,
)
from colab_rotator import ColabAccountManager, BLACKLISTED_EMAILS
from colab_orchestrator import ColabVoiceOrchestrator, MANDATED_ACCOUNT_EMAIL


# ==============================================================================
# Helper Functions for Audio Synthesis & Test Data Generation
# ==============================================================================

def create_synthetic_wav(
    filepath: Path,
    duration: float,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = 1,
    audio_type: str = "speech_harmonics",
    amplitude: float = 3000.0,
    clipping: bool = False,
) -> Path:
    """
    Generates test WAV files with precise acoustic profiles:
    - 'silence': zero amplitude (RMS = 0)
    - 'pure_sine': 160Hz synthetic tone (no high frequencies)
    - 'speech_harmonics': multi-harmonic speech-like signal (rich energy at 2.5kHz and 5kHz)
    - 'clipped': speech signal with clipped peaks
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    num_samples = int(duration * sample_rate)
    frames = []

    for i in range(num_samples):
        t = float(i) / float(sample_rate)
        if audio_type == "silence":
            sample_val = 0.0
        elif audio_type == "pure_sine":
            # 160Hz pure tone - zero energy above 2.5kHz
            sample_val = amplitude * math.sin(2.0 * math.pi * 160.0 * t)
        elif audio_type == "speech_harmonics":
            # Fundamental + harmonics up to 6kHz simulating Chinese vowels
            f0 = 220.0
            sample_val = (
                amplitude * 0.5 * math.sin(2.0 * math.pi * f0 * t)
                + amplitude * 0.3 * math.sin(2.0 * math.pi * (2 * f0) * t)
                + amplitude * 0.15 * math.sin(2.0 * math.pi * 2600.0 * t) # > 2.5kHz
                + amplitude * 0.05 * math.sin(2.0 * math.pi * 5200.0 * t) # > 5.0kHz
            )
        else:
            sample_val = 0.0

        if clipping:
            # Force severe digital clipping on > 2% of samples
            if i % 10 == 0:
                sample_val = 32767.0

        # Clamp to 16-bit signed integer range
        int_sample = max(-32768, min(32767, int(sample_val)))
        frames.append(int_sample)

    raw_bytes = struct.pack(f"<{len(frames)}h", *frames)
    if channels == 2:
        # Interleave for stereo
        stereo_frames = []
        for s in frames:
            stereo_frames.extend([s, s])
        raw_bytes = struct.pack(f"<{len(stereo_frames)}h", *stereo_frames)

    with wave.open(str(filepath), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2) # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(raw_bytes)

    return filepath


def make_jwt_token(email: str) -> str:
    """Creates a mock JWT token string with the given email in payload claims."""
    header = base64.urlsafe_b64encode(json.dumps({"alg": "RS256", "typ": "JWT"}).encode()).decode().rstrip("=")
    payload = base64.urlsafe_b64encode(json.dumps({"email": email, "sub": "12345"}).encode()).decode().rstrip("=")
    signature = base64.urlsafe_b64encode(b"mock_signature").decode().rstrip("=")
    return f"{header}.{payload}.{signature}"


# ==============================================================================
# 1. GPU Enforcement & Impossibility of CPU Fallback Tests
# ==============================================================================

class TestGPUEnforcement:
    """Adversarial stress tests for GPU enforcement and CPU fallback elimination."""

    def test_gpu_allocation_failure_raises_runtime_error_immediately(self):
        """Verify that when T4 GPU allocation fails, RuntimeError is raised without CPU fallback."""
        mock_mgr = MagicMock()
        def mock_run_cmd(alias, cmd, timeout=None):
            if "sessions" in cmd:
                return 0, "No active sessions found", ""
            if "new" in cmd and "--gpu" in cmd:
                return 1, "", "ResourceExhausted: No NVIDIA T4 GPU available in region us-central1"
            return 0, "", ""

        mock_mgr.run_colab_command.side_effect = mock_run_cmd
        orchestrator = ColabVoiceOrchestrator(pool_manager=mock_mgr)

        with pytest.raises(RuntimeError) as exc_info:
            orchestrator._execute_colab_synthesis(
                account_alias="test_acc",
                row_id=2,
                ref_wav_path=Path("/tmp/ref.wav"),
                ref_txt_path=Path("/tmp/ref.txt"),
                force_gpu=True
            )

        err_msg = str(exc_info.value)
        assert "Colab GPU (T4) provisioning failed" in err_msg
        assert "CPU fallback is strictly disabled" in err_msg

        # EMPIRICAL PROOF: Verify colab new was NEVER called without --gpu T4
        for call_item in mock_mgr.run_colab_command.call_args_list:
            cmd = call_item[0][1]
            if "new" in cmd:
                assert "--gpu" in cmd, f"VIOLATION: Found CPU fallback command invocation: {cmd}"
                assert "T4" in cmd, f"VIOLATION: GPU type was not T4: {cmd}"

    def test_force_gpu_default_values_across_all_interfaces(self):
        """Verify that force_gpu defaults to True across all public and internal interfaces."""
        # 1. synthesize_row signature
        sig_synth = inspect.signature(ColabVoiceOrchestrator.synthesize_row)
        assert sig_synth.parameters["force_gpu"].default is True

        # 2. run signature
        sig_run = inspect.signature(ColabVoiceOrchestrator.run)
        assert sig_run.parameters["force_gpu"].default is True

        # 3. _execute_colab_synthesis signature
        sig_exec = inspect.signature(ColabVoiceOrchestrator._execute_colab_synthesis)
        assert sig_exec.parameters["force_gpu"].default is True

    def test_even_if_force_gpu_false_passed_no_cpu_fallback_occurs(self):
        """Even if force_gpu=False is passed, verify CPU fallback is eliminated from the codebase."""
        mock_mgr = MagicMock()
        mock_mgr.run_colab_command.side_effect = lambda alias, cmd, timeout=None: (
            (0, "No active sessions found", "") if "sessions" in cmd else
            (1, "", "GPU unavailable") if "new" in cmd else (0, "", "")
        )
        orchestrator = ColabVoiceOrchestrator(pool_manager=mock_mgr)

        with pytest.raises(RuntimeError) as exc_info:
            orchestrator._execute_colab_synthesis(
                account_alias="test_acc",
                row_id=2,
                ref_wav_path=Path("/tmp/ref.wav"),
                ref_txt_path=Path("/tmp/ref.txt"),
                force_gpu=False  # deliberately pass False
            )

        assert "Colab GPU (T4) provisioning failed" in str(exc_info.value)
        # Verify no CPU fallback command was executed
        for call_item in mock_mgr.run_colab_command.call_args_list:
            cmd = call_item[0][1]
            if "new" in cmd:
                assert "--gpu" in cmd

    def test_colab_worker_script_cuda_enforcement(self):
        """Verify that scripts/colab_worker_synth.py strictly asserts CUDA availability."""
        worker_path = REPO_ROOT / "scripts" / "colab_worker_synth.py"
        assert worker_path.exists()
        content = worker_path.read_text(encoding="utf-8")

        assert "if not torch.cuda.is_available():" in content
        assert "raise RuntimeError" in content
        assert "CPU execution is forbidden per policy" in content
        assert 'device = "cuda:0"' in content


# ==============================================================================
# 2. Account Security & Blacklist Inviolability Tests
# ==============================================================================

class TestAccountSecurity:
    """Adversarial stress tests for permanent blacklisting of aleron.dt@gmail.com."""

    def test_direct_registration_blacklist_rejection(self, tmp_path):
        """Verify registration of blacklisted email is immediately blocked."""
        mgr = ColabAccountManager(base_dir=str(tmp_path / "profiles"))
        with pytest.raises(PermissionError) as exc_info:
            mgr.register_account("hacker_alias", "aleron.dt@gmail.com")
        assert "PERMANENTLY BLACKLISTED" in str(exc_info.value)

    @pytest.mark.parametrize("evasion_email", [
        "Aleron.DT@gmail.com",
        "ALERON.DT@GMAIL.COM",
        "Aleron.DT@GoogleMail.com",
        "aleron.dt@customdomain.org",
        "prefix.aleron.dt.suffix@gmail.com",
        "aleron.dt",
    ])
    def test_case_and_substring_evasion_attempts_blocked(self, tmp_path, evasion_email):
        """Verify case manipulation and substring embedding attempts are blocked."""
        mgr = ColabAccountManager(base_dir=str(tmp_path / "profiles"))
        with pytest.raises(PermissionError) as exc_info:
            mgr.register_account("evasion_alias", evasion_email)
        assert "PERMANENTLY BLACKLISTED" in str(exc_info.value)

    def test_registry_file_tampering_auto_purged_on_initialization(self, tmp_path):
        """If an attacker manually edits registry.json to add blacklisted account, verify auto-purge."""
        base_dir = tmp_path / "profiles"
        base_dir.mkdir(parents=True)
        registry_file = base_dir / "registry.json"

        # Manually create poisoned registry
        poisoned_data = {
            "version": 1,
            "accounts": {
                "legit_account": {"email": "hothihuong113@gmail.com", "status": "READY"},
                "poisoned_account": {"email": "aleron.dt@gmail.com", "status": "READY"},
                "case_poisoned": {"email": "ALERON.DT@GMAIL.COM", "status": "READY"}
            },
            "last_active_account": None,
            "blacklisted_emails": ["aleron.dt@gmail.com", "aleron.dt"]
        }
        with open(registry_file, "w") as f:
            json.dump(poisoned_data, f)

        # Initialize manager - must trigger _enforce_blacklist()
        mgr = ColabAccountManager(base_dir=str(base_dir))
        active_accounts = mgr.list_accounts()
        aliases = [a["alias"] for a in active_accounts]

        assert "legit_account" in aliases
        assert "poisoned_account" not in aliases
        assert "case_poisoned" not in aliases

        # Inspect on-disk registry
        with open(registry_file, "r") as f:
            cleaned = json.load(f)
        assert "poisoned_account" not in cleaned["accounts"]
        assert "case_poisoned" not in cleaned["accounts"]

    def test_trojan_jwt_token_claims_detection_and_purge(self, tmp_path):
        """
        Trojan Attack: Account is registered under innocent alias and fake email,
        but disk token contains an id_token with email = aleron.dt@gmail.com.
        Verify that orchestrator decodes token claims, catches the trojan, purges it, and raises PermissionError.
        """
        base_dir = tmp_path / "profiles"
        mgr = ColabAccountManager(base_dir=str(base_dir))

        # Register seemingly innocent alias
        alias = "innocent_trojan"
        prof_dir = mgr.register_account(alias, "innocent@example.com")

        # Inject trojan token.json containing JWT with blacklisted email
        token_dir = prof_dir / ".config" / "colab-cli"
        token_dir.mkdir(parents=True, exist_ok=True)
        jwt_str = make_jwt_token("aleron.dt@gmail.com")
        with open(token_dir / "token.json", "w") as f:
            json.dump({"id_token": jwt_str, "access_token": "xyz"}, f)

        orchestrator = ColabVoiceOrchestrator(pool_manager=mgr)

        with pytest.raises(PermissionError) as exc_info:
            orchestrator._verify_and_validate_account(alias)

        assert ("CRITICAL ERROR" in str(exc_info.value) or "CRITICAL SECURITY VIOLATION" in str(exc_info.value))
        assert "aleron.dt@gmail.com" in str(exc_info.value)

        # Verify account was eradicated from registry
        reg = mgr._read_registry()
        assert alias not in reg["accounts"]

    def test_orchestrator_run_terminates_immediately_on_security_violation(self):
        """Verify orchestrator.run aborts immediately when a PermissionError is raised."""
        mock_mgr = MagicMock()
        mock_mgr.get_ready_account.return_value = "bad_alias"

        orchestrator = ColabVoiceOrchestrator(pool_manager=mock_mgr)
        with patch.object(orchestrator, "_verify_and_validate_account", side_effect=PermissionError("Blacklist breach")):
            with pytest.raises(PermissionError):
                orchestrator.run(row_id=2)

        # Synthesis must never have been called
        assert orchestrator.active_session_name is None


# ==============================================================================
# 3. Session Cleanup Resilience Tests
# ==============================================================================

class TestTeardownResilience:
    """Adversarial stress tests for VM teardown resilience, retry logic, and verification."""

    def test_teardown_retries_on_transient_failure_and_succeeds(self):
        """Verify that _safe_teardown_session retries on failure and succeeds on subsequent attempt."""
        mock_mgr = MagicMock()
        calls = []

        def mock_run_cmd(alias, cmd, timeout=None):
            calls.append(cmd)
            if "stop" in cmd:
                if len(calls) == 1:
                    return 1, "", "Connection reset by peer"
                return 0, "Session voice_worker_123 stopped successfully", ""
            return 0, "", ""

        mock_mgr.run_colab_command.side_effect = mock_run_cmd
        orchestrator = ColabVoiceOrchestrator(pool_manager=mock_mgr)

        with patch("time.sleep", return_value=None):
            success = orchestrator._safe_teardown_session("test_acc", "voice_worker_123", max_retries=3)

        assert success is True
        # Verify 2 stop calls were made
        stop_calls = [c for c in calls if "stop" in c]
        assert len(stop_calls) == 2

    @pytest.mark.parametrize("idempotent_msg", [
        (0, "Session voice_worker_123 stopped"),
        (1, "Session not found"),
        (1, "Session already terminated"),
        (1, "No active session matching voice_worker_123"),
    ])
    def test_teardown_recognizes_idempotent_stop_messages(self, idempotent_msg):
        """Verify that idempotent messages ('not found', 'terminated', etc.) are treated as clean stops immediately."""
        mock_mgr = MagicMock()
        code, out = idempotent_msg
        mock_mgr.run_colab_command.return_value = (code, out, "")
        orchestrator = ColabVoiceOrchestrator(pool_manager=mock_mgr)

        success = orchestrator._safe_teardown_session("test_acc", "voice_worker_123", max_retries=3)
        assert success is True
        # Should stop after first attempt without retrying
        assert mock_mgr.run_colab_command.call_count == 1

    def test_teardown_post_verification_via_colab_sessions(self):
        """Verify post-teardown verification queries 'colab sessions' when stop attempts fail."""
        mock_mgr = MagicMock()

        # Case A: 3 stop failures, but sessions query confirms session is absent -> returns True
        def mock_cmd_absent(alias, cmd, timeout=None):
            if "stop" in cmd:
                return 1, "", "Server error 500"
            if "sessions" in cmd:
                return 0, "No active sessions found", ""
            return 0, "", ""

        mock_mgr.run_colab_command.side_effect = mock_cmd_absent
        orchestrator = ColabVoiceOrchestrator(pool_manager=mock_mgr)
        with patch("time.sleep", return_value=None):
            result = orchestrator._safe_teardown_session("test_acc", "voice_worker_123", max_retries=3)
        assert result is True

        # Case B: 3 stop failures, and sessions query confirms session STILL active -> returns False
        def mock_cmd_still_active(alias, cmd, timeout=None):
            if "stop" in cmd:
                return 1, "", "Server error 500"
            if "sessions" in cmd:
                return 0, "[voice_worker_123] https://colab.research.google.com/...", ""
            return 0, "", ""

        mock_mgr.run_colab_command.side_effect = mock_cmd_still_active
        with patch("time.sleep", return_value=None):
            result_failed = orchestrator._safe_teardown_session("test_acc", "voice_worker_123", max_retries=3)
        assert result_failed is False

    def test_finally_block_guarantees_teardown_on_any_exception(self):
        """Verify that _execute_colab_synthesis tears down session in finally: block even on worker failure."""
        mock_mgr = MagicMock()
        mock_teardown = MagicMock(return_value=True)

        def mock_run_cmd(alias, cmd, timeout=None):
            if "sessions" in cmd:
                return 0, "No active sessions found", ""
            if "new" in cmd:
                return 0, "VM provisioned", ""
            if "upload" in cmd:
                return 0, "uploaded", ""
            if "exec" in cmd:
                # Worker crashes
                return 1, "", "RuntimeError: Out of Memory during inference"
            return 0, "", ""

        mock_mgr.run_colab_command.side_effect = mock_run_cmd
        orchestrator = ColabVoiceOrchestrator(pool_manager=mock_mgr)
        orchestrator._safe_teardown_session = mock_teardown

        with pytest.raises(RuntimeError):
            orchestrator._execute_colab_synthesis(
                account_alias="test_acc",
                row_id=2,
                ref_wav_path=Path("/tmp/ref.wav"),
                ref_txt_path=Path("/tmp/ref.txt"),
                force_gpu=True
            )

        # Teardown must have been called in finally block
        assert mock_teardown.called
        assert orchestrator.active_session_name is None

    def test_pre_flight_cleanup_identifies_and_stops_orphan_sessions(self):
        """Verify that _pre_flight_cleanup parses active sessions and terminates orphan VMs."""
        mock_mgr = MagicMock()
        mock_mgr.run_colab_command.side_effect = [
            # 1. Initial sessions query with 2 orphans
            (0, "[voice_worker_1111] https://colab.research.google.com/1\n[voice_worker_2222] https://colab.research.google.com/2", ""),
            # 2. Stop orphan 1
            (0, "stopped", ""),
            # 3. Stop orphan 2
            (0, "stopped", ""),
            # 4. Post-cleanup check
            (0, "No active sessions found", "")
        ]

        orchestrator = ColabVoiceOrchestrator(pool_manager=mock_mgr)
        orchestrator._pre_flight_cleanup("test_acc")

        # Verify stop commands were issued for both orphans
        calls = mock_mgr.run_colab_command.call_args_list
        stopped_sessions = [c[0][1][2] for c in calls if c[0][1][0] == "stop"]
        assert "voice_worker_1111" in stopped_sessions
        assert "voice_worker_2222" in stopped_sessions


# ==============================================================================
# 4. Audio QC Duration Calibration & Acoustic Integrity Tests
# ==============================================================================

class TestAudioQCCalibration:
    """Adversarial stress tests for DEFAULT_MIN_DURATION = 0.25 calibration and signal integrity."""

    def test_default_min_duration_calibrated_to_025(self):
        """Verify DEFAULT_MIN_DURATION constant is exactly 0.25."""
        assert audio_qc.DEFAULT_MIN_DURATION == 0.25

    def test_short_valid_speech_accepted_without_false_positives(self, tmp_path):
        """
        Verify that short genuine speech (0.25s - 0.39s) passes check_wav_file with 0.25 threshold,
        whereas the old 0.40 threshold falsely rejected it.
        """
        for dur in [0.25, 0.27, 0.30, 0.35]:
            wav_file = create_synthetic_wav(
                tmp_path / f"speech_{int(dur*100)}.wav",
                duration=dur,
                audio_type="speech_harmonics",
                amplitude=3000.0
            )

            # Test with calibrated threshold (0.25s default) -> MUST PASS
            valid, reason, meta = check_wav_file(str(wav_file))
            assert valid is True, f"Failed at duration {dur}s with reason: {reason}"
            assert meta["duration"] == dur
            assert meta["rms"] >= 500.0

            # Test with old threshold (0.40s) -> PROVE IT WOULD HAVE FAILED
            valid_old, reason_old, _ = check_wav_file(str(wav_file), min_duration=0.40)
            assert valid_old is False
            assert "below minimum threshold 0.4" in reason_old

    def test_silence_rejection_preserved_across_all_durations(self, tmp_path):
        """Verify that silence (RMS < 500) is strictly rejected regardless of duration."""
        for dur in [0.25, 0.30, 0.50, 1.00, 2.00]:
            silence_file = create_synthetic_wav(
                tmp_path / f"silence_{int(dur*100)}.wav",
                duration=dur,
                audio_type="silence",
                amplitude=0.0
            )
            valid, reason, meta = check_wav_file(str(silence_file))
            assert valid is False, f"Silence of duration {dur}s was incorrectly accepted!"
            assert "below silence threshold 500" in reason
            assert meta["rms"] < 500.0

    def test_fake_synthetic_tone_rejection_preserved(self, tmp_path):
        """
        Verify that synthetic pure sine wave tones (160Hz / 440Hz buzzers)
        are strictly rejected by Check 7 FFT spectral check, despite high RMS and duration.
        """
        for dur in [0.25, 0.30, 0.50, 1.00]:
            tone_file = create_synthetic_wav(
                tmp_path / f"tone_{int(dur*100)}.wav",
                duration=dur,
                audio_type="pure_sine",
                amplitude=4000.0
            )
            valid, reason, meta = check_wav_file(str(tone_file))
            assert valid is False, f"Synthetic tone of duration {dur}s was incorrectly accepted!"
            assert "Synthetic tone / sine-wave facade rejected" in reason
            # Verify high-frequency energy ratio is below required threshold
            assert meta["spectral_energy_2500hz_pct"] < 0.05 or meta["spectral_energy_5000hz_pct"] == 0.0

    def test_sub_threshold_truncated_blips_strictly_rejected(self, tmp_path):
        """Verify that audio shorter than 0.25s (e.g. 0.10s, 0.20s, 0.24s) is strictly rejected."""
        for dur in [0.05, 0.10, 0.15, 0.20, 0.24]:
            blip_file = create_synthetic_wav(
                tmp_path / f"blip_{int(dur*100)}.wav",
                duration=dur,
                audio_type="speech_harmonics",
                amplitude=3000.0
            )
            valid, reason, meta = check_wav_file(str(blip_file))
            assert valid is False, f"Sub-threshold blip of duration {dur}s was incorrectly accepted!"
            assert "below minimum threshold 0.25" in reason

    def test_digital_clipping_rejection_preserved(self, tmp_path):
        """Verify that files with digital clipping > 1% are strictly rejected."""
        clipped_file = create_synthetic_wav(
            tmp_path / "clipped.wav",
            duration=0.50,
            audio_type="speech_harmonics",
            clipping=True
        )
        valid, reason, meta = check_wav_file(str(clipped_file))
        assert valid is False
        assert "Severe digital clipping detected" in reason
        assert meta["clipping_ratio"] > 0.01

    def test_sample_rate_and_channels_enforcement_preserved(self, tmp_path):
        """Verify that non-24000Hz or non-mono files are strictly rejected."""
        # Test 16000Hz
        sr_file = create_synthetic_wav(tmp_path / "16khz.wav", duration=0.5, sample_rate=16000)
        v, r, _ = check_wav_file(str(sr_file))
        assert v is False and "Invalid sample rate" in r

        # Test stereo (channels=2)
        stereo_file = create_synthetic_wav(tmp_path / "stereo.wav", duration=0.5, channels=2)
        v2, r2, _ = check_wav_file(str(stereo_file))
        assert v2 is False and "Invalid channels" in r2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
