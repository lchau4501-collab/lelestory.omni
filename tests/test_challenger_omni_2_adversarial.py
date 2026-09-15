"""
Empirical Adversarial Test Suite for Challenger Omni-2
Milestone M1: LeLe Storybook OmniVoice Voiceover Pipeline (Row #3 《井底之蛙》)

Focus Areas:
1. Adversarial Blacklist Security Enforcement against aleron.dt@gmail.com
2. Colab Account Pool Stress Testing & 0 Active Sessions Invariant
3. Google Drive Folder 1D8ZZpzMiXTo_S_q56jpj0c5p6YKih72q File Integrity & MD5 Matching
4. Google Sheet story Tab 21px Row Height Invariant & Row #3 State Verification
"""

import os
import sys
import json
import time
import shutil
import base64
import hashlib
import tempfile
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from colab_rotator import (
    ColabAccountManager,
    BLACKLISTED_EMAILS,
    DEFAULT_COOLDOWN_SECONDS
)
from colab_orchestrator import ColabVoiceOrchestrator, MANDATED_ACCOUNT_EMAIL
from drive_resolver import extract_drive_folder_id, get_service_account_credentials

VOICE_FOLDER_ID_ROW_3 = "1D8ZZpzMiXTo_S_q56jpj0c5p6YKih72q"
LOCAL_VOICE_DIR_ROW_3 = REPO_ROOT / "artifacts" / "voice_row_3"
SPREADSHEET_ID = "1b6LNl7JHRiCsjK1w9VuD86GLqAfmSOtDUOm5whrGdH0"

EXPECTED_17_WAV_NAMES = {
    "title.wav",
    "scene1.wav", "scene2.wav", "scene3.wav", "scene4.wav", "scene5.wav",
    "scene6.wav", "scene7.wav", "scene8.wav", "scene9.wav", "scene10.wav",
    "vocab_1.wav", "vocab_2.wav", "vocab_3.wav", "vocab_4.wav", "vocab_5.wav",
    "outro_loop.wav"
}


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def isolated_manager():
    temp_dir = tempfile.mkdtemp(prefix="test_mgr_iso_")
    mgr = ColabAccountManager(base_dir=temp_dir)
    yield mgr
    shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================================
# 1. Adversarial Blacklist Security Enforcement Tests
# ============================================================================

class TestAdversarialBlacklistSecurity:
    """Rigorous stress-testing of 4-layer blacklist against aleron.dt@gmail.com."""

    def test_blacklist_invariants(self):
        """Verify blacklist constants strictly include prohibited identifiers."""
        assert "aleron.dt@gmail.com" in BLACKLISTED_EMAILS
        assert "aleron.dt" in BLACKLISTED_EMAILS

    @pytest.mark.parametrize("bad_email", [
        "aleron.dt@gmail.com",
        "ALERON.DT@GMAIL.COM",
        "  aleron.dt@gmail.com  ",
        "aleron.dt@googlemail.com",
        "prefix_aleron.dt@gmail.com",
        "aleron.dt+test@gmail.com",
    ])
    def test_layer1_register_rejection_emails(self, isolated_manager, bad_email):
        """Layer 1: Reject any variation of blacklisted email at registration."""
        with pytest.raises(PermissionError) as exc_info:
            isolated_manager.register_account("test_bad_alias", bad_email)
        assert "PERMANENTLY BLACKLISTED" in str(exc_info.value)
        # Profile directory must NOT be created
        prof = isolated_manager.get_account_profile_dir("test_bad_alias")
        assert not prof.exists()

    @pytest.mark.parametrize("bad_alias", [
        "aleron.dt",
        "aleron.dt@gmail.com",
        "ALERON.DT",
        "my_aleron.dt_acc",
    ])
    def test_layer1_register_rejection_aliases(self, isolated_manager, bad_alias):
        """Layer 1: Reject any alias containing blacklisted identifier."""
        with pytest.raises(PermissionError) as exc_info:
            isolated_manager.register_account(bad_alias, "innocent_user@example.com")
        assert "PERMANENTLY BLACKLISTED" in str(exc_info.value)

    def test_layer2_filesystem_and_registry_purge(self, isolated_manager):
        """Layer 2: Corrupted registry or rogue filesystem folders are wiped instantly on init."""
        # Inject corrupted account into registry
        reg = isolated_manager._read_registry()
        reg["accounts"]["corrupted_aleron"] = {
            "email": "ALERON.DT@GMAIL.COM",
            "status": "READY"
        }
        isolated_manager._write_registry(reg)

        # Create rogue on-disk directory matching blacklist
        rogue_dir = isolated_manager.base_dir / "aleron.dt_rogue"
        rogue_dir.mkdir(parents=True, exist_ok=True)
        (rogue_dir / "secret_key.txt").write_text("compromised")

        # Create corrupted account profile dir
        corrupted_dir = isolated_manager.get_account_profile_dir("corrupted_aleron")
        corrupted_dir.mkdir(parents=True, exist_ok=True)
        (corrupted_dir / "token.json").write_text("malicious")

        # Re-initialize manager
        fresh_mgr = ColabAccountManager(base_dir=str(isolated_manager.base_dir))

        # Both rogue directory and corrupted profile directory must be wiped completely
        assert not rogue_dir.exists(), "Rogue blacklisted directory was not wiped from disk!"
        assert not corrupted_dir.exists(), "Corrupted blacklisted profile was not wiped from disk!"
        assert "corrupted_aleron" not in [a["alias"] for a in fresh_mgr.list_accounts()]

    def test_layer3_trojan_token_jwt_claim_trap(self, isolated_manager):
        """Layer 3: Trojan OAuth JWT token containing blacklisted email is trapped and directory destroyed."""
        alias = "innocent_looking_account"
        prof_dir = isolated_manager.register_account(alias, "innocent@gmail.com")
        token_dir = prof_dir / ".config" / "colab-cli"
        token_dir.mkdir(parents=True, exist_ok=True)

        # Forge JWT id_token with blacklisted email claim
        header_b64 = base64.urlsafe_b64encode(b'{"alg":"RS256"}').decode().rstrip("=")
        payload_b64 = base64.urlsafe_b64encode(b'{"email":"aleron.dt@gmail.com","sub":"12345"}').decode().rstrip("=")
        trojan_jwt = f"{header_b64}.{payload_b64}.fakesig"

        token_file = token_dir / "token.json"
        token_file.write_text(json.dumps({"id_token": trojan_jwt, "access_token": "ya29.xyz"}))

        # verify_account_token_email MUST raise PermissionError and wipe prof_dir
        with pytest.raises(PermissionError) as exc_info:
            isolated_manager.verify_account_token_email(alias)

        assert "BLACKLISTED" in str(exc_info.value)
        assert not prof_dir.exists(), "Profile directory must be physically deleted upon trojan token detection!"
        assert alias not in [a["alias"] for a in isolated_manager.list_accounts()]

    @pytest.mark.parametrize("env_var", ["COLAB_USER", "USER_EMAIL", "GOOGLE_ACCOUNT"])
    def test_layer4_execution_env_trap(self, isolated_manager, monkeypatch, env_var):
        """Layer 4: Execution refuses command if hostile environment variable contains blacklisted email."""
        alias = "valid_runner"
        prof_dir = isolated_manager.register_account(alias, "valid@gmail.com")

        monkeypatch.setenv(env_var, "aleron.dt@gmail.com")

        with pytest.raises(PermissionError) as exc_info:
            isolated_manager.run_colab_command(alias, ["sessions"])

        assert "STRICT POLICY" in str(exc_info.value)
        assert not prof_dir.exists(), f"Profile dir must be deleted when {env_var} triggers blacklist!"

    def test_orchestrator_verification_defense_in_depth(self, isolated_manager):
        """ColabVoiceOrchestrator._verify_and_validate_account traps blacklisted email and purges account."""
        alias = "bad_orch_account"
        # Force inject into registry to bypass layer 1
        reg = isolated_manager._read_registry()
        reg["accounts"][alias] = {
            "email": "aleron.dt@gmail.com",
            "status": "READY"
        }
        isolated_manager._write_registry(reg)
        prof_dir = isolated_manager.get_account_profile_dir(alias)
        prof_dir.mkdir(parents=True, exist_ok=True)

        orch = ColabVoiceOrchestrator(pool_manager=isolated_manager)
        with pytest.raises(PermissionError) as exc_info:
            orch._verify_and_validate_account(alias)

        assert "CRITICAL SECURITY VIOLATION" in str(exc_info.value)
        assert not prof_dir.exists()

    def test_orchestrator_rclone_remote_isolation(self):
        """Verify codebase strictly uses authorized remote analib-vpsg24gb: and never gdrive_rd:."""
        orch_source = (REPO_ROOT / "src" / "colab_orchestrator.py").read_text(encoding="utf-8")
        assert "analib-vpsg24gb:" in orch_source
        assert "gdrive_rd:" not in orch_source


# ============================================================================
# 2. Colab Account Pool Stress Testing & Zero Active Sessions
# ============================================================================

class TestColabAccountPoolStress:
    """Stress-testing rotation state transitions, cooldown timers, and live VM session state."""

    def test_single_account_cooldown_failover(self, isolated_manager):
        """When account 1 enters cooldown, selector immediately pivots to account 2."""
        isolated_manager.register_account("acc_1", "u1@example.com")
        isolated_manager.register_account("acc_2", "u2@example.com")

        for a in ["acc_1", "acc_2"]:
            p = isolated_manager.get_account_profile_dir(a) / ".config" / "colab-cli" / "token.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({"token": "valid"}))

        isolated_manager.mark_cooldown("acc_1", duration_seconds=1800, reason="Quota Exceeded")
        assert isolated_manager.select_active_account() == "acc_2"

    def test_cascade_all_accounts_cooldown(self, isolated_manager):
        """When all accounts are in cooldown, select_active_account returns None and earliest returns lowest wait."""
        aliases = [f"worker_{i}" for i in range(1, 6)]
        for idx, a in enumerate(aliases, 1):
            isolated_manager.register_account(a, f"user{idx}@example.com")
            p = isolated_manager.get_account_profile_dir(a) / ".config" / "colab-cli" / "token.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({"token": "valid"}))
            # Varying cooldowns: worker_1 has lowest cooldown (100s), worker_5 highest (500s)
            isolated_manager.mark_cooldown(a, duration_seconds=100 * idx, reason=f"Limit {idx}")

        # No account is ready
        assert isolated_manager.select_active_account() is None

        # Earliest available must be worker_1 with ~100s remaining
        earliest_alias, rem_sec = isolated_manager.get_earliest_available_account()
        assert earliest_alias == "worker_1"
        assert 90 <= rem_sec <= 100

    def test_cooldown_automatic_recovery(self, isolated_manager):
        """Account in cooldown automatically flips to READY once cooldown_until passes."""
        alias = "quick_cooldown_acc"
        isolated_manager.register_account(alias, "quick@example.com")
        p = isolated_manager.get_account_profile_dir(alias) / ".config" / "colab-cli" / "token.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"token": "valid"}))

        isolated_manager.mark_cooldown(alias, duration_seconds=1, reason="Transient test")
        accs = isolated_manager.list_accounts()
        assert accs[0]["status"] == "COOLING_DOWN"

        # Sleep past cooldown
        time.sleep(1.2)
        fresh_accs = isolated_manager.list_accounts()
        assert fresh_accs[0]["status"] == "READY"
        assert isolated_manager.select_active_account() == alias

    def test_prioritization_by_failure_and_success_counts(self, isolated_manager):
        """Accounts are prioritized: fewer failures first, then more successes."""
        isolated_manager.register_account("acc_high_fail", "hf@example.com")
        isolated_manager.register_account("acc_clean", "clean@example.com")
        isolated_manager.register_account("acc_veteran", "vet@example.com")

        for a in ["acc_high_fail", "acc_clean", "acc_veteran"]:
            p = isolated_manager.get_account_profile_dir(a) / ".config" / "colab-cli" / "token.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({"token": "valid"}))

        reg = isolated_manager._read_registry()
        reg["accounts"]["acc_high_fail"]["failure_count"] = 3
        reg["accounts"]["acc_clean"]["failure_count"] = 0
        reg["accounts"]["acc_clean"]["success_count"] = 1
        reg["accounts"]["acc_veteran"]["failure_count"] = 0
        reg["accounts"]["acc_veteran"]["success_count"] = 10
        isolated_manager._write_registry(reg)

        # acc_veteran has 0 failures and 10 successes -> should be chosen first
        selected = isolated_manager.select_active_account()
        assert selected == "acc_veteran"

    def test_live_zero_active_colab_sessions_across_all_5_accounts(self):
        """Empirical live check: query Google Colab CLI on host for all 5 pool accounts; assert 0 active sessions."""
        live_mgr = ColabAccountManager()
        accounts = live_mgr.list_accounts()
        assert len(accounts) == 5, f"Expected 5 accounts in live pool, found {len(accounts)}"

        for acc in accounts:
            alias = acc["alias"]
            email = acc["email"]
            code, stdout, stderr = live_mgr.run_colab_command(alias, ["sessions"], timeout=25)
            combined = f"{stdout} {stderr}".strip()

            assert code == 0, f"Account [{alias}] colab sessions query failed (code {code}): {combined}"
            assert "[colab] No active sessions found on server." in combined, (
                f"Account [{alias}] ({email}) has active or lingering Colab sessions! Found: {combined}"
            )


# ============================================================================
# 3. Google Drive File Integrity & Byte-Level Verification
# ============================================================================

class TestGoogleDriveFileIntegrity:
    """Empirical inspection of Google Drive folder 1D8ZZpzMiXTo_S_q56jpj0c5p6YKih72q."""

    @pytest.fixture(scope="class")
    def drive_items(self):
        """Fetch file metadata directly from Google Drive via rclone lsjson."""
        cmd = [
            "rclone", "lsjson",
            "analib-vpsg24gb:",
            "--drive-root-folder-id", VOICE_FOLDER_ID_ROW_3
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        assert res.returncode == 0, f"rclone lsjson failed: {res.stderr}"
        items = json.loads(res.stdout)
        return items

    def test_drive_item_count_exactly_17(self, drive_items):
        """Verify exactly 17 files exist in the remote voice folder."""
        assert len(drive_items) == 17, f"Expected exactly 17 files, got {len(drive_items)}"

    def test_drive_file_names_conform_to_spec(self, drive_items):
        """Verify names of all 17 files match the canonical Row #3 specification."""
        names = {item["Name"] for item in drive_items}
        missing = EXPECTED_17_WAV_NAMES - names
        extra = names - EXPECTED_17_WAV_NAMES
        assert not missing, f"Missing files on Google Drive: {missing}"
        assert not extra, f"Unexpected files on Google Drive: {extra}"

    def test_drive_no_zero_byte_files(self, drive_items):
        """Verify no files on Google Drive are empty (0 bytes)."""
        for item in drive_items:
            assert item["Size"] > 0, f"File {item['Name']} on Google Drive has 0 bytes!"
            assert item["Size"] >= 30_000, f"File {item['Name']} is suspiciously small: {item['Size']} bytes"

    def test_drive_mime_types(self, drive_items):
        """Verify all remote items have valid audio/wav or audio/x-wav MIME type."""
        for item in drive_items:
            mime = item.get("MimeType", "")
            assert mime in ("audio/wav", "audio/x-wav"), f"File {item['Name']} has invalid MIME: {mime}"

    def test_drive_md5_byte_for_byte_match_with_local_artifacts(self, drive_items):
        """Verify 100% cryptographic MD5 match between Google Drive files and local artifacts."""
        assert LOCAL_VOICE_DIR_ROW_3.exists(), f"Local voice dir {LOCAL_VOICE_DIR_ROW_3} missing"

        cmd = [
            "rclone", "md5sum",
            "analib-vpsg24gb:",
            "--drive-root-folder-id", VOICE_FOLDER_ID_ROW_3
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        assert res.returncode == 0, f"rclone md5sum failed: {res.stderr}"

        remote_md5_map = {}
        for line in res.stdout.strip().splitlines():
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                remote_md5_map[parts[1].strip()] = parts[0].strip()

        assert len(remote_md5_map) == 17, f"Expected 17 remote MD5 entries, got {len(remote_md5_map)}"

        for fname in EXPECTED_17_WAV_NAMES:
            local_path = LOCAL_VOICE_DIR_ROW_3 / fname
            assert local_path.exists(), f"Local file {fname} does not exist in artifacts!"

            with open(local_path, "rb") as f:
                local_md5 = hashlib.md5(f.read()).hexdigest()

            remote_md5 = remote_md5_map.get(fname)
            assert remote_md5 is not None, f"Remote MD5 missing for {fname}"
            assert local_md5 == remote_md5, (
                f"MD5 mismatch for {fname}! Local: {local_md5}, Remote: {remote_md5}"
            )


# ============================================================================
# 4. Google Sheet Invariant & State Stress Testing
# ============================================================================

class TestGoogleSheetInvariants:
    """Empirical verification of Google Sheets API v4 metadata and row height invariants."""

    @pytest.fixture(scope="class")
    def sheets_service(self):
        from googleapiclient.discovery import build
        creds = get_service_account_credentials()
        assert creds is not None, "Failed to load Google Service Account credentials"
        return build("sheets", "v4", credentials=creds)

    def test_21px_row_height_invariant_rows_1_to_20(self, sheets_service):
        """Assert that rows 1 through 20 on tab 'story' strictly maintain pixelSize = 21."""
        meta = sheets_service.spreadsheets().get(
            spreadsheetId=SPREADSHEET_ID,
            ranges=["story!A1:Q20"],
            fields="sheets(properties(sheetId,title),data(rowMetadata(pixelSize)))"
        ).execute()

        sheet = meta["sheets"][0]
        assert sheet["properties"]["title"] == "story"

        row_metadata = sheet["data"][0].get("rowMetadata", [])
        assert len(row_metadata) >= 20, f"Expected at least 20 row metadata entries, got {len(row_metadata)}"

        violations = []
        for idx in range(20):
            row_num = idx + 1
            px = row_metadata[idx].get("pixelSize")
            if px != 21:
                violations.append((row_num, px))

        assert not violations, f"Row height invariant violated! Non-21px rows: {violations}"

    def test_story_row_3_values_integrity(self, sheets_service):
        """Assert Story Row #3 has Status='Voice', Col G=Drive URL, Col Q=GK3 Passed."""
        vals = sheets_service.spreadsheets().values().get(
            spreadsheetId=SPREADSHEET_ID,
            range="story!A3:Q3"
        ).execute().get("values", [[]])[0]

        assert len(vals) >= 17, f"Expected at least 17 columns in Row #3, got {len(vals)}"

        col_a_id = vals[0]
        col_b_title = vals[1]
        col_d_status = vals[3]
        col_g_voice_url = vals[6]
        col_q_notes = vals[16]

        assert col_a_id == "3", f"Expected Col A to be '3', got '{col_a_id}'"
        assert col_b_title == "井底之蛙", f"Expected Col B to be '井底之蛙', got '{col_b_title}'"
        assert col_d_status == "Voice", f"Expected Col D (Status) to be 'Voice', got '{col_d_status}'"
        assert VOICE_FOLDER_ID_ROW_3 in col_g_voice_url, (
            f"Expected Col G to point to folder {VOICE_FOLDER_ID_ROW_3}, got '{col_g_voice_url}'"
        )
        assert "GK3 Passed" in col_q_notes, (
            f"Expected Col Q to contain 'GK3 Passed', got '{col_q_notes}'"
        )
        assert "17 WAV files verified" in col_q_notes, (
            f"Expected Col Q to confirm 17 WAV files, got '{col_q_notes}'"
        )
