import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import pytest
import shutil
import tempfile
import json
import base64
import time
from pathlib import Path
from colab_rotator import ColabAccountManager, BLACKLISTED_EMAILS, DEFAULT_COOLDOWN_SECONDS, DEFAULT_MAX_RETRIES


@pytest.fixture
def temp_manager():
    temp_dir = tempfile.mkdtemp()
    mgr = ColabAccountManager(base_dir=temp_dir)
    yield mgr
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_blacklist_hardcoded():
    assert "aleron.dt@gmail.com" in BLACKLISTED_EMAILS
    assert "aleron.dt" in BLACKLISTED_EMAILS


def test_cannot_register_blacklisted_email(temp_manager):
    with pytest.raises(PermissionError):
        temp_manager.register_account("bad_account", "aleron.dt@gmail.com")

    with pytest.raises(PermissionError):
        temp_manager.register_account("aleron.dt", "innocent@gmail.com")


def test_blacklist_purge_on_init(temp_manager):
    # Simulate a corrupted registry with blacklisted email
    reg = temp_manager._read_registry()
    reg["accounts"]["sneaky"] = {"email": "aleron.dt@gmail.com", "status": "READY"}
    temp_manager._write_registry(reg)

    # Also create sneaky profile directory
    sneaky_dir = temp_manager.get_account_profile_dir("sneaky")
    sneaky_dir.mkdir(parents=True, exist_ok=True)

    # Re-init should enforce blacklist and purge it from both registry and disk
    new_mgr = ColabAccountManager(base_dir=str(temp_manager.base_dir))
    assert "sneaky" not in [a["alias"] for a in new_mgr.list_accounts()]
    assert not sneaky_dir.exists()


def test_register_and_list_accounts(temp_manager):
    p1 = temp_manager.register_account("gmail_1", "user1@gmail.com")
    assert p1.exists()

    accs = temp_manager.list_accounts()
    assert len(accs) == 1
    assert accs[0]["alias"] == "gmail_1"
    assert accs[0]["status"] == "NO_TOKEN"


def test_cooldown_and_rotation(temp_manager):
    temp_manager.register_account("acc1", "user1@gmail.com")
    temp_manager.register_account("acc2", "user2@gmail.com")

    # Mock tokens
    for alias in ["acc1", "acc2"]:
        p = temp_manager.get_account_profile_dir(alias) / ".config" / "colab-cli" / "token.json"
        p.write_text(json.dumps({"mock": True}))

    accs = temp_manager.list_accounts()
    assert all(a["status"] == "READY" for a in accs)

    # Put acc1 in cooldown
    temp_manager.mark_cooldown("acc1", duration_seconds=3600, reason="Quota exceeded")
    
    # Selection should pick acc2
    active = temp_manager.select_active_account()
    assert active == "acc2"


def test_default_1800s_cooldown_tracking(temp_manager):
    assert DEFAULT_COOLDOWN_SECONDS == 1800
    temp_manager.register_account("acc1", "user1@gmail.com")
    p = temp_manager.get_account_profile_dir("acc1") / ".config" / "colab-cli" / "token.json"
    p.write_text(json.dumps({"mock": True}))

    # Mark cooldown with default duration
    temp_manager.mark_cooldown("acc1")
    accs = temp_manager.list_accounts()
    assert accs[0]["status"] == "COOLING_DOWN"
    assert 1700 <= accs[0]["cooldown_remaining_sec"] <= 1800

    # Earliest available
    earliest_alias, rem_sec = temp_manager.get_earliest_available_account()
    assert earliest_alias == "acc1"
    assert 1700 <= rem_sec <= 1800


def test_layer_3_trojan_token_purge(temp_manager):
    alias = "trojan_acc"
    prof_dir = temp_manager.register_account(alias, "innocent@gmail.com")
    token_dir = prof_dir / ".config" / "colab-cli"
    token_dir.mkdir(parents=True, exist_ok=True)

    header = base64.urlsafe_b64encode(json.dumps({"alg": "HS256"}).encode()).decode().rstrip("=")
    payload = base64.urlsafe_b64encode(json.dumps({"email": "aleron.dt@gmail.com"}).encode()).decode().rstrip("=")
    fake_jwt = f"{header}.{payload}.signature"
    (token_dir / "token.json").write_text(json.dumps({"id_token": fake_jwt}))

    # Verify layer 3 catches and instantly deletes profile directory
    with pytest.raises(PermissionError):
        temp_manager.verify_account_token_email(alias)

    assert not prof_dir.exists()
    assert alias not in [a["alias"] for a in temp_manager.list_accounts()]


def test_layer_4_execution_guard_with_env(temp_manager, monkeypatch):
    alias = "test_acc"
    prof_dir = temp_manager.register_account(alias, "valid@gmail.com")

    # Set environment variable matching blacklisted email
    monkeypatch.setenv("COLAB_USER", "aleron.dt@gmail.com")

    with pytest.raises(PermissionError) as exc_info:
        temp_manager.run_colab_command(alias, ["sessions"])
    assert "STRICT POLICY" in str(exc_info.value)
    assert not prof_dir.exists()
