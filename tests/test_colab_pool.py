import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import pytest
import shutil
import tempfile
from pathlib import Path
from colab_rotator import ColabAccountManager, BLACKLISTED_EMAILS


@pytest.fixture
def temp_manager():
    temp_dir = tempfile.mkdtemp()
    mgr = ColabAccountManager(base_dir=temp_dir)
    yield mgr
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_blacklist_hardcoded():
    assert "aleron.dt@gmail.com" in BLACKLISTED_EMAILS


def test_cannot_register_blacklisted_email(temp_manager):
    with pytest.raises(PermissionError):
        temp_manager.register_account("bad_account", "aleron.dt@gmail.com")


def test_blacklist_purge_on_init(temp_manager):
    # Simulate a corrupted registry with blacklisted email
    reg = temp_manager._read_registry()
    reg["accounts"]["sneaky"] = {"email": "aleron.dt@gmail.com", "status": "READY"}
    temp_manager._write_registry(reg)

    # Re-init should enforce blacklist and purge it
    new_mgr = ColabAccountManager(base_dir=str(temp_manager.base_dir))
    assert "sneaky" not in [a["alias"] for a in new_mgr.list_accounts()]


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
        p.write_text("{\"mock\": true}")

    accs = temp_manager.list_accounts()
    assert all(a["status"] == "READY" for a in accs)

    # Put acc1 in cooldown
    temp_manager.mark_cooldown("acc1", duration_seconds=3600, reason="Quota exceeded")
    
    # Selection should pick acc2
    active = temp_manager.select_active_account()
    assert active == "acc2"
