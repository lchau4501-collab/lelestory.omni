"""
Colab Account Rotation Pool Manager for LeLe Storybook Video Engine.
Manages a pool of Google Colab user profiles to rotate across multiple Gmail accounts (gmail_1 to gmail_5),
handling quota limits, 1800s cooldown tracking, run counters, 3 retries, and running colab commands with isolated environments.

STRICT HARDENING POLICY:
The account "aleron.dt@gmail.com" is PERMANENTLY FORBIDDEN and BLACKLISTED across 4 layers:
1. Registration & alias validation
2. Registry & filesystem auto-purging
3. Token JWT claim and whoami inspection
4. Command execution and environment runtime checks
Any violation causes instant profile directory deletion and a fatal abort (PermissionError).
"""

import os
import sys
import json
import time
import shutil
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import shlex

logger = logging.getLogger("lelestory.omni.colab_rotator")

PROFILES_BASE_DIR = os.path.expanduser("~/.config/colab_profiles")
REGISTRY_FILE = os.path.join(PROFILES_BASE_DIR, "registry.json")
COLAB_EXECUTABLE = os.path.expanduser("~/.local/bin/colab")

# User Mandate: aleron.dt@gmail.com must NEVER be used in Colab under any circumstances
BLACKLISTED_EMAILS = {
    "aleron.dt@gmail.com",
    "aleron.dt"
}

# Strict User Policy: 30-minute (1800s) cooldown tracking
DEFAULT_COOLDOWN_SECONDS = 1800
DEFAULT_MAX_RETRIES = 3


class ColabAccountManager:
    """Manages multi-account profiles and rotation for google-colab-cli with 4-layer blacklist protection."""

    def __init__(self, base_dir: str = PROFILES_BASE_DIR):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.base_dir / "registry.json"
        self._ensure_registry()
        self._enforce_blacklist()

    def _ensure_registry(self) -> None:
        if not self.registry_path.exists():
            default_data = {
                "version": 1,
                "accounts": {},
                "last_active_account": None,
                "blacklisted_emails": list(BLACKLISTED_EMAILS)
            }
            self._write_registry(default_data)

    def _read_registry(self) -> Dict[str, Any]:
        try:
            with open(self.registry_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error reading registry: {e}. Rebuilding...")
            self._ensure_registry()
            with open(self.registry_path, "r", encoding="utf-8") as f:
                return json.load(f)

    def _write_registry(self, data: Dict[str, Any]) -> None:
        data["blacklisted_emails"] = list(BLACKLISTED_EMAILS)
        temp_file = self.registry_path.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        temp_file.replace(self.registry_path)

    def _is_blacklisted(self, identifier: str) -> bool:
        """Helper to test any string (email, alias, env var) against the permanent blacklist."""
        if not identifier:
            return False
        clean = identifier.strip().lower()
        return any(b in clean for b in BLACKLISTED_EMAILS)

    def _enforce_blacklist(self) -> None:
        """
        Layer 2 Blacklist Guard:
        Purges any blacklisted accounts from registry AND deletes their physical directories from disk.
        """
        reg = self._read_registry()
        accounts = reg.get("accounts", {})
        purged = []
        for alias, info in list(accounts.items()):
            email = (info.get("email") or "").strip().lower()
            if self._is_blacklisted(email) or self._is_blacklisted(alias):
                logger.critical(f"POLICY VIOLATION DETECTED: Account {alias} ({email}) is blacklisted! Purging...")
                purged.append(alias)
                del accounts[alias]
                prof_dir = self.get_account_profile_dir(alias)
                if prof_dir.exists():
                    shutil.rmtree(prof_dir, ignore_errors=True)

        # Also inspect on-disk directory names in base_dir
        if self.base_dir.exists():
            for child in self.base_dir.iterdir():
                if child.is_dir() and self._is_blacklisted(child.name):
                    logger.critical(f"POLICY VIOLATION DETECTED: Rogue directory {child} matches blacklist! Deleting...")
                    shutil.rmtree(child, ignore_errors=True)

        if purged:
            reg["accounts"] = accounts
            if reg.get("last_active_account") in purged:
                reg["last_active_account"] = None
            self._write_registry(reg)

    def get_account_profile_dir(self, account_alias: str) -> Path:
        return self.base_dir / account_alias

    def list_accounts(self) -> List[Dict[str, Any]]:
        """Returns all registered accounts with live status (excluding blacklisted)."""
        self._enforce_blacklist()
        reg = self._read_registry()
        accounts = []
        now = time.time()
        for alias, info in reg.get("accounts", {}).items():
            email = (info.get("email") or "").strip().lower()
            if self._is_blacklisted(email) or self._is_blacklisted(alias):
                continue

            profile_dir = self.get_account_profile_dir(alias)
            token_file = profile_dir / ".config" / "colab-cli" / "token.json"
            has_token = token_file.exists() and token_file.stat().st_size > 0

            status = info.get("status", "UNCONFIGURED")
            cooldown_until = info.get("cooldown_until", 0)

            if cooldown_until > now:
                status = "COOLING_DOWN"
            elif has_token and status == "COOLING_DOWN":
                status = "READY"
            elif has_token and status != "READY":
                status = "READY"
            elif not has_token:
                status = "NO_TOKEN"

            accounts.append({
                "alias": alias,
                "email": info.get("email", "Unknown"),
                "status": status,
                "cooldown_remaining_sec": max(0, int(cooldown_until - now)),
                "success_count": info.get("success_count", 0),
                "failure_count": info.get("failure_count", 0),
                "last_used": info.get("last_used", None),
                "profile_dir": str(profile_dir)
            })
        return accounts

    def register_account(self, alias: str, email: str = "") -> Path:
        """
        Layer 1 Blacklist Guard:
        Creates directory structure for a new account profile, strictly checking blacklist.
        Raises PermissionError on violation.
        """
        if self._is_blacklisted(email) or self._is_blacklisted(alias):
            raise PermissionError(
                f"STRICT POLICY VIOLATION: Account {email or alias} is PERMANENTLY BLACKLISTED and forbidden from Colab!"
            )

        profile_dir = self.get_account_profile_dir(alias)
        (profile_dir / ".config" / "colab-cli").mkdir(parents=True, exist_ok=True)

        reg = self._read_registry()
        reg["accounts"][alias] = {
            "email": email,
            "status": "REGISTERED",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_used": None,
            "cooldown_until": 0,
            "success_count": 0,
            "failure_count": 0
        }
        self._write_registry(reg)
        logger.info(f"Registered Colab account: {alias} at {profile_dir}")
        return profile_dir

    def verify_account_token_email(self, alias: str) -> Optional[str]:
        """
        Layer 3 Blacklist Guard:
        Inspects the token for account alias (decoding JWT id_token claims and whoami).
        If blacklisted, deletes the profile directory immediately from disk and raises PermissionError.
        """
        profile_dir = self.get_account_profile_dir(alias)
        token_file = profile_dir / ".config" / "colab-cli" / "token.json"
        if not token_file.exists():
            return None

        detected_email = ""
        try:
            with open(token_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                import base64
                id_token = data.get("id_token")
                if id_token and "." in id_token:
                    payload = id_token.split(".")[1]
                    padded = payload + "=" * (-len(payload) % 4)
                    claims = json.loads(base64.urlsafe_b64decode(padded).decode("utf-8", errors="ignore"))
                    detected_email = (claims.get("email") or "").strip().lower()
        except Exception:
            pass

        if not detected_email:
            code, stdout, _ = self.run_colab_command(alias, ["whoami"], timeout=10, max_retries=1)
            if code == 0 and "email:" in stdout.lower():
                for line in stdout.splitlines():
                    if "email:" in line.lower():
                        detected_email = line.split(":", 1)[1].strip().lower()
                        break

        if self._is_blacklisted(detected_email) or self._is_blacklisted(alias):
            logger.critical(f"FATAL: Token for account {alias} belongs to BLACKLISTED EMAIL {detected_email}! Deleting profile instantly!")
            self.remove_account(alias)
            raise PermissionError(f"CRITICAL ERROR: Account {detected_email or alias} is BLACKLISTED and cannot be used!")

        if detected_email:
            reg = self._read_registry()
            if alias in reg["accounts"]:
                reg["accounts"][alias]["email"] = detected_email
                self._write_registry(reg)

        return detected_email

    def remove_account(self, alias: str) -> bool:
        """Removes an account profile and deletes its directory immediately."""
        reg = self._read_registry()
        if alias in reg["accounts"]:
            del reg["accounts"][alias]
            if reg.get("last_active_account") == alias:
                reg["last_active_account"] = None
            self._write_registry(reg)

        profile_dir = self.get_account_profile_dir(alias)
        if profile_dir.exists():
            shutil.rmtree(profile_dir, ignore_errors=True)
        logger.info(f"Removed Colab account: {alias}")
        return True

    def mark_cooldown(
        self,
        alias: str,
        duration_seconds: int = DEFAULT_COOLDOWN_SECONDS,
        reason: str = "Quota/Rate Limit"
    ) -> None:
        """Places an account in cooldown (default: 1800s / 30 minutes)."""
        reg = self._read_registry()
        if alias in reg["accounts"]:
            reg["accounts"][alias]["status"] = "COOLING_DOWN"
            reg["accounts"][alias]["cooldown_until"] = time.time() + duration_seconds
            reg["accounts"][alias]["last_error"] = reason
            reg["accounts"][alias]["failure_count"] = reg["accounts"][alias].get("failure_count", 0) + 1
            self._write_registry(reg)
            logger.warning(f"Account {alias} entered COOLING_DOWN for {duration_seconds}s. Reason: {reason}")

    def mark_success(self, alias: str, cooldown_seconds: int = 0) -> None:
        """
        Records successful job execution for an account.
        Increments success_count, updates last_used, and optionally sets cooldown.
        """
        reg = self._read_registry()
        if alias in reg["accounts"]:
            now_iso = datetime.now(timezone.utc).isoformat()
            reg["accounts"][alias]["last_used"] = now_iso
            reg["accounts"][alias]["success_count"] = reg["accounts"][alias].get("success_count", 0) + 1
            reg["last_active_account"] = alias

            if cooldown_seconds > 0:
                reg["accounts"][alias]["status"] = "COOLING_DOWN"
                reg["accounts"][alias]["cooldown_until"] = time.time() + cooldown_seconds
                logger.info(f"Account {alias} completed job. In cooldown for {cooldown_seconds}s.")
            else:
                reg["accounts"][alias]["status"] = "READY"
                reg["accounts"][alias]["cooldown_until"] = 0

            self._write_registry(reg)

    def record_run_result(self, alias: str, success: bool, error_msg: str = "") -> None:
        """Records run result: marks success if true, else triggers cooldown."""
        if success:
            self.mark_success(alias)
        else:
            self.mark_cooldown(alias, reason=error_msg)

    def select_active_account(self) -> Optional[str]:
        """Picks the best available non-blacklisted account that is READY."""
        accounts = self.list_accounts()
        ready_accounts = [a for a in accounts if a["status"] == "READY"]

        if not ready_accounts:
            return None

        ready_accounts.sort(key=lambda a: (a["failure_count"], -a["success_count"]))
        return ready_accounts[0]["alias"]

    def get_earliest_available_account(self) -> Tuple[Optional[str], int]:
        """Returns the alias of the account that will become available earliest and seconds remaining."""
        accounts = self.list_accounts()
        if not accounts:
            return None, 0

        ready = [a for a in accounts if a["status"] == "READY"]
        if ready:
            return ready[0]["alias"], 0

        accounts.sort(key=lambda a: a["cooldown_remaining_sec"])
        return accounts[0]["alias"], accounts[0]["cooldown_remaining_sec"]

    def run_colab_command(
        self,
        account_alias: str,
        cmd_args: Union[List[str], str],
        timeout: Optional[int] = None,
        capture_output: bool = True,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_delay: float = 2.0
    ) -> Tuple[int, str, str]:
        """
        Layer 4 Blacklist Guard & Execution Runner with 3 Retries:
        Executes a colab CLI command in the isolated context of the specified account.
        Strictly refuses execution if account, email, or environment matches blacklist.
        Retries up to max_retries on transient errors.
        """
        reg = self._read_registry()
        acct_email = (reg.get("accounts", {}).get(account_alias, {}).get("email") or "").lower()

        # Check blacklist on account email, alias, and environment variables
        env_checks = [
            acct_email,
            account_alias,
            os.environ.get("COLAB_USER", ""),
            os.environ.get("USER_EMAIL", ""),
            os.environ.get("GOOGLE_ACCOUNT", "")
        ]
        for val in env_checks:
            if self._is_blacklisted(val):
                prof_dir = self.get_account_profile_dir(account_alias)
                if prof_dir.exists():
                    shutil.rmtree(prof_dir, ignore_errors=True)
                raise PermissionError(f"STRICT POLICY: Refusing execution with blacklisted email/identifier: {val}")

        profile_dir = self.get_account_profile_dir(account_alias)
        env = os.environ.copy()
        env["HOME"] = str(profile_dir)
        local_bin = os.path.expanduser("~/.local/bin")
        env["PATH"] = f"{local_bin}:{env.get('PATH', '')}"

        if isinstance(cmd_args, str):
            args = shlex.split(cmd_args)
        else:
            args = list(cmd_args)

        if args and args[0] == "colab":
            args = args[1:]

        executable = COLAB_EXECUTABLE if os.path.exists(COLAB_EXECUTABLE) else "colab"
        full_cmd = [executable, "--auth", "oauth2"] + args

        cmd_str = " ".join(full_cmd)
        logger.info(f"[{account_alias}] Running: {cmd_str}")

        last_code = -1
        last_out = ""
        last_err = ""

        for attempt in range(1, max_retries + 1):
            try:
                res = subprocess.run(
                    full_cmd,
                    env=env,
                    capture_output=capture_output,
                    text=True,
                    timeout=timeout
                )
                last_code = res.returncode
                last_out = res.stdout or ""
                last_err = res.stderr or ""

                if last_code == 0:
                    return last_code, last_out, last_err

                # Non-zero return code
                if attempt < max_retries:
                    logger.warning(f"[{account_alias}] Command failed (attempt {attempt}/{max_retries}, code {last_code}): {last_err or last_out}. Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    return last_code, last_out, last_err

            except subprocess.TimeoutExpired:
                logger.error(f"[{account_alias}] Command timed out after {timeout}s (attempt {attempt}/{max_retries}): {cmd_str}")
                last_code = -1
                last_out = ""
                last_err = f"TimeoutExpired after {timeout}s"
                if attempt < max_retries:
                    time.sleep(retry_delay)
                else:
                    return last_code, last_out, last_err

            except Exception as e:
                logger.error(f"[{account_alias}] Execution error (attempt {attempt}/{max_retries}): {e}")
                last_code = -1
                last_out = ""
                last_err = str(e)
                if attempt < max_retries:
                    time.sleep(retry_delay)
                else:
                    return last_code, last_out, last_err

        return last_code, last_out, last_err


if __name__ == "__main__":
    mgr = ColabAccountManager()
    accs = mgr.list_accounts()
    print(f"Colab Account Pool: {len(accs)} account(s) registered.")
    print(f"Permanent Blacklist: {list(BLACKLISTED_EMAILS)}")
    for a in accs:
        print(f"  - [{a['status']}] {a['alias']} ({a['email']}) | Success: {a['success_count']} | Cooldown: {a['cooldown_remaining_sec']}s")
