#!/usr/bin/env python3
"""
Interactive CLI Tool to Add & Manage Colab Account Pool.
Facilitates one-time OAuth authentication for each Gmail account with isolated token storage
and strict blacklist enforcement (aleron.dt@gmail.com is forbidden).
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from colab_rotator import ColabAccountManager, BLACKLISTED_EMAILS


def cmd_list(mgr: ColabAccountManager):
    accs = mgr.list_accounts()
    print("\n=======================================================")
    print(f"  Google Colab Account Pool ({len(accs)} account(s) registered)")
    print("=======================================================")
    print(f"  [STRICT POLICY] Blacklisted: {list(BLACKLISTED_EMAILS)}")
    print("-------------------------------------------------------")
    if not accs:
        print("  (Chưa có tài khoản nào được đăng ký. Dùng lệnh `add <alias>` để thêm)")
    for a in accs:
        cooldown_str = f" | Cooldown: {a['cooldown_remaining_sec']}s" if a['cooldown_remaining_sec'] > 0 else ""
        print(f"  * [{a['status']:<12}] Alias: {a['alias']:<15} Email: {a['email']:<25} Success: {a['success_count']}{cooldown_str}")
    print("=======================================================\n")


def cmd_add(mgr: ColabAccountManager, alias: str):
    alias = alias.strip().lower()
    if not alias:
        print("❌ Error: Alias cannot be empty.")
        return

    profile_dir = mgr.get_account_profile_dir(alias)
    token_file = profile_dir / ".config" / "colab-cli" / "token.json"

    if token_file.exists():
        resp = input(f"Tài khoản '{alias}' đã có token. Bạn có muốn cấp quyền lại (re-auth) không? [y/N]: ").strip().lower()
        if resp != "y":
            print("Hủy bỏ thao tác.")
            return
        token_file.unlink(missing_ok=True)

    mgr.register_account(alias)
    print(f"\n🔐 Đang khởi tạo phiên xác thực OAuth2 cho profile: '{alias}'...")

    env = os.environ.copy()
    env["HOME"] = str(profile_dir)
    local_bin = os.path.expanduser("~/.local/bin")
    env["PATH"] = f"{local_bin}:{env.get('PATH', '')}"

    colab_bin = os.path.expanduser("~/.local/bin/colab")
    cmd = [colab_bin, "--auth", "oauth2", "sessions"]

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    auth_url = None
    output_lines = []

    # Read until auth URL is printed
    while True:
        line = proc.stderr.readline()
        if not line:
            line = proc.stdout.readline()
        if not line and proc.poll() is not None:
            break
        if line:
            output_lines.append(line.strip())
            if "https://accounts.google.com/o/oauth2/auth" in line:
                for part in line.split():
                    if part.startswith("https://accounts.google.com"):
                        auth_url = part.strip()
                        break
                if auth_url:
                    break

    if not auth_url:
        print("❌ Không tìm thấy URL xác thực. Đầu ra từ colab-cli:")
        print("\n".join(output_lines))
        proc.kill()
        return

    print("\n" + "="*70)
    print("👉 BƯỚC 1: Mở đường link sau trên trình duyệt (Chrome/Safari...):")
    print(f"\n  {auth_url}\n")
    print("⚠️ LƯU Ý: KHÔNG ĐƯỢC CHỌN tài khoản aleron.dt@gmail.com (tài khoản này bị cấm sử dụng)!")
    print("👉 BƯỚC 2: Chọn tài khoản Gmail của bạn, bấm Tiếp tục / Cho phép.")
    print("👉 BƯỚC 3: Sao chép Mã xác nhận (Authorization Code) mà Google hiển thị trên màn hình.")
    print("="*70 + "\n")

    code = input("👉 BƯỚC 4: Dán Mã xác nhận vào đây và nhấn Enter: ").strip()
    if not code:
        print("❌ Mã xác thực rỗng. Hủy bỏ.")
        proc.kill()
        return

    try:
        stdout_data, stderr_data = proc.communicate(input=code + "\n", timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        print("❌ Quá thời gian chờ phản hồi từ máy chủ Google.")
        return

    if not token_file.exists():
        print("❌ Xác thực không thành công. Token không được tạo. Chi tiết lỗi:")
        print(stderr_data or stdout_data)
        return

    # Verify token does not belong to blacklisted account
    try:
        email = mgr.verify_account_token_email(alias)
        print(f"\n✅ XÁC THỰC THÀNH CÔNG! Tài khoản: {email or alias} đã sẵn sàng trong Pool.")
    except PermissionError as pe:
        print(f"\n🚫 {pe}")
        return


def cmd_test(mgr: ColabAccountManager, alias: str):
    code, stdout, stderr = mgr.run_colab_command(alias, ["sessions"], timeout=15)
    if code == 0:
        print(f"✅ Tài khoản '{alias}' kết nối Colab thành công!")
        mgr.mark_success(alias)
        # Check active sessions
        code_w, out_w, _ = mgr.run_colab_command(alias, ["whoami"], timeout=10)
        if code_w == 0:
            print(f"   Thông tin: {out_w.strip()}")
    else:
        print(f"❌ Tài khoản '{alias}' kết nối thất bại (code {code}):")
        print(stderr or stdout)


def cmd_remove(mgr: ColabAccountManager, alias: str):
    mgr.remove_account(alias)
    print(f"🗑️ Đã xoá tài khoản '{alias}' khỏi Pool.")


def main():
    parser = argparse.ArgumentParser(description="Colab Account Pool Authentication & Management")
    subparsers = parser.add_subparsers(dest="subcommand", help="Sub-command")

    subparsers.add_parser("list", help="Liệt kê tất cả tài khoản trong Pool")

    add_p = subparsers.add_parser("add", help="Thêm tài khoản mới vào Pool")
    add_p.add_argument("alias", help="Tên gợi nhớ cho tài khoản (ví dụ: gmail1, gmail2, colab_work)")

    test_p = subparsers.add_parser("test", help="Kiểm tra kết nối của tài khoản")
    test_p.add_argument("alias", help="Tên gợi nhớ của tài khoản")

    rm_p = subparsers.add_parser("remove", help="Xoá tài khoản khỏi Pool")
    rm_p.add_argument("alias", help="Tên gợi nhớ của tài khoản")

    args = parser.parse_args()
    mgr = ColabAccountManager()

    if args.subcommand == "list" or not args.subcommand:
        cmd_list(mgr)
    elif args.subcommand == "add":
        cmd_add(mgr, args.alias)
    elif args.subcommand == "test":
        cmd_test(mgr, args.alias)
    elif args.subcommand == "remove":
        cmd_remove(mgr, args.alias)


if __name__ == "__main__":
    main()
