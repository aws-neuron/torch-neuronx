import importlib.util
import socket
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "redact.py"
SPEC = importlib.util.spec_from_file_location("redact", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_redacts_home_paths_account_key_secret_and_hostname():
    home_one = "/" + "home/person/project/file.py"
    home_two = "/local/" + "home/person/work"
    home_three = "/" + "Users/person/work"
    root_home = "/" + "root/private.txt"
    account = "123456" * 2
    key_id = "AKIA" + "A" * 16
    hostname = socket.gethostname()
    source = (
        f"{home_one} {home_two} {home_three} {root_home} {account} {key_id} "
        f"api_token=plain SECRET_KEY: 'quoted value' host={hostname}"
    )
    redacted = MODULE.redact_text(source)
    for value in (home_one, home_two, home_three, root_home, account, key_id, hostname):
        assert value not in redacted
    assert redacted.count("<redacted-home>") == 4
    assert "<redacted-account>" in redacted
    assert "<redacted-key-id>" in redacted
    assert redacted.count("<redacted-secret>") == 2
    assert "<redacted-hostname>" in redacted


def test_redaction_is_stable_and_leaves_safe_text_unchanged():
    text = "RuntimeError: expected float input"
    assert MODULE.redact_text(text) == text
    sensitive = "token=hidden"
    assert MODULE.redact_text(MODULE.redact_text(sensitive)) == MODULE.redact_text(sensitive)


def test_cli_redacts_stdin_and_help():
    source = ("123456" * 2) + "\n"
    result = subprocess.run(
        [sys.executable, str(SCRIPT)], input=source, text=True, capture_output=True, check=False
    )
    assert result.returncode == 0
    assert result.stdout == "<redacted-account>\n"

    help_result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"], text=True, capture_output=True, check=False
    )
    assert help_result.returncode == 0
    assert "usage:" in help_result.stdout.lower()
