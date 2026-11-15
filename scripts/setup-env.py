"""Create private configuration for a new installation without overwriting it."""
import os
from pathlib import Path
import re
import secrets


def main():
    root = Path(__file__).resolve().parents[1]
    destination = root / ".env"
    if destination.exists():
        match = re.search(r"^POSTGRES_PASSWORD\s*=([^\n]*)$", destination.read_text(), re.MULTILINE)
        password = match.group(1).strip().strip("\"'") if match else ""
        if not password or password.startswith("#"):
            raise SystemExit(
                ".env already exists. Set POSTGRES_PASSWORD to the existing database's "
                "password, or generate one with `openssl rand -hex 32` for a new database. "
                "No files changed."
            )
        print("Keeping existing .env configuration.")
        return
    template = (root / ".env.example").read_text()
    configured = template.replace("POSTGRES_PASSWORD=\n", f"POSTGRES_PASSWORD={secrets.token_hex(32)}\n")
    with open(destination, "x", opener=lambda path, flags: os.open(path, flags, 0o600)) as output:
        output.write(configured)
    print("Created .env with a unique database password. Keep this file private.")


if __name__ == "__main__":
    main()
