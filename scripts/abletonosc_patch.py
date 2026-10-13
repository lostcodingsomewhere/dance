"""Check or apply dance's AbletonOSC patches (vendor/abletonosc/patches).

    python scripts/abletonosc_patch.py --check   # report; change nothing
    python scripts/abletonosc_patch.py           # back up, apply what's missing

Exit codes: 0 = everything applied, 1 = patches still pending (``--check``),
2 = a patch conflicts with the install (nothing changed), 3 = no install found.
After applying, fully quit and reopen Live.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dance.osc import fork


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--target", type=Path, default=fork.DEFAULT_TARGET, help="AbletonOSC clone")
    ap.add_argument("--check", action="store_true", help="report only")
    ap.add_argument(
        "--backup-root",
        type=Path,
        default=Path.home() / ".dance" / "backups",
        help="where the pre-patch copy goes (default ~/.dance/backups)",
    )
    args = ap.parse_args(argv)

    target: Path = args.target.expanduser()
    if not (target / "abletonosc").is_dir():
        print(f"No AbletonOSC install at {target} — see docs/abletonosc_setup.md.")
        return 3

    upstream = fork.read_upstream()
    head = fork.head_commit(target)
    pinned = upstream.get("commit", "")
    if head is None:
        print(f"{target} is not a git clone; patches are checked against the files as they are.")
    elif head != pinned:
        print(f"Note: install is at {head[:10]}, patches are pinned to {pinned[:10]}.")

    patches = fork.patch_files()
    if args.check:
        statuses = fork.classify(target, patches)
        for s in statuses:
            print(f"{s.status:9s} {s.patch.name}" + (f"\n          {s.detail}" if s.detail else ""))
        if any(s.status == fork.CONFLICT for s in statuses):
            return 2
        return 1 if any(s.status == fork.PENDING for s in statuses) else 0

    try:
        statuses, backup = fork.apply_patches(target, patches, args.backup_root.expanduser())
    except fork.PatchConflictError as exc:
        print(f"Nothing changed — a patch conflicts with the install: {exc}")
        return 2
    for s in statuses:
        verb = "already applied" if s.status == fork.APPLIED else "applied"
        print(f"{verb:15s} {s.patch.name}")
    if backup is None:
        print("Nothing to do.")
        return 0
    print(f"Backup of the previous abletonosc/ package: {backup}")
    print("Now fully quit and reopen Live (toggling the Control Surface isn't enough).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
