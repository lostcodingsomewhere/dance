#!/usr/bin/env bash
# One-shot bootstrap for a fresh machine.
#
# Idempotent — safe to re-run. Walks you through:
#   1. Python venv + dev deps
#   2. Database migrations
#   3. Companion app deps
#   4. ~/.dance/ scaffold + .env from .env.example
#   5. Host-tool check (yt-dlp, ffmpeg) with Homebrew install hint
#   6. Optional cookies.txt + Spotify creds nudge
#
# Run from the repo root:
#   ./bin/setup.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

cyan() { printf "\033[36m%s\033[0m\n" "$1"; }
green() { printf "\033[32m%s\033[0m\n" "$1"; }
yellow() { printf "\033[33m%s\033[0m\n" "$1"; }
red() { printf "\033[31m%s\033[0m\n" "$1"; }
section() {
  echo
  cyan "── $1 ──"
}

# ---------------------------------------------------------------------------
section "Python venv + dev install"
# ---------------------------------------------------------------------------

if ! command -v python3 >/dev/null; then
  red "python3 not on PATH. Install Python 3.10+ first (e.g. \`brew install python@3.11\`)."
  exit 1
fi

PY_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
  red "Python $PY_VERSION found — need 3.10 or newer."
  exit 1
fi
echo "Python $PY_VERSION ✓"

if [ ! -d .venv ]; then
  echo "Creating .venv …"
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip >/dev/null
pip install -e ".[dev]" >/dev/null
green "Python deps installed into .venv"

# ---------------------------------------------------------------------------
section "Database migrations"
# ---------------------------------------------------------------------------

if alembic upgrade head; then
  green "Schema up to date"
else
  red "Alembic migration failed — see error above."
  exit 1
fi

# ---------------------------------------------------------------------------
section "Companion app"
# ---------------------------------------------------------------------------

if ! command -v npm >/dev/null; then
  yellow "npm not on PATH — skipping companion-app install."
  yellow "Install Node 20+ (e.g. \`brew install node\`) then run:"
  yellow "    cd companion-app && npm install"
else
  (cd companion-app && npm install --silent)
  green "Companion-app deps installed"
fi

# ---------------------------------------------------------------------------
section "Per-user data dir + env"
# ---------------------------------------------------------------------------

mkdir -p "$HOME/.dance"

if [ ! -f "$HOME/.dance/.env" ]; then
  if [ -f .env.example ]; then
    cp .env.example "$HOME/.dance/.env"
    green "Created ~/.dance/.env from .env.example"
    yellow "Edit it to add Spotify creds + your playlist URL when ready."
  else
    touch "$HOME/.dance/.env"
    yellow "Created empty ~/.dance/.env — see README for keys you may want."
  fi
else
  green "~/.dance/.env already exists"
fi

# ---------------------------------------------------------------------------
section "Host tools (yt-dlp, ffmpeg)"
# ---------------------------------------------------------------------------

NEED_BREW=()
command -v yt-dlp >/dev/null && green "yt-dlp ✓ ($(yt-dlp --version))" || NEED_BREW+=("yt-dlp")
command -v ffmpeg >/dev/null && green "ffmpeg ✓" || NEED_BREW+=("ffmpeg")

if [ ${#NEED_BREW[@]} -gt 0 ]; then
  yellow "Missing host tools: ${NEED_BREW[*]}"
  if command -v brew >/dev/null; then
    yellow "Install with:"
    yellow "    brew install ${NEED_BREW[*]}"
  else
    yellow "Install Homebrew first (https://brew.sh), then:"
    yellow "    brew install ${NEED_BREW[*]}"
  fi
fi

# ---------------------------------------------------------------------------
section "Optional: YouTube cookies"
# ---------------------------------------------------------------------------

if [ -f "$HOME/.dance/cookies.txt" ]; then
  green "~/.dance/cookies.txt found"
else
  yellow "~/.dance/cookies.txt not present — yt-dlp will run unauthenticated."
  yellow "Export cookies (recommended) via the 'Get cookies.txt LOCALLY'"
  yellow "Chrome extension, save to ~/.dance/cookies.txt."
fi

# ---------------------------------------------------------------------------
section "Done"
# ---------------------------------------------------------------------------

cat <<'EOF'

Next steps:

  # Activate the venv in this shell:
  source .venv/bin/activate

  # Start the backend:
  uvicorn dance.api:create_app --factory --host 127.0.0.1 --port 8000

  # In another shell, start the companion app:
  cd companion-app && npm run dev

Then open http://localhost:5173.

The MasterStrip's tiny dot (top-right) shows host-deps health — green = ready,
amber = optional missing, red = required missing. Click it for the checklist.
EOF
