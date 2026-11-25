#!/usr/bin/env bash
# One-shot setup of the KVACK backing services on the Linux laptop:
# Postgres + Qdrant + Langfuse.
#
#   ./infra/setup-laptop.sh [--install-docker] [--windows-ip <ip>]
#
#   --install-docker   apt-install Docker, git, curl (Debian/Ubuntu; asks for sudo)
#   --windows-ip IP    open ports 5432, 6333, 3000 to that IP only (needs ufw + sudo)
#
# Safe to re-run.
set -euo pipefail

cd "$(dirname "$0")"

INSTALL=0
WIN_IP=""
while [ $# -gt 0 ]; do
  case "$1" in
    --install-docker) INSTALL=1 ;;
    --windows-ip) WIN_IP="${2:?--windows-ip needs an address}"; shift ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

if [ "$INSTALL" = 1 ]; then
  sudo apt-get update
  sudo apt-get install -y docker.io docker-compose-v2 git curl openssl
  sudo systemctl enable --now docker
  sudo usermod -aG docker "$USER"
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is missing. Re-run with --install-docker (Debian/Ubuntu)." >&2
  exit 1
fi
if ! docker info >/dev/null 2>&1; then
  echo "Cannot talk to Docker as $USER. If you just installed it, log out and back in" >&2
  echo "(or run 'newgrp docker'), then re-run this script." >&2
  exit 1
fi

if [ -n "$WIN_IP" ]; then
  if command -v ufw >/dev/null 2>&1; then
    for port in 5432 6333 3000; do
      sudo ufw allow from "$WIN_IP" to any port "$port" proto tcp
    done
    sudo ufw status | head -5 || true
    echo "Note: if ufw is inactive these rules do nothing; enable it with 'sudo ufw enable'"
    echo "(allow SSH first: 'sudo ufw allow ssh') if you want the firewall on."
  else
    echo "ufw not installed; skipping firewall rules." >&2
  fi
fi

./setup-services.sh
./setup-langfuse.sh

echo
echo "=============================================================="
echo "All services are up. Copy the two blocks printed above into"
echo "qa-copilot/.env on the Windows machine (Postgres/Qdrant, then Langfuse)."
echo "Re-print them any time:  cat infra/.env infra/.langfuse/secrets.env"
echo "=============================================================="
