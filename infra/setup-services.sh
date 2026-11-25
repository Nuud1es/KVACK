#!/usr/bin/env bash
# Starts Postgres + Qdrant for KVACK on this machine (run on the Linux laptop).
#   ./infra/setup-services.sh
# Safe to re-run: keeps an existing password and data volumes.
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is not installed. On Debian/Ubuntu:  sudo apt install docker.io docker-compose-v2" >&2
  echo "Then:  sudo usermod -aG docker \$USER   (log out and in again)" >&2
  exit 1
fi
if ! docker compose version >/dev/null 2>&1; then
  echo "The 'docker compose' plugin is missing:  sudo apt install docker-compose-v2" >&2
  exit 1
fi

# Generate a database password once; reuse it on later runs.
if [ ! -f .env ]; then
  umask 077
  echo "POSTGRES_PASSWORD=$(head -c 24 /dev/urandom | base64 | tr -d '/+=\n' | head -c 24)" > .env
  echo "Created infra/.env with a new database password."
fi
# shellcheck disable=SC1091
source .env

docker compose up -d

echo -n "Waiting for Postgres"
for _ in $(seq 1 30); do
  if docker compose exec -T postgres pg_isready -U qacopilot >/dev/null 2>&1; then break; fi
  echo -n "."; sleep 2
done
echo
echo -n "Waiting for Qdrant"
for _ in $(seq 1 30); do
  if curl -fsS http://localhost:6333/readyz >/dev/null 2>&1; then break; fi
  echo -n "."; sleep 2
done
echo

# Primary IPv4 address; never fails the script (hostname -I is not available everywhere).
IP=$( (hostname -I 2>/dev/null || true) | awk '{print $1}')
if [ -z "$IP" ]; then
  IP=$( (ip -4 route get 1.1.1.1 2>/dev/null || true) | awk '{for(i=1;i<=NF;i++) if($i=="src"){print $(i+1); exit}}')
fi
IP=${IP:-<laptop-ip>}
cat <<MSG

Services are up. Put these in qa-copilot/.env on the Windows machine:

  QACOPILOT_DATABASE_URL=postgresql+psycopg://qacopilot:${POSTGRES_PASSWORD}@${IP}:5432/qacopilot
  QACOPILOT_QDRANT_URL=http://${IP}:6333

Firewall: allow ports 5432 and 6333 from the Windows machine only, e.g.
  sudo ufw allow from <windows-ip> to any port 5432 proto tcp
  sudo ufw allow from <windows-ip> to any port 6333 proto tcp
MSG
