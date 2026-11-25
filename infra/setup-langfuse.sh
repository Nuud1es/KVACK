#!/usr/bin/env bash
# Starts a self-hosted Langfuse (web, worker, Postgres, ClickHouse, Redis, MinIO) on this machine.
#   ./infra/setup-langfuse.sh [admin-email]
# Safe to re-run: keeps generated secrets and data. Uses Langfuse's own docker-compose.yml
# (cloned once into infra/.langfuse/upstream) plus a generated override that sets our secrets,
# a pre-created project and API keys, and turns telemetry off.
set -euo pipefail

cd "$(dirname "$0")"
DIR=.langfuse
EMAIL="${1:-admin@qacopilot.local}"
mkdir -p "$DIR"

for cmd in docker git curl; do
  command -v "$cmd" >/dev/null 2>&1 || { echo "Missing: $cmd" >&2; exit 1; }
done

if [ ! -d "$DIR/upstream/.git" ]; then
  git clone --depth 1 https://github.com/langfuse/langfuse.git "$DIR/upstream"
fi
UPSTREAM="$DIR/upstream/docker-compose.yml"

# Fail clearly if upstream renamed its services (the override depends on these names).
SERVICES=$(docker compose -p langfuse -f "$UPSTREAM" config --services)
for s in langfuse-web langfuse-worker; do
  if ! grep -qx "$s" <<<"$SERVICES"; then
    echo "Upstream compose has no '$s' service (found: $(echo $SERVICES)). Edit this script." >&2
    exit 1
  fi
done

rand_hex() { head -c "$1" /dev/urandom | od -An -tx1 | tr -d ' \n'; }

# Generate secrets once.
if [ ! -f "$DIR/secrets.env" ]; then
  umask 077
  {
    echo "NEXTAUTH_SECRET=$(rand_hex 32)"
    echo "SALT=$(rand_hex 32)"
    echo "ENCRYPTION_KEY=$(rand_hex 32)"
    echo "LANGFUSE_PUBLIC_KEY=pk-lf-$(rand_hex 16)"
    echo "LANGFUSE_SECRET_KEY=sk-lf-$(rand_hex 16)"
    echo "LANGFUSE_ADMIN_EMAIL=$EMAIL"
    echo "LANGFUSE_ADMIN_PASSWORD=$(rand_hex 12)"
  } > "$DIR/secrets.env"
  echo "Generated Langfuse secrets in infra/$DIR/secrets.env"
fi
# shellcheck disable=SC1091
source "$DIR/secrets.env"

# Primary IPv4 address; never fails the script (hostname -I is not available everywhere).
IP=$( (hostname -I 2>/dev/null || true) | awk '{print $1}')
if [ -z "$IP" ]; then
  IP=$( (ip -4 route get 1.1.1.1 2>/dev/null || true) | awk '{for(i=1;i<=NF;i++) if($i=="src"){print $(i+1); exit}}')
fi
IP=${IP:-localhost}

umask 077
cat > "$DIR/override.yml" <<YAML
services:
  langfuse-web:
    environment:
      NEXTAUTH_URL: http://${IP}:3000
      NEXTAUTH_SECRET: ${NEXTAUTH_SECRET}
      SALT: ${SALT}
      ENCRYPTION_KEY: ${ENCRYPTION_KEY}
      TELEMETRY_ENABLED: "false"
      LANGFUSE_INIT_ORG_ID: gressingham
      LANGFUSE_INIT_ORG_NAME: Gressingham
      LANGFUSE_INIT_PROJECT_ID: qa-copilot
      LANGFUSE_INIT_PROJECT_NAME: KVACK
      LANGFUSE_INIT_PROJECT_PUBLIC_KEY: ${LANGFUSE_PUBLIC_KEY}
      LANGFUSE_INIT_PROJECT_SECRET_KEY: ${LANGFUSE_SECRET_KEY}
      LANGFUSE_INIT_USER_EMAIL: ${LANGFUSE_ADMIN_EMAIL}
      LANGFUSE_INIT_USER_NAME: Admin
      LANGFUSE_INIT_USER_PASSWORD: ${LANGFUSE_ADMIN_PASSWORD}
  langfuse-worker:
    environment:
      SALT: ${SALT}
      ENCRYPTION_KEY: ${ENCRYPTION_KEY}
      TELEMETRY_ENABLED: "false"
  postgres:
    # don't publish 5432: the project's own Postgres (setup-services.sh) already uses it
    ports: !override []
YAML

docker compose -p langfuse -f "$UPSTREAM" -f "$DIR/override.yml" up -d

echo -n "Waiting for Langfuse (first start can take a couple of minutes)"
for _ in $(seq 1 60); do
  if curl -fsS http://localhost:3000/api/public/health >/dev/null 2>&1; then echo " up"; break; fi
  echo -n "."; sleep 3
done

cat <<MSG

Langfuse UI:  http://${IP}:3000
Login:        ${LANGFUSE_ADMIN_EMAIL} / ${LANGFUSE_ADMIN_PASSWORD}

Put these in qa-copilot/.env on the Windows machine:

  QACOPILOT_LANGFUSE_HOST=http://${IP}:3000
  QACOPILOT_LANGFUSE_PUBLIC_KEY=${LANGFUSE_PUBLIC_KEY}
  QACOPILOT_LANGFUSE_SECRET_KEY=${LANGFUSE_SECRET_KEY}

Note: Langfuse's internal Postgres/ClickHouse/Redis/MinIO keep upstream's default passwords.
Upstream binds them to 127.0.0.1 only; do not publish those ports. Only port 3000 is network-facing.
MSG
