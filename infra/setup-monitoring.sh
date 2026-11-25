#!/usr/bin/env bash
# Starts Prometheus + Grafana for KVACK on this machine (run on the Linux laptop).
#   ./infra/setup-monitoring.sh [api-host:port] [http|https]
# e.g. ./infra/setup-monitoring.sh my-windows-pc.tailnet.ts.net:8000
# With no argument it scrapes host.docker.internal:8000 (an API running on this machine).
# Safe to re-run: keeps the generated passwords and the data volumes, and re-writes the scrape
# config, so re-running with a new address is how you change where the API is.
set -euo pipefail

cd "$(dirname "$0")"
MON=monitoring
TARGET="${1:-host.docker.internal:8000}"
SCHEME="${2:-http}"

if ! command -v docker >/dev/null 2>&1 || ! docker compose version >/dev/null 2>&1; then
  echo "Docker with the compose plugin is needed (see setup-services.sh)." >&2
  exit 1
fi
if [ "$SCHEME" != "http" ] && [ "$SCHEME" != "https" ]; then
  echo "Scheme must be http or https, not '$SCHEME'." >&2
  exit 1
fi

rand() { head -c 24 /dev/urandom | base64 | tr -d '/+=\n' | head -c 32; }

# The token a scraper must send, kept in a file Prometheus reads (never inside prometheus.yml).
# World-readable on purpose: Prometheus runs as an unprivileged user inside its container.
TOKEN_FILE="$MON/prometheus/metrics_token"
if [ ! -s "$TOKEN_FILE" ]; then
  printf '%s' "$(rand)" > "$TOKEN_FILE"
  chmod 644 "$TOKEN_FILE"
  echo "Created a metrics token."
fi
METRICS_TOKEN=$(cat "$TOKEN_FILE")

# Grafana's admin password lives in infra/.env beside the database password.
touch .env && chmod 600 .env
if ! grep -q '^GRAFANA_ADMIN_PASSWORD=' .env; then
  echo "GRAFANA_ADMIN_PASSWORD=$(rand)" >> .env
  echo "Created a Grafana admin password in infra/.env."
fi
GRAFANA_PW=$(grep '^GRAFANA_ADMIN_PASSWORD=' .env | cut -d= -f2-)

sed -e "s#__API_TARGET__#${TARGET}#" -e "s#__API_SCHEME__#${SCHEME}#" \
  "$MON/prometheus/prometheus.yml.template" > "$MON/prometheus/prometheus.yml"

# Alerts go to an ntfy topic. Set NTFY_URL / NTFY_TOPIC to use another server or topic; the default
# is an ntfy on this machine (port 8091). Without a reachable ntfy, alerts only show in
# Prometheus and Grafana.
NTFY_URL="${NTFY_URL:-http://127.0.0.1:8091}"
NTFY_TOPIC="${NTFY_TOPIC:-kvack-alerts}"
if [ ! -f "$MON/alertmanager/alertmanager.yml" ] || [ "${NTFY_URL}" != "$(cat "$MON/alertmanager/.ntfy_url" 2>/dev/null)" ]; then
  python3 - "$NTFY_URL" "$NTFY_TOPIC" "$MON" <<'PY'
import sys, urllib.parse
url, topic, mon = sys.argv[1:4]
title = '{{ if eq .status "firing" }}ALERT{{ else }}Resolved{{ end }}: {{ (index .alerts 0).labels.alertname }}'
message = '{{ range .alerts }}{{ .annotations.summary }}\n{{ end }}'
query = urllib.parse.urlencode(
    {"template": "yes", "title": title, "message": message, "tags": "rotating_light"},
    quote_via=urllib.parse.quote,
)
hook = f"{url.rstrip('/')}/{topic}?{query}"
text = open(f"{mon}/alertmanager/alertmanager.yml.template").read().replace("__NTFY_WEBHOOK__", hook)
open(f"{mon}/alertmanager/alertmanager.yml", "w").write(text)
open(f"{mon}/alertmanager/.ntfy_url", "w").write(url)
PY
fi

# Regenerate the dashboard so it always matches make_dashboard.py.
python3 "$MON/make_dashboard.py" >/dev/null

docker compose -f "$MON/docker-compose.monitoring.yml" --env-file .env up -d
# A changed prometheus.yml is only read on a reload or restart.
docker compose -f "$MON/docker-compose.monitoring.yml" --env-file .env restart prometheus >/dev/null

echo -n "Waiting for Prometheus"
for _ in $(seq 1 30); do
  curl -fsS http://127.0.0.1:9092/-/ready >/dev/null 2>&1 && break; echo -n "."; sleep 2
done
echo
echo -n "Waiting for Grafana"
for _ in $(seq 1 45); do
  curl -fsS http://127.0.0.1:3001/api/health >/dev/null 2>&1 && break; echo -n "."; sleep 2
done
echo

# Primary IPv4 address; never fails the script (hostname -I is not available everywhere).
IP=$( (hostname -I 2>/dev/null || true) | awk '{print $1}')
if [ -z "$IP" ]; then
  IP=$( (ip -4 route get 1.1.1.1 2>/dev/null || true) | awk '{for(i=1;i<=NF;i++) if($i=="src"){print $(i+1); exit}}')
fi
IP=${IP:-<laptop-ip>}
cat <<MSG

Monitoring is up.

  Grafana     http://${IP}:3001      login: admin / ${GRAFANA_PW}
  Prometheus  http://127.0.0.1:9092   (this machine only: Status > Targets shows if the API is reached)

Prometheus scrapes ${SCHEME}://${TARGET}/metrics. On the machine that runs the API, add to
qa-copilot/.env and restart the API:

  QACOPILOT_METRICS_TOKEN=${METRICS_TOKEN}

If the API is not on this machine it must listen beyond localhost, e.g.
  uvicorn qa_api.main:app --app-dir services/api/src --host 0.0.0.0 --port 8000
and the firewall should let only this machine reach that port. The token protects /metrics, but
the rest of the API is protected only by its API keys, so prefer HTTPS or a private network (the
tailnet) rather than the open internet.

Alert rules are in infra/monitoring/prometheus/alerts.yml. They show in Prometheus and on the
dashboard, and Alertmanager sends them to ntfy: ${NTFY_URL}/${NTFY_TOPIC}
(open that address in a browser, or subscribe to the topic in the ntfy phone app).
MSG
