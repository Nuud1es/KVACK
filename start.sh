#!/usr/bin/env bash
# Sets up and starts everything KVACK needs on this laptop. Safe to re-run: it only does
# what is missing, so the same command is both first-time setup and "start it again".
#
#   ./start.sh              set up what is missing and start the API
#   ./start.sh stop         stop the API (the Docker services keep running)
#   ./start.sh stop --all   stop the API and the Docker services (Postgres, Qdrant, Langfuse, monitoring)
#   ./start.sh status       what is running
#   ./start.sh --reinstall  also reinstall the Python packages
#
# Environment: QA_HOST (default: the Docker bridge address if there is one, else 127.0.0.1) and
# QA_PORT (default 8000) choose where the API listens.
#
# Needs: Docker, python3 (with the venv module: "sudo apt install python3-venv"), and Node/npm to
# build the web page. Nothing here needs sudo or touches system services.
set -euo pipefail

cd "$(dirname "$0")"
ROOT="$PWD"
PORT="${QA_PORT:-8000}"
HOST="${QA_HOST:-}"
PY="$ROOT/.venv/bin/python"
LOG="$ROOT/logs/api.log"
PROJECTS=(infra langfuse qacopilot-monitoring)  # Docker Compose projects this app uses

say() { printf '\n==> %s\n' "$*"; }
die() { printf 'Error: %s\n' "$*" >&2; exit 1; }

api_pid() {  # the process listening on the API port, if any
  ss -ltnp "sport = :$PORT" 2>/dev/null | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2 || true
}

containers() {  # container ids of one Compose project (running or stopped)
  docker ps -aq --filter "label=com.docker.compose.project=$1" 2>/dev/null || true
}

default_host() {
  if [ -n "$HOST" ]; then echo "$HOST"; return; fi
  local docker0
  docker0="$(ip -4 -o addr show docker0 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -1 || true)"
  echo "${docker0:-127.0.0.1}"
}

cmd_status() {
  echo "API:"
  if [ -n "$(api_pid)" ]; then echo "  running (pid $(api_pid)) on port $PORT"; else echo "  not running"; fi
  echo "Docker services:"
  for p in "${PROJECTS[@]}"; do
    local total running
    total="$(containers "$p" | wc -l)"
    running="$(docker ps -q --filter "label=com.docker.compose.project=$p" | wc -l)"
    echo "  $p: $running of $total containers running"
  done
}

cmd_stop() {
  local pid
  pid="$(api_pid)"
  if [ -n "$pid" ]; then kill "$pid" && echo "API stopped."; else echo "The API was not running."; fi
  if [ "${1:-}" = "--all" ]; then
    for p in "${PROJECTS[@]}"; do
      ids="$(docker ps -q --filter "label=com.docker.compose.project=$p")"
      [ -n "$ids" ] && docker stop $ids >/dev/null && echo "Stopped $p."
    done
  fi
}

cmd_up() {
  local reinstall=0
  [ "${1:-}" = "--reinstall" ] && reinstall=1

  say "Checking tools"
  command -v docker >/dev/null || die "Docker is not installed. Run ./infra/setup-laptop.sh --install-docker"
  docker info >/dev/null 2>&1 || die "Docker is not running, or this user cannot use it (try: sudo systemctl start docker)."
  command -v python3 >/dev/null || die "python3 is not installed."

  say "Python environment"
  if [ ! -x "$PY" ]; then
    python3 -m venv .venv || die "Could not create .venv (on Debian/Ubuntu: sudo apt install python3-venv)."
    reinstall=1
  fi
  if [ "$reinstall" = 1 ] || ! "$PY" -c "import qa_api, qacopilot" >/dev/null 2>&1; then
    "$PY" -m pip install --quiet --upgrade pip
    "$PY" -m pip install --quiet \
      -e "packages/qacopilot[dev,tracing,ingest,retrieval,db,agent,metrics]" -e "services/api[dev]"
    echo "Packages installed."
  else
    echo "Packages already installed (./start.sh --reinstall to redo)."
  fi

  say "Settings (.env)"
  if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env from the template. Fill in the database, Qdrant and LLM settings, then run ./start.sh again."
    echo "(Services are local: see 'Developing on the Linux laptop' in CLAUDE.md for the values.)"
    exit 1
  fi

  say "Docker services (Postgres, Qdrant, Langfuse)"
  if [ -z "$(containers infra)" ]; then
    echo "First run: creating them (this also generates passwords into infra/.env)."
    ./infra/setup-laptop.sh
  else
    for p in infra langfuse; do
      stopped="$(docker ps -aq --filter "label=com.docker.compose.project=$p" --filter status=exited --filter status=created)"
      [ -n "$stopped" ] && docker start $stopped >/dev/null && echo "Started $p."
    done
  fi
  for _ in $(seq 1 30); do
    docker exec infra-postgres-1 pg_isready -q 2>/dev/null && break
    sleep 1
  done
  docker exec infra-postgres-1 pg_isready -q 2>/dev/null || die "Postgres did not come up. See: docker logs infra-postgres-1"
  echo "Postgres is ready."

  say "Database tables"
  "$PY" -m alembic -c packages/qacopilot/alembic.ini upgrade head 2>&1 | grep -v -i warn | tail -2 || true

  say "Web page"
  if ! command -v npm >/dev/null; then
    echo "npm is not installed, so the web page cannot be built (the API still runs)."
  elif [ ! -f web/dist/index.html ] || [ -n "$(find web/src web/index.html web/package.json -newer web/dist/index.html -print -quit 2>/dev/null)" ]; then
    (cd web && { [ -d node_modules ] || npm ci --silent; } && npm run build --silent)
    echo "Built web/dist."
  else
    echo "web/dist is up to date."
  fi

  say "Monitoring (optional)"
  if [ -n "$(containers qacopilot-monitoring)" ]; then
    stopped="$(docker ps -aq --filter "label=com.docker.compose.project=qacopilot-monitoring" --filter status=exited --filter status=created)"
    [ -n "$stopped" ] && docker start $stopped >/dev/null
    echo "Grafana http://localhost:3001  Prometheus http://127.0.0.1:9092"
  else
    echo "Not set up. Add it with: ./infra/setup-monitoring.sh host.docker.internal:$PORT"
  fi

  say "API"
  if [ -n "$(api_pid)" ]; then
    echo "Already running (pid $(api_pid)) on port $PORT."
  else
    local host
    host="$(default_host)"
    mkdir -p logs
    # Always the venv's own Python: another Python on the machine may lack the packages.
    nohup "$PY" -m uvicorn qa_api.main:app --app-dir services/api/src --host "$host" --port "$PORT" >>"$LOG" 2>&1 &
    for _ in $(seq 1 40); do
      curl -s -m2 -o /dev/null "http://$host:$PORT/health" && break
      sleep 1
    done
    curl -s -m2 -o /dev/null "http://$host:$PORT/health" || die "The API did not start. Last lines of $LOG:\n$(tail -15 "$LOG")"
    echo "Running at http://$host:$PORT (log: logs/api.log)."
  fi

  say "Done"
  echo "Stop the API:       ./start.sh stop"
  echo "Stop everything:    ./start.sh stop --all"
  echo "The LLM server is separate; check it with ./infra/check-llm.sh"
}

case "${1:-up}" in
  up | --reinstall) cmd_up "${1:-}" ;;
  stop) cmd_stop "${2:-}" ;;
  status) cmd_status ;;
  -h | --help | help) sed -n '2,16p' "$0" ;;
  *) die "Unknown command '$1'. Try: ./start.sh help" ;;
esac
