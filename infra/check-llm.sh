#!/usr/bin/env bash
# Is the LLM server reachable? Reads QACOPILOT_LLM_BASE_URL from .env (or the environment).
# Usage: ./infra/check-llm.sh [--wait [seconds]]   (--wait polls until it is up, default 300s)
# Exit 0 = up and answering, 1 = not.
set -uo pipefail
cd "$(dirname "$0")/.."

url="${QACOPILOT_LLM_BASE_URL:-}"
if [ -z "$url" ] && [ -f .env ]; then
  url=$(grep -E '^QACOPILOT_LLM_BASE_URL=' .env | head -1 | cut -d= -f2- | tr -d '"'"'"' \r')
fi
[ -n "$url" ] || { echo "QACOPILOT_LLM_BASE_URL is not set (env or .env)"; exit 1; }
url="${url%/}"

key="${QACOPILOT_LLM_API_KEY:-}"
if [ -z "$key" ] && [ -f .env ]; then
  key=$(grep -E '^QACOPILOT_LLM_API_KEY=' .env | head -1 | cut -d= -f2- | tr -d '"'"'"' \r')
fi
# "not-needed" is the placeholder for a server with no key; send a real key if there is one
auth=()
if [ -n "$key" ] && [ "$key" != "not-needed" ]; then auth=(-H "Authorization: Bearer $key"); fi
host=$(echo "$url" | sed -E 's#^[a-z]+://([^/:]+).*#\1#')

# Is the server open to anyone who can reach it? A POST with an empty body and NO key should be
# turned away (401/403). If it reaches the handler (400/422) the LLM is unprotected, which matters
# most when it is exposed with Tailscale Funnel. Nothing is generated: the body is empty.
auth_check() {
  local c
  c=$(curl -s -m 10 -o /dev/null -w "%{http_code}" -X POST -H 'content-type: application/json' -d '{}' "$url/chat/completions" 2>/dev/null) || c=000
  case "$c" in
    401|403) echo "  auth:      refuses requests that carry no API key (good)" ;;
    000)     echo "  auth:      could not test" ;;
    *)       echo "  auth:      WARNING: answered HTTP $c to a request with NO key. Anyone who can reach"
             echo "             this URL can use the LLM. Start llama-server with --api-key <secret>." ;;
  esac
}

check() {
  local ok=0
  if command -v tailscale >/dev/null 2>&1 && [[ "$host" == *.ts.net ]]; then
    if tailscale status >/dev/null 2>&1; then
      echo "  tailscale: logged in"
    else
      echo "  tailscale: NOT running/logged in here (sudo tailscale up)"; return 1
    fi
  fi
  if getent hosts "$host" >/dev/null 2>&1; then
    echo "  dns:       $host resolves"
  else
    echo "  dns:       $host does NOT resolve"; return 1
  fi
  local out code t
  out=$(curl -s -m 10 -o /dev/null -w "%{http_code} %{time_total}" "${auth[@]}" "$url/models" 2>/dev/null) || out="000 10"
  code=${out%% *}; t=${out##* }
  if [[ "$code" == 2* ]]; then
    echo "  http:      $url/models -> $code in ${t}s"; ok=1
  elif [[ "$code" == 401 || "$code" == 403 ]]; then
    echo "  http:      $url/models -> $code: the server is up but REJECTED the API key (QACOPILOT_LLM_API_KEY)"
  else
    echo "  http:      $url/models -> no answer (code $code): PC asleep, llama.cpp not running, or tailscale serve/funnel down"
  fi
  [ "$ok" = 1 ] && auth_check
  [ "$ok" = 1 ]
}

if [ "${1:-}" = "--wait" ]; then
  limit="${2:-300}"; end=$((SECONDS + limit))
  while true; do
    echo "[$(date +%T)] checking $url"
    check && { echo "LLM is UP"; exit 0; }
    [ $SECONDS -ge $end ] && { echo "gave up after ${limit}s"; exit 1; }
    sleep 10
  done
fi

echo "checking $url"
if check; then echo "LLM is UP"; else echo "LLM is DOWN"; exit 1; fi
