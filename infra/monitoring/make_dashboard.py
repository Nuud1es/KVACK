#!/usr/bin/env python3
"""Writes grafana/dashboards/qa-copilot.json. Edit the panels here and re-run; the JSON is what
Grafana loads (provisioned, read-only in the UI). Standard library only.

    python3 infra/monitoring/make_dashboard.py
"""

import json
from pathlib import Path

DS = {"type": "prometheus", "uid": "prometheus"}
RATE = "$__rate_interval"
_id = 0


def _next() -> int:
    global _id
    _id += 1
    return _id


def _targets(items: list[tuple[str, str]]) -> list[dict]:
    return [
        {"datasource": DS, "expr": expr, "legendFormat": legend, "refId": chr(65 + i)}
        for i, (expr, legend) in enumerate(items)
    ]


def row(title: str, y: int) -> dict:
    return {"type": "row", "title": title, "id": _next(), "collapsed": False,
            "gridPos": {"h": 1, "w": 24, "x": 0, "y": y}, "panels": []}


def stat(title: str, expr: str, x: int, y: int, w: int = 4, h: int = 4, *, unit: str = "short",
         steps: list[tuple[float | None, str]] | None = None, mappings: list[dict] | None = None,
         desc: str = "") -> dict:
    steps = steps or [(None, "green")]
    return {
        "type": "stat", "title": title, "id": _next(), "description": desc, "datasource": DS,
        "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "targets": _targets([(expr, "")]),
        "options": {"reduceOptions": {"calcs": ["lastNotNull"]}, "colorMode": "background",
                    "graphMode": "none", "textMode": "auto"},
        "fieldConfig": {"defaults": {
            "unit": unit, "mappings": mappings or [], "noValue": "no data",
            "thresholds": {"mode": "absolute",
                           "steps": [{"color": c, "value": v} for v, c in steps]}}, "overrides": []},
    }


UP_DOWN = [{"type": "value", "options": {"1": {"text": "UP", "color": "green"},
                                         "0": {"text": "DOWN", "color": "red"}}}]
UP_STEPS = [(None, "red"), (1, "green")]


def ts(title: str, items: list[tuple[str, str]], x: int, y: int, w: int = 8, h: int = 7, *,
       unit: str = "short", stack: bool = False, desc: str = "", fill: int = 10) -> dict:
    return {
        "type": "timeseries", "title": title, "id": _next(), "description": desc, "datasource": DS,
        "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "targets": _targets(items),
        "options": {"legend": {"displayMode": "list", "placement": "bottom"},
                    "tooltip": {"mode": "multi", "sort": "desc"}},
        "fieldConfig": {"defaults": {
            "unit": unit, "custom": {"lineWidth": 2, "fillOpacity": fill, "showPoints": "never",
                                     "stacking": {"mode": "normal" if stack else "none"}}},
            "overrides": []},
    }


def build() -> dict:
    p: list[dict] = []
    y = 0
    p.append(row("Health", y)); y += 1
    p += [
        stat("API", 'up{job="qa-copilot"}', 0, y, unit="none", mappings=UP_DOWN, steps=UP_STEPS,
             desc="Can Prometheus scrape the API?"),
        stat("Postgres", 'qacopilot_dependency_up{service="postgres"}', 4, y, unit="none",
             mappings=UP_DOWN, steps=UP_STEPS, desc="Approvals, review queue and due dates need it"),
        stat("Qdrant", 'qacopilot_dependency_up{service="qdrant"}', 8, y, unit="none",
             mappings=UP_DOWN, steps=UP_STEPS, desc="Document search"),
        stat("LLM", 'qacopilot_dependency_up{service="llm"}', 12, y, unit="none",
             mappings=UP_DOWN, steps=UP_STEPS, desc="Answers, extraction and reminder intros"),
        stat("Alerts firing", 'count(ALERTS{alertstate="firing",alertname=~"QACopilot.*"}) or vector(0)',
             16, y, unit="none", steps=[(None, "green"), (1, "red")],
             desc="Rules are in infra/monitoring/prometheus/alerts.yml"),
        stat("Queue data age", "time() - qacopilot_metrics_refreshed_timestamp_seconds", 20, y,
             unit="s", steps=[(None, "green"), (120, "orange"), (300, "red")],
             desc="How old the review/approval numbers are; high means the API's refresh is failing"),
    ]
    y += 4
    p.append(row("Work waiting for people", y)); y += 1
    p += [
        stat("Readings to review", "qacopilot_certificates_needing_review", 0, y, 5,
             steps=[(None, "green"), (1, "orange"), (25, "red")],
             desc="Poor scans and ungrounded values nobody has confirmed"),
        stat("Awaiting approval", "qacopilot_agent_runs_awaiting_approval", 5, y, 5,
             steps=[(None, "green"), (1, "orange")]),
        stat("Oldest approval waiting", "qacopilot_agent_oldest_awaiting_approval_seconds", 10, y, 5,
             unit="s", steps=[(None, "green"), (14400, "orange"), (86400, "red")]),
        stat("Certificates overdue", "qacopilot_certificates_overdue", 15, y, 4,
             steps=[(None, "green"), (1, "red")]),
        stat("Due in 30 days", "qacopilot_certificates_due_30d", 19, y, 5,
             steps=[(None, "green"), (1, "orange")], desc="Includes the overdue ones"),
    ]
    y += 4
    p.append(ts("Queues over time", [
        ("qacopilot_certificates_needing_review", "readings to review"),
        ("qacopilot_agent_runs_awaiting_approval", "awaiting approval"),
        ("qacopilot_certificates_overdue", "overdue certificates"),
    ], 0, y, 24, 6)); y += 6

    p.append(row("Traffic", y)); y += 1
    p += [
        ts("Requests per second by route",
           [(f"sum by (route) (rate(qacopilot_http_requests_total[{RATE}]))", "{{route}}")],
           0, y, unit="reqps", stack=True),
        ts("Server errors (5xx) per second",
           [(f'sum(rate(qacopilot_http_requests_total{{status=~"5.."}}[{RATE}]))', "5xx"),
            (f'sum(rate(qacopilot_http_requests_total{{status=~"4.."}}[{RATE}]))', "4xx")],
           8, y, unit="reqps", desc="4xx are mostly sign-in and validation; 5xx are faults"),
        ts("Latency p95 by route",
           [(f"histogram_quantile(0.95, sum by (le, route) (rate(qacopilot_http_request_seconds_bucket[{RATE}])))",
             "{{route}}")], 16, y, unit="s",
           desc="Streamed answers are timed until the stream ends"),
    ]
    y += 7
    p.append(row("Ask", y)); y += 1
    p += [
        ts("Answers by path",
           [(f'sum by (via) (rate(qacopilot_ask_total{{outcome="answered"}}[{RATE}]))', "{{via}}")],
           0, y, unit="reqps", stack=True,
           desc="documents = search + LLM; certificates and coa_check = computed in code"),
        ts("Outcomes",
           [(f"sum by (outcome) (rate(qacopilot_ask_total[{RATE}]))", "{{outcome}}")],
           8, y, unit="reqps", stack=True, desc="refused = the documents do not cover it"),
        ts("Time to answer p95 by path",
           [(f"histogram_quantile(0.95, sum by (le, via) (rate(qacopilot_ask_seconds_bucket[{RATE}])))",
             "{{via}}")], 16, y, unit="s"),
    ]
    y += 7
    p.append(row("LLM", y)); y += 1
    p += [
        ts("LLM calls by operation and outcome",
           [(f"sum by (operation, outcome) (rate(qacopilot_llm_requests_total[{RATE}]))",
             "{{operation}} {{outcome}}")], 0, y, unit="reqps", stack=True,
           desc="unreachable = could not connect or timed out; error = the server answered badly"),
        ts("LLM latency",
           [(f"histogram_quantile(0.5, sum by (le) (rate(qacopilot_llm_request_seconds_bucket[{RATE}])))", "p50"),
            (f"histogram_quantile(0.95, sum by (le) (rate(qacopilot_llm_request_seconds_bucket[{RATE}])))", "p95")],
           8, y, unit="s"),
        ts("Tokens per second",
           [(f"sum by (kind) (rate(qacopilot_llm_tokens_total[{RATE}]))", "{{kind}}")], 16, y,
           stack=True),
    ]
    y += 7
    p.append(row("Agent and ingest", y)); y += 1
    p += [
        ts("Agent runs started by status",
           [(f"sum by (status) (increase(qacopilot_agent_runs_total[{RATE}]))", "{{status}}")],
           0, y, stack=True),
        ts("Approvals and rejections",
           [(f"sum by (decision) (increase(qacopilot_agent_decisions_total[{RATE}]))", "{{decision}}")],
           8, y, stack=True),
        ts("Documents ingested by type",
           [(f"sum by (doc_type) (increase(qacopilot_documents_ingested_total[{RATE}]))", "{{doc_type}}")],
           16, y, stack=True),
    ]
    return {
        "uid": "qacopilot", "title": "KVACK", "tags": ["qa-copilot"], "timezone": "browser",
        "schemaVersion": 39, "version": 1, "refresh": "30s", "editable": False,
        "time": {"from": "now-6h", "to": "now"}, "panels": p, "templating": {"list": []},
        "annotations": {"list": []},
        "description": "Health, traffic, answers, LLM and the queues that wait for people.",
    }


if __name__ == "__main__":
    out = Path(__file__).parent / "grafana" / "dashboards" / "qa-copilot.json"
    out.write_text(json.dumps(build(), indent=2) + "\n")
    print("wrote", out)
