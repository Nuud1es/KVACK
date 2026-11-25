# LLM feasibility spike (Phase 1)

Run 2026-11-18 from the Linux laptop against the work LLM machine
(`Darkstar-Nemotron-3.5-Lightning-30B-A3B-Abliterated-BF16.Q8_0.gguf`, behind a proxy).
One document per check: this shows the model works, it is not an accuracy measurement
(that comes from the golden set in Phase 2).

| Check | Result |
|---|---|
| Tool calling (`get_weather`) | PASS, 1.0s |
| Schema extraction (calibration cert) | PASS, 5.9s; serial, ISO dates and result correct |
| Thinking off | `enable_thinking` ignored; reasoning returned separately in `reasoning_content`, `content` is clean |
| `response_format` JSON schema | Not enforced (answered `14 March 2026`) |

## Decision
Keep the 30B-A3B model. `extract()` already compensates for the missing enforcement
(schema in the prompt, JSON pulled from prose/fences, Pydantic validation, retry with field
errors). Every date field in extraction schemas must say ISO `YYYY-MM-DD` in its description.

## Notes
- Langfuse v4 (events-only): `/api/public/traces` returns 404; query `/api/public/v2/observations`.
- Spike calls were traced (observations visible in Langfuse).
