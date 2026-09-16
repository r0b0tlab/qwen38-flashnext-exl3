# Metrics — Qwen3.8-Flash-Next EXL3 2.50 bpw (RTX 3090 campaign)

<!-- PUBLISH CHECKLIST: fill <<<...>>> tokens at publication; keep raw traces in the local
     run dirs (~/r0b0bench-q200v2/runs/), per r0b0bench rules. -->

| file | what |
| --- | --- |
| `q200v2/summary.json` | Kit-produced run summary (frozen Q200v2 text-180): scores, per-family transport/grade counts, identity hashes, response-budget audit. |
| `q200v2/throughput-digest.json` | E2E throughput per PROCEDURES section 4: `completion_tokens / elapsed_seconds` per row, mean / p50 / aggregate. |
| `q200v2/telemetry.tsv` + `telemetry-digest.json` | Serve-host telemetry at 2 s cadence (power, temp, util, clock, throttle, VRAM, MemAvailable, swap); digest = load-only mean/max + min MemAvailable + throttle states. |
| `q200v2/manual-evidence.json` | Independent review of the 20 `hard_reasoning` rows (schema `r0b0tlab.qwen38.manual_evidence.v1`, hash-bound to response bytes; validated by the kit loader). |
| `niah/niah-2n.json`, `niah/niah-3n.json` | Max-context multi-needle NIAH at 262,080 tokens (33/66 % and 33/66/90 %, answer = last). |
| `niah/telemetry.tsv` + `telemetry-digest.json` | Telemetry for the NIAH phase. |

Method notes (fill at publication):

- Q200v2 identity: dataset sha256 `<<<MEASURED:dataset_sha256>>>`, run identity
  `<<<MEASURED:run_identity_sha256>>>`; chat kwargs
  `{enable_thinking, thinking, reasoning_effort=low}`; max_tokens 8192; 1 worker;
  admission: single in-flight, serialized serve.
- Serve config: `-mcs 320 -mct 6`, MTP on, cq3 cache, 262,144-token native context
  (no rope scaling), OpenAI-compatible endpoint via `scripts/serve_openai.py`.
- NIAH deviations disclosed (if any): `<<<MEASURED:niah_deviations>>>`.
- Transport failures / ceiling rows disclosed: `<<<MEASURED:transport_notes>>>`.
