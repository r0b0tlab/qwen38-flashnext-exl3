# Flash-Next pipeline — execution progress

Plan: `~/.hermes/plans/2026-09-16_092727-qwen38-flash-next-exl3.md` (v6, single-quant 2.50 bpw)

## Status board (updated 2026-09-16 ~10:15 local)

| phase | state | notes |
| --- | --- | --- |
| 0.1 workspace | DONE | `~/exl3-flash-next`, commits c9793bd…2488a01 |
| 0.2 budget gates | DONE | disk FITS (468/582), vram mcs ≥ 321 (comfort), ram 34.0 OK; tests 4/4 |
| shard survey | DONE | 180.0B params BF16; experts 288 tensors/96 files; ngram shards 5-37 (137 tensors) |
| 1.1 download | RUNNING | `hf download` hf_transfer, 16 workers, log `notes/download3.log`; ~6-10 MB/s; ETA ~10-14 h; notification armed |
| 1.2 convert (2.50bpw) | TODO | launch on download completion: `-b 2.50 -mb 4 -vb 6 -hq -ngb 3 -cr 250` → `models/qwen38-flash-next-exl3-b250`, `-w work/target-b250`; ~3-6 h; notification |
| 2.1 serve harness | DONE (code) | `scripts/serve_openai.py` (offload + MTP + face-validations), test 2/2 passed |
| 2.2 load gate | TODO | mcs 320 / mct 6 / --mtp, VRAM ≤ 22.5 GiB, RSS ≤ 42 GiB |
| 3.x smokes | TODO | MTP acceptance ≥ 1.5; vision caption; no-YaRN assert in convert post-checks |
| 4.x perf search | scripts READY | `cpu_probe.py`, `perf_sweep.py` (rows A-F), `long_context_check.py` |
| 5.x evals | runner READY | `run_q200v2.sh` (frozen kit, dataset sha256 `66a75701…` verified), `niah_multikey.py`, `digest.py` |
| 6.x publication | scripts READY | `hf_publish.sh`; container set + `build.sh` staged; GitHub repo TBD after evals |

## Environment facts

- GPU idle; 582 GB free disk; 47 GiB RAM available; 6-core Zen 5 (mct 6 default).
- Engine: `~/exl3-qwen38-dflash2/exllamav3` (dflash2-pathway = v1.5.0 + DFlash2 commits; Flash-Next arch supported incl. PLE + MTP).
- mul1 codebook eligibility for CPU offload CONFIRMED (conversion default; required by the offload path per doc/convert.md:38).
- HF auth: `r0b0tlab` (token in `~/.hermes/.env`); GHCR login from 27B campaign.
- User directive: repos AFTER evals pass; user flips GHCR package public at the end.

## Critical constants

- Experts: 4.9152e6 params each; per-expert bytes at 2.50 bpw = 1.536 MB.
- Preset: mcs 320, mct 6, MTP on, cq3, cache 270336 tokens, native 262144 context (NO rope flags ever).
- Sweep env: `EXL3_MOE_CPU_PIN=1`, `EXL3_MOE_CPU_SWIZZLE=1`, `EXL3_MOE_FUSED_DET=1` (ISA auto → expect vbmi).
