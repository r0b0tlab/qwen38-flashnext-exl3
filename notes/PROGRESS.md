# Flash-Next pipeline — execution progress

Plan: `~/.hermes/plans/2026-09-16_092727-qwen38-flash-next-exl3.md` (v6, single-quant 2.50 bpw)

## NEXT ACTIONS (in order — resume here after any interruption)

1. **Download complete** (background proc fires; log `notes/download3.log` ends `download exit=0`):
   - verify: `ls models/qwen38-flash-next-hf/model-*.safetensors | wc -l` == 131
   - verify: `du -sh models/qwen38-flash-next-hf` ≈ 335 GiB; if short, re-run `hf download` once (gap-fill, fast)
   - then `hf download` one more time to confirm nothing pending (it exits immediately when complete)
2. **Launch conversion** (background, notify):
   `cd /home/am/exl3-flash-next && bash scripts/convert.sh` (canonical command; logs `notes/convert.log`)
   - watch after layer 10: GPU ≤ 22 GiB; torch RSS < 40 GiB; `df -h /` ≥ 150 GiB free
   - interrupted → `bash scripts/convert.sh -r`
3. **Post-conversion checks** (plan Task 1.2): no-YaRN assert (262144 + no rope_scaling),
   `ngram_embedding.safetensors` present, `mtp`/`visual` keys in index, `du -sh` ≈ 58 GiB.
4. **Load gate**: `python3 scripts/serve_openai.py --target models/qwen38-flash-next-exl3-b250
   --moe-cpu-split 320 --moe-cpu-threads 6 --mtp --cache-tokens 270336 --cq 3 --port 8890 --model-name qwen38-flash-next-exl3`
   (background; `[serve] READY` expected; VRAM ≤ 22.5 GiB; RSS ≤ 42 GiB; chat sanity → answers `4`).
5. **Smokes**: `scripts/acceptance_check.py --draft mtp --moe-cpu-split 320` (acceptance ≥ 1.5); vision example caption.
6. **Perf**: `scripts/cpu_probe.py` (calibration), then `scripts/perf_sweep.py --rows A,B,E` → C,D,F; then `-ndt` sweep on winner.
7. **Evals**: NIAH 2n + 3n at 262080 against the serve; Q200v2 via `scripts/run_q200v2.sh <run-id>`;
   digests via `scripts/digest.py <run_dir>`; manual-evidence review for hard_reasoning.
8. **Container build** (deferred ~13:55 — Docker Hub pull stalled under download pressure): rerun `bash container/build.sh` once the download finishes; then push to GHCR at publication.
9. **Publication** (only after evals pass): fill `<<<MEASURED:...>>>` tokens in `hf-cards/README-flash-next.md`
   (grep '<<<' → must be empty), then `scripts/hf_publish.sh models/qwen38-flash-next-exl3-b250 r0b0tlab/Qwen3.8-Flash-Next-EXL3-2.50bpw hf-cards/README-flash-next.md`;
   container build+push via `container/build.sh` + GHCR push; GitHub results repo `r0b0tlab/qwen38-flashnext-exl3`; user flips package public at the end.

## Status board (updated 2026-09-17 ~05:00 local)

| phase | state | notes |
| --- | --- | --- |
| 0.1 workspace | DONE | commits c9793bd…287e87d |
| 0.2 budget gates | DONE | disk FITS (468/582), vram mcs ≥ 321 (comfort), ram 34.0 OK; tests 4/4 |
| shard survey | DONE | 180.0B params BF16; experts 288 tensors/96 files; ngram shards 5-37 |
| pre-flight checks | DONE | arch ✓; mul1 default codebook ✓; `-b` float ✓; ngram inline ✓; `-hq` ✓; source config 262144/no-rope ✓; tokenizer kit-kwargs render PASS (47 vs 59 tok) |
| 1.1 download | DONE | 131/131 shards, 335.3 GiB, 0 incomplete, gap-fill scan clean |
| 1.2 convert 2.50 | DONE | `-- All done`; 6 shards + 19.8 GB ngram table (61 GB total); no-YaRN + mtp/visual asserts PASS; work dir freed |
| 2.x serve load gate | PARTIAL | mcs 320 loads OK in smokes (vbmi worker, 6 threads); full serve gate next |
| 3.x smokes | DONE | MTP acceptance 4.06 (cold 6.8 tok/s); vision PASS (screenshot described, 26.7 tok/s); both smokes ran mcs 320 |
| 4.x perf search | RUNNING | cpu_probe + sweep rows A,B,E (proc_6f75778e2e34); then C,D,F + ndt + PINNED_ARENA A/B |
| 5.x evals | DONE | NIAH x2 PASS @262080; Q200v2 173/6/1 (gsm8k 98.75, HR 20/20, HE 100, ifeval 87.18); e2e mean 51.2 tok/s; digests written |
| 6.x publication | IN FLIGHT | GitHub repo PUSHED; HF model uploading (notify armed); GHCR image PUSHED + verified via docker manifest inspect (index sha256:746ad74a..., tags 2.50bpw + latest; package PRIVATE until owner flips) |

## Environment facts

- GPU idle; disk 552 GB free; 47 GiB RAM; 6-core Zen 5 (mct 6 default).
- Engine: `~/exl3-qwen38-dflash2/exllamav3` (v1.5.0 + DFlash2 commits; Flash-Next arch incl. PLE + MTP).
- HF auth `r0b0tlab` (token `~/.hermes/.env`); GHCR login from 27B campaign.
- User directive: repos AFTER evals pass; user flips GHCR package public at the end.

## Critical constants

- Experts 4.9152e6 params; per-expert bytes @2.50bpw = 1.536 MB.
- Preset mcs 320 / mct 6 / MTP on / cq3 / cache 270336 / native 262144 (NO rope flags ever).
- Sweep env: `EXL3_MOE_CPU_PIN=1`, `EXL3_MOE_CPU_SWIZZLE=1`, `EXL3_MOE_FUSED_DET=1`.