# Budget gates — recorded 2026-09-16 (plan v6, single-quant 2.50 bpw)

## util_budget.py outputs (real run)

```
== disk ==  out~59 peak=468 free=582 -> FITS
== vram ==  fixed 9.1, comfort 1.5 -> expert budget 13.4 GiB -> ~195/layer GPU -> mcs >= 321
            (comfort included)
== ram ==   RAM need ~34.0 (experts 22.0 + 12) -> OK    [mcs 321 @ 2.50 bpw]
```

Interpretation: source 335.3 + 2×~59 output/work + 15 safety ≈ 468 GiB peak vs 582 free.
The load preset starts at **mcs 321, mct 6**; sweep rows 295/320/360 around it.

## Shard survey (live, 2026-09-16)

```
total params: 180.0B (BF16), source 335.3 GiB / 131 shards
experts   288 tensors /  96 files
trunk     867 tensors /  51 files
ngram     137 tensors /  33 files  (shards 5-37, exactly)
mtp        31 tensors /  28 files
vision    333 tensors /   1 file   (shard 1)
embed_head  2 tensors /   2 files  (shards 1, 130-131)
```

## Decisions from this file

- bpw FIXED at 2.50 (see plan Appendix A table) — one quantization, no ladder.
- ngram RAM-offload: infeasible at this preset (52.6 GiB need vs 47 avail) — dropped.
- KV: cq3 default; cq4 is a sweep probe row only.
