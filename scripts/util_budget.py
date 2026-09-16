#!/usr/bin/env python3
"""Disk / VRAM / RAM gates for the Flash-Next pipeline (bpw fixed at 2.50). Pure stdlib."""
import argparse, sys

SRC_GIB, TABLE_GIB = 335.3, {4: 24.2, 3: 18.6, 2: 13.4}
TRUNK_B, PER_EXPERT_B, SAFETY = 125.0, 4.9152e6, 15.0
COMFORT_GIB = 1.5          # owner rule: never run the card at the wire


def disk(bits, ngb, free):
    out = 120.8e9 * bits / 8 / 2**30 + 3.5e9 * (bits + 0.6) / 8 / 2**30 + TABLE_GIB[ngb] + 4.0
    need = SRC_GIB + 2 * out + SAFETY
    ok = need <= free
    print(f"out~{out:.0f} peak={need:.0f} free={free:.0f} -> {'FITS' if ok else f'STOP deficit {need-free:.0f}'}")
    return 0 if ok else 1


def vram(bits, cq):
    kv = {3: 2.25, 4: 3.0, 16: 6.0}[cq]
    fixed = 1.7 + 2.6 + kv + 2.5            # trunk + head/MTP/vision + KV + scratch
    budget = 24 - fixed - COMFORT_GIB
    per_expert_mib = PER_EXPERT_B * bits / 8 / 2**20
    n_gpu = int(budget * 1024 / (48 * per_expert_mib))
    mcs = 512 - n_gpu + 4
    print(f"fixed {fixed:.1f}, comfort {COMFORT_GIB} -> expert budget {budget:.1f} GiB "
          f"-> ~{n_gpu}/layer GPU -> mcs >= {mcs} (comfort included)")
    return 0


def ram(mcs, bits):
    cpu = 48 * mcs * PER_EXPERT_B * bits / 8 / 2**30
    need = cpu + 12
    ok = need <= 47
    print(f"RAM need ~{need:.1f} (experts {cpu:.1f} + 12) -> {'OK' if ok else 'over'}")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices = ["disk", "vram", "ram"], required = True)
    ap.add_argument("--free-gib", type = float, default = 584)
    ap.add_argument("--bits", type = float, default = 2.5)
    ap.add_argument("--ngram-bits", type = int, default = 3)
    ap.add_argument("--mcs", type = int, default = 512)
    ap.add_argument("--cq", type = int, default = 3)
    a = ap.parse_args()
    sys.exit({"disk": lambda: disk(a.bits, a.ngram_bits, a.free_gib),
              "vram": lambda: vram(a.bits, a.cq),
              "ram":  lambda: ram(a.mcs, a.bits)}[a.mode]())
