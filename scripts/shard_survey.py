#!/usr/bin/env python3
"""Shard survey for Qwen3.8-Flash-Next: group tensors/files, print the inventory.

Read-only against the live HF repo. Results used to fix the bpw pre-conversion
(see plan Appendix A). Records to notes/shard-survey.txt when run from the
workspace root.
"""
import json, re, urllib.request
from collections import defaultdict

ENV = '/home/am/.hermes/.env'

def main():
    tok = re.search(r'^HF_TOKEN=(.+)$', open(ENV).read(), re.M).group(1).strip()

    def get(u):
        req = urllib.request.Request(u, headers = {'Authorization': f'Bearer {tok}'})
        return json.load(urllib.request.urlopen(req))

    info = get('https://huggingface.co/api/models/Qwen/Qwen3.8-Flash-Next')
    print('dtype totals:', info.get('safetensors', {}).get('parameters'))
    print('total params:', info.get('safetensors', {}).get('total'))

    idx = get('https://huggingface.co/Qwen/Qwen3.8-Flash-Next/raw/main/model.safetensors.index.json')
    wm = idx['weight_map']
    print('tensors:', len(wm), 'total_size(GiB):', round(idx['metadata']['total_size'] / 2**30, 1))

    g = defaultdict(lambda: [0, set()])
    for k, v in wm.items():
        if 'ngram' in k or '.ple' in k: t = 'ngram'
        elif 'mtp' in k: t = 'mtp'
        elif 'visual' in k: t = 'vision'
        elif 'expert' in k: t = 'experts'
        elif 'embed' in k or 'lm_head' in k: t = 'embed_head'
        else: t = 'trunk'
        g[t][0] += 1; g[t][1].add(v)
    for t, (n, f) in sorted(g.items()):
        print(f'{t:10s} tensors={n:5d} files={len(f):3d}')

if __name__ == "__main__":
    main()
