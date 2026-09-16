#!/usr/bin/env bash
# Container entrypoint: launch the validated chat.py path with env-var-driven flags.
# chat.py is an interactive console app; PROMPT runs a single turn and exits.
#
# Click-run: if the model dir is empty (e.g. a fresh named volume mounted at /models),
# the EXL3 model is downloaded from Hugging Face into the volume on first start.
set -euo pipefail

TARGET_REPO="${TARGET_REPO:-r0b0tlab/Qwen3.8-Flash-Next-EXL3-2.50bpw}"

ensure_model () {
    local dir="$1" repo="$2"
    if [ -f "${dir}/config.json" ]; then
        return 0
    fi
    if [ "${AUTO_DOWNLOAD:-1}" != "1" ]; then
        echo "[entrypoint] model missing at ${dir} and AUTO_DOWNLOAD=0" >&2
        exit 1
    fi
    echo "[entrypoint] downloading ${repo} -> ${dir} (first run; this takes a while)"
    mkdir -p "${dir}"
    hf download "${repo}" --local-dir "${dir}"
}

ensure_model "${MODEL_DIR:-/models/qwen38-flash-next-exl3}" "${TARGET_REPO}"

ARGS=(
    -m "${MODEL_DIR:-/models/qwen38-flash-next-exl3}"
    -mode "${MODE:-chatml}"
    -cs "${CTX:-262144}"
    -cq "${CQ:-3}"
    -mcs "${MCS:-320}"
    -mct "${MCT:-6}"
)

[ "${MTP:-1}" = "1" ] && ARGS+=(-mtp)

[ -n "${PROMPT:-}" ] && ARGS+=(-prompt "${PROMPT}")

# Extras passed through from the host, e.g. EXTRA_ARGS="-basic -tps"
# shellcheck disable=SC2206
[ -n "${EXTRA_ARGS:-}" ] && ARGS+=(${EXTRA_ARGS})

cd /opt/exllamav3
exec python examples/chat.py "${ARGS[@]}"
