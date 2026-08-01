#!/usr/bin/env bash
# eval-summary.sh — run `just eval` across every prefix and print only the
# aggregate WER/CER, labeled by speaker. Full output is kept in $LOGDIR.
#
#   ./eval-summary.sh                 # run everything
#   LOGDIR=./eval-logs ./eval-summary.sh
#   ./eval-summary.sh manual/Ryan     # or pass explicit prefixes

set -uo pipefail

SPEAKERS=(speaker_017_11ec372417df speaker_023_9d57aeaf822c speaker_095_1daf6547c74f)
CS_SETS=(cs-2 cs-4)
MANUAL=(Johnny Ryan Sakamoto)

LOGDIR=${LOGDIR:-$(mktemp -d -t evalsum.XXXXXX)}
mkdir -p "$LOGDIR"

# ---- build the prefix list -------------------------------------------------
if (($# > 0)); then
  prefixes=("$@")
else
  prefixes=()
  for s in "${SPEAKERS[@]}"; do
    for cs in "${CS_SETS[@]}"; do
      prefixes+=("synthetic/$cs/$s")
    done
  done
  for m in "${MANUAL[@]}"; do
    prefixes+=("manual/$m")
  done
fi

# ---- helpers ---------------------------------------------------------------
# grab <line-prefix> <output>  ->  "<wer> <cer>"
grab() {
  local line
  line=$(grep -m1 "^$1" <<<"$2") || return 1
  line=${line#*:}                     # drop everything up to the colon
  awk 'NF>=2 {print $1, $2}' <<<"${line//\// }"
}

delta() { awk -v a="$1" -v b="$2" 'BEGIN{printf "%+.4f", a-b}'; }
fmt()   { awk -v v="$1" 'BEGIN{printf "%.4f", v}'; }

rows=()
fail=0

# ---- run -------------------------------------------------------------------
for p in "${prefixes[@]}"; do
  printf '\r\033[2Krunning %s ...' "$p" >&2
  log="$LOGDIR/${p//\//_}.log"

  if ! out=$(just eval --prefix "$p" 2>&1 | tee "$log"); then
    rows+=("$p|-|ERR|ERR|ERR|ERR|-|-")
    fail=1
    continue
  fi

  read -r base_wer base_cer < <(grab 'Base Aggregate' "$out")
  read -r pipe_wer pipe_cer < <(grab 'Pipeline Aggregate' "$out")
  files=$(grep -m1 '^Evaluated files' <<<"$out" | awk '{print $NF}')

  if [[ -z ${base_wer:-} || -z ${pipe_wer:-} ]]; then
    rows+=("$p|${files:--}|NOPARSE|NOPARSE|NOPARSE|NOPARSE|-|-")
    fail=1
    continue
  fi

  rows+=("$p|${files:--}|$(fmt "$base_wer")|$(fmt "$base_cer")|$(fmt "$pipe_wer")|$(fmt "$pipe_cer")|$(delta "$pipe_wer" "$base_wer")|$(delta "$pipe_cer" "$base_cer")")
done
printf '\r\033[2K' >&2

# ---- report ----------------------------------------------------------------
row_fmt='%-46s %5s %9s %9s %9s %9s %9s %9s\n'
# shellcheck disable=SC2059
printf "$row_fmt" PREFIX FILES BASE_WER BASE_CER PIPE_WER PIPE_CER D_WER D_CER
printf '%s\n' "$(printf '%.0s-' {1..118})"
for r in "${rows[@]}"; do
  IFS='|' read -r a b c d e f g h <<<"$r"
  # shellcheck disable=SC2059
  printf "$row_fmt" "$a" "$b" "$c" "$d" "$e" "$f" "$g" "$h"
done

echo
echo "full logs: $LOGDIR"
exit "$fail"
