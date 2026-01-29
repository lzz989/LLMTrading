#!/usr/bin/env bash
set -euo pipefail

# Public repo guard: fail fast if we accidentally track secrets or personal artifacts.
# IMPORTANT: Only print file paths (not matched content) to avoid leaking secrets to CI logs.

fail=0

_err() {
  echo "ERROR: $*" >&2
  fail=1
}

_check_tracked_path() {
  local path="$1"
  if git ls-files --error-unmatch "$path" >/dev/null 2>&1; then
    _err "tracked file should NOT be committed: $path"
  fi
}

_check_tracked_glob_nonempty() {
  local glob="$1"
  local hits
  hits="$(git ls-files "$glob" | head -n 20 || true)"
  if [[ -n "${hits}" ]]; then
    _err "tracked paths should NOT be committed (showing up to 20): pattern=$glob"
    echo "${hits}" >&2
  fi
}

_check_pattern_files_only() {
  local pat="$1"
  local desc="$2"
  local hits
  # Exclude this script itself to avoid self-matching the regex string literals.
  hits="$(git grep -lE "${pat}" -- ':!scripts/sensitive_scan.sh' || true)"
  if [[ -n "${hits}" ]]; then
    _err "${desc} (pattern=${pat})"
    echo "${hits}" >&2
  fi
}

# 1) Never commit local env/data/output artifacts
_check_tracked_path ".env"
_check_tracked_glob_nonempty "data/*"
_check_tracked_glob_nonempty "outputs/*"

# 2) Common secret patterns (file names only)
_check_pattern_files_only "sk-[A-Za-z0-9]{20,}" "possible API key (sk-...) found"
_check_pattern_files_only "BEGIN( RSA)? PRIVATE KEY" "private key block found"

# TuShare proxy tutorial leftovers: forbid numeric hardcode of token & known private domain keywords.
_check_pattern_files_only "pro\\._DataApi__token\\s*=\\s*['\\\"][0-9]{10,}['\\\"]" "hardcoded tushare token found"
_check_pattern_files_only "xiximiao|aihubproxy" "private proxy domain/keyword found"

if [[ "${fail}" -ne 0 ]]; then
  echo >&2
  echo "Sensitive scan FAILED. Fix the files above before pushing public." >&2
  exit 2
fi

echo "Sensitive scan OK."
