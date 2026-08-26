#!/usr/bin/env bash
# Build and probe one exact H3 endpoint release archive.
#
#   scripts/h3-release.sh --artifact-release <name> [--endpoint org/name]
#       [--runtime-sha <40-hex>] [--tensorfs-sha <40-hex>]
#       [--cozy-eval-sha <40-hex>] [--out <dir>]
#
# This repo owns the serving BUNDLE, not a per-endpoint image, cache mount, or worker
# receipt. The bundle is one source commit, one committed uv lock, three source-built peer
# wheels, and the existing release.json archive declaration. `build-pins.json` records the
# inputs th-042 must materialize; it does not claim that materialization happened on a
# worker.
set -euo pipefail

RUNTIME_REPO="${RUNTIME_REPO:-$HOME/cozy_v2/cozy-runtime}"
TENSORFS_REPO="${TENSORFS_REPO:-$HOME/cozy_v2/tensorfs}"
COZY_EVAL_REPO="${COZY_EVAL_REPO:-$HOME/cozy_v2/cozy-eval}"

# Full commits only. Abbreviations are a moving lookup, not a release input.
RUNTIME_SHA="${RUNTIME_SHA:-96b8c235fe1ce48e392256610eb47ad318dd6850}"
RUNTIME_FLOOR="d78a9a0fddb34c11c334019cde83df6a5260bdee"
TENSORFS_SHA="${TENSORFS_SHA:-c6c598baca3799508f64b4ef02af542c9181dc27}"
COZY_EVAL_SHA="${COZY_EVAL_SHA:-61560ff5cc0bce2403b7e2bc9e6bff753be58ca1}"

BUILD_PYTHON="3.12.12"
REQUIRED_UV="0.9.18"
REQUIRED_MATURIN="1.14.1"
OUT="${OUT:-$HOME/.cache/cozy/se-011}"
VERSION="${VERSION:-1.0.0}"
ENDPOINT="cozy/minimax-h3"
ARTIFACT_RELEASE=""

usage() {
  echo "usage: $0 --artifact-release <name> [--endpoint org/name]" >&2
  echo "          [--runtime-sha <40-hex>] [--tensorfs-sha <40-hex>]" >&2
  echo "          [--cozy-eval-sha <40-hex>] [--out <dir>]" >&2
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --endpoint) ENDPOINT="$2"; shift 2 ;;
    --artifact-release) ARTIFACT_RELEASE="$2"; shift 2 ;;
    --runtime-sha) RUNTIME_SHA="$2"; shift 2 ;;
    --tensorfs-sha) TENSORFS_SHA="$2"; shift 2 ;;
    --cozy-eval-sha) COZY_EVAL_SHA="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac
done

if [[ ! "$ENDPOINT" =~ ^[a-z0-9][a-z0-9._-]*/[a-z0-9][a-z0-9._-]*$ ]]; then
  echo "REFUSED: endpoint must be an org/name slug, got $ENDPOINT" >&2
  exit 2
fi
if [[ ! "$ARTIFACT_RELEASE" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "REFUSED: --artifact-release is required and must be an opaque release name" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -n "$(git -C "$ROOT" status --porcelain)" ]; then
  echo "REFUSED: $ROOT is dirty. Release source is an exact commit; commit or remove only" >&2
  echo "         the intended H3 changes before building." >&2
  exit 3
fi

exact_commit() {
  local repo="$1" pin="$2" label="$3" resolved
  if [[ ! "$pin" =~ ^[0-9a-f]{40}$ ]]; then
    echo "REFUSED: $label pin is not a full 40-hex commit: $pin" >&2
    return 1
  fi
  if ! resolved="$(git -C "$repo" rev-parse --verify "${pin}^{commit}" 2>/dev/null)"; then
    echo "REFUSED: $label commit $pin is absent from $repo" >&2
    return 1
  fi
  if [ "$resolved" != "$pin" ]; then
    echo "REFUSED: $label resolved to $resolved instead of exact input $pin" >&2
    return 1
  fi
  printf '%s\n' "$resolved"
}

SOURCE_SHA="$(git -C "$ROOT" rev-parse HEAD)"
RUNTIME_SHA="$(exact_commit "$RUNTIME_REPO" "$RUNTIME_SHA" cozy-runtime)"
TENSORFS_SHA="$(exact_commit "$TENSORFS_REPO" "$TENSORFS_SHA" tensorfs)"
COZY_EVAL_SHA="$(exact_commit "$COZY_EVAL_REPO" "$COZY_EVAL_SHA" cozy-eval)"
if ! git -C "$RUNTIME_REPO" merge-base --is-ancestor "$RUNTIME_FLOOR" "$RUNTIME_SHA"; then
  echo "REFUSED: runtime $RUNTIME_SHA predates the binding-record floor $RUNTIME_FLOOR" >&2
  exit 3
fi

if [ "$(uv --version)" != "uv $REQUIRED_UV" ]; then
  echo "REFUSED: this lock was cut with uv $REQUIRED_UV; found $(uv --version)" >&2
  exit 3
fi
if [ "$(maturin --version)" != "maturin $REQUIRED_MATURIN" ]; then
  echo "REFUSED: TensorFS wheel requires maturin $REQUIRED_MATURIN; found $(maturin --version)" >&2
  exit 3
fi
if ! BUILD_PYTHON_PATH="$(uv python find "$BUILD_PYTHON" 2>/dev/null)"; then
  echo "REFUSED: uv-managed Python $BUILD_PYTHON is not installed" >&2
  exit 3
fi
for command in ffmpeg ffprobe; do
  if ! command -v "$command" >/dev/null; then
    echo "REFUSED: deterministic media probes require $command on PATH" >&2
    exit 3
  fi
done

mkdir -p "$OUT"
WORK="$(mktemp -d "$OUT/.h3-build.XXXXXX")"
TENSORFS_TREE="/tmp/cozy-h3-tensorfs-$TENSORFS_SHA"
if [[ ! "$TENSORFS_TREE" =~ ^/tmp/cozy-h3-tensorfs-[0-9a-f]{40}$ ]]; then
  echo "REFUSED: unsafe canonical TensorFS build path: $TENSORFS_TREE" >&2
  exit 3
fi
exec 9>"$TENSORFS_TREE.lock"
if ! flock -n 9; then
  echo "REFUSED: another exact H3 TensorFS build owns $TENSORFS_TREE" >&2
  exit 3
fi
if [ -e "$TENSORFS_TREE" ]; then
  rm -rf -- "$TENSORFS_TREE"
fi
cleanup() {
  case "$WORK" in
    "$OUT"/.h3-build.*) rm -rf -- "$WORK" ;;
    *) echo "REFUSED cleanup outside the build root: $WORK" >&2 ;;
  esac
  case "$TENSORFS_TREE" in
    /tmp/cozy-h3-tensorfs-[0-9a-f]*) rm -rf -- "$TENSORFS_TREE" ;;
    *) echo "REFUSED cleanup outside the canonical TensorFS root: $TENSORFS_TREE" >&2 ;;
  esac
}
trap cleanup EXIT

RUNTIME_TREE="$WORK/cozy-runtime"
COZY_EVAL_TREE="$WORK/cozy-eval"
SOURCE_TREE="$WORK/source"
TREE="$WORK/tree"
WHEELS="$WORK/wheels"
mkdir -p "$RUNTIME_TREE" "$TENSORFS_TREE" "$COZY_EVAL_TREE" "$SOURCE_TREE" \
  "$TREE/vendor" "$WHEELS"

# Every peer is read by commit archive. Concurrent worktrees and untracked build products
# cannot enter the release, and no peer checkout is mutated.
git -C "$RUNTIME_REPO" archive "$RUNTIME_SHA" | tar -x -C "$RUNTIME_TREE"
git -C "$TENSORFS_REPO" archive "$TENSORFS_SHA" | tar -x -C "$TENSORFS_TREE"
git -C "$COZY_EVAL_REPO" archive "$COZY_EVAL_SHA" | tar -x -C "$COZY_EVAL_TREE"
# Probe from the whole exact source commit. A hand-maintained script list is another
# closure that drifts; it already omitted h3-keys.py and made the first bundle probe red.
git -C "$ROOT" archive "$SOURCE_SHA" | tar -x -C "$SOURCE_TREE"

RUNTIME_EPOCH="$(git -C "$RUNTIME_REPO" show -s --format=%ct "$RUNTIME_SHA")"
TENSORFS_EPOCH="$(git -C "$TENSORFS_REPO" show -s --format=%ct "$TENSORFS_SHA")"
COZY_EVAL_EPOCH="$(git -C "$COZY_EVAL_REPO" show -s --format=%ct "$COZY_EVAL_SHA")"
SOURCE_DATE_EPOCH="$RUNTIME_EPOCH" nice -n 19 uv build --wheel \
  --python "$BUILD_PYTHON_PATH" --project "$RUNTIME_TREE" \
  --out-dir "$WHEELS" >/dev/null
SOURCE_DATE_EPOCH="$COZY_EVAL_EPOCH" nice -n 19 uv build --wheel \
  --python "$BUILD_PYTHON_PATH" --project "$COZY_EVAL_TREE" \
  --out-dir "$WHEELS" >/dev/null
TENSORFS_BUILD_CACHE="${H3_TENSORFS_BUILD_CACHE:-$HOME/.cache/cozy/se-011-tensorfs-target}"
mkdir -p "$TENSORFS_BUILD_CACHE"
(cd "$TENSORFS_TREE" && SOURCE_DATE_EPOCH="$TENSORFS_EPOCH" \
  CARGO_TARGET_DIR="$TENSORFS_BUILD_CACHE" nice -n 19 maturin build --release \
  --interpreter "$BUILD_PYTHON_PATH" --out "$WHEELS" >/dev/null)

RUNTIME_WHEEL="cozy_runtime-0.0.1-py3-none-any.whl"
TENSORFS_WHEEL="tensorfs-0.0.1-cp311-abi3-manylinux_2_34_x86_64.whl"
COZY_EVAL_WHEEL="cozy_eval-2.3.0-py3-none-any.whl"
for wheel in "$RUNTIME_WHEEL" "$TENSORFS_WHEEL" "$COZY_EVAL_WHEEL"; do
  if [ ! -s "$WHEELS/$wheel" ]; then
    echo "REFUSED: exact source build did not produce $wheel" >&2
    exit 4
  fi
  install -m 0644 "$WHEELS/$wheel" "$TREE/vendor/$wheel"
done

# The committed H3 tree is the release tree. No generated module, dependency heredoc, or
# second handler exists.
cp -a "$SOURCE_TREE/h3/." "$TREE/"
install -m 0644 "$SOURCE_TREE/LICENSE" "$SOURCE_TREE/NOTICE" "$TREE/"

BINDINGS="$(grep -c '^\[bindings\.' "$TREE/endpoint.toml")"
BEFORE="$(grep -c '^release = ' "$TREE/endpoint.toml")"
sed -i "s|^release = .*|release = \"$ARTIFACT_RELEASE\"|" "$TREE/endpoint.toml"
AFTER="$(grep -c "^release = \"$ARTIFACT_RELEASE\"$" "$TREE/endpoint.toml")"
if [ "$BINDINGS" != "$BEFORE" ] || [ "$BINDINGS" != "$AFTER" ]; then
  echo "REFUSED: $BINDINGS bindings, $BEFORE release lines before, $AFTER after" >&2
  exit 4
fi

sha256_file() {
  sha256sum "$1" | cut -d' ' -f1
}
RUNTIME_WHEEL_SHA="$(sha256_file "$TREE/vendor/$RUNTIME_WHEEL")"
TENSORFS_WHEEL_SHA="$(sha256_file "$TREE/vendor/$TENSORFS_WHEEL")"
COZY_EVAL_WHEEL_SHA="$(sha256_file "$TREE/vendor/$COZY_EVAL_WHEEL")"
"$BUILD_PYTHON_PATH" - "$TREE/build-pins.json" \
  "$SOURCE_SHA" "$ARTIFACT_RELEASE" "$BUILD_PYTHON" "$REQUIRED_UV" "$REQUIRED_MATURIN" \
  "$RUNTIME_SHA" "$RUNTIME_WHEEL" "$RUNTIME_WHEEL_SHA" \
  "$TENSORFS_SHA" "$TENSORFS_WHEEL" "$TENSORFS_WHEEL_SHA" \
  "$COZY_EVAL_SHA" "$COZY_EVAL_WHEEL" "$COZY_EVAL_WHEEL_SHA" <<'PY'
import json
import pathlib
import sys

(
    out,
    source_sha,
    artifact_release,
    python_version,
    uv_version,
    maturin_version,
    runtime_sha,
    runtime_wheel,
    runtime_wheel_sha,
    tensorfs_sha,
    tensorfs_wheel,
    tensorfs_wheel_sha,
    eval_sha,
    eval_wheel,
    eval_wheel_sha,
) = sys.argv[1:]
payload = {
    "artifact_release": artifact_release,
    "endpoint_source_sha": source_sha,
    "platform": "linux-x86_64",
    "python_version": python_version,
    "uv_version": uv_version,
    "maturin_version": maturin_version,
    "system_commands": ["ffmpeg", "ffprobe"],
    "torch": {"version": "2.9.1+cu129", "torchvision": "0.24.1+cu129", "cuda": "12.9"},
    "peers": {
        "cozy-runtime": {
            "source_sha": runtime_sha,
            "wheel": runtime_wheel,
            "wheel_sha256": runtime_wheel_sha,
        },
        "tensorfs": {
            "source_sha": tensorfs_sha,
            "wheel": tensorfs_wheel,
            "wheel_sha256": tensorfs_wheel_sha,
        },
        "cozy-eval": {
            "source_sha": eval_sha,
            "wheel": eval_wheel,
            "wheel_sha256": eval_wheel_sha,
        },
    },
}
pathlib.Path(out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

# A lock that parses is not a closure proof. Materialize it, import every direct runtime
# dependency, check the CUDA build identity, derive the descriptor, and run the deterministic
# contract/vision/control-plane probes before any archive is emitted.
uv lock --check --project "$TREE" --python "$BUILD_PYTHON_PATH" >/dev/null
nice -n 19 uv sync --locked --no-install-project --project "$TREE" \
  --python "$BUILD_PYTHON_PATH" >/dev/null
"$TREE/.venv/bin/python" - <<'PY'
import importlib.metadata as metadata
import av
import cozy_eval
import diffusers
import msgspec
import numpy
import PIL
import tensorfs
import torch
import torchvision
import transformers

expected = {
    "cozy-eval": "2.3.0",
    "cozy-runtime": "0.0.1",
    "tensorfs": "0.0.1",
    "torch": "2.9.1+cu129",
    "torchvision": "0.24.1+cu129",
}
actual = {name: metadata.version(name) for name in expected}
if actual != expected:
    raise SystemExit(f"installed direct versions disagree: {actual!r} != {expected!r}")
if torch.version.cuda != "12.9":
    raise SystemExit(f"torch CUDA build is {torch.version.cuda!r}, expected '12.9'")
print("install probe:", actual, "cuda", torch.version.cuda)
PY
"$TREE/.venv/bin/cozy-runtime" describe --dir "$TREE" --check
nice -n 19 "$TREE/.venv/bin/python" "$SOURCE_TREE/scripts/h3-conform.py"
nice -n 19 "$TREE/.venv/bin/python" "$SOURCE_TREE/scripts/h3-vision-conform.py"
nice -n 19 "$TREE/.venv/bin/python" "$SOURCE_TREE/scripts/h3-live.py"

SLUG="${ENDPOINT#*/}"
CANDIDATE="$WORK/$SLUG-$VERSION.tar.gz"
ARCHIVE="$OUT/$SLUG-$VERSION-${SOURCE_SHA:0:12}.tar.gz"
python3 "$SOURCE_TREE/scripts/pack.py" "$TREE" "$ENDPOINT" "$VERSION" "$CANDIDATE" \
  | sed 's/^/  /'
if [ -e "$ARCHIVE" ]; then
  if ! cmp -s "$CANDIDATE" "$ARCHIVE"; then
    echo "REFUSED: $ARCHIVE already exists with different bytes for the same source commit" >&2
    exit 5
  fi
else
  install -m 0644 "$CANDIDATE" "$ARCHIVE"
fi

echo "  archive:          $ARCHIVE"
echo "  endpoint source:  $SOURCE_SHA"
echo "  model release:    $ARTIFACT_RELEASE ($BINDINGS bindings)"
echo "  runtime source:   $RUNTIME_SHA"
echo "  tensorfs source:  $TENSORFS_SHA"
echo "  cozy-eval source: $COZY_EVAL_SHA"
echo "  digest:           sha256:$(sha256_file "$ARCHIVE")"
