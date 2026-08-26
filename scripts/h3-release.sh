#!/usr/bin/env bash
# Build and inspect the ONE importable H3 project wheel.
#
# This is deliberately not a serving-bundle author. The canonical bundle,
# WheelhouseManifest, EndpointEnvironmentSpec and InstalledEnvironmentReceipt do not yet
# have an implementation to call. This script stops at the endpoint-owned facts those
# authorities need: deterministic project-wheel bytes, the exact dependency lock and
# descriptor, exact source-built peer wheels, and an installed-wheel describe/import
# transcript. It emits no source archive and no H3-specific image.
#
#   scripts/h3-release.sh --artifact-release <name> [--endpoint org/name]
#       [--runtime-sha <40-hex>] [--tensorfs-sha <40-hex>]
#       [--cozy-eval-sha <40-hex>] [--creator-sha <40-hex>] [--out <dir>]
set -euo pipefail

RUNTIME_REPO="${RUNTIME_REPO:-$HOME/cozy_v2/cozy-runtime}"
TENSORFS_REPO="${TENSORFS_REPO:-$HOME/cozy_v2/tensorfs}"
COZY_EVAL_REPO="${COZY_EVAL_REPO:-$HOME/cozy_v2/cozy-eval}"
CREATOR_REPO="${CREATOR_REPO:-$HOME/cozy_v2/cozy-creator}"

# Full commits only. Abbreviations are moving lookups, not build inputs.
RUNTIME_SHA="${RUNTIME_SHA:-96b8c235fe1ce48e392256610eb47ad318dd6850}"
RUNTIME_FLOOR="d78a9a0fddb34c11c334019cde83df6a5260bdee"
TENSORFS_SHA="${TENSORFS_SHA:-c6c598baca3799508f64b4ef02af542c9181dc27}"
COZY_EVAL_SHA="${COZY_EVAL_SHA:-61560ff5cc0bce2403b7e2bc9e6bff753be58ca1}"
CREATOR_SHA="${CREATOR_SHA:-be714cfd1e8f3324046955a0037be7c9cd46039b}"

BUILD_PYTHON="3.12.12"
REQUIRED_UV="0.9.18"
REQUIRED_MATURIN="1.14.1"
VERSION="${VERSION:-1.0.0}"
OUT="${OUT:-$HOME/.cache/cozy/se-011}"
ENDPOINT="cozy/minimax-h3"
ARTIFACT_RELEASE=""

usage() {
  echo "usage: $0 --artifact-release <name> [--endpoint org/name]" >&2
  echo "          [--runtime-sha <40-hex>] [--tensorfs-sha <40-hex>]" >&2
  echo "          [--cozy-eval-sha <40-hex>] [--creator-sha <40-hex>] [--out <dir>]" >&2
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --endpoint) ENDPOINT="$2"; shift 2 ;;
    --artifact-release) ARTIFACT_RELEASE="$2"; shift 2 ;;
    --runtime-sha) RUNTIME_SHA="$2"; shift 2 ;;
    --tensorfs-sha) TENSORFS_SHA="$2"; shift 2 ;;
    --cozy-eval-sha) COZY_EVAL_SHA="$2"; shift 2 ;;
    --creator-sha) CREATOR_SHA="$2"; shift 2 ;;
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
# Only h3/ enters project_wheel_digest. Unrelated work elsewhere in this shared checkout
# is not an input and must not make this lane absorb or discard somebody else's files.
if ! git -C "$ROOT" diff --quiet -- h3 ||
   ! git -C "$ROOT" diff --cached --quiet -- h3 ||
   [ -n "$(git -C "$ROOT" ls-files --others --exclude-standard -- h3)" ]; then
  echo "REFUSED: $ROOT/h3 is dirty; the project wheel requires one exact source commit" >&2
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
CREATOR_SHA="$(exact_commit "$CREATOR_REPO" "$CREATOR_SHA" cozy-creator)"
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
for command in ffmpeg ffprobe go; do
  if ! command -v "$command" >/dev/null; then
    echo "REFUSED: installed-wheel proof requires $command on PATH" >&2
    exit 3
  fi
done

mkdir -p "$OUT"
WORK="$(mktemp -d "$OUT/.h3-wheel.XXXXXX")"
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
    "$OUT"/.h3-wheel.*) rm -rf -- "$WORK" ;;
    *) echo "REFUSED cleanup outside build root: $WORK" >&2 ;;
  esac
  case "$TENSORFS_TREE" in
    /tmp/cozy-h3-tensorfs-[0-9a-f]*) rm -rf -- "$TENSORFS_TREE" ;;
    *) echo "REFUSED cleanup outside TensorFS root: $TENSORFS_TREE" >&2 ;;
  esac
}
trap cleanup EXIT

RUNTIME_TREE="$WORK/cozy-runtime"
COZY_EVAL_TREE="$WORK/cozy-eval"
CREATOR_TREE="$WORK/cozy-creator"
SOURCE_TREE="$WORK/source"
PROJECT_TREE="$WORK/project"
ENV_INPUT="$WORK/environment-input"
PEER_WHEELS="$WORK/peer-wheels"
mkdir -p "$RUNTIME_TREE" "$TENSORFS_TREE" "$COZY_EVAL_TREE" "$CREATOR_TREE" \
  "$SOURCE_TREE" "$PROJECT_TREE" "$ENV_INPUT/vendor" "$PEER_WHEELS"

# Every repository is read through its exact commit archive. Dirty worktrees and
# concurrent untracked outputs cannot enter the wheel or its inspection environment.
git -C "$RUNTIME_REPO" archive "$RUNTIME_SHA" | tar -x -C "$RUNTIME_TREE"
git -C "$TENSORFS_REPO" archive "$TENSORFS_SHA" | tar -x -C "$TENSORFS_TREE"
git -C "$COZY_EVAL_REPO" archive "$COZY_EVAL_SHA" | tar -x -C "$COZY_EVAL_TREE"
git -C "$CREATOR_REPO" archive "$CREATOR_SHA" | tar -x -C "$CREATOR_TREE"
git -C "$ROOT" archive "$SOURCE_SHA" | tar -x -C "$SOURCE_TREE"
cp -a "$SOURCE_TREE/h3/." "$PROJECT_TREE/"

BINDINGS="$(grep -c '^\[bindings\.' "$PROJECT_TREE/endpoint.toml")"
BEFORE="$(grep -c '^release = ' "$PROJECT_TREE/endpoint.toml")"
sed -i "s|^release = .*|release = \"$ARTIFACT_RELEASE\"|" "$PROJECT_TREE/endpoint.toml"
AFTER="$(grep -c "^release = \"$ARTIFACT_RELEASE\"$" "$PROJECT_TREE/endpoint.toml")"
if [ "$BINDINGS" != "$BEFORE" ] || [ "$BINDINGS" != "$AFTER" ]; then
  echo "REFUSED: $BINDINGS bindings, $BEFORE release rows before, $AFTER after" >&2
  exit 4
fi

# Build the existing Cozy packer from the exact source pin. No local reimplementation and
# no PEP 517 backend participates in project-wheel production.
PACKER="$WORK/cozy"
(cd "$CREATOR_TREE" && CGO_ENABLED=0 nice -n 19 go build -trimpath -buildvcs=false \
  -ldflags "-X github.com/cozy-creator/cozy-creator-v2/internal/app.commit=$CREATOR_SHA" \
  -o "$PACKER" ./cmd/cozy)

mkdir -p "$WORK/pack-a" "$WORK/pack-b"
nice -n 19 "$PACKER" pack "$PROJECT_TREE" --out "$WORK/pack-a" >"$WORK/pack-a.log"
(umask 077; TZ=Pacific/Kiritimati LC_ALL=C SOURCE_DATE_EPOCH=1 \
  nice -n 19 "$PACKER" pack "$PROJECT_TREE" --out "$WORK/pack-b") >"$WORK/pack-b.log"
PROJECT_WHEEL="$(find "$WORK/pack-a" -maxdepth 1 -type f -name '*.whl' -print -quit)"
SECOND_WHEEL="$(find "$WORK/pack-b" -maxdepth 1 -type f -name '*.whl' -print -quit)"
if [ -z "$PROJECT_WHEEL" ] || [ -z "$SECOND_WHEEL" ] || ! cmp -s "$PROJECT_WHEEL" "$SECOND_WHEEL"; then
  echo "REFUSED: two Cozy packer runs did not emit one byte-identical project wheel" >&2
  exit 4
fi

RUNTIME_EPOCH="$(git -C "$RUNTIME_REPO" show -s --format=%ct "$RUNTIME_SHA")"
TENSORFS_EPOCH="$(git -C "$TENSORFS_REPO" show -s --format=%ct "$TENSORFS_SHA")"
COZY_EVAL_EPOCH="$(git -C "$COZY_EVAL_REPO" show -s --format=%ct "$COZY_EVAL_SHA")"
SOURCE_DATE_EPOCH="$RUNTIME_EPOCH" nice -n 19 uv build --wheel \
  --python "$BUILD_PYTHON_PATH" --project "$RUNTIME_TREE" --out-dir "$PEER_WHEELS" >/dev/null
SOURCE_DATE_EPOCH="$COZY_EVAL_EPOCH" nice -n 19 uv build --wheel \
  --python "$BUILD_PYTHON_PATH" --project "$COZY_EVAL_TREE" --out-dir "$PEER_WHEELS" >/dev/null
TENSORFS_BUILD_CACHE="${H3_TENSORFS_BUILD_CACHE:-$HOME/.cache/cozy/se-011-tensorfs-target}"
mkdir -p "$TENSORFS_BUILD_CACHE"
(cd "$TENSORFS_TREE" && SOURCE_DATE_EPOCH="$TENSORFS_EPOCH" \
  CARGO_TARGET_DIR="$TENSORFS_BUILD_CACHE" nice -n 19 maturin build --release \
  --interpreter "$BUILD_PYTHON_PATH" --out "$PEER_WHEELS" >/dev/null)

RUNTIME_WHEEL="cozy_runtime-0.0.1-py3-none-any.whl"
TENSORFS_WHEEL="tensorfs-0.0.1-cp311-abi3-manylinux_2_34_x86_64.whl"
COZY_EVAL_WHEEL="cozy_eval-2.3.0-py3-none-any.whl"
for wheel in "$RUNTIME_WHEEL" "$TENSORFS_WHEEL" "$COZY_EVAL_WHEEL"; do
  if [ ! -s "$PEER_WHEELS/$wheel" ]; then
    echo "REFUSED: exact source build did not produce $wheel" >&2
    exit 4
  fi
  install -m 0644 "$PEER_WHEELS/$wheel" "$ENV_INPUT/vendor/$wheel"
done
install -m 0644 "$PROJECT_TREE/pyproject.toml" "$PROJECT_TREE/uv.lock" "$ENV_INPUT/"

# This is the expected red arm until the canonical WheelhouseManifest and its blobs
# exist: an empty cache plus uv's network-off mode cannot materialize the lock. Success
# here would mean the arm no longer proves the missing platform authority.
mkdir -p "$WORK/empty-cache"
set +e
UV_CACHE_DIR="$WORK/empty-cache" nice -n 19 uv sync --offline --locked \
  --no-install-project --project "$ENV_INPUT" --python "$BUILD_PYTHON_PATH" \
  >"$WORK/offline-red.log" 2>&1
OFFLINE_RC=$?
set -e
if [ "$OFFLINE_RC" -eq 0 ]; then
  echo "REFUSED: empty-cache offline materialization unexpectedly succeeded" >&2
  exit 5
fi
if ! grep -Eiq 'offline|cache|not found' "$WORK/offline-red.log"; then
  echo "REFUSED: offline red arm failed for an unrelated reason" >&2
  sed -n '1,20p' "$WORK/offline-red.log" >&2
  exit 5
fi

# Diagnostic installation is intentionally online/cache-assisted. It proves the actual
# project wheel, descriptor and dependency versions; it does NOT claim the still-missing
# offline EnvMaterializer or InstalledEnvironmentReceipt.
nice -n 19 uv sync --locked --no-install-project --project "$ENV_INPUT" \
  --python "$BUILD_PYTHON_PATH" >/dev/null
nice -n 19 uv pip install --python "$ENV_INPUT/.venv/bin/python" --no-deps "$PROJECT_WHEEL" \
  >/dev/null
SITE_PACKAGES="$("$ENV_INPUT/.venv/bin/python" -I - <<'PY'
import sysconfig
print(sysconfig.get_paths()["purelib"])
PY
)"
"$ENV_INPUT/.venv/bin/python" -I - "$SITE_PACKAGES" "$VERSION" <<'PY'
import importlib.metadata as metadata
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
version = sys.argv[2]
expected = {
    "cozy-eval": "2.3.0",
    "cozy-runtime": "0.0.1",
    "diffusers": "0.40.0",
    "h3-endpoint": version,
    "tensorfs": "0.0.1",
    "torch": "2.9.1+cu129",
    "torchvision": "0.24.1+cu129",
}
actual = {name: metadata.version(name) for name in expected}
if actual != expected:
    raise SystemExit(f"installed versions disagree: {actual!r} != {expected!r}")
for name in ("gates", "h3", "h3_arch", "h3_ref"):
    module = __import__(name)
    origin = pathlib.Path(module.__file__).resolve()
    if not origin.is_relative_to(root):
        raise SystemExit(f"{name} imported from {origin}, outside installed wheel root {root}")
import torch
if torch.version.cuda != "12.9":
    raise SystemExit(f"torch CUDA build is {torch.version.cuda!r}, expected '12.9'")
print("installed wheel:", actual, "cuda", torch.version.cuda, "site", root)
PY
"$ENV_INPUT/.venv/bin/cozy-runtime" describe --dir "$SITE_PACKAGES" --check

sha256_file() {
  sha256sum "$1" | cut -d' ' -f1
}
SLUG="${ENDPOINT#*/}"
DEST="$OUT/$SLUG-$VERSION-$ARTIFACT_RELEASE-${SOURCE_SHA:0:12}"
mkdir -p "$DEST"
FINAL_WHEEL="$DEST/$(basename "$PROJECT_WHEEL")"
if [ -e "$FINAL_WHEEL" ]; then
  if ! cmp -s "$PROJECT_WHEEL" "$FINAL_WHEEL"; then
    echo "REFUSED: $FINAL_WHEEL already has different bytes" >&2
    exit 5
  fi
else
  install -m 0644 "$PROJECT_WHEEL" "$FINAL_WHEEL"
fi

echo "  project wheel:        $FINAL_WHEEL"
echo "  project wheel digest: sha256:$(sha256_file "$FINAL_WHEEL")"
echo "  endpoint source:      $SOURCE_SHA"
echo "  model release:        $ARTIFACT_RELEASE ($BINDINGS bindings)"
echo "  descriptor digest:    sha256:$(sha256_file "$PROJECT_TREE/endpoint.descriptor.json")"
echo "  lock digest:          sha256:$(sha256_file "$PROJECT_TREE/uv.lock")"
echo "  Cozy packer source:   $CREATOR_SHA"
echo "  runtime source:       $RUNTIME_SHA"
echo "  TensorFS source:      $TENSORFS_SHA"
echo "  cozy-eval source:     $COZY_EVAL_SHA"
echo "  PlatformTarget facts: linux/amd64, manylinux_2_34 floor, cp312, cuda, cu129"
echo "  offline red arm:      observed empty-cache --offline refusal (exit $OFFLINE_RC)"
echo "  handoff:              BLOCKED on canonical endpoint_bundle_digest and wheelhouse_manifest_digest"
echo "                        therefore no EndpointEnvironmentSpec/environment_spec_digest was minted"
