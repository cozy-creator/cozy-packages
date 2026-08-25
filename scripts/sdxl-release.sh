#!/usr/bin/env bash
# Build the SDXL ENDPOINT RELEASE — the archive `cozy install --from --digest` verifies.
#
#   scripts/sdxl-release.sh [--artifact-release <name>] [--endpoint org/name]
#                           [--runtime-sha <sha>] [--out <dir>]
#
# The endpoint SOURCE is this repo's `sdxl/` directory, unmodified: one module, its
# `endpoint.toml`, its committed descriptor, and the two bundled tokenizer vocabularies.
# Everything else in the tree is the CLOSURE — the two wheels that are not on an index yet
# and the lock over all of it.
#
# `--artifact-release` rewrites the `[bindings]` table's `release=`, which is how the SAME
# endpoint source is released against a different RUNG of the same model repo (`fp8` is
# cr-006's encoded UNet over the same three other components). Bindings state SELECTION;
# the source is byte-identical across rungs, which is the point of the two-sided contract.
#
# TWO PINS, both by read-only `git archive`, because a verification whose peer can move
# underneath it measures nothing — and because the tensorfs wheel is where cozytensors'
# ENCODING REGISTRY lives (decisions #391), so the release's pinned tensorfs is what
# decides which encodings this endpoint can read.
set -euo pipefail

RUNTIME_REPO="${RUNTIME_REPO:-$HOME/cozy_v2/cozy-runtime}"
# 8f58549: cr-008b/cr-007's defect-fix head. It is what makes a SERVING endpoint survive
# its second, third and Nth 1024px request (decisions #401/#402), which is se-008's whole
# subject — an endpoint that renders once is a demo.
RUNTIME_SHA="${RUNTIME_SHA:-8f58549}"
TENSORFS_WHEEL="${TENSORFS_WHEEL:-/tmp/cozy-wheels-cl003/tensorfs-0.0.1-cp311-abi3-linux_x86_64.whl}"
OUT="${OUT:-$HOME/.cache/cozy/se-008}"
VERSION="${VERSION:-1.0.0}"
ENDPOINT="cozy/sdxl"
ARTIFACT_RELEASE=""

while [ $# -gt 0 ]; do
  case "$1" in
    --endpoint) ENDPOINT="$2"; shift 2 ;;
    --artifact-release) ARTIFACT_RELEASE="$2"; shift 2 ;;
    --runtime-sha) RUNTIME_SHA="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    *) echo "usage: $0 [--endpoint org/name] [--artifact-release <name>] [--runtime-sha <sha>] [--out <dir>]" >&2; exit 2 ;;
  esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SLUG="${ENDPOINT#*/}"
RT="$OUT/runtime"
T="$OUT/tree-$SLUG"
rm -rf "$RT" "$T"
mkdir -p "$RT" "$T/vendor"

FULL="$(git -C "$RUNTIME_REPO" rev-parse "$RUNTIME_SHA")"
# READ-ONLY. That repo has concurrent writers; a checkout of its working tree would make
# this release a photograph of somebody else's edit.
git -C "$RUNTIME_REPO" archive "$FULL" | tar -x -C "$RT"

nice -n 19 uv build --wheel --project "$RT" --out-dir "$T/vendor" >/dev/null
rm -f "$T/vendor/.gitignore"
cp "$TENSORFS_WHEEL" "$T/vendor/"
TFS_WHEEL_NAME="$(basename "$TENSORFS_WHEEL")"

# The endpoint's own source, copied whole. `sdxl/` IS the release: there is no generated
# module, no rewritten import and no second copy of the handler anywhere.
cp "$ROOT/sdxl/sdxl.py" "$ROOT/sdxl/endpoint.toml" "$ROOT/sdxl/endpoint.descriptor.json" "$T/"
for sub in tokenizer tokenizer_2; do
  mkdir -p "$T/$sub"
  cp "$ROOT/sdxl/$sub"/* "$T/$sub/"
done

if [ -n "$ARTIFACT_RELEASE" ]; then
  sed -i "s|^release = .*|release = \"$ARTIFACT_RELEASE\"|" "$T/endpoint.toml"
fi

cat > "$T/pyproject.toml" <<TOML
# The endpoint as a RELEASE. \`cozy install\` builds its venv with \`uv sync --locked\`
# against this lock, so the cozy-runtime that serves the endpoint is the one the release
# pinned — never the host's.
#
# The MODEL LIBRARIES are the endpoint's own dependencies and are declared here as such
# (cozy-runtime.md §1.1: the runtime defines no model architecture). cr-008b's corpus
# fixture reached them through \`cozy-runtime[corpus]\`, which is the RUNTIME's fixture
# extra; a product endpoint that depended on it would be pinning its architecture to
# somebody else's test closure.
[project]
name = "sdxl-endpoint"
version = "$VERSION"
requires-python = ">=3.11"
dependencies = [
    "cozy-runtime==0.0.1",
    "tensorfs==0.0.1",
    "torch>=2.6",
    "diffusers>=0.36",
    "transformers>=4.40,<5",
]

# uv records path sources RELATIVE to the project root, so an in-tree wheel relocates with
# the archive and an out-of-tree one does not (cl-009's finding).
[tool.uv.sources]
cozy-runtime = { path = "vendor/cozy_runtime-0.0.1-py3-none-any.whl" }
tensorfs = { path = "vendor/$TFS_WHEEL_NAME" }
TOML

( cd "$T" && nice -n 19 uv lock --quiet )

ARCHIVE="$OUT/$SLUG-$VERSION.tar.gz"
nice -n 19 python3 "$ROOT/scripts/pack.py" "$T" "$ENDPOINT" "$VERSION" "$ARCHIVE" | sed 's/^/  /'
echo "  runtime:  $FULL (git archive, read-only)"
echo "  tensorfs: $TFS_WHEEL_NAME"
echo "  binding:  $(grep -E '^(repo|release) =' "$T/endpoint.toml" | tr '\n' ' ')"
