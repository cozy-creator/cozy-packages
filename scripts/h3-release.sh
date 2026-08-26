#!/usr/bin/env bash
# Build the H3 ENDPOINT RELEASE — the archive `cozy install --from --digest` verifies.
#
#   scripts/h3-release.sh [--artifact-release <name>] [--endpoint org/name]
#                         [--runtime-sha <sha>] [--out <dir>]
#
# The endpoint SOURCE is this repo's `h3/` directory, unmodified: one module, its
# `h3_arch/` model library, its `endpoint.toml`, its committed descriptor, and the bundled
# Qwen vocabulary. Everything else is the CLOSURE — the two wheels that are not on an index
# and the lock over all of it.
#
# THE FOOT-GUN THIS SCRIPT IS WRITTEN AGAINST. se-008 found that rebuilding an fp8 archive
# needs `--artifact-release fp8` and not just `--endpoint`, because otherwise `endpoint.toml`
# still selects the fp16 artifact and the release is silently the wrong rung. H3's analogue
# is WORSE, because it has TWO binding tables — `Fl2VAModel` and `Ref2VAModel` — and a
# rewrite that edited one and not the other would produce a release whose two task slots
# bind DIFFERENT artifacts, which is a legal deployment and not the one anyone asked for.
# So the rewrite here is applied to EVERY `release =` line and then COUNTED: if the number
# of lines rewritten is not the number of binding tables, this script refuses rather than
# shipping a half-rewritten selection. The count is printed on every build, green or not.
#
# TWO PINS, both by read-only `git archive`, because a verification whose peer can move
# underneath it measures nothing.
set -euo pipefail

RUNTIME_REPO="${RUNTIME_REPO:-$HOME/cozy_v2/cozy-runtime}"
# f1625f9 is the FLOOR, not a preference: it is the record-key break
# (`entrypoint_binding_plan_id`, `model_binding_path`, `model_parameter_name`, real lists),
# and a release pinned before it speaks a dead record. Anything at or after is legal.
RUNTIME_SHA="${RUNTIME_SHA:-be33325}"
RUNTIME_FLOOR="${RUNTIME_FLOOR:-f1625f9}"
TENSORFS_WHEEL="${TENSORFS_WHEEL:-/tmp/cozy-wheels-cl003/tensorfs-0.0.1-cp311-abi3-linux_x86_64.whl}"
OUT="${OUT:-$HOME/.cache/cozy/se-001}"
VERSION="${VERSION:-1.0.0}"
ENDPOINT="cozy/minimax-h3"
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
# THE PIN FLOOR, CHECKED rather than commented: an ancestor of the record-key break speaks
# a record the coordinator no longer writes, and the failure would appear as a binding that
# resolves to nothing on a rented card.
if ! git -C "$RUNTIME_REPO" merge-base --is-ancestor "$RUNTIME_FLOOR" "$FULL"; then
  echo "REFUSED: runtime pin $RUNTIME_SHA does not contain $RUNTIME_FLOOR — the record-key" >&2
  echo "         break. A release pinned before it binds against a dead record shape." >&2
  exit 3
fi

# READ-ONLY. That repo has concurrent writers; a checkout of its working tree would make
# this release a photograph of somebody else's edit.
git -C "$RUNTIME_REPO" archive "$FULL" | tar -x -C "$RT"

nice -n 19 uv build --wheel --project "$RT" --out-dir "$T/vendor" >/dev/null
rm -f "$T/vendor/.gitignore"
cp "$TENSORFS_WHEEL" "$T/vendor/"
TFS_WHEEL_NAME="$(basename "$TENSORFS_WHEEL")"

# The endpoint's own source, copied whole. `h3/` IS the release: no generated module, no
# rewritten import, no second copy of the handler anywhere.
cp "$ROOT/h3/h3.py" "$ROOT/h3/endpoint.toml" "$ROOT/h3/endpoint.descriptor.json" "$T/"
cp "$ROOT/LICENSE" "$ROOT/NOTICE" "$T/"
mkdir -p "$T/h3_arch" "$T/h3_ref" "$T/tokenizer"
cp "$ROOT/h3/h3_arch"/*.py "$T/h3_arch/"
cp "$ROOT/h3/h3_ref"/*.py "$T/h3_ref/"
cp "$ROOT/h3/tokenizer"/* "$T/tokenizer/"

BINDINGS="$(grep -c '^\[bindings\.' "$T/endpoint.toml")"
if [ -n "$ARTIFACT_RELEASE" ]; then
  BEFORE="$(grep -c '^release = ' "$T/endpoint.toml")"
  sed -i "s|^release = .*|release = \"$ARTIFACT_RELEASE\"|" "$T/endpoint.toml"
  AFTER="$(grep -c "^release = \"$ARTIFACT_RELEASE\"$" "$T/endpoint.toml")"
  if [ "$BEFORE" != "$BINDINGS" ] || [ "$AFTER" != "$BINDINGS" ]; then
    echo "REFUSED: $BINDINGS binding tables, $BEFORE release lines before, $AFTER after —" >&2
    echo "         a partial rewrite would bind the two task slots to different artifacts." >&2
    exit 4
  fi
fi
echo "  bindings: $BINDINGS table(s), all selecting $(grep -m1 '^release = ' "$T/endpoint.toml" | cut -d'"' -f2)"

cat > "$T/pyproject.toml" <<TOML
# The endpoint as a RELEASE. \`cozy install\` builds its venv with \`uv sync --locked\`
# against this lock, so the cozy-runtime that serves the endpoint is the one the release
# pinned — never the host's.
#
# TWO MODEL LIBRARIES SHIP HERE FOR NOW, and that is a migration state rather than a
# design. \`h3_ref/\` is the REFERENCE one (#531): a construction layer over the official
# H3 implementation diffusers merged on 2026-08-05, which is where every architecture
# question is answered from now on. \`h3_arch/\` is the hand port it replaces, kept only
# until the upstream path is proven against it on a card — the port is oracle-proven and
# the replacement is not yet, and deleting proven code ahead of its unproven successor is
# how a licence fix becomes a correctness regression. NOTICE records the terms; the port
# is GPL-adapted and the replacement is not, which is the second reason it goes.
#
# THE DIFFUSERS PIN IS A SHA, NEVER A BRANCH: \`main\` moves and builds must be
# reproducible. This is the same commit v1's H3 endpoint pins, chosen for that reason and
# not for its date — it is the first pin both lanes can be compared at, and a newer SHA
# would fold unreviewed upstream drift into a commit whose whole claim is a rebase onto
# reviewed upstream code. It ships \`MiniMaxH3Transformer3DModel\`,
# \`AutoencoderKLMiniMaxH3\`, \`AutoencoderKLMiniMaxH3Audio\` and \`MiniMaxH3Scheduler\`;
# it collapses to \`diffusers>=0.40\` once 0.40.0 ships them in a release.
#
# TRANSFORMERS MOVED TO 5.x, and it is a FIX rather than a bump: the conditioner is
# Qwen3-VL, whose \`Qwen3VLForConditionalGeneration\` does not exist below 5.x. The
# previous \`<5\` cap was a floor copied from a family that does not use it, and an archive
# built under it could not have constructed the conditioner at all.
[project]
name = "h3-endpoint"
version = "$VERSION"
requires-python = ">=3.11"
dependencies = [
    "cozy-runtime==0.0.1",
    "tensorfs==0.0.1",
    "torch>=2.6",
    "transformers>=5.13,<6",
    "diffusers @ git+https://github.com/huggingface/diffusers@50e7158093710f9c1b4ea9ff100137a91c9228f3",
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
echo "  runtime:  $FULL (git archive, read-only; contains $RUNTIME_FLOOR)"
echo "  tensorfs: $TFS_WHEEL_NAME"
echo "  binding:  $(grep -E '^(repo|release) =' "$T/endpoint.toml" | tr '\n' ' ')"
