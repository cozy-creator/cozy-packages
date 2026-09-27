#!/usr/bin/env sh
# Same-account packages name the `tensorhub` uv index and never spell its URL; Creator
# writes it for the command's Hub and account (`cozy package lock`, publish, install).
# CI qualifies the committed locks with raw uv, so it writes the index they are bound to.
#   scripts/bind-account-index.sh <index-url> <project>...
set -eu
url="$1"
shift
for project in "$@"; do
  grep -q 'index = "tensorhub"' "$project/pyproject.toml" || continue
  printf '\n[[tool.uv.index]]\nname = "tensorhub"\nurl = "%s"\nexplicit = true\n' "$url" >>"$project/pyproject.toml"
done
