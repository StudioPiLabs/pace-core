#!/usr/bin/env bash
# Build the field specification: regenerate its body from the schema, then
# typeset. The body is never edited by hand; tests/test_spec_document.py fails
# when the checked-in copy and the generator disagree.
set -euo pipefail
here="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$here/../.."
uv run python -m pace_core.spec_appendix -o docs/spec/spec_appendix.tex
cd docs/spec
for _pass in 1 2; do
  xelatex -interaction=nonstopmode -halt-on-error pace_spec.tex > /dev/null
done
if grep -Eq 'Overfull \\[hv]box|undefined references|Fatal error' pace_spec.log; then
  echo 'pace_spec built with overfull boxes, undefined references or errors' >&2
  exit 1
fi
echo "built docs/spec/pace_spec.pdf ($(pdfinfo pace_spec.pdf | awk '/^Pages/{print $2}') pages)"
