#!/usr/bin/env bash
# Publish a built journal to the gh-pages branch under its own folder, keeping the other folders there.
# FOLDER "." publishes the sandbox home page at the site root (its files only; folders are kept).
# The branch holds plain files (no Git LFS), because GitHub Pages serves LFS pointers, not their content.
# Usage: scripts/publish_pages.sh SITE_DIR FOLDER    e.g. build/journals/synthetic_slam synthetic_slam
# Build first: uv run --locked python scripts/build_journals.py
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SITE=$(cd "${1:?site folder}" && pwd)
FOLDER=${2:?publish folder, or . for the home page}
REMOTE=$(git -C "$ROOT" remote get-url origin)
SOURCE=$(git -C "$ROOT" rev-parse --short HEAD)
if [ -n "$(git -C "$ROOT" status --porcelain -- site scripts src previews docs)" ]; then
    echo "Commit the site sources before publishing; the build records the source commit." >&2
    exit 1
fi
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
if git ls-remote --exit-code --heads "$REMOTE" gh-pages >/dev/null 2>&1; then
    GIT_LFS_SKIP_SMUDGE=1 git clone --quiet --depth 1 --branch gh-pages "$REMOTE" "$STAGE"
else
    git -C "$STAGE" init --quiet -b gh-pages
    git -C "$STAGE" remote add origin "$REMOTE"
fi
if [ "$FOLDER" = "." ]; then
    cp -R "$SITE"/. "$STAGE/"
else
    rm -rf "${STAGE:?}/$FOLDER"
    mkdir -p "$STAGE/$FOLDER"
    cp -R "$SITE"/. "$STAGE/$FOLDER/"
fi
touch "$STAGE/.nojekyll"
if [ ! -f "$STAGE/index.html" ]; then
    # Until a journals index exists, the site root points to the first journal.
    cat > "$STAGE/index.html" <<EOF
<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>World Sandbox Journals</title><meta http-equiv="refresh" content="0; url=$FOLDER/">
<style>body{margin:0;background:#f7f5fa;color:#23243a;font:17px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;display:grid;place-items:center;min-height:100vh}a{color:#646da0}</style>
</head><body><p><a href="$FOLDER/">$FOLDER</a></p></body></html>
EOF
fi
git -C "$STAGE" add -A
if git -C "$STAGE" diff --cached --quiet; then
    echo "gh-pages already up to date"
    exit 0
fi
git -C "$STAGE" -c user.name="$(git -C "$ROOT" config user.name)" -c user.email="$(git -C "$ROOT" config user.email)" \
    commit --quiet -m "Publish ${FOLDER/#./home page} from $SOURCE"
git -C "$STAGE" push --quiet origin gh-pages
echo "published ${FOLDER/#./home page} from $SOURCE"
