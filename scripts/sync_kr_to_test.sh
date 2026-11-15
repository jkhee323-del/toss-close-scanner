#!/usr/bin/env bash
# Copy committed KRX history from a known main commit to the test branch.
# Optional company-name cache must not break the sync when absent.
set -euo pipefail

source_commit="${1:?Usage: sync_kr_to_test.sh SOURCE_COMMIT [TEST_BRANCH]}"
test_branch="${2:-accuracy-lab}"

if ! git cat-file -e "${source_commit}:data/history" 2>/dev/null; then
  echo "::error::KR history missing from source commit ${source_commit}"
  exit 1
fi

git fetch origin "${test_branch}"
git switch -C "${test_branch}" "origin/${test_branch}"
git restore --source="${source_commit}" --staged --worktree -- data/history

if git cat-file -e "${source_commit}:data/kr_names.csv" 2>/dev/null; then
  git restore --source="${source_commit}" --staged --worktree -- data/kr_names.csv
else
  echo "No KR name cache in source; keeping any test-branch copy unchanged."
fi

if git diff --cached --quiet; then
  echo "No KR cache changes to sync."
else
  git commit -m "Auto-sync Korea market data to test site"
  git push origin "HEAD:${test_branch}"
fi
