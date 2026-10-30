# usage: bash backup_push.sh "<snapshot commit message first line>"
# Stage the small files, merge candidate master into rl exp/memory-chain-v3 (+ the refreshed snapshot), fast-forward
# exp/memory-chain-v3-timing to a-chain, push both. The rl main worktree is not touched (a temporary worktree).
set -e
B=D:/Projects/fortune/Isaac/temp/memory-chain-v3/scripts/backup
W=D:/Projects/fortune/Isaac/temp/rl-exp-memory-chain-v3
python -X utf8 $B/stage_backup.py
cd D:/Projects/fortune/Isaac/rl
git remote add memory-chain-v3-candidate D:/Projects/fortune/Isaac/temp/memory-chain-v3/candidate
git fetch -q memory-chain-v3-candidate
git merge-base --is-ancestor exp/memory-chain-v3-timing memory-chain-v3-candidate/a-chain
git branch -f exp/memory-chain-v3-timing memory-chain-v3-candidate/a-chain
git branch -f exp/memory-chain-v3-replay memory-chain-v3-candidate/replay-v1 2>/dev/null || git branch exp/memory-chain-v3-replay memory-chain-v3-candidate/replay-v1
git branch --unset-upstream exp/memory-chain-v3-replay 2>/dev/null || true
git branch -f exp/memory-chain-v3-truncated memory-chain-v3-candidate/truncated-v1 2>/dev/null || git branch exp/memory-chain-v3-truncated memory-chain-v3-candidate/truncated-v1
git branch --unset-upstream exp/memory-chain-v3-truncated 2>/dev/null || true
git branch --unset-upstream exp/memory-chain-v3-timing 2>/dev/null || true
git worktree add -q $W exp/memory-chain-v3
cd $W
git merge -q --no-ff -m "Merge candidate master $(git rev-parse --short memory-chain-v3-candidate/master)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" memory-chain-v3-candidate/master
rm -rf memory-chain-v3
cp -r D:/Projects/fortune/Isaac/temp/rl-stage-memory-chain-v3/memory-chain-v3 memory-chain-v3
git add -A memory-chain-v3
git commit -q -m "$1

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" 2>&1 | grep -v '^warning' || true
git log --oneline -3
cd D:/Projects/fortune/Isaac/rl
git worktree remove $W
git remote remove memory-chain-v3-candidate
timeout 600 git push origin exp/memory-chain-v3 exp/memory-chain-v3-timing exp/memory-chain-v3-replay exp/memory-chain-v3-truncated 2>&1 | tail -3
git ls-remote origin "refs/heads/exp/memory-chain-v3*"
git branch --show-current; git status --short | wc -l
