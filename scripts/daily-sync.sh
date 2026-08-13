#!/usr/bin/env bash
# daily-sync.sh — 每天结束一键同步：提交本地改动 → 拉取远程 → 推送
# 用法（在 vault 根目录，Git Bash 里）：
#   bash scripts/daily-sync.sh
#
# 机器专属配置（.obsidian 那 6 个）已在 .gitignore，本脚本永远不会误传它们。
# 两台电脑都能用；每天在你干完活的那台跑一次即可。
set -uo pipefail

# 切到脚本上一级 = vault 根目录，保证相对路径正确（无论从哪运行）
cd "$(dirname "$0")/.." || { echo "✗ 无法定位 vault 根目录"; exit 1; }
echo "▶ vault: $(pwd)"

# 1) 暂存所有改动（机器专属配置已忽略，安全）
git add -A

# 2) 有改动才提交
if git diff --cached --quiet; then
  echo "① 本地无改动，跳过提交"
else
  msg="sync: $(date '+%Y-%m-%d %H:%M') @$(hostname)"
  git commit -m "$msg" && echo "① 已提交：$msg"
fi

# 3) 拉取远程，把本地提交叠加在最新代码之上（线性历史）
echo "② 拉取远程 (git pull --rebase)"
if ! git pull --rebase; then
  echo "✗ 拉取时有冲突，已暂停。你的改动已安全提交在本地。"
  echo "  把上面的报错发我，我帮你解冲突后再 push。"
  exit 1
fi

# 4) 推送
echo "③ 推送 (git push)"
if git push; then
  echo "✓ 同步完成"
else
  echo "✗ push 失败，把上面的报错发我。"
  exit 1
fi
