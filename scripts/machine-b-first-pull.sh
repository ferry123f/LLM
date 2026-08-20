#!/usr/bin/env bash
# machine-b-first-pull.sh
# ─────────────────────────────────────────────────────────────
# 「另一台电脑」第一次拉取「配置停同步」那次提交时，运行一次即可。
# 作用：备份本机 .obsidian 里将被停同步的配置 → 安全 pull → 恢复本机配置。
# 之后这些配置永远只留在本机、不再同步，本脚本无需再运行。
#
# 用法（在 vault 根目录，Git Bash 里）：
#   bash scripts/machine-b-first-pull.sh
#
# 注意：不要直接 `git pull` 后才想起来跑本脚本——要先跑本脚本，它内部会替你 pull。
# 获取本脚本而不触发危险 pull 的方法见 README 或对话记录（git fetch + git show）。
# ─────────────────────────────────────────────────────────────

# 需要保护的机器专属文件（与 .gitignore 里停同步的清单一致）
FILES=(
  ".obsidian/app.json"
  ".obsidian/appearance.json"
  ".obsidian/core-plugins.json"
  ".obsidian/community-plugins.json"
  ".obsidian/graph.json"
  ".obsidian/plugins/realclaudian/data.json"
)

# 必须在 git 仓库根目录运行
if [ ! -d .git ]; then
  echo "✗ 请在 vault 根目录（含 .git 的那层）运行本脚本"
  exit 1
fi

echo "① 备份本机配置 → *.bak"
for f in "${FILES[@]}"; do
  if [ -f "$f" ]; then cp -f "$f" "$f.bak" && echo "  备份 $f"; fi
done

echo "② 丢弃这些文件的本地 git 改动（真内容已在 .bak），让 pull 干净通过"
git checkout -- "${FILES[@]}" 2>/dev/null || true

echo "③ git pull"
if git pull; then
  echo "  pull 成功"
else
  echo "  ⚠ pull 未顺利完成——请把上面的报错发我。你的配置已备份在 *.bak，是安全的。"
fi

echo "④ 从备份恢复本机配置（此时它们已被 gitignore，恢复后只留本机、不再同步）"
for f in "${FILES[@]}"; do
  if [ -f "$f.bak" ]; then mv -f "$f.bak" "$f" && echo "  恢复 $f"; fi
done

echo "✓ 完成。现在跑 git status 应显示干净（这 6 个文件已被忽略）。"
