#!/usr/bin/env bash
# 收集当天 wiki/ 下的笔记改动，输出规整的「文件 → 新增内容」结构。
#
# 用法：  bash collect_diff.sh [vault根目录]
# 默认  ： 当前工作目录
# 环境变量：MAX_LINES  每个文件最多输出多少行（默认 120，超出显式报告截断行数）
#                      写日报够用；觉得信息不足就调高重跑，例如 MAX_LINES=400
#
# 处理三种情况：
#   1. 未提交的改动（工作区 + 暂存区 + 未跟踪的新笔记）
#   2. 当天已提交的改动（--since=midnight），逐 commit 分段
#   3. 两者都有 —— 合并输出，分区标注
#
# 两个已踩过的坑，别改回去：
#   * core.quotePath=false —— 否则中文文件名被转义成 \345\255\246 这种八进制
#   * 每个来源只调一次 git，用 awk 切分 —— 这个 vault 有定时自动 commit
#     （"vault backup: <时间戳>"），先 --name-only 再逐文件 diff 会撞上竞态：
#     列表拿到了，逐文件 diff 时改动已被自动提交走，结果全是空的。
#
# 退出码：
#   0  有改动，已输出
#   1  今天 wiki/ 无改动（调用方应转而询问用户）
#   2  参数或环境错误（非 git 仓库等）

set -uo pipefail

VAULT="${1:-$(pwd)}"
MAX_LINES="${MAX_LINES:-120}"

git_q() { git -c core.quotePath=false -C "$VAULT" "$@"; }

if ! git_q rev-parse --git-dir >/dev/null 2>&1; then
  echo "ERROR: '$VAULT' 不是 git 仓库" >&2
  exit 2
fi

if [ ! -d "$VAULT/wiki" ]; then
  echo "ERROR: '$VAULT/wiki' 不存在" >&2
  exit 2
fi

# 把 unified diff 切成「文件 → 新增行」分段。
# 只留新增行（+），剥前缀、去空行；每文件超过 MAX 行就截断并报告真实行数。
SPLIT_AWK='
function flush() {
  if (file != "" && total > shown)
    printf "…（该文件本次共新增 %d 行，已显示前 %d 行，剩余 %d 行未显示）\n", total, shown, total - shown
}
/^diff --git / {
  flush()
  # "a/PATH b/PATH" —— 两侧路径相同，按长度取中点，这样含空格的中文名也不会切错
  s = substr($0, 12)
  L = int((length(s) - 5) / 2)
  file = substr(s, 3, L)
  printf "\n--- 文件：%s ---\n", file
  total = 0; shown = 0
  next
}
/^new file mode/ { print "（★ 全新文件，以下即全文）"; next }
/^\+\+\+ /       { next }
/^\+/ {
  line = substr($0, 2)
  if (line ~ /^[[:space:]]*$/) next
  total++
  if (shown < MAX) { print line; shown++ }
  next
}
END { flush() }
'

split_diff() { awk -v MAX="$MAX_LINES" "$SPLIT_AWK"; }

HAS_HEAD=0
git_q rev-parse HEAD >/dev/null 2>&1 && HAS_HEAD=1

# ---------- 1. 未提交的改动 ----------
# 一次调用拿到完整 diff，不再逐文件复查
if [ "$HAS_HEAD" = 1 ]; then
  DIRTY_DIFF=$(git_q diff HEAD -- wiki/ 2>/dev/null)
else
  DIRTY_DIFF=$(git_q diff -- wiki/ 2>/dev/null)
fi

UNTRACKED=$(git_q ls-files --others --exclude-standard -- wiki/ 2>/dev/null)

if [ -n "$DIRTY_DIFF" ] || [ -n "$UNTRACKED" ]; then
  echo "=============================================="
  echo "未提交的改动（工作区）"
  echo "=============================================="

  [ -n "$DIRTY_DIFF" ] && printf '%s\n' "$DIRTY_DIFF" | split_diff

  if [ -n "$UNTRACKED" ]; then
    while IFS= read -r f; do
      [ -z "$f" ] && continue
      echo
      echo "--- 文件：$f ---"
      echo "（★ 全新文件，尚未 git add，以下即全文）"
      sed '/^[[:space:]]*$/d' "$VAULT/$f" 2>/dev/null \
        | awk -v MAX="$MAX_LINES" '
            { total++; if (shown < MAX) { print; shown++ } }
            END { if (total > shown)
                    printf "…（该文件共 %d 行，已显示前 %d 行，剩余 %d 行未显示）\n", total, shown, total - shown }'
    done <<< "$UNTRACKED"
  fi
  echo
fi

# ---------- 2. 当天已提交的改动 ----------
COMMITS=$(git_q log --since=midnight --format=%H -- wiki/ 2>/dev/null)

if [ -n "$COMMITS" ]; then
  N=$(printf '%s\n' "$COMMITS" | grep -c .)
  echo "=============================================="
  echo "今天已提交的改动（共 $N 个 commit，从新到旧）"
  echo "=============================================="
  echo
  echo "注意：本 vault 有定时自动备份提交，同一篇笔记的内容常被拆散在多个"
  echo "      commit 里。判断「学了什么」要把这些片段合起来看，不要按 commit 数量"
  echo "      估工作量。"

  while IFS= read -r c; do
    [ -z "$c" ] && continue
    echo
    echo "########## commit $(git_q log -1 --format='%h %ad %s' --date=format:'%H:%M' "$c") ##########"
    git_q show "$c" --format= -- wiki/ 2>/dev/null | split_diff
  done <<< "$COMMITS"
  echo
fi

# ---------- 收尾 ----------
if [ -z "$DIRTY_DIFF" ] && [ -z "$UNTRACKED" ] && [ -z "$COMMITS" ]; then
  echo "NO_CHANGES: 今天 wiki/ 下没有检测到改动"
  exit 1
fi

exit 0
