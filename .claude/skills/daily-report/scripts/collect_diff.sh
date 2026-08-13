#!/usr/bin/env bash
# 收集当天 wiki/ 下的笔记改动，输出规整的「文件 → 新增内容」结构。
#
# 用法：  bash collect_diff.sh [vault根目录]
# 默认  ： 当前工作目录
# 环境变量：MAX_LINES  单个文件/提交最多输出多少行（默认 200，超出显式报告截断）
#
# 处理三种情况（这正是要固化的判断，不留给模型现推）：
#   1. 未提交的改动（工作区 + 暂存区 + 未跟踪的新笔记）
#   2. 当天已提交的改动（--since=midnight）
#   3. 两者都有 —— 合并输出，分区标注
#
# 退出码：
#   0  有改动，已输出
#   1  今天 wiki/ 无改动（调用方应转而询问用户）
#   2  参数或环境错误（非 git 仓库等）

set -uo pipefail

VAULT="${1:-$(pwd)}"
MAX_LINES="${MAX_LINES:-200}"

# core.quotePath=false —— 否则中文文件名会被转义成 \345\255\246 这种八进制，
# 拿去当路径参数再 diff 一次就取不到内容了。这个坑必须在脚本里堵死。
git_q() { git -c core.quotePath=false -C "$VAULT" "$@"; }

if ! git_q rev-parse --git-dir >/dev/null 2>&1; then
  echo "ERROR: '$VAULT' 不是 git 仓库" >&2
  exit 2
fi

if [ ! -d "$VAULT/wiki" ]; then
  echo "ERROR: '$VAULT/wiki' 不存在" >&2
  exit 2
fi

# 只保留 diff 的新增行，剥掉 +/+++ 前缀，去掉纯空行。
# 这是给模型读的，不需要保留 diff 语法噪声。
added_lines_only() {
  grep '^+' | grep -v '^+++' | sed 's/^+//' | sed '/^[[:space:]]*$/d'
}

# 截断并显式告知 —— 静默截断会让日报误以为「就这么多」。
cap() {
  local label="$1"
  local buf
  buf=$(cat)
  local n
  n=$(printf '%s\n' "$buf" | wc -l | tr -d ' ')
  if [ "$n" -gt "$MAX_LINES" ]; then
    printf '%s\n' "$buf" | head -n "$MAX_LINES"
    echo "…（$label 共 $n 行，已截断，剩余 $((n - MAX_LINES)) 行未显示）"
  else
    printf '%s\n' "$buf"
  fi
}

HAS_HEAD=0
git_q rev-parse HEAD >/dev/null 2>&1 && HAS_HEAD=1

# ---------- 1. 未提交的改动 ----------
if [ "$HAS_HEAD" = 1 ]; then
  UNCOMMITTED=$(git_q diff HEAD --name-only -- wiki/ 2>/dev/null)
else
  UNCOMMITTED=$(git_q diff --name-only -- wiki/ 2>/dev/null)
fi

# 未跟踪的新笔记也要算 —— 新建的文章不在 diff 里
UNTRACKED=$(git_q ls-files --others --exclude-standard -- wiki/ 2>/dev/null)

if [ -n "$UNCOMMITTED" ] || [ -n "$UNTRACKED" ]; then
  echo "=============================================="
  echo "未提交的改动"
  echo "=============================================="
  echo

  if [ -n "$UNCOMMITTED" ]; then
    while IFS= read -r f; do
      [ -z "$f" ] && continue
      echo "--- 文件：$f （已有文件，新增内容如下）---"
      if [ "$HAS_HEAD" = 1 ]; then
        git_q diff HEAD -- "$f" 2>/dev/null | added_lines_only | cap "$f"
      else
        git_q diff -- "$f" 2>/dev/null | added_lines_only | cap "$f"
      fi
      echo
    done <<< "$UNCOMMITTED"
  fi

  if [ -n "$UNTRACKED" ]; then
    while IFS= read -r f; do
      [ -z "$f" ] && continue
      echo "--- 文件：$f （★ 全新文件，全文如下）---"
      sed '/^[[:space:]]*$/d' "$VAULT/$f" 2>/dev/null | cap "$f"
      echo
    done <<< "$UNTRACKED"
  fi
fi

# ---------- 2. 当天已提交的改动 ----------
COMMITS=$(git_q log --since=midnight --format=%H -- wiki/ 2>/dev/null)

if [ -n "$COMMITS" ]; then
  echo "=============================================="
  echo "今天已提交的改动"
  echo "=============================================="
  echo

  while IFS= read -r c; do
    [ -z "$c" ] && continue
    SUBJECT=$(git_q log -1 --format='%h %s' "$c")
    echo "--- commit: $SUBJECT ---"
    echo "改动文件："
    git_q show "$c" --format= --name-only -- wiki/ 2>/dev/null | sed 's/^/  /'
    echo "新增内容："
    git_q show "$c" --format= -- wiki/ 2>/dev/null | added_lines_only | cap "commit $SUBJECT"
    echo
  done <<< "$COMMITS"
fi

# ---------- 收尾 ----------
if [ -z "$UNCOMMITTED" ] && [ -z "$UNTRACKED" ] && [ -z "$COMMITS" ]; then
  echo "NO_CHANGES: 今天 wiki/ 下没有检测到改动"
  exit 1
fi

exit 0
